from __future__ import annotations

import csv
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pickle

import torch
from model_cnn import (
    build_cnn_model,
    build_simple_cnn_model,
    build_myrtle_cnn_model,
    load_pretrained_cnn,
    make_optimizer_and_scheduler,
    make_criterion,
    train_epoch_cnn,
    TorchModelAdapter,
    _EpochShuffler,
)

from generate_dataset import DatasetBundle, build_dataset_bundle
from classifier_metrics import (
    append_classifier_csv_row,
    collect_advanced_classifier_metrics,
    collect_classifier_epoch,
    finalize_classifier_dashboard,
    finalize_classifier_simplified,
    initialize_classifier_csv,
)
from metrics import (
    plot_example_distribution_dynamics,
    plot_2d_decision_boundaries,
    plot_training_report,
)
from neural_collapse import (
    append_nc_csv_row,
    build_nc_class_pairs,
    collect_nc_raw_epoch,
    finalize_nc_metrics,
    initialize_nc_csv,
)
from separability_measures import (
    SepEpochRaw,
    append_sep_csv_row,
    collect_sep_raw_epoch,
    finalize_sep_metrics,
    initialize_sep_csv,
)
from PCA_analysis import (
    append_pca_csv_row,
    collect_pca_epoch,
    finalize_pca_analysis,
    initialize_pca_csv,
    normalize_projected_dims,
)
from PCA_geometric_overlapping import (
    append_geo_csv_row,
    collect_geo_epoch,
    finalize_geo_plots,
    finalize_geo_plots_simplified,
    initialize_geo_csv,
)
from hyperplanes import (
    append_hyperplane_csv_row,
    collect_hyperplane_epoch,
    finalize_hyperplane_plots,
    initialize_hyperplane_csv,
)

from projection_PCA_analysis import (
        append_proj_nc_csv_row,
        collect_proj_nc_epoch,
        finalize_proj_nc_plots,
        initialize_proj_nc_csv,
    )

from metrics_for_multiple_classes import (
    append_multiclass_csv_row,
    collect_multiclass_epoch,
    finalize_multiclass_plots,
    finalize_shape_metrics_plots,
    initialize_multiclass_csv,
    #compute_batched_logits,
)

from model import JAXModel, ParamTree, build_model


@dataclass
class RunningStats:
    loss_sum: float
    correct_sum: int
    count_sum: int
    class_loss_sum: np.ndarray
    class_correct_sum: np.ndarray
    class_count_sum: np.ndarray


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _analysis_output_enabled(
    analysis_cfg: Dict[str, Any],
    keys: tuple[str, ...],
    default: bool = True,
) -> bool:
    outputs_cfg = analysis_cfg.get("outputs", analysis_cfg.get("metrics", {}))
    if isinstance(outputs_cfg, dict):
        for key in keys:
            if key in outputs_cfg:
                return _as_bool(outputs_cfg[key])
    for key in keys:
        if key in analysis_cfg:
            return _as_bool(analysis_cfg[key])
    return default


def _normalize_plot_version(raw: Any) -> str:
    if raw is None:
        return "full"
    if isinstance(raw, str):
        text = raw.strip().lower()
        if text in {"", "full"}:
            return "full"
        if text == "simplified":
            return "simplified"
    raise ValueError("analysis.plot_version must be 'full' or 'simplified'.")


def _normalize_freeze_part(raw: Any) -> str | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        text = raw.strip().lower()
        if text in {"", "none", "null", "false", "off"}:
            return None
        if text in {"classifier", "head"}:
            return "classifier"
        if text in {"preclassifier", "feature_extractor", "features", "hidden_layers", "encoder"}:
            return "preclassifier"
    raise ValueError(
        "training.freeze_part must be one of: classifier, preclassifier (or feature_extractor)."
    )


def _parse_freeze_after_steps(raw: Any) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, str) and raw.strip().lower() in {"", "none", "null"}:
        return None
    steps = int(raw)
    if steps < 0:
        raise ValueError("training.freeze_after_steps must be >= 0")
    return steps


def _build_grad_mask(params: ParamTree, frozen_part: str | None) -> ParamTree:
    mask = jax.tree_util.tree_map(lambda x: jnp.ones_like(x), params)
    if frozen_part == "classifier":
        mask["classifier"] = jax.tree_util.tree_map(
            lambda x: jnp.zeros_like(x),
            params["classifier"],
        )
    elif frozen_part == "preclassifier":
        mask["hidden_layers"] = jax.tree_util.tree_map(
            lambda x: jnp.zeros_like(x),
            params["hidden_layers"],
        )
    return mask


def _apply_grad_mask(grads: ParamTree, mask: ParamTree) -> ParamTree:
    return jax.tree_util.tree_map(lambda g, m: g * m, grads, mask)


def _parse_bump_order(raw: Any) -> list[int] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        text = raw.strip()
        if text == "" or text.lower() in {"none", "null"}:
            return None
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        if text.strip() == "":
            return []
        parts = [part.strip() for part in text.split(",") if part.strip()]
        return [int(part) for part in parts]
    if isinstance(raw, (list, tuple, np.ndarray)):
        return [int(part) for part in raw]
    raise ValueError("dynamics.bump_order must be null, a list of ints, or a comma-separated string.")


def _validate_bump_order(bump_order: list[int] | None, num_classes: int) -> list[int] | None:
    if bump_order is None:
        return None
    if len(bump_order) != num_classes:
        raise ValueError(
            f"dynamics.bump_order must have length equal to num_classes ({num_classes}). "
            f"Got length {len(bump_order)}."
        )
    for idx, class_id in enumerate(bump_order):
        if class_id < 0 or class_id >= num_classes:
            raise ValueError(
                f"dynamics.bump_order has invalid class id {class_id} at index {idx}. "
                f"Valid range is [0, {num_classes - 1}]."
            )
    return bump_order


def _build_class_to_indices(targets: np.ndarray, num_classes: int) -> Dict[int, np.ndarray]:
    return {
        class_id: np.where(targets == class_id)[0].astype(np.int64, copy=False)
        for class_id in range(num_classes)
    }


def _build_metric_subset(
    inputs: np.ndarray,
    targets: np.ndarray,
    class_to_indices: Dict[int, np.ndarray],
    num_classes: int,
    subset_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, Dict[int, np.ndarray]]:
    total_samples = int(targets.shape[0])
    subset_size = min(int(subset_size), total_samples)
    if subset_size <= 0:
        raise ValueError("analysis.metric_subset_size must be > 0")

    if subset_size < num_classes:
        chosen = rng.choice(total_samples, size=subset_size, replace=False)
        subset_inputs = inputs[chosen]
        subset_targets = targets[chosen]
        return subset_inputs, subset_targets, _build_class_to_indices(subset_targets, num_classes)

    counts = np.array(
        [len(class_to_indices[class_id]) for class_id in range(num_classes)],
        dtype=np.int64,
    )
    base = subset_size // num_classes
    per_class = np.minimum(base, counts)
    remaining = subset_size - int(per_class.sum())
    if remaining > 0:
        available = counts - per_class
        if int(available.sum()) > 0:
            extra = rng.multinomial(remaining, available / available.sum())
            per_class += extra

    chosen_indices: list[np.ndarray] = []
    for class_id, count in enumerate(per_class):
        if count <= 0:
            continue
        class_indices = class_to_indices[class_id]
        replace = int(count) > class_indices.shape[0]
        chosen = rng.choice(class_indices, size=int(count), replace=replace)
        chosen_indices.append(chosen.astype(np.int64, copy=False))

    if not chosen_indices:
        raise ValueError("Metric subset selection produced an empty sample.")

    chosen_all = np.concatenate(chosen_indices, axis=0)
    chosen_all = rng.permutation(chosen_all)
    subset_inputs = inputs[chosen_all]
    subset_targets = targets[chosen_all]
    return subset_inputs, subset_targets, _build_class_to_indices(subset_targets, num_classes)


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _devices_for(platform: str) -> list[jax.Device]:
    try:
        return list(jax.devices(platform))
    except Exception:
        return []


def _pin_jax_platform(platform: str) -> None:
    normalized = str(platform).strip().lower()
    if normalized == "gpu":
        normalized = "cuda"
    if normalized not in {"cpu", "cuda"}:
        return

    os.environ["JAX_PLATFORMS"] = normalized
    os.environ["JAX_PLATFORM_NAME"] = normalized
    try:
        jax.config.update("jax_platforms", normalized)
    except Exception:
        pass
    try:
        jax.config.update("jax_platform_name", normalized)
    except Exception:
        pass


def _resolve_device(device_raw: Any) -> jax.Device:
    requested = str(device_raw if device_raw is not None else "auto").strip().lower()

    if requested in {"auto", ""}:
        gpu_devices = _devices_for("gpu")
        if gpu_devices:
            return gpu_devices[0]
        _pin_jax_platform("cpu")
        return jax.devices("cpu")[0]

    if requested in {"cuda", "gpu"}:
        _pin_jax_platform("cuda")
        gpu_devices = _devices_for("gpu")
        if not gpu_devices:
            raise RuntimeError("Config requested CUDA/GPU but no GPU device is available to JAX.")
        return gpu_devices[0]

    if requested == "cpu":
        _pin_jax_platform("cpu")
        return jax.devices("cpu")[0]

    raise ValueError(f"Unsupported device '{requested}'. Use auto, cuda/gpu, or cpu.")


def _device_label(device: jax.Device) -> str:
    return f"{device.platform}:{device.id}"


def _init_stats(num_classes: int) -> RunningStats:
    return RunningStats(
        loss_sum=0.0,
        correct_sum=0,
        count_sum=0,
        class_loss_sum=np.zeros(num_classes, dtype=np.float64),
        class_correct_sum=np.zeros(num_classes, dtype=np.float64),
        class_count_sum=np.zeros(num_classes, dtype=np.float64),
    )


def _update_stats(
    stats: RunningStats,
    per_sample_loss: jax.Array,
    preds: jax.Array,
    target: jax.Array,
    num_classes: int,
    compute_per_class: bool = True, # <-- New flag
) -> None:
    loss_np = np.asarray(per_sample_loss, dtype=np.float64)
    preds_np = np.asarray(preds, dtype=np.int64)
    target_np = np.asarray(target, dtype=np.int64)
    correct_np = (preds_np == target_np).astype(np.float64)

    stats.loss_sum += float(np.sum(loss_np))
    stats.correct_sum += int(np.sum(correct_np))
    stats.count_sum += int(target_np.shape[0])

    if compute_per_class:
        class_count = np.bincount(target_np, minlength=num_classes).astype(np.float64)
        class_loss = np.bincount(target_np, weights=loss_np, minlength=num_classes).astype(np.float64)
        class_correct = np.bincount(target_np, weights=correct_np, minlength=num_classes).astype(np.float64)

        stats.class_count_sum += class_count
        stats.class_loss_sum += class_loss
        stats.class_correct_sum += class_correct


def _finalize_stats(stats: RunningStats) -> Dict[str, Any]:
    denom = max(1, stats.count_sum)
    accuracy = stats.correct_sum / denom
    training_error = 1.0 - accuracy
    per_class_loss = np.full_like(stats.class_loss_sum, np.nan)
    per_class_acc = np.full_like(stats.class_correct_sum, np.nan)
    valid = stats.class_count_sum > 0
    per_class_loss[valid] = stats.class_loss_sum[valid] / stats.class_count_sum[valid]
    per_class_acc[valid] = stats.class_correct_sum[valid] / stats.class_count_sum[valid]

    return {
        "loss": stats.loss_sum / denom,
        "accuracy": accuracy,
        "training_error": training_error,
        "zero_training_error": bool(stats.count_sum > 0 and stats.correct_sum == stats.count_sum),
        "per_class_loss": per_class_loss,
        "per_class_accuracy": per_class_acc,
    }


def _compute_focus_weight(period_step: int, period_length: int, w_max: float) -> tuple[float, float]:
    t = float(period_step)
    period = float(period_length)
    slope = 2.0 * (w_max - 1.0) / period
    if t < period / 2.0:
        focus_weight = 1.0 + t * slope
    else:
        focus_weight = 2.0 * w_max - t * slope - 1.0
    phase = t / period
    return focus_weight, phase


def _compute_class_probabilities(
    step: int,
    num_classes: int,
    period_length: int,
    w_max: float,
    bump_order: list[int] | None = None,
) -> tuple[np.ndarray, int, float, float]:
    if bump_order is None:
        focus_class = (step // period_length) % num_classes
    else:
        cycle_index = (step // period_length) % len(bump_order)
        focus_class = int(bump_order[cycle_index])
    period_step = step % period_length
    focus_weight, bump_phase = _compute_focus_weight(period_step, period_length, w_max)

    weights = np.ones(num_classes, dtype=np.float64)
    weights[focus_class] = focus_weight
    probabilities = weights / weights.sum()
    return probabilities, int(focus_class), float(focus_weight), float(bump_phase)


def _sample_batch_by_class_counts(
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    class_to_indices: Dict[int, np.ndarray],
    class_counts: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    class_counts_int = np.asarray(class_counts, dtype=np.int64)
    sampled_index_parts: list[np.ndarray] = []
    for class_id, count in enumerate(class_counts_int):
        if count <= 0:
            continue
        class_indices = class_to_indices[class_id]
        chosen = rng.choice(class_indices, size=int(count), replace=True)
        sampled_index_parts.append(chosen.astype(np.int64, copy=False))

    if not sampled_index_parts:
        raise ValueError(
            "Deterministic class_counts produced an empty batch. "
            "Increase training.batch_size or adjust dynamics settings."
        )

    sampled_indices = np.concatenate(sampled_index_parts, axis=0)
    return train_inputs[sampled_indices], train_targets[sampled_indices], class_counts_int


def _iterate_minibatches(
    inputs: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    num_samples = int(inputs.shape[0])
    for start in range(0, num_samples, batch_size):
        end = min(num_samples, start + batch_size)
        yield inputs[start:end], targets[start:end]


def _collect_pre_classifier_outputs(
    model: JAXModel,
    params: ParamTree,
    inputs: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    device: jax.Device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    logits_list: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    for batch_inputs, batch_targets in _iterate_minibatches(inputs, targets, batch_size):
        x = jax.device_put(jnp.asarray(batch_inputs, dtype=jnp.float32), device=device)
        logits, intermediates = model.apply(params, x, return_intermediates=True)
        pre_classifier = intermediates.get("pre_classifier")
        if pre_classifier is None:
            raise RuntimeError("Failed to capture activations for layer 'pre_classifier'.")
        features.append(np.asarray(pre_classifier, dtype=np.float64))
        logits_list.append(np.asarray(logits, dtype=np.float64))
        labels.append(np.asarray(batch_targets, dtype=np.int64))

    if not features:
        raise ValueError("No samples provided for pre-classifier collection.")

    features_arr = np.concatenate(features, axis=0)
    logits_arr = np.concatenate(logits_list, axis=0)
    labels_arr = np.concatenate(labels, axis=0)
    if features_arr.shape[0] != inputs.shape[0]:
        raise ValueError("Collected activations do not match expected sample count.")

    return features_arr, logits_arr, labels_arr


def evaluate_arrays(
    params: ParamTree,
    predict_step: Callable[[ParamTree, jax.Array], jax.Array],
    inputs: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    device: jax.Device,
    num_classes: int,
    compute_per_class: bool = True, # <-- New flag
) -> Dict[str, Any]:
    stats = _init_stats(num_classes)

    for batch_inputs, batch_targets in _iterate_minibatches(inputs, targets, batch_size):
        x = jax.device_put(jnp.asarray(batch_inputs, dtype=jnp.float32), device=device)
        y = jax.device_put(jnp.asarray(batch_targets, dtype=jnp.int32), device=device)
        logits = predict_step(params, x)
        per_sample_loss = optax.softmax_cross_entropy_with_integer_labels(logits, y)
        preds = jnp.argmax(logits, axis=1)
        
        # Pass the flag down
        _update_stats(stats, per_sample_loss, preds, y, num_classes, compute_per_class)

    return _finalize_stats(stats)


def _l2_penalty(model: JAXModel, params: ParamTree) -> jax.Array:
    total = jnp.array(0.0, dtype=jnp.float32)
    for _name, param in model.iter_named_parameters(params):
        total = total + jnp.sum(jnp.square(param))
    return total


def _classifier_weight_grads(
    pre_classifier: jax.Array,
    logits: jax.Array,
    targets: jax.Array,
    num_classes: int,
) -> jax.Array:
    probs = jax.nn.softmax(logits, axis=1)
    one_hot = jax.nn.one_hot(targets, num_classes, dtype=logits.dtype)
    diff = probs - one_hot
    grad_kernel = jnp.einsum("bi,bj->bij", pre_classifier, diff)
    return jnp.transpose(grad_kernel, (0, 2, 1))


def train_epoch(
    params: ParamTree,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    model: JAXModel,
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    class_to_indices: Dict[int, np.ndarray],
    batch_size: int,
    device: jax.Device,
    num_classes: int,
    global_step: int,
    steps_this_epoch: int,
    total_steps: int,
    bumps_enabled: bool,
    bumps_after_freeze: bool,
    reinitialize_after_freeze: bool,
    period_length: int,
    w_max: float,
    bump_order: list[int] | None,
    weight_decay: float,
    cumulative_weight_distance: float,
    prev_params: ParamTree,
    rng: np.random.Generator,
    freeze_part: str | None,
    freeze_after_steps: int | None,
    freeze_applied: bool,
    grad_mask: ParamTree,
    reinit_key: jax.Array,
    initial_params: ParamTree,
    compute_per_class: bool = True, # <-- NEW ARGUMENT
) -> tuple[
    ParamTree,
    optax.OptState,
    Dict[str, Any],
    int,
    Dict[str, Any],
    np.ndarray,
    float,
    ParamTree,
    jax.Array | None,
    ParamTree,
    bool,
    ParamTree,
    jax.Array,
    int | None,
]:
    stats = _init_stats(num_classes)

    @jax.jit
    def _train_step(
        step_params: ParamTree,
        step_opt_state: optax.OptState,
        batch_x: jax.Array,
        batch_y: jax.Array,
        mask: ParamTree,
    ) -> tuple[ParamTree, optax.OptState, jax.Array, jax.Array, jax.Array]:
        def _loss_fn(loss_params: ParamTree) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
            logits, intermediates = model.apply(loss_params, batch_x, return_intermediates=True)
            pre_classifier = intermediates["pre_classifier"]
            per_sample = optax.softmax_cross_entropy_with_integer_labels(logits, batch_y)
            data_loss = jnp.mean(per_sample)
            if weight_decay > 0.0:
                data_loss = data_loss + 0.5 * weight_decay * _l2_penalty(model, loss_params)
            return data_loss, (logits, per_sample, pre_classifier)

        (_loss, (logits, per_sample_loss, pre_classifier)), grads = jax.value_and_grad(_loss_fn, has_aux=True)(
            step_params
        )
        masked_grads = _apply_grad_mask(grads, mask)
        updates, next_opt_state = optimizer.update(masked_grads, step_opt_state, step_params)
        next_params = optax.apply_updates(step_params, updates)
        preds = jnp.argmax(logits, axis=1)
        classifier_grads = _classifier_weight_grads(pre_classifier, logits, batch_y, num_classes)
        return next_params, next_opt_state, per_sample_loss, preds, classifier_grads

    bump_state: Dict[str, Any] = {
        "active": False,
        "focus_class": -1,
        "focus_weight": 1.0,
        "phase": 0.0,
    }
    sampled_distributions: list[np.ndarray] = []
    last_classifier_grads: jax.Array | None = None
    freeze_step: int | None = None

    for _ in range(steps_this_epoch):
        if (
            (not freeze_applied)
            and freeze_after_steps is not None
            and freeze_part is not None
            and global_step >= freeze_after_steps
        ):
            classifier_reset = False
            if reinitialize_after_freeze:
                reinit_key, subkey = jax.random.split(reinit_key)
                new_params = model.init(subkey)
                updated_params: ParamTree = dict(params)
                if freeze_part == "classifier":
                    updated_params["hidden_layers"] = new_params["hidden_layers"]
                elif freeze_part == "preclassifier":
                    updated_params["classifier"] = new_params["classifier"]
                    classifier_reset = True

                params = jax.device_put(updated_params, device=device)
                opt_state = optimizer.init(params)
            grad_mask = _build_grad_mask(params, freeze_part)
            prev_params = params
            freeze_applied = True
            freeze_step = int(global_step)
            if classifier_reset:
                cumulative_weight_distance = 0.0
                initial_params = params

        if freeze_applied:
            allow_bumps = bumps_after_freeze
        else:
            allow_bumps = bumps_enabled

        in_uniform_tail = global_step >= (0.95 * float(total_steps))
        if allow_bumps and (not in_uniform_tail):
            class_probs, focus_class, focus_weight, bump_phase = _compute_class_probabilities(
                step=global_step,
                num_classes=num_classes,
                period_length=period_length,
                w_max=w_max,
                bump_order=bump_order,
            )
            bump_state = {
                "active": True,
                "focus_class": focus_class,
                "focus_weight": focus_weight,
                "phase": bump_phase,
            }
        else:
            class_probs = np.full(num_classes, 1.0 / num_classes, dtype=np.float64)
            bump_state = {
                "active": False,
                "focus_class": -1,
                "focus_weight": 1.0,
                "phase": 0.0,
            }

        class_counts = (class_probs * float(batch_size)).astype(np.int64)
        batch_data, batch_target, sampled_class_counts = _sample_batch_by_class_counts(
            train_inputs=train_inputs,
            train_targets=train_targets,
            class_to_indices=class_to_indices,
            class_counts=class_counts,
            rng=rng,
        )
        sampled_count_total = int(sampled_class_counts.sum())
        sampled_distributions.append(sampled_class_counts.astype(np.float64) / float(max(1, sampled_count_total)))

        x = jax.device_put(jnp.asarray(batch_data, dtype=jnp.float32), device=device)
        y = jax.device_put(jnp.asarray(batch_target, dtype=jnp.int32), device=device)

        prev_w = model.classifier_weight_matrix(prev_params)
        params, opt_state, per_sample_loss, preds, classifier_grads = _train_step(
            params,
            opt_state,
            x,
            y,
            grad_mask,
        )
        current_w = model.classifier_weight_matrix(params)
        dist = float(jnp.linalg.norm(current_w - prev_w))
        cumulative_weight_distance += dist
        prev_params = params
        last_classifier_grads = classifier_grads

        _update_stats(stats, per_sample_loss, preds, y, num_classes, compute_per_class)
        global_step += 1

    if sampled_distributions:
        sampled_distribution_arr = np.stack(sampled_distributions, axis=0)
    else:
        sampled_distribution_arr = np.zeros((0, num_classes), dtype=np.float64)

    return (
        params,
        opt_state,
        _finalize_stats(stats),
        global_step,
        bump_state,
        sampled_distribution_arr,
        cumulative_weight_distance,
        prev_params,
        last_classifier_grads,
        initial_params,
        freeze_applied,
        grad_mask,
        reinit_key,
        freeze_step,
    )


def _csv_header(num_classes: int, training_results: str = "all") -> list[str]:
    base = [
        "epoch",
        "global_step",
        "tpt_reached",
        "tpt_step",
        "bumps_active",
        "bump_focus_class",
        "bump_focus_weight",
        "bump_phase",
        "train_loss",
        "train_accuracy",
        "train_error",
        "test_loss",
        "test_accuracy",
    ]
    if training_results == "all":
        for class_id in range(num_classes):
            base.append(f"train_loss_class_{class_id}")
        for class_id in range(num_classes):
            base.append(f"train_accuracy_class_{class_id}")
        for class_id in range(num_classes):
            base.append(f"test_loss_class_{class_id}")
        for class_id in range(num_classes):
            base.append(f"test_accuracy_class_{class_id}")
    return base


def _metric_row(
    epoch: int,
    global_step: int,
    tpt_reached: bool,
    tpt_step: int,
    bump_state: Dict[str, Any],
    train_metrics: Dict[str, Any],
    test_metrics: Dict[str, Any],
    num_classes: int,
    training_results: str = "all",
) -> list[Any]:
    row: list[Any] = [
        epoch,
        global_step,
        int(tpt_reached),
        tpt_step,
        int(bool(bump_state["active"])),
        int(bump_state["focus_class"]),
        float(bump_state["focus_weight"]),
        float(bump_state["phase"]),
        float(train_metrics["loss"]),
        float(train_metrics["accuracy"]),
        float(train_metrics["training_error"]),
        float(test_metrics["loss"]),
        float(test_metrics["accuracy"]),
    ]
    if training_results == "all":
        for class_id in range(num_classes):
            row.append(float(train_metrics["per_class_loss"][class_id]))
        for class_id in range(num_classes):
            row.append(float(train_metrics["per_class_accuracy"][class_id]))
        for class_id in range(num_classes):
            row.append(float(test_metrics["per_class_loss"][class_id]))
        for class_id in range(num_classes):
            row.append(float(test_metrics["per_class_accuracy"][class_id]))
    return row


def run_training(config: dict, run_dir: Path, run_label: str) -> Dict[str, Any]:
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    dynamics_cfg = config.get("dynamics", {})
    optimizer_cfg = config.get("optimizer", {})
    analysis_cfg = config.get("analysis", {})

    training_results = str(analysis_cfg.get("compute_results", "all")).strip().lower()
    if training_results not in {"all", "average"}:
        training_results = "all"

    compute_per_class_flag = (training_results == "all")

    dataset_name = str(data_cfg.get("dataset_name", "spiral"))
    dataset_key = dataset_name.strip().lower()

    random_seed = int(training_cfg.get("random_seed", config.get("random_seed", 42)))
    _set_global_seed(random_seed)

    device = _resolve_device(config.get("device", "auto"))

    total_steps = int(training_cfg.get("training_steps", training_cfg.get("total_steps", 25000)))
    save_metrics_every_n_steps = int(training_cfg.get("save_metrics_every_n_steps", 50))
    batch_size = int(training_cfg.get("batch_size", 128))
    eval_batch_size = int(training_cfg.get("eval_batch_size", batch_size))

    if total_steps <= 0:
        raise ValueError("training.training_steps must be > 0")
    if save_metrics_every_n_steps <= 0:
        raise ValueError("training.save_metrics_every_n_steps must be > 0")
    if batch_size <= 0:
        raise ValueError("training.batch_size must be > 0")
    if eval_batch_size <= 0:
        raise ValueError("training.eval_batch_size must be > 0")

    optimizer_name = str(optimizer_cfg.get("optimizer_type", training_cfg.get("optimizer", "adam"))).strip().lower()
    loss_name = str(training_cfg.get("loss", "cross_entropy")).strip().lower()
    if optimizer_name not in {"adam", "sgd"}:
        raise ValueError(f"Unsupported optimizer '{optimizer_name}'. Supported: adam, sgd.")
    if loss_name not in {"cross_entropy", "cross-entropy", "ce"}:
        raise ValueError(f"Only cross-entropy loss is supported in this runner. Got '{loss_name}'.")

    learning_rate = float(optimizer_cfg.get("learning_rate", training_cfg.get("learning_rate", 0.002)))
    beta1 = float(optimizer_cfg.get("beta1", training_cfg.get("beta1", 0.9)))
    beta2 = float(optimizer_cfg.get("beta2", training_cfg.get("beta2", 0.999)))
    eps = float(optimizer_cfg.get("eps", training_cfg.get("eps", 1e-8)))
    momentum = float(optimizer_cfg.get("momentum", training_cfg.get("momentum", 0.0)))
    nesterov = _as_bool(optimizer_cfg.get("nesterov", training_cfg.get("nesterov", False)))
    weight_decay = float(
        optimizer_cfg.get(
            "weight_decay",
            optimizer_cfg.get("l2_reg", training_cfg.get("weight_decay", training_cfg.get("l2_reg", 0.0))),
        )
    )

    gradient_clipping_raw = optimizer_cfg.get("gradient_clipping", training_cfg.get("gradient_clipping"))
    gradient_clipping = None if gradient_clipping_raw in (None, "", "null") else float(gradient_clipping_raw)

    freeze_part = _normalize_freeze_part(training_cfg.get("freeze_part"))
    freeze_after_steps = _parse_freeze_after_steps(training_cfg.get("freeze_after_steps"))
    if freeze_part is None and freeze_after_steps is not None:
        raise ValueError("training.freeze_part must be set when training.freeze_after_steps is provided")
    if freeze_part is not None and freeze_after_steps is None:
        raise ValueError("training.freeze_after_steps must be set when training.freeze_part is provided")
    if freeze_after_steps is not None and freeze_after_steps >= total_steps:
        freeze_part = None
        freeze_after_steps = None

    bumps_before_tpt = _as_bool(dynamics_cfg.get("bumps_before_tpt", False))
    bumps_at_tpt = _as_bool(dynamics_cfg.get("bumps_at_tpt", False))
    bumps_after_freeze = _as_bool(dynamics_cfg.get("bumps_after_freeze", True))
    reinitialize_after_freeze = _as_bool(dynamics_cfg.get("reinitialize", True))
    period_length = int(dynamics_cfg.get("period_length", 250))
    w_max = float(dynamics_cfg.get("w_max", 50.0))

    if period_length <= 0:
        raise ValueError("dynamics.period_length must be > 0")
    if w_max < 1.0:
        raise ValueError("dynamics.w_max must be >= 1.0")

    dataset_bundle: DatasetBundle = build_dataset_bundle(config, run_dir)
    num_classes = int(data_cfg.get("num_classes", dataset_bundle.num_classes))
    if num_classes != dataset_bundle.num_classes:
        raise ValueError(
            f"Configured num_classes ({num_classes}) does not match dataset labels ({dataset_bundle.num_classes})."
        )

    bump_order = _validate_bump_order(_parse_bump_order(dynamics_cfg.get("bump_order")), num_classes)

    metric_inputs = dataset_bundle.train_inputs
    metric_targets = dataset_bundle.train_targets
    test_metric_inputs = dataset_bundle.test_inputs
    test_metric_targets = dataset_bundle.test_targets
    metric_subset_size: int | None = None
    if dataset_key in {"mnist", "cifar10", "cifar100", "tiny_imagenet"}:
        metric_subset_size = int(
            analysis_cfg.get("metric_subset_size", data_cfg.get("metric_subset_size", 5000))
        )
        metric_subset_size = max(metric_subset_size, num_classes)

        train_subset_size = min(metric_subset_size, dataset_bundle.train_inputs.shape[0])
        if train_subset_size < dataset_bundle.train_inputs.shape[0]:
            metric_rng = np.random.default_rng(random_seed + 101)
            metric_inputs, metric_targets, _ = _build_metric_subset(
                inputs=dataset_bundle.train_inputs,
                targets=dataset_bundle.train_targets,
                class_to_indices=dataset_bundle.class_to_indices,
                num_classes=num_classes,
                subset_size=train_subset_size,
                rng=metric_rng,
            )

        test_subset_size = min(metric_subset_size, dataset_bundle.test_inputs.shape[0])
        if test_subset_size < dataset_bundle.test_inputs.shape[0]:
            test_metric_rng = np.random.default_rng(random_seed + 202)
            test_class_to_indices = _build_class_to_indices(dataset_bundle.test_targets, num_classes)
            test_metric_inputs, test_metric_targets, _ = _build_metric_subset(
                inputs=dataset_bundle.test_inputs,
                targets=dataset_bundle.test_targets,
                class_to_indices=test_class_to_indices,
                num_classes=num_classes,
                subset_size=test_subset_size,
                rng=test_metric_rng,
            )

    architecture_str = str(model_cfg.get("architecture", "mlp")).strip().lower()
    # "resnet18_cnn" kept for backward compatibility with existing configs;
    # "cnn" is the new generic entry point that reads model.cnn.backbone.
    use_pytorch_cnn = architecture_str in {"resnet18_cnn", "cnn"}
    if use_pytorch_cnn:
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cnn_cfg = model_cfg.get("cnn", {}) or {}
        default_backbone = "resnet18" if architecture_str == "resnet18_cnn" else "simple_cnn"
        backbone_name = str(cnn_cfg.get("backbone", default_backbone)).strip().lower()

        # image datasets (e.g. mnist) have input_shape like (C, H, W); tabular
        # datasets (spiral, blobs, rings, checkerboard, ...) have input_shape
        # like (D,) -- a flat feature count with no spatial dims. Only the
        # former can feed conv1 directly; the latter needs a stem that
        # projects flat vectors into a small image-like tensor first.
        is_tabular_input = len(dataset_bundle.input_shape) == 1
        if is_tabular_input:
            input_ch = int(cnn_cfg.get("stem_channels", 1))
            input_dim = int(dataset_bundle.input_shape[0])
        else:
            input_ch = int(dataset_bundle.input_shape[0])
            input_dim = None

        pretrained_checkpoint = cnn_cfg.get("pretrained_checkpoint")

        if backbone_name == "resnet18":
            resnet_cfg = cnn_cfg.get("resnet", {}) or {}
            blocks_per_stage = tuple(
                int(b) for b in resnet_cfg.get("blocks_per_stage", [2, 2, 2, 2])
            )
            num_stages = int(resnet_cfg.get("num_stages", 4))
            width_mult = float(resnet_cfg.get("width_mult", 1.0))

            if pretrained_checkpoint:
                torch_model, classifier, feature_capture = load_pretrained_cnn(
                    checkpoint_path=str(pretrained_checkpoint),
                    pretrained_num_classes=int(cnn_cfg.get("pretrained_num_classes", 10)),
                    new_num_classes=num_classes, input_ch=input_ch, device=torch_device,
                    backbone="resnet18", input_dim=input_dim,
                    stem_spatial_size=int(cnn_cfg.get("stem_spatial_size", 8)),
                    blocks_per_stage=blocks_per_stage, num_stages=num_stages,
                    width_mult=width_mult,
                    freeze_backbone=_as_bool(cnn_cfg.get("freeze_backbone", False)),
                )
            else:
                torch_model, classifier, feature_capture = build_cnn_model(
                    num_classes=num_classes, input_ch=input_ch, device=torch_device,
                    input_dim=input_dim,
                    stem_spatial_size=int(cnn_cfg.get("stem_spatial_size", 8)),
                    blocks_per_stage=blocks_per_stage,
                    num_stages=num_stages,
                    width_mult=width_mult,
                )
        elif backbone_name == "simple_cnn":
            fc_hidden_dim_raw = cnn_cfg.get("fc_hidden_dim")
            fc_hidden_dim = int(fc_hidden_dim_raw) if fc_hidden_dim_raw else None
            channels = [int(c) for c in cnn_cfg.get("channels", [16, 32])]
            kernel_size = int(cnn_cfg.get("kernel_size", 3))
            use_batchnorm = _as_bool(cnn_cfg.get("use_batchnorm", True))
            pool_every_block = _as_bool(cnn_cfg.get("pool_every_block", True))
            dropout = float(cnn_cfg.get("dropout", 0.0))
            stem_spatial_size = int(cnn_cfg.get("stem_spatial_size", 16))

            if pretrained_checkpoint:
                torch_model, classifier, feature_capture = load_pretrained_cnn(
                    checkpoint_path=str(pretrained_checkpoint),
                    pretrained_num_classes=int(cnn_cfg.get("pretrained_num_classes", 10)),
                    new_num_classes=num_classes, input_ch=input_ch, device=torch_device,
                    backbone="simple_cnn", input_dim=input_dim,
                    stem_spatial_size=stem_spatial_size, channels=channels,
                    kernel_size=kernel_size, use_batchnorm=use_batchnorm,
                    pool_every_block=pool_every_block, dropout=dropout,
                    fc_hidden_dim=fc_hidden_dim,
                    freeze_backbone=_as_bool(cnn_cfg.get("freeze_backbone", False)),
                )
            else:
                torch_model, classifier, feature_capture = build_simple_cnn_model(
                    num_classes=num_classes, input_ch=input_ch, device=torch_device,
                    channels=channels, kernel_size=kernel_size,
                    use_batchnorm=use_batchnorm, pool_every_block=pool_every_block,
                    dropout=dropout, fc_hidden_dim=fc_hidden_dim,
                    input_dim=input_dim, stem_spatial_size=stem_spatial_size,
                )
        elif backbone_name == "myrtle":
            # Configurable Myrtle-family CNN (Shankar et al., 2020), the
            # network used for the CIFAR-10 experiments in Ruiz-Garcia et al.
            # 2021 ("Tilting the playing field"). Only the architecture is
            # ported over here -- no dynamical loss / Gamma-weighting; the
            # existing bump-sampling oscillation mechanism below is used
            # instead, unchanged, since it is architecture-agnostic.
            myrtle_cfg = cnn_cfg.get("myrtle", {}) or {}
            base_width = int(myrtle_cfg.get("base_width", 64))
            myrtle_num_stages = int(myrtle_cfg.get("num_stages", 4))
            blocks_per_stage_raw = myrtle_cfg.get("blocks_per_stage")
            myrtle_blocks_per_stage = (
                [int(b) for b in blocks_per_stage_raw] if blocks_per_stage_raw else None
            )
            channels_raw = myrtle_cfg.get("channels")
            myrtle_channels = [int(c) for c in channels_raw] if channels_raw else None
            kernel_size = int(myrtle_cfg.get("kernel_size", 3))
            use_batchnorm = _as_bool(myrtle_cfg.get("use_batchnorm", True))
            pool_last_stage = _as_bool(myrtle_cfg.get("pool_last_stage", True))
            dropout = float(myrtle_cfg.get("dropout", 0.0))
            stem_spatial_size = int(cnn_cfg.get("stem_spatial_size", 32))

            if pretrained_checkpoint:
                torch_model, classifier, feature_capture = load_pretrained_cnn(
                    checkpoint_path=str(pretrained_checkpoint),
                    pretrained_num_classes=int(cnn_cfg.get("pretrained_num_classes", 10)),
                    new_num_classes=num_classes, input_ch=input_ch, device=torch_device,
                    backbone="myrtle", input_dim=input_dim,
                    stem_spatial_size=stem_spatial_size,
                    base_width=base_width, myrtle_num_stages=myrtle_num_stages,
                    myrtle_blocks_per_stage=myrtle_blocks_per_stage, channels=myrtle_channels,
                    kernel_size=kernel_size, use_batchnorm=use_batchnorm,
                    pool_last_stage=pool_last_stage, dropout=dropout,
                    freeze_backbone=_as_bool(cnn_cfg.get("freeze_backbone", False)),
                )
            else:
                torch_model, classifier, feature_capture = build_myrtle_cnn_model(
                    num_classes=num_classes, input_ch=input_ch, device=torch_device,
                    base_width=base_width, num_stages=myrtle_num_stages,
                    blocks_per_stage=myrtle_blocks_per_stage, channels=myrtle_channels,
                    kernel_size=kernel_size, use_batchnorm=use_batchnorm,
                    pool_last_stage=pool_last_stage, dropout=dropout,
                    input_dim=input_dim, stem_spatial_size=stem_spatial_size,
                )
        else:
            raise ValueError(
                f"Unknown model.cnn.backbone '{backbone_name}'. Expected 'resnet18', 'simple_cnn', or 'myrtle'."
            )
        initial_classifier_weight = (classifier.weight.detach().cpu().clone().numpy())
        last_classifier_weight = initial_classifier_weight.copy()
        model = TorchModelAdapter(torch_model, classifier, feature_capture, torch_device, init_type=str(model_cfg.get("init_type", "pytorch_default")),)
        params = None
        initial_params = None
        cumulative_weight_distance = 0.0
        prev_params = None
        grad_mask = None
        freeze_applied = False
        freeze_step_applied = None
        reinit_key = None
    else:
        built_model = build_model(
            model_cfg=model_cfg, input_shape=dataset_bundle.input_shape,
            num_classes=num_classes, random_seed=random_seed,
        )
        model = built_model.model
        params = jax.device_put(built_model.params, device=device)
        initial_params = params
        initial_classifier_weight = np.asarray(model.classifier_weight_matrix(initial_params), dtype=np.float64,)
        last_classifier_weight = initial_classifier_weight.copy()
        cumulative_weight_distance = 0.0
        prev_params = params
        grad_mask = _build_grad_mask(params, None)
        freeze_applied = False
        freeze_step_applied = None
        reinit_key = jax.random.PRNGKey(int(random_seed) + 1)

    total_epochs = int(math.ceil(total_steps / save_metrics_every_n_steps))

    if use_pytorch_cnn:
        cnn_optimizer, cnn_scheduler = make_optimizer_and_scheduler(
            torch_model, lr=learning_rate, momentum=momentum,
            weight_decay=weight_decay, epochs=total_epochs,
        )
        criterion = make_criterion(training_cfg.get("loss_name", "CrossEntropyLoss"))
        optimizer = None
        opt_state = None
    else:
        if optimizer_name == "adam":
            base_optimizer = optax.adam(learning_rate=learning_rate, b1=beta1, b2=beta2, eps=eps)
        else:
            base_optimizer = optax.sgd(learning_rate=learning_rate, momentum=momentum, nesterov=nesterov)
        optimizer = optax.chain(optax.clip_by_global_norm(gradient_clipping), base_optimizer) if gradient_clipping is not None else base_optimizer
        opt_state = optimizer.init(params)

    csv_path = run_dir / "training_metrics.csv"
    figure_path = run_dir / "training_report.png"
    plot_version = _normalize_plot_version(analysis_cfg.get("plot_version"))
    enable_classifier_metrics = _analysis_output_enabled(analysis_cfg, ("classifier_metrics",))
    enable_nc_metrics = _analysis_output_enabled(analysis_cfg, ("nc_metrics",))
    enable_sep_metrics = _analysis_output_enabled(analysis_cfg, ("separability_metrics",))
    enable_pca_analysis = _analysis_output_enabled(analysis_cfg, ("pca_analysis",))
    enable_geo_metrics = _analysis_output_enabled(analysis_cfg, ("PCA_geometric", "pca_geometric", "pca_geometric_overlapping"),)
    enable_hp_metrics = _analysis_output_enabled(analysis_cfg, ("hyperplane_metrics", "hyperplanes"))
    enable_proj_nc = _analysis_output_enabled(analysis_cfg, ("proj_nc_analysis",))
    enable_multiclass_metrics = _analysis_output_enabled(analysis_cfg, ("multiclass_metrics",))

    collect_pre_classifier = (
        enable_classifier_metrics or enable_pca_analysis
        or enable_geo_metrics or enable_hp_metrics
        or enable_proj_nc or enable_multiclass_metrics
    )

    nc_class_pairs = build_nc_class_pairs(num_classes)

    nc_csv_path: Path | None = None
    neural_collapse_figure_path: Path | None = None
    if enable_nc_metrics:
        nc_csv_path = run_dir / "nc_metrics.csv"
        neural_collapse_figure_path = run_dir / "neural_collapse.png"
        initialize_nc_csv(
            nc_csv_path=nc_csv_path,
            num_classes=num_classes,
            class_pairs=nc_class_pairs,
        )

    sep_csv_path: Path | None = None
    separability_figure_path: Path | None = None
    if enable_sep_metrics:
        sep_csv_path = run_dir / "separability_metrics.csv"
        separability_figure_path = run_dir / "separability_measures.png"
        initialize_sep_csv(
            sep_csv_path=sep_csv_path,
            num_classes=num_classes,
            class_pairs=nc_class_pairs,
        )

    classifier_csv_path: Path | None = None
    classifier_figure_path: Path | None = None
    if enable_classifier_metrics:
        classifier_csv_path = run_dir / "classifier_metrics.csv"
        classifier_figure_path = run_dir / "classifier_metrics.png"
        initialize_classifier_csv(
            csv_path=classifier_csv_path,
            num_classes=num_classes,
        )

    pca_csv_path: Path | None = None
    pca_variance_path: Path | None = None
    pca_projected_path: Path | None = None
    feature_dim: int | None = None
    projected_dims: list[int] | None = None
    if enable_pca_analysis:
        pca_cfg = analysis_cfg.get("pca", {})
        feature_dim = int(model.classifier_weight_matrix(params).shape[1])
        projected_dims = normalize_projected_dims(pca_cfg.get("projected_dims", [1, 2, 3, 5]), feature_dim)
        pca_csv_path = run_dir / "pca_analysis.csv"
        pca_variance_path = run_dir / "pca_explained_variance.png"
        pca_projected_path = run_dir / "pca_projected_metrics.png"
        initialize_pca_csv(
            csv_path=pca_csv_path,
            feature_dim=feature_dim,
            projected_dims=projected_dims,
        )

    geo_csv_path: Path | None = None
    geo_figure_path: Path | None = None
    if enable_geo_metrics:
        geo_csv_path = run_dir / "PCA_geometric.csv"
        geo_figure_path = run_dir / "PCA_geometric_overlapping.png"
        initialize_geo_csv(
            csv_path=geo_csv_path,
            num_classes=num_classes,
            class_pairs=nc_class_pairs,
        )
    
    hp_csv_path: Path | None = None
    hp_figure_path: Path | None = None
    if enable_hp_metrics:
        hp_csv_path  = run_dir / "hyperplanes.csv"
        hp_figure_path = run_dir / "hyperplanes.png"
        initialize_hyperplane_csv(csv_path=hp_csv_path, num_classes=num_classes)

    proj_nc_csv_path: Path | None = None
    proj_nc_figure_path: Path | None = None
    if enable_proj_nc:
        proj_nc_csv_path = run_dir / "proj_nc_metrics.csv"
        proj_nc_figure_path = run_dir / "proj_nc_metrics.png"
        initialize_proj_nc_csv(
            csv_path=proj_nc_csv_path,
            num_classes=num_classes,
            class_pairs=nc_class_pairs,
        )
    
    multiclass_csv_path: Path | None = None
    multiclass_figure_path: Path | None = None
    multiclass_shape_path: Path | None = None
    if enable_multiclass_metrics:
        multiclass_csv_path = run_dir / "metrics_for_multiple_classes.csv"
        multiclass_figure_path = run_dir / "metrics_for_multiple_classes.png"
        multiclass_shape_path = run_dir / "collapse_shapes.png"
        initialize_multiclass_csv(multiclass_csv_path)

    if use_pytorch_cnn:
        predict_step = lambda step_params, batch_x: model.apply(step_params, batch_x)
    else:
        predict_step = jax.jit(lambda step_params, batch_x: model.apply(step_params, batch_x))

    last_classifier_grads = None
    global_step = 0
    tpt_reached = False
    tpt_step = -1
    rng = np.random.default_rng(random_seed)
    class_distribution_history: list[np.ndarray] = []

    cnn_shuffler = None
    if use_pytorch_cnn:
        cnn_shuffler = _EpochShuffler(
            num_samples=dataset_bundle.train_inputs.shape[0],
            batch_size=batch_size,
            rng=rng,
        )

    last_train_metrics: Dict[str, Any] = {
        "loss": np.nan,
        "accuracy": np.nan,
        "per_class_loss": np.full(num_classes, np.nan),
        "per_class_accuracy": np.full(num_classes, np.nan),
    }
    last_test_metrics: Dict[str, Any] = {
        "loss": np.nan,
        "accuracy": np.nan,
        "per_class_loss": np.full(num_classes, np.nan),
        "per_class_accuracy": np.full(num_classes, np.nan),
    }

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(_csv_header(num_classes, training_results))
        f.flush()

        for epoch in range(1, total_epochs + 1):
            steps_this_epoch = min(save_metrics_every_n_steps, total_steps - global_step)
            if steps_this_epoch <= 0:
                break

            bumps_enabled = bumps_at_tpt if tpt_reached else bumps_before_tpt

            if use_pytorch_cnn:
                cnn_result = train_epoch_cnn(
                    model=torch_model,
                    criterion=criterion,
                    device=torch_device,
                    num_classes=num_classes,
                    train_inputs=dataset_bundle.train_inputs,
                    train_targets=dataset_bundle.train_targets,
                    class_to_indices=dataset_bundle.class_to_indices,
                    batch_size=batch_size,
                    optimizer=cnn_optimizer,
                    global_step=global_step,
                    steps_this_epoch=steps_this_epoch,
                    total_steps=total_steps,
                    bumps_enabled=bumps_enabled,
                    period_length=period_length,
                    w_max=w_max,
                    bump_order=bump_order,
                    rng=rng,
                    shuffler=cnn_shuffler,
                )
                cnn_scheduler.step()
                global_step = cnn_result["global_step"]
                bump_state = cnn_result["bump_state"]
                step_distributions = cnn_result["step_distributions"]
                freeze_step = None

            else:
                (params, opt_state, _step_train_metrics, global_step, bump_state, step_distributions,
                 cumulative_weight_distance, prev_params, last_classifier_grads, initial_params,
                 freeze_applied, grad_mask, reinit_key, freeze_step) = train_epoch(
                params=params,
                opt_state=opt_state,
                optimizer=optimizer,
                model=model,
                train_inputs=dataset_bundle.train_inputs,
                train_targets=dataset_bundle.train_targets,
                class_to_indices=dataset_bundle.class_to_indices,
                batch_size=batch_size,
                device=device,
                num_classes=num_classes,
                global_step=global_step,
                steps_this_epoch=steps_this_epoch,
                total_steps=total_steps,
                bumps_enabled=bumps_enabled,
                bumps_after_freeze=bumps_after_freeze,
                reinitialize_after_freeze=reinitialize_after_freeze,
                period_length=period_length,
                w_max=w_max,
                bump_order=bump_order,
                weight_decay=weight_decay,
                cumulative_weight_distance=cumulative_weight_distance,
                prev_params=prev_params,
                rng=rng,
                freeze_part=freeze_part,
                freeze_after_steps=freeze_after_steps,
                freeze_applied=freeze_applied,
                grad_mask=grad_mask,
                reinit_key=reinit_key,
                initial_params=initial_params,
                compute_per_class=compute_per_class_flag
            )
            if freeze_step is not None and freeze_step_applied is None:
                freeze_step_applied = int(freeze_step)
                print(
                    f"[{run_label}] freeze applied at step {freeze_step_applied} (frozen {freeze_part})"
                )
            if step_distributions.size > 0:
                class_distribution_history.extend(step_distributions)

            train_metrics = evaluate_arrays(
                params=params,
                predict_step=predict_step,
                inputs=metric_inputs,
                targets=metric_targets,
                batch_size=eval_batch_size,
                device=device,
                num_classes=num_classes,
                compute_per_class=compute_per_class_flag, # Pass it here
            )
            test_metrics = evaluate_arrays(
                params=params,
                predict_step=predict_step,
                inputs=test_metric_inputs,
                targets=test_metric_targets,
                batch_size=eval_batch_size,
                device=device,
                num_classes=num_classes,
                compute_per_class=compute_per_class_flag, # Pass it here
            )

            if enable_nc_metrics:
                nc_raw = collect_nc_raw_epoch(
                    model=model,
                    params=params,
                    inputs=metric_inputs,
                    targets=metric_targets,
                    num_classes=num_classes,
                    class_pairs=nc_class_pairs,
                    eval_batch_size=eval_batch_size,
                )
                append_nc_csv_row(
                    nc_csv_path=nc_csv_path,
                    epoch=epoch,
                    global_step=global_step,
                    raw=nc_raw,
                    num_classes=num_classes,
                )

            if enable_sep_metrics:
                sep_raw: SepEpochRaw = collect_sep_raw_epoch(
                    model=model,
                    params=params,
                    inputs=metric_inputs,
                    targets=metric_targets,
                    num_classes=num_classes,
                    class_pairs=nc_class_pairs,
                    eval_batch_size=eval_batch_size,
                )
                append_sep_csv_row(
                    sep_csv_path=sep_csv_path,
                    epoch=epoch,
                    global_step=global_step,
                    raw=sep_raw,
                    num_classes=num_classes,
                )

            pre_classifier: np.ndarray | None = None
            logits: np.ndarray | None = None
            labels: np.ndarray | None = None
            if collect_pre_classifier:
                pre_classifier, logits, labels = _collect_pre_classifier_outputs(
                    model=model,
                    params=params,
                    inputs=metric_inputs,
                    targets=metric_targets,
                    batch_size=eval_batch_size,
                    device=device,
                )

            if enable_classifier_metrics:
                if pre_classifier is None or logits is None or labels is None:
                    raise RuntimeError("Classifier metrics enabled but no pre-classifier outputs were collected.")
                
                initial_weight_matrix = initial_classifier_weight

                if use_pytorch_cnn:
                    weight_matrix = (classifier.weight.detach().cpu().numpy().astype(np.float64))
                else:
                    weight_matrix = np.asarray(model.classifier_weight_matrix(params), dtype=np.float64,)
                    
                classifier_raw = collect_classifier_epoch(
                    pre_classifier=pre_classifier,
                    logits=logits,
                    targets=labels,
                    weight_matrix=weight_matrix,
                )
                
                advanced_classifier_raw = collect_advanced_classifier_metrics(
                    weight_matrix=weight_matrix,
                    initial_weight_matrix=initial_weight_matrix,
                    cumulative_weight_distance=cumulative_weight_distance,
                    logits=logits,
                    targets=labels,
                    grads=last_classifier_grads,
                    previous_weight_matrix=last_classifier_weight,
                )

                last_classifier_weight = weight_matrix.copy()
                append_classifier_csv_row(
                    csv_path=classifier_csv_path,
                    epoch=epoch,
                    global_step=global_step,
                    raw=classifier_raw,
                    num_classes=num_classes,
                    advanced_raw=advanced_classifier_raw,
                )

            if enable_pca_analysis:
                if pre_classifier is None or labels is None:
                    raise RuntimeError("PCA analysis enabled but no pre-classifier outputs were collected.")
                if feature_dim is None or projected_dims is None:
                    raise RuntimeError("PCA analysis enabled but PCA configuration is missing.")
                pca_raw = collect_pca_epoch(
                    pre_classifier=pre_classifier,
                    targets=labels,
                    num_classes=num_classes,
                    class_pairs=nc_class_pairs,
                    eval_batch_size=eval_batch_size,
                    projected_dims=projected_dims,
                )
                append_pca_csv_row(
                    csv_path=pca_csv_path,
                    epoch=epoch,
                    global_step=global_step,
                    raw=pca_raw,
                    feature_dim=feature_dim,
                    projected_dims=projected_dims,
                )

            if enable_geo_metrics:
                if pre_classifier is None or labels is None:
                    raise RuntimeError("PCA geometric metrics enabled but no pre-classifier outputs were collected.")
                geo_raw = collect_geo_epoch(
                    pre_classifier=pre_classifier,
                    targets=labels,
                    num_classes=num_classes,
                    class_pairs=nc_class_pairs,
                )
                append_geo_csv_row(
                    csv_path=geo_csv_path,
                    epoch=epoch,
                    global_step=global_step,
                    raw=geo_raw,
                    num_classes=num_classes,
                )

            if enable_proj_nc:
                if pre_classifier is None or labels is None:
                    raise RuntimeError(
                        "proj_nc analysis enabled but no pre-classifier outputs were collected."
                    )
                proj_raw = collect_proj_nc_epoch(
                    pre_classifier=pre_classifier,
                    targets=labels,
                    num_classes=num_classes,
                    class_pairs=nc_class_pairs,
                )
                append_proj_nc_csv_row(
                    csv_path=proj_nc_csv_path,
                    epoch=epoch,
                    global_step=global_step,
                    raw=proj_raw,
                    num_classes=num_classes,
                    class_pairs=nc_class_pairs,
                )
            
            if enable_hp_metrics:
                if pre_classifier is None or labels is None:
                    raise RuntimeError(
                        "Hyperplane metrics enabled but no pre-classifier outputs were collected."
                    )
                weight_matrix = np.asarray(model.classifier_weight_matrix(params), dtype=np.float64)
                try:
                    bias_vec = np.asarray(params["classifier"]["bias"], dtype=np.float64)
                except (KeyError, TypeError):
                    bias_vec = None
                hp_raw = collect_hyperplane_epoch(
                    pre_classifier=pre_classifier,
                    targets=labels,
                    weight_matrix=weight_matrix,
                    bias=bias_vec,
                    num_classes=num_classes,
                )
                append_hyperplane_csv_row(
                    csv_path=hp_csv_path,
                    epoch=epoch,
                    global_step=global_step,
                    raw=hp_raw,
                    num_classes=num_classes,
                )

            if enable_multiclass_metrics:
                if pre_classifier is None or labels is None or logits is None:
                    raise RuntimeError(
                        "multiclass_metrics enabled but pre-classifier/logits were not collected."
                    )
                #test_logits, test_labels = compute_batched_logits(
                #    model=model,
                #    params=params,
                #    inputs=test_metric_inputs,
                #    targets=test_metric_targets,
                #    eval_batch_size=eval_batch_size,
                #)
                initial_weight_matrix = initial_classifier_weight

                if use_pytorch_cnn:
                    weight_matrix = (classifier.weight.detach().cpu().numpy().astype(np.float64))
                else:
                    weight_matrix = np.asarray(model.classifier_weight_matrix(params), dtype=np.float64)

                mc_raw = collect_multiclass_epoch(
                    pre_classifier=pre_classifier,
                    targets=labels,
                    #train_logits=logits,
                    #test_logits=test_logits,
                    #test_targets=test_labels,
                    weight_matrix=weight_matrix,
                    #initial_weight_matrix=initial_weight_matrix,
                    #cumulative_weight_distance=cumulative_weight_distance,
                    num_classes=num_classes,
                    previous_weight_matrix=last_classifier_weight,
                )

                last_classifier_weight = weight_matrix.copy()
                append_multiclass_csv_row(
                    csv_path=multiclass_csv_path,
                    epoch=epoch,
                    global_step=global_step,
                    raw=mc_raw,
                )

            if (not tpt_reached) and bool(train_metrics.get("zero_training_error", False)):
                tpt_reached = True
                tpt_step = global_step

            row = _metric_row(
                epoch=epoch,
                global_step=global_step,
                tpt_reached=tpt_reached,
                tpt_step=tpt_step,
                bump_state=bump_state,
                train_metrics=train_metrics,
                test_metrics=test_metrics,
                num_classes=num_classes,
                training_results=training_results,
            )
            writer.writerow(row)
            f.flush()

            last_train_metrics = train_metrics
            last_test_metrics = test_metrics

            focus_text = str(bump_state["focus_class"]) if bump_state["active"] else "-"
            print(
                f"[{run_label}] epoch {epoch:04d}/{total_epochs:04d} | "
                f"step {global_step:06d}/{total_steps:06d} | "
                f"train loss {train_metrics['loss']:.4f} acc {train_metrics['accuracy']:.4f} | "
                f"test loss {test_metrics['loss']:.4f} acc {test_metrics['accuracy']:.4f} | "
                f"bumps {'on' if bump_state['active'] else 'off'} focus {focus_text}"
            )

    architecture = str(model_cfg.get("architecture", "mlp"))
    plot_training_report(
        csv_path=csv_path,
        output_path=figure_path,
        num_classes=num_classes,
        title=f"{run_label} | {dataset_name} | {architecture}",
    )

    distribution_figure_path = run_dir / "example_distribution_dynamics.png"
    if class_distribution_history:
        distribution_history_arr = np.asarray(class_distribution_history, dtype=np.float64)
    else:
        distribution_history_arr = np.zeros((0, num_classes), dtype=np.float64)
    plot_example_distribution_dynamics(
        class_distribution_history=distribution_history_arr,
        output_path=distribution_figure_path,
        title=f"{run_label} | Example Distribution Dynamics",
        tpt_step=tpt_step if tpt_reached else -1,
    )

    if enable_nc_metrics:
        finalize_nc_metrics(
            nc_csv_path=nc_csv_path,
            output_path=neural_collapse_figure_path,
            num_classes=num_classes,
            class_pairs=nc_class_pairs,
            tpt_step=tpt_step if tpt_reached else -1,
        )
    if enable_sep_metrics:
        finalize_sep_metrics(
            sep_csv_path=sep_csv_path,
            output_path=separability_figure_path,
            num_classes=num_classes,
            class_pairs=nc_class_pairs,
            tpt_step=tpt_step if tpt_reached else -1,
        )
    if enable_classifier_metrics:
        if plot_version == "simplified":
            finalize_classifier_simplified(
                csv_path=classifier_csv_path,
                output_path=classifier_figure_path,
                num_classes=num_classes,
                tpt_step=tpt_step if tpt_reached else -1,
            )
        else:
            finalize_classifier_dashboard(
                csv_path=classifier_csv_path,
                output_path=classifier_figure_path,
                num_classes=num_classes,
                tpt_step=tpt_step if tpt_reached else -1,
            )
    if enable_pca_analysis:
        finalize_pca_analysis(
            csv_path=pca_csv_path,
            variance_output_path=pca_variance_path,
            projected_output_path=pca_projected_path,
            feature_dim=feature_dim,
            projected_dims=projected_dims,
            tpt_step=tpt_step if tpt_reached else -1,
        )
    if enable_geo_metrics:
        if plot_version == "simplified":
            finalize_geo_plots_simplified(
                csv_path=geo_csv_path,
                output_path=geo_figure_path,
                num_classes=num_classes,
                class_pairs=nc_class_pairs,
                tpt_step=tpt_step if tpt_reached else -1,
            )
        else:
            finalize_geo_plots(
                csv_path=geo_csv_path,
                output_path=geo_figure_path,
                num_classes=num_classes,
                class_pairs=nc_class_pairs,
                tpt_step=tpt_step if tpt_reached else -1,
            )
    if enable_hp_metrics:
        finalize_hyperplane_plots(
            csv_path=hp_csv_path,
            output_path=hp_figure_path,
            num_classes=num_classes,
            tpt_step=tpt_step if tpt_reached else -1,
        )
    if enable_proj_nc:
        finalize_proj_nc_plots(
            csv_path=proj_nc_csv_path,
            output_path=proj_nc_figure_path,
            num_classes=num_classes,
            class_pairs=nc_class_pairs,
            tpt_step=tpt_step if tpt_reached else -1,
        )
    if enable_multiclass_metrics:
        finalize_multiclass_plots(
            csv_path=multiclass_csv_path,
            output_path=multiclass_figure_path,
            tpt_step=tpt_step if tpt_reached else -1,
        )
        finalize_shape_metrics_plots(
            csv_path=multiclass_csv_path,
            output_path=multiclass_shape_path,
            tpt_step=tpt_step if tpt_reached else -1,
        )

    boundary_figure_path: Path | None = None
    if dataset_key in {"spiral", "gaussian_blobs", "blobs", "rings", "checkerboard", "random_checkerboard", "dartboard"}:
        boundary_figure_path = run_dir / f"{dataset_key}_decision_boundaries.png"
        plot_2d_decision_boundaries(
            model=model,
            params=params,
            train_inputs=dataset_bundle.train_inputs,
            train_targets=dataset_bundle.train_targets,
            test_inputs=dataset_bundle.test_inputs,
            test_targets=dataset_bundle.test_targets,
            output_path=boundary_figure_path,
            title=f"{run_label} | {dataset_name} | {architecture} | Decision Boundaries",
        )

    summary = {
        "run_label": run_label,
        "device": _device_label(device),
        "optimizer": optimizer_name,
        "init_type": model.init_type,
        "total_steps": total_steps,
        "save_metrics_every_n_steps": save_metrics_every_n_steps,
        "epochs": total_epochs,
        "freeze_part": freeze_part,
        "freeze_after_steps": freeze_after_steps,
        "freeze_applied": bool(freeze_step_applied is not None),
        "freeze_step": int(freeze_step_applied) if freeze_step_applied is not None else None,
        "bumps_after_freeze": bool(bumps_after_freeze),
        "reinitialize_after_freeze": bool(reinitialize_after_freeze),
        "tpt_reached": bool(tpt_reached),
        "tpt_step": int(tpt_step),
        "metric_subset_size": int(metric_subset_size) if metric_subset_size is not None else None,
        "metric_subset_count": int(metric_inputs.shape[0]),
        "csv_path": str(csv_path),
        "nc_csv_path": str(nc_csv_path) if nc_csv_path is not None else None,
        "figure_path": str(figure_path),
        "neural_collapse_figure_path": str(neural_collapse_figure_path)
        if neural_collapse_figure_path is not None
        else None,
        "separability_figure_path": str(separability_figure_path)
        if separability_figure_path is not None
        else None,
        "classifier_csv_path": str(classifier_csv_path) if classifier_csv_path is not None else None,
        "classifier_figure_path": str(classifier_figure_path)
        if classifier_figure_path is not None
        else None,
        "pca_csv_path": str(pca_csv_path) if pca_csv_path is not None else None,
        "pca_variance_path": str(pca_variance_path) if pca_variance_path is not None else None,
        "pca_projected_path": str(pca_projected_path) if pca_projected_path is not None else None,
        "geo_csv_path": str(geo_csv_path) if geo_csv_path is not None else None,
        "geo_figure_path": str(geo_figure_path) if geo_figure_path is not None else None,
        "hp_csv_path":    str(hp_csv_path)    if hp_csv_path    is not None else None,
        "hp_figure_path": str(hp_figure_path) if hp_figure_path is not None else None,
        "distribution_figure_path": str(distribution_figure_path),
        "decision_boundary_path": str(boundary_figure_path) if boundary_figure_path is not None else None,
        "final_train_loss": float(last_train_metrics["loss"]),
        "final_train_accuracy": float(last_train_metrics["accuracy"]),
        "final_test_loss": float(last_test_metrics["loss"]),
        "final_test_accuracy": float(last_test_metrics["accuracy"]),
    }

    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # --- NEW CODE: Save dataset and models ---
    
    # 1. Save the dataset bundles so we can plot the exact same train/test points
    np.savez(
        run_dir / "dataset_bundle.npz",
        train_inputs=dataset_bundle.train_inputs,
        train_targets=dataset_bundle.train_targets,
        test_inputs=dataset_bundle.test_inputs,
        test_targets=dataset_bundle.test_targets
    )

    # 2. Save the model weights
    if use_pytorch_cnn:
        torch.save({
            'torch_model': torch_model.state_dict(),
            'classifier': classifier.state_dict()
        }, run_dir / "model_weights.pt")
    else:
        # Save JAX parameters using standard pickle
        with open(run_dir / "jax_params.pkl", "wb") as f:
            pickle.dump(params, f)
            
    # -----------------------------------------

    return summary