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

from generate_dataset import DatasetBundle, build_dataset_bundle
from metrics import (
    plot_example_distribution_dynamics,
    plot_spiral_decision_boundaries,
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


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


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
) -> None:
    loss_np = np.asarray(per_sample_loss, dtype=np.float64)
    preds_np = np.asarray(preds, dtype=np.int64)
    target_np = np.asarray(target, dtype=np.int64)
    correct_np = (preds_np == target_np).astype(np.float64)

    stats.loss_sum += float(np.sum(loss_np))
    stats.correct_sum += int(np.sum(correct_np))
    stats.count_sum += int(target_np.shape[0])

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
) -> tuple[np.ndarray, int, float, float]:
    focus_class = (step // period_length) % num_classes
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


def evaluate_arrays(
    params: ParamTree,
    predict_step: Callable[[ParamTree, jax.Array], jax.Array],
    inputs: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    device: jax.Device,
    num_classes: int,
) -> Dict[str, Any]:
    stats = _init_stats(num_classes)

    for batch_inputs, batch_targets in _iterate_minibatches(inputs, targets, batch_size):
        x = jax.device_put(jnp.asarray(batch_inputs, dtype=jnp.float32), device=device)
        y = jax.device_put(jnp.asarray(batch_targets, dtype=jnp.int32), device=device)
        logits = predict_step(params, x)
        per_sample_loss = optax.softmax_cross_entropy_with_integer_labels(logits, y)
        preds = jnp.argmax(logits, axis=1)
        _update_stats(stats, per_sample_loss, preds, y, num_classes)

    return _finalize_stats(stats)


def _l2_penalty(model: JAXModel, params: ParamTree) -> jax.Array:
    total = jnp.array(0.0, dtype=jnp.float32)
    for _name, param in model.iter_named_parameters(params):
        total = total + jnp.sum(jnp.square(param))
    return total


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
    period_length: int,
    w_max: float,
    weight_decay: float,
    rng: np.random.Generator,
) -> tuple[ParamTree, optax.OptState, Dict[str, Any], int, Dict[str, Any], np.ndarray]:
    stats = _init_stats(num_classes)

    @jax.jit
    def _train_step(
        step_params: ParamTree,
        step_opt_state: optax.OptState,
        batch_x: jax.Array,
        batch_y: jax.Array,
    ) -> tuple[ParamTree, optax.OptState, jax.Array, jax.Array]:
        def _loss_fn(loss_params: ParamTree) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
            logits = model.apply(loss_params, batch_x)
            per_sample = optax.softmax_cross_entropy_with_integer_labels(logits, batch_y)
            data_loss = jnp.mean(per_sample)
            if weight_decay > 0.0:
                data_loss = data_loss + 0.5 * weight_decay * _l2_penalty(model, loss_params)
            return data_loss, (logits, per_sample)

        (_loss, (logits, per_sample_loss)), grads = jax.value_and_grad(_loss_fn, has_aux=True)(step_params)
        updates, next_opt_state = optimizer.update(grads, step_opt_state, step_params)
        next_params = optax.apply_updates(step_params, updates)
        preds = jnp.argmax(logits, axis=1)
        return next_params, next_opt_state, per_sample_loss, preds

    bump_state: Dict[str, Any] = {
        "active": False,
        "focus_class": -1,
        "focus_weight": 1.0,
        "phase": 0.0,
    }
    sampled_distributions: list[np.ndarray] = []

    for _ in range(steps_this_epoch):
        in_uniform_tail = global_step >= (0.95 * float(total_steps))
        if bumps_enabled and (not in_uniform_tail):
            class_probs, focus_class, focus_weight, bump_phase = _compute_class_probabilities(
                step=global_step,
                num_classes=num_classes,
                period_length=period_length,
                w_max=w_max,
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

        params, opt_state, per_sample_loss, preds = _train_step(params, opt_state, x, y)

        _update_stats(stats, per_sample_loss, preds, y, num_classes)
        global_step += 1

    if sampled_distributions:
        sampled_distribution_arr = np.stack(sampled_distributions, axis=0)
    else:
        sampled_distribution_arr = np.zeros((0, num_classes), dtype=np.float64)

    return params, opt_state, _finalize_stats(stats), global_step, bump_state, sampled_distribution_arr


def _csv_header(num_classes: int) -> list[str]:
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

    bumps_before_tpt = _as_bool(dynamics_cfg.get("bumps_before_TPT", False))
    bumps_at_tpt = _as_bool(dynamics_cfg.get("bumps_at_TPT", False))
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

    built_model = build_model(
        model_cfg=model_cfg,
        input_shape=dataset_bundle.input_shape,
        num_classes=num_classes,
        random_seed=random_seed,
    )
    model = built_model.model
    params = jax.device_put(built_model.params, device=device)

    if optimizer_name == "adam":
        base_optimizer = optax.adam(
            learning_rate=learning_rate,
            b1=beta1,
            b2=beta2,
            eps=eps,
        )
    else:
        base_optimizer = optax.sgd(
            learning_rate=learning_rate,
            momentum=momentum,
            nesterov=nesterov,
        )

    if gradient_clipping is not None:
        optimizer = optax.chain(optax.clip_by_global_norm(gradient_clipping), base_optimizer)
    else:
        optimizer = base_optimizer
    opt_state = optimizer.init(params)

    csv_path = run_dir / "training_metrics.csv"
    nc_csv_path = run_dir / "nc_metrics.csv"
    figure_path = run_dir / "training_report.png"
    neural_collapse_figure_path = run_dir / "neural_collapse.png"

    nc_class_pairs = build_nc_class_pairs(num_classes)
    initialize_nc_csv(
        nc_csv_path=nc_csv_path,
        num_classes=num_classes,
        class_pairs=nc_class_pairs,
    )
    sep_csv_path = run_dir / "separability_metrics.csv"
    separability_figure_path = run_dir / "separability_measures.png"
    initialize_sep_csv(
        sep_csv_path=sep_csv_path,
        num_classes=num_classes,
        class_pairs=nc_class_pairs,
    )

    predict_step = jax.jit(lambda step_params, batch_x: model.apply(step_params, batch_x))

    total_epochs = int(math.ceil(total_steps / save_metrics_every_n_steps))
    global_step = 0
    tpt_reached = False
    tpt_step = -1
    rng = np.random.default_rng(random_seed)
    class_distribution_history: list[np.ndarray] = []

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
        writer.writerow(_csv_header(num_classes))
        f.flush()

        for epoch in range(1, total_epochs + 1):
            steps_this_epoch = min(save_metrics_every_n_steps, total_steps - global_step)
            if steps_this_epoch <= 0:
                break

            bumps_enabled = bumps_at_tpt if tpt_reached else bumps_before_tpt
            params, opt_state, _step_train_metrics, global_step, bump_state, step_distributions = train_epoch(
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
                period_length=period_length,
                w_max=w_max,
                weight_decay=weight_decay,
                rng=rng,
            )
            if step_distributions.size > 0:
                class_distribution_history.extend(step_distributions)

            train_metrics = evaluate_arrays(
                params=params,
                predict_step=predict_step,
                inputs=dataset_bundle.train_inputs,
                targets=dataset_bundle.train_targets,
                batch_size=eval_batch_size,
                device=device,
                num_classes=num_classes,
            )
            test_metrics = evaluate_arrays(
                params=params,
                predict_step=predict_step,
                inputs=dataset_bundle.test_inputs,
                targets=dataset_bundle.test_targets,
                batch_size=eval_batch_size,
                device=device,
                num_classes=num_classes,
            )

            nc_raw = collect_nc_raw_epoch(
                model=model,
                params=params,
                inputs=dataset_bundle.train_inputs,
                targets=dataset_bundle.train_targets,
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

            sep_raw: SepEpochRaw = collect_sep_raw_epoch(
                model=model,
                params=params,
                inputs=dataset_bundle.train_inputs,
                targets=dataset_bundle.train_targets,
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

    dataset_name = str(data_cfg.get("dataset_name", "spiral"))
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

    finalize_nc_metrics(
        nc_csv_path=nc_csv_path,
        output_path=neural_collapse_figure_path,
        num_classes=num_classes,
        class_pairs=nc_class_pairs,
        tpt_step=tpt_step if tpt_reached else -1,
    )
    finalize_sep_metrics(
        sep_csv_path=sep_csv_path,
        output_path=separability_figure_path,
        num_classes=num_classes,
        class_pairs=nc_class_pairs,
        tpt_step=tpt_step if tpt_reached else -1,
    )

    boundary_figure_path: Path | None = None
    if dataset_name.strip().lower() == "spiral":
        boundary_figure_path = run_dir / "spiral_decision_boundaries.png"
        plot_spiral_decision_boundaries(
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
        "tpt_reached": bool(tpt_reached),
        "tpt_step": int(tpt_step),
        "csv_path": str(csv_path),
        "nc_csv_path": str(nc_csv_path),
        "figure_path": str(figure_path),
        "neural_collapse_figure_path": str(neural_collapse_figure_path),
        "separability_figure_path": str(separability_figure_path),
        "distribution_figure_path": str(distribution_figure_path),
        "decision_boundary_path": str(boundary_figure_path) if boundary_figure_path is not None else None,
        "final_train_loss": float(last_train_metrics["loss"]),
        "final_train_accuracy": float(last_train_metrics["accuracy"]),
        "final_test_loss": float(last_test_metrics["loss"]),
        "final_test_accuracy": float(last_test_metrics["accuracy"]),
    }

    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary