from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from generate_dataset import DatasetBundle, build_dataset_bundle
from metrics import (
    plot_example_distribution_dynamics,
    plot_spiral_decision_boundaries,
    plot_training_report,
)
from neural_collapse import (
    append_nc_csv_row,
    build_nc_class_pairs,
    build_nc_layer_specs,
    collect_nc_raw_epoch,
    finalize_nc_metrics,
    initialize_nc_csv,
)
from model import build_model


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
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device_raw: Any) -> torch.device:
    requested = str(device_raw if device_raw is not None else "auto").strip().lower()
    if requested in {"auto", ""}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested in {"cuda", "gpu"}:
        if not torch.cuda.is_available():
            raise RuntimeError("Config requested CUDA but no CUDA device is available.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device '{requested}'. Use auto, cuda, or cpu.")


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
    per_sample_loss: torch.Tensor,
    preds: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
) -> None:
    correct = (preds == target).to(dtype=torch.float32)

    stats.loss_sum += float(per_sample_loss.sum().item())
    stats.correct_sum += int(correct.sum().item())
    stats.count_sum += int(target.shape[0])

    class_count = torch.bincount(target, minlength=num_classes)
    class_loss = torch.bincount(target, weights=per_sample_loss, minlength=num_classes)
    class_correct = torch.bincount(target, weights=correct, minlength=num_classes)

    stats.class_count_sum += class_count.detach().cpu().numpy().astype(np.float64)
    stats.class_loss_sum += class_loss.detach().cpu().numpy().astype(np.float64)
    stats.class_correct_sum += class_correct.detach().cpu().numpy().astype(np.float64)


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
    T = float(period_length)
    slope = 2.0 * (w_max - 1.0) / T
    if t < T / 2.0:
        focus_weight = 1.0 + t * slope
    else:
        focus_weight = 2.0 * w_max - t * slope - 1.0
    phase = t / T
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
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    class_to_indices: Dict[int, np.ndarray],
    class_counts: np.ndarray,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
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
    index_tensor = torch.from_numpy(sampled_indices)
    return train_inputs[index_tensor], train_targets[index_tensor], class_counts_int


@torch.no_grad()
def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> Dict[str, Any]:
    model.eval()
    stats = _init_stats(num_classes)
    use_non_blocking = device.type == "cuda"

    for data, target in loader:
        data = data.to(device, non_blocking=use_non_blocking)
        target = target.to(device, non_blocking=use_non_blocking)
        logits = model(data)
        per_sample_loss = F.cross_entropy(logits, target, reduction="none")
        preds = torch.argmax(logits, dim=1)
        _update_stats(stats, per_sample_loss, preds, target, num_classes)

    return _finalize_stats(stats)


def train_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_dataset: TensorDataset,
    class_to_indices: Dict[int, np.ndarray],
    batch_size: int,
    device: torch.device,
    num_classes: int,
    global_step: int,
    steps_this_epoch: int,
    total_steps: int,
    bumps_enabled: bool,
    period_length: int,
    w_max: float,
    gradient_clipping: float | None,
    rng: np.random.Generator,
) -> tuple[Dict[str, Any], int, Dict[str, Any], np.ndarray]:
    model.train()
    stats = _init_stats(num_classes)

    train_inputs, train_targets = train_dataset.tensors
    use_non_blocking = device.type == "cuda"

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

        batch_data = batch_data.to(device, non_blocking=use_non_blocking)
        batch_target = batch_target.to(device, non_blocking=use_non_blocking)

        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_data)
        per_sample_loss = F.cross_entropy(logits, batch_target, reduction="none")
        loss = per_sample_loss.mean()
        loss.backward()

        if gradient_clipping is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clipping)

        optimizer.step()

        preds = torch.argmax(logits, dim=1)
        _update_stats(stats, per_sample_loss.detach(), preds.detach(), batch_target.detach(), num_classes)
        global_step += 1

    if sampled_distributions:
        sampled_distribution_arr = np.stack(sampled_distributions, axis=0)
    else:
        sampled_distribution_arr = np.zeros((0, num_classes), dtype=np.float64)

    return _finalize_stats(stats), global_step, bump_state, sampled_distribution_arr


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
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    total_steps = int(training_cfg.get("training_steps", training_cfg.get("total_steps", 25000)))
    steps_per_epoch = int(training_cfg.get("steps_per_epoch", 50))
    batch_size = int(training_cfg.get("batch_size", 128))
    eval_batch_size = int(training_cfg.get("eval_batch_size", batch_size))
    num_workers = int(training_cfg.get("num_workers", 0))

    if total_steps <= 0:
        raise ValueError("training.training_steps must be > 0")
    if steps_per_epoch <= 0:
        raise ValueError("training.steps_per_epoch must be > 0")
    if batch_size <= 0:
        raise ValueError("training.batch_size must be > 0")

    optimizer_name = str(optimizer_cfg.get("optimizer_type", training_cfg.get("optimizer", "adam"))).strip().lower()
    loss_name = str(training_cfg.get("loss", "cross_entropy")).strip().lower()
    if optimizer_name != "adam":
        raise ValueError(f"Only Adam optimizer is supported in this runner. Got '{optimizer_name}'.")
    if loss_name not in {"cross_entropy", "cross-entropy", "ce"}:
        raise ValueError(f"Only cross-entropy loss is supported in this runner. Got '{loss_name}'.")

    learning_rate = float(optimizer_cfg.get("learning_rate", training_cfg.get("learning_rate", 0.002)))
    beta1 = float(optimizer_cfg.get("beta1", training_cfg.get("beta1", 0.9)))
    beta2 = float(optimizer_cfg.get("beta2", training_cfg.get("beta2", 0.999)))
    eps = float(optimizer_cfg.get("eps", training_cfg.get("eps", 1e-8)))
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

    model = build_model(model_cfg, dataset_bundle.input_shape, num_classes, device)
    nc_layer_specs = build_nc_layer_specs(model)
    nc_layer_names = [spec.name for spec in nc_layer_specs]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=(beta1, beta2),
        eps=eps,
        weight_decay=weight_decay,
    )

    test_loader = DataLoader(
        dataset_bundle.test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    # Report metrics from full-dataset evaluation so plots are independent of bump-weighted batches.
    train_eval_loader = DataLoader(
        dataset_bundle.train_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    nc_loader = DataLoader(
        dataset_bundle.train_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    csv_path = run_dir / "training_metrics.csv"
    nc_csv_path = run_dir / "nc_metrics.csv"
    figure_path = run_dir / "training_report.png"
    neural_collapse_figure_path = run_dir / "neural_collapse.png"
    nc_class_pairs = build_nc_class_pairs(num_classes)
    initialize_nc_csv(
        nc_csv_path=nc_csv_path,
        layer_names=nc_layer_names,
        num_classes=num_classes,
        class_pairs=nc_class_pairs,
    )

    total_epochs = int(math.ceil(total_steps / steps_per_epoch))
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
            steps_this_epoch = min(steps_per_epoch, total_steps - global_step)
            if steps_this_epoch <= 0:
                break

            bumps_enabled = bumps_at_tpt if tpt_reached else bumps_before_tpt
            _step_train_metrics, global_step, bump_state, step_distributions = train_epoch(
                model=model,
                optimizer=optimizer,
                train_dataset=dataset_bundle.train_dataset,
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
                gradient_clipping=gradient_clipping,
                rng=rng,
            )
            if step_distributions.size > 0:
                class_distribution_history.extend(step_distributions)
            train_metrics = evaluate_loader(model, train_eval_loader, device, num_classes)
            test_metrics = evaluate_loader(model, test_loader, device, num_classes)

            nc_raw = collect_nc_raw_epoch(
                model=model,
                loader=nc_loader,
                device=device,
                num_classes=num_classes,
                class_pairs=nc_class_pairs,
                layer_specs=nc_layer_specs,
            )
            append_nc_csv_row(
                nc_csv_path=nc_csv_path,
                epoch=epoch,
                global_step=global_step,
                raw=nc_raw,
                layer_names=nc_layer_names,
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
        layer_names=nc_layer_names,
        num_classes=num_classes,
        class_pairs=nc_class_pairs,
        tpt_step=tpt_step if tpt_reached else -1,
    )

    boundary_figure_path: Path | None = None
    if dataset_name.strip().lower() == "spiral":
        boundary_figure_path = run_dir / "spiral_decision_boundaries.png"
        plot_spiral_decision_boundaries(
            model=model,
            train_dataset=dataset_bundle.train_dataset,
            test_dataset=dataset_bundle.test_dataset,
            output_path=boundary_figure_path,
            title=f"{run_label} | {dataset_name} | {architecture} | Decision Boundaries",
        )

    summary = {
        "run_label": run_label,
        "device": str(device),
        "total_steps": total_steps,
        "steps_per_epoch": steps_per_epoch,
        "epochs": total_epochs,
        "tpt_reached": bool(tpt_reached),
        "tpt_step": int(tpt_step),
        "csv_path": str(csv_path),
        "nc_csv_path": str(nc_csv_path),
        "figure_path": str(figure_path),
        "neural_collapse_figure_path": str(neural_collapse_figure_path),
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
