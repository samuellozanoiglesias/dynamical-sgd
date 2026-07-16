from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


EPS = 1e-12


@dataclass
class ClassifierEpochRaw:
    weight_mean_alignment: np.ndarray
    weight_norms: np.ndarray
    margin_mean_by_class: np.ndarray
    margin_var_by_class: np.ndarray
    condition_number: float


@dataclass
class ClassifierAdvancedRaw:
    correct_logit_mean_by_class: np.ndarray
    max_wrong_logit_mean_by_class: np.ndarray
    weight_orthogonality: float
    stable_rank: float
    path_curvature_ratio: float
    gsnr_by_class: np.ndarray
    sensitive_param_fraction: float
    mean_weight_step_distance: float


def _flatten_features(x: np.ndarray) -> np.ndarray:
    return np.reshape(x, (x.shape[0], -1))


def _extract_classifier_weight_matrix(params: dict) -> jnp.ndarray:
    try:
        kernel = params["classifier"]["kernel"]
    except Exception as exc:  # pragma: no cover - guards against mismatched params.
        raise ValueError("params must include params['classifier']['kernel']") from exc
    return jnp.transpose(jnp.asarray(kernel), (1, 0))


def collect_classifier_epoch(
    pre_classifier: np.ndarray,
    logits: np.ndarray,
    targets: np.ndarray,
    weight_matrix: np.ndarray,
    eps: float = EPS,
) -> ClassifierEpochRaw:
    features = _flatten_features(np.asarray(pre_classifier, dtype=np.float64))
    logits_arr = np.asarray(logits, dtype=np.float64)
    targets_arr = np.asarray(targets, dtype=np.int64)
    weights = np.asarray(weight_matrix, dtype=np.float64)

    num_samples = int(features.shape[0])
    if num_samples == 0:
        raise ValueError("No samples provided for classifier metrics.")

    num_classes = int(weights.shape[0])
    if logits_arr.ndim != 2 or logits_arr.shape[1] != num_classes:
        logits_arr = features @ weights.T

    class_counts = np.bincount(targets_arr, minlength=num_classes).astype(np.float64)
    class_sums = np.zeros((num_classes, features.shape[1]), dtype=np.float64)
    for class_id in range(num_classes):
        mask = targets_arr == class_id
        if np.any(mask):
            class_sums[class_id] = np.sum(features[mask], axis=0)

    class_means = np.zeros_like(class_sums)
    valid = class_counts > 0
    class_means[valid] = class_sums[valid] / class_counts[valid, None]

    weight_norms = np.linalg.norm(weights, axis=1)
    mean_norms = np.linalg.norm(class_means, axis=1)
    alignment = np.sum(weights * class_means, axis=1) / (weight_norms * mean_norms + eps)
    alignment[~valid] = np.nan

    margin_mean = np.full(num_classes, np.nan, dtype=np.float64)
    margin_var = np.full(num_classes, np.nan, dtype=np.float64)
    if num_classes > 1:
        correct_logits = logits_arr[np.arange(num_samples), targets_arr]
        masked = np.array(logits_arr, copy=True)
        masked[np.arange(num_samples), targets_arr] = -np.inf
        max_other = np.max(masked, axis=1)
        margins = correct_logits - max_other
        for class_id in range(num_classes):
            mask = targets_arr == class_id
            if np.any(mask):
                vals = margins[mask]
                margin_mean[class_id] = float(np.mean(vals))
                margin_var[class_id] = float(np.var(vals))

    if weights.size == 0:
        condition_number = float("nan")
    else:
        singular_vals = np.linalg.svd(weights, compute_uv=False)
        if singular_vals.size == 0:
            condition_number = float("nan")
        else:
            condition_number = float(singular_vals[0] / max(float(singular_vals[-1]), eps))

    return ClassifierEpochRaw(
        weight_mean_alignment=alignment.astype(np.float64),
        weight_norms=weight_norms.astype(np.float64),
        margin_mean_by_class=margin_mean.astype(np.float64),
        margin_var_by_class=margin_var.astype(np.float64),
        condition_number=condition_number,
    )


def collect_advanced_classifier_metrics(
    weight_matrix: dict,
    initial_weight_matrix: dict,
    cumulative_weight_distance: float,
    logits: np.ndarray,
    targets: np.ndarray,
    grads: np.ndarray | None,
    eps: float = EPS,
    previous_weight_matrix: np.ndarray | None = None,
    sensitivity_threshold: float = 1e-3,
) -> ClassifierAdvancedRaw:
    logits_arr = np.asarray(logits, dtype=np.float64)
    targets_arr = np.asarray(targets, dtype=np.int64)

    weights = np.asarray(weight_matrix, dtype=np.float64)
    initial_weights = np.asarray(initial_weight_matrix, dtype=np.float64)
    num_classes = int(weights.shape[0])

    correct_logit_mean = np.full(num_classes, np.nan, dtype=np.float64)
    max_wrong_logit_mean = np.full(num_classes, np.nan, dtype=np.float64)
    if logits_arr.ndim == 2 and logits_arr.shape[1] == num_classes:
        num_samples = logits_arr.shape[0]
        if num_samples > 0 and targets_arr.shape[0] == num_samples:
            correct_logits = logits_arr[np.arange(num_samples), targets_arr]
            masked = np.array(logits_arr, copy=True)
            masked[np.arange(num_samples), targets_arr] = -np.inf
            max_wrong = np.max(masked, axis=1)
            for class_id in range(num_classes):
                mask = targets_arr == class_id
                if np.any(mask):
                    correct_logit_mean[class_id] = float(np.mean(correct_logits[mask]))
                    max_wrong_logit_mean[class_id] = float(np.mean(max_wrong[mask]))

    if num_classes <= 1:
        weight_orthogonality = float("nan")
    else:
        norms = jnp.linalg.norm(weights, axis=1)
        denom = norms[:, None] * norms[None, :] + eps
        cos = (weights @ weights.T) / denom
        sum_off_diag = jnp.sum(cos) - jnp.sum(jnp.diag(cos))
        weight_orthogonality = float(sum_off_diag / (num_classes * (num_classes - 1)))

    if weights.size == 0:
        stable_rank = float("nan")
    else:
        singular_vals = jnp.linalg.svd(weights, compute_uv=False)
        if singular_vals.size == 0:
            stable_rank = float("nan")
        else:
            fro_sq = jnp.sum(jnp.square(singular_vals))
            spectral_sq = jnp.square(jnp.max(singular_vals))
            stable_rank = float(fro_sq / (spectral_sq + eps))

    delta = weights - initial_weights
    delta_norm = float(jnp.linalg.norm(delta))
    path_curvature_ratio = float(float(cumulative_weight_distance) / (delta_norm + eps))

    gsnr_by_class = np.full(num_classes, np.nan, dtype=np.float64)
    if grads is not None:
        grads_arr = np.asarray(grads, dtype=np.float64)
        if grads_arr.ndim == 2:
            grads_arr = grads_arr[None, ...]
        if grads_arr.ndim == 3:
            if grads_arr.shape[1] != num_classes and grads_arr.shape[2] == num_classes:
                grads_arr = np.transpose(grads_arr, (0, 2, 1))
            if grads_arr.shape[1] == num_classes:
                for class_id in range(num_classes):
                    class_grads = grads_arr[:, class_id, :]
                    mean_grad = np.mean(class_grads, axis=0)
                    var_grad = np.var(class_grads, axis=0)
                    gsnr_by_class[class_id] = float(np.mean(np.square(mean_grad) / (var_grad + eps)))

    # --- Step-wise weight movement (this step vs. the previous recorded step) ---
    # "Sensitive" parameters are ones whose weight moved by more than
    # `sensitivity_threshold` since the last measurement; mean_weight_step_distance
    # is the average |delta| across ALL parameters, which is expected to shrink
    # as training converges (large early steps, near-zero later steps).
    if previous_weight_matrix is None:
        sensitive_param_fraction = float("nan")
        mean_weight_step_distance = float("nan")
    else:
        previous_weights = np.asarray(previous_weight_matrix, dtype=np.float64)
        if previous_weights.shape != weights.shape or weights.size == 0:
            sensitive_param_fraction = float("nan")
            mean_weight_step_distance = float("nan")
        else:
            step_abs_delta = np.abs(weights - previous_weights)
            sensitive_param_fraction = float(np.mean(step_abs_delta > sensitivity_threshold))
            mean_weight_step_distance = float(np.mean(step_abs_delta))

    return ClassifierAdvancedRaw(
        correct_logit_mean_by_class=correct_logit_mean,
        max_wrong_logit_mean_by_class=max_wrong_logit_mean,
        weight_orthogonality=weight_orthogonality,
        stable_rank=stable_rank,
        path_curvature_ratio=path_curvature_ratio,
        gsnr_by_class=gsnr_by_class,
        sensitive_param_fraction=sensitive_param_fraction,
        mean_weight_step_distance=mean_weight_step_distance,
    )


def initialize_classifier_csv(csv_path: Path, num_classes: int) -> None:
    header = ["epoch", "global_step", "condition_number"]
    header.extend([f"weight_mean_alignment_class_{class_id}" for class_id in range(num_classes)])
    header.extend([f"weight_norm_class_{class_id}" for class_id in range(num_classes)])
    header.extend([f"logit_margin_mean_class_{class_id}" for class_id in range(num_classes)])
    header.extend([f"logit_margin_var_class_{class_id}" for class_id in range(num_classes)])
    header.extend([f"logit_correct_mean_class_{class_id}" for class_id in range(num_classes)])
    header.extend([f"logit_max_wrong_mean_class_{class_id}" for class_id in range(num_classes)])
    header.append("weight_orthogonality")
    header.append("stable_rank")
    header.append("path_curvature_ratio")
    header.append("sensitive_param_fraction")
    header.append("mean_weight_step_distance")
    header.extend([f"gsnr_class_{class_id}" for class_id in range(num_classes)])

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)


def append_classifier_csv_row(
    csv_path: Path,
    epoch: int,
    global_step: int,
    raw: ClassifierEpochRaw,
    num_classes: int,
    advanced_raw: ClassifierAdvancedRaw | None = None,
) -> None:
    row: list[float | int] = [epoch, global_step, float(raw.condition_number)]
    row.extend(raw.weight_mean_alignment[:num_classes].tolist())
    row.extend(raw.weight_norms[:num_classes].tolist())
    row.extend(raw.margin_mean_by_class[:num_classes].tolist())
    row.extend(raw.margin_var_by_class[:num_classes].tolist())

    if advanced_raw is None:
        correct_logit = [float("nan")] * num_classes
        max_wrong_logit = [float("nan")] * num_classes
        weight_orthogonality = float("nan")
        stable_rank = float("nan")
        path_curvature_ratio = float("nan")
        sensitive_param_fraction = float("nan")
        mean_weight_step_distance = float("nan")
        gsnr = [float("nan")] * num_classes
    else:
        correct_logit = advanced_raw.correct_logit_mean_by_class[:num_classes].tolist()
        max_wrong_logit = advanced_raw.max_wrong_logit_mean_by_class[:num_classes].tolist()
        weight_orthogonality = float(advanced_raw.weight_orthogonality)
        stable_rank = float(advanced_raw.stable_rank)
        path_curvature_ratio = float(advanced_raw.path_curvature_ratio)
        sensitive_param_fraction = float(advanced_raw.sensitive_param_fraction)
        mean_weight_step_distance = float(advanced_raw.mean_weight_step_distance)
        gsnr = advanced_raw.gsnr_by_class[:num_classes].tolist()

    row.extend(correct_logit)
    row.extend(max_wrong_logit)
    row.append(weight_orthogonality)
    row.append(stable_rank)
    row.append(path_curvature_ratio)
    row.append(sensitive_param_fraction)
    row.append(mean_weight_step_distance)
    row.extend(gsnr)

    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def _load_classifier_columns(csv_path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            casted: dict[str, float] = {}
            for key, value in row.items():
                if value is None or value == "":
                    casted[key] = float("nan")
                    continue
                if key in {"epoch", "global_step"}:
                    casted[key] = float(int(float(value)))
                else:
                    casted[key] = float(value)
            rows.append(casted)
    return rows


def finalize_classifier_metrics(
    csv_path: Path,
    output_path: Path,
    num_classes: int,
    tpt_step: int = -1,
) -> None:
    rows = _load_classifier_columns(csv_path)
    if not rows:
        raise ValueError(f"No classifier rows found in {csv_path}")

    steps = np.asarray([int(row["global_step"]) for row in rows], dtype=np.int64)
    alignment = np.asarray(
        [[row[f"weight_mean_alignment_class_{class_id}"] for class_id in range(num_classes)] for row in rows],
        dtype=np.float64,
    )
    weight_norms = np.asarray(
        [[row[f"weight_norm_class_{class_id}"] for class_id in range(num_classes)] for row in rows],
        dtype=np.float64,
    )
    margin_mean = np.asarray(
        [[row[f"logit_margin_mean_class_{class_id}"] for class_id in range(num_classes)] for row in rows],
        dtype=np.float64,
    )
    margin_var = np.asarray(
        [[row[f"logit_margin_var_class_{class_id}"] for class_id in range(num_classes)] for row in rows],
        dtype=np.float64,
    )
    condition_number = np.asarray([row["condition_number"] for row in rows], dtype=np.float64)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True)

    ax = axes[0, 0]
    for class_id in range(num_classes):
        ax.plot(steps, alignment[:, class_id], linewidth=1.6, label=f"class {class_id}")
    ax.set_title("Weight-Mean Alignment")
    ax.set_ylabel("Cosine similarity")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, alpha=0.3)
    if num_classes <= 12:
        ax.legend(fontsize=7, ncol=2)

    ax = axes[0, 1]
    for class_id in range(num_classes):
        ax.plot(steps, weight_norms[:, class_id], linewidth=1.6, label=f"class {class_id}")
    ax.set_title("Classifier Weight Norms")
    ax.set_ylabel("L2 norm")
    ax.grid(True, alpha=0.3)
    if num_classes <= 12:
        ax.legend(fontsize=7, ncol=2)

    ax = axes[1, 0]
    for class_id in range(num_classes):
        color = None
        if num_classes <= 10:
            color = plt.get_cmap("tab10")(class_id % 10)
        ax.plot(steps, margin_mean[:, class_id], linewidth=1.6, color=color)
        ax.plot(steps, margin_var[:, class_id], linewidth=1.2, linestyle="--", alpha=0.7, color=color)
    ax.set_title("Effective Logit Margin (solid=mean, dashed=var)")
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Margin")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(steps, condition_number, linewidth=1.8, label="condition number")
    ax.set_title("Classifier Condition Number")
    ax.set_xlabel("Global Step")
    ax.set_ylabel("kappa(W) (log)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    if tpt_step >= 0:
        for axis in axes.flat:
            axis.axvline(tpt_step, color="black", linestyle="-", linewidth=2.0)

    fig.suptitle("Classifier Metrics", fontsize=16)
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def finalize_classifier_simplified(
    csv_path: Path,
    output_path: Path,
    num_classes: int,
    tpt_step: int = -1,
) -> None:
    rows = _load_classifier_columns(csv_path)
    if not rows:
        raise ValueError(f"No classifier rows found in {csv_path}")

    steps = np.asarray([int(row["global_step"]) for row in rows], dtype=np.int64)
    correct_logits = np.asarray(
        [
            [row.get(f"logit_correct_mean_class_{class_id}", float("nan")) for class_id in range(num_classes)]
            for row in rows
        ],
        dtype=np.float64,
    )
    max_wrong_logits = np.asarray(
        [
            [row.get(f"logit_max_wrong_mean_class_{class_id}", float("nan")) for class_id in range(num_classes)]
            for row in rows
        ],
        dtype=np.float64,
    )
    path_curvature_ratio = np.asarray(
        [row.get("path_curvature_ratio", float("nan")) for row in rows],
        dtype=np.float64,
    )
    weight_norms = np.asarray(
        [
            [row.get(f"weight_norm_class_{class_id}", float("nan")) for class_id in range(num_classes)]
            for row in rows
        ],
        dtype=np.float64,
    )
    condition_number = np.asarray(
        [row.get("condition_number", float("nan")) for row in rows],
        dtype=np.float64,
    )

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True)

    ax = axes[0, 0]
    for class_id in range(num_classes):
        color = None
        if num_classes <= 10:
            color = plt.get_cmap("tab10")(class_id % 10)
        ax.plot(steps, correct_logits[:, class_id], linewidth=1.6, color=color, label=f"class {class_id} correct")
        ax.plot(
            steps,
            max_wrong_logits[:, class_id],
            linewidth=1.2,
            linestyle="--",
            alpha=0.8,
            color=color,
            label=f"class {class_id} max wrong",
        )
    ax.set_title("Logit Decomposition")
    ax.set_ylabel("Logit")
    ax.set_ylim(-5.0, 25.0)
    ax.grid(True, alpha=0.3)
    if num_classes <= 6:
        ax.legend(fontsize=7, ncol=2)

    ax = axes[0, 1]
    ax.plot(steps, path_curvature_ratio, linewidth=1.8)
    ax.set_title("Path Curvature Ratio")
    ax.set_ylabel("cumulative / ||W - W0||")
    ax.set_ylim(0.0, 25.0)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    for class_id in range(num_classes):
        ax.plot(steps, weight_norms[:, class_id], linewidth=1.6, label=f"class {class_id}")
    ax.set_title("Classifier Weight Norms")
    ax.set_xlabel("Global Step")
    ax.set_ylabel("L2 norm")
    ax.set_ylim(0.0, 300.0)
    ax.grid(True, alpha=0.3)
    if num_classes <= 12:
        ax.legend(fontsize=7, ncol=2)

    ax = axes[1, 1]
    ax.plot(steps, condition_number, linewidth=1.8, label="condition number")
    ax.set_title("Classifier Condition Number")
    ax.set_xlabel("Global Step")
    ax.set_ylabel("kappa(W)")
    ax.set_ylim(0.0, 15.0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    if tpt_step >= 0:
        for axis in axes.flat:
            axis.axvline(tpt_step, color="black", linestyle="-", linewidth=2.0)

    fig.suptitle("Classifier Metrics", fontsize=16)
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def finalize_classifier_dashboard(
    csv_path: Path,
    output_path: Path,
    num_classes: int,
    tpt_step: int = -1,
) -> None:
    rows = _load_classifier_columns(csv_path)
    if not rows:
        raise ValueError(f"No classifier rows found in {csv_path}")

    steps = np.asarray([int(row["global_step"]) for row in rows], dtype=np.int64)
    alignment = np.asarray(
        [[row[f"weight_mean_alignment_class_{class_id}"] for class_id in range(num_classes)] for row in rows],
        dtype=np.float64,
    )
    weight_norms = np.asarray(
        [[row[f"weight_norm_class_{class_id}"] for class_id in range(num_classes)] for row in rows],
        dtype=np.float64,
    )
    correct_logits = np.asarray(
        [[row.get(f"logit_correct_mean_class_{class_id}", float("nan")) for class_id in range(num_classes)] for row in rows],
        dtype=np.float64,
    )
    max_wrong_logits = np.asarray(
        [[row.get(f"logit_max_wrong_mean_class_{class_id}", float("nan")) for class_id in range(num_classes)] for row in rows],
        dtype=np.float64,
    )
    weight_orthogonality = np.asarray(
        [row.get("weight_orthogonality", float("nan")) for row in rows],
        dtype=np.float64,
    )
    stable_rank = np.asarray([row.get("stable_rank", float("nan")) for row in rows], dtype=np.float64)
    condition_number = np.asarray([row.get("condition_number", float("nan")) for row in rows], dtype=np.float64)
    path_curvature_ratio = np.asarray(
        [row.get("path_curvature_ratio", float("nan")) for row in rows],
        dtype=np.float64,
    )
    sensitive_param_fraction = np.asarray(
        [row.get("sensitive_param_fraction", float("nan")) for row in rows],
        dtype=np.float64,
    )
    mean_weight_step_distance = np.asarray(
        [row.get("mean_weight_step_distance", float("nan")) for row in rows],
        dtype=np.float64,
    )
    gsnr = np.asarray(
        [[row.get(f"gsnr_class_{class_id}", float("nan")) for class_id in range(num_classes)] for row in rows],
        dtype=np.float64,
    )

    fig, axes = plt.subplots(5, 2, figsize=(16, 27), sharex=True)

    ax = axes[0, 0]
    for class_id in range(num_classes):
        color = None
        if num_classes <= 10:
            color = plt.get_cmap("tab10")(class_id % 10)
        ax.plot(steps, correct_logits[:, class_id], linewidth=1.6, color=color, label=f"class {class_id} correct")
        ax.plot(
            steps,
            max_wrong_logits[:, class_id],
            linewidth=1.2,
            linestyle="--",
            alpha=0.8,
            color=color,
            label=f"class {class_id} max wrong",
        )
    ax.set_title("Logit Decomposition")
    ax.set_ylabel("Logit")
    ax.grid(True, alpha=0.3)
    if num_classes <= 6:
        ax.legend(fontsize=7, ncol=2)

    ax = axes[0, 1]
    ax.plot(steps, weight_orthogonality, linewidth=1.8)
    ax.set_title("Mean Weight Orthogonality")
    ax.set_ylabel("Avg cosine similarity")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(steps, stable_rank, linewidth=1.8)
    ax.set_title("Stable Rank")
    ax.set_ylabel("sr(W)")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(steps, path_curvature_ratio, linewidth=1.8)
    ax.set_title("Path Curvature Ratio")
    ax.set_ylabel("cumulative / ||W - W0||")
    ax.grid(True, alpha=0.3)

    ax = axes[2, 0]
    for class_id in range(num_classes):
        color = None
        if num_classes <= 10:
            color = plt.get_cmap("tab10")(class_id % 10)
        ax.plot(steps, gsnr[:, class_id], linewidth=1.4, color=color, label=f"class {class_id}")
    ax.set_title("Gradient SNR")
    ax.set_xlabel("Global Step")
    ax.set_ylabel("GSNR")
    ax.grid(True, alpha=0.3)
    if num_classes <= 10:
        ax.legend(fontsize=7, ncol=2)

    ax = axes[2, 1]
    for class_id in range(num_classes):
        ax.plot(steps, weight_norms[:, class_id], linewidth=1.6, label=f"class {class_id}")
    ax.set_title("Classifier Weight Norms")
    ax.set_xlabel("Global Step")
    ax.set_ylabel("L2 norm")
    ax.grid(True, alpha=0.3)
    if num_classes <= 12:
        ax.legend(fontsize=7, ncol=2)

    ax = axes[3, 0]
    ax.plot(steps, sensitive_param_fraction, linewidth=1.8, color="tab:red")
    ax.set_title("Sensitive Parameter Fraction")
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Fraction with |Δw| > threshold")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)

    ax = axes[3, 1]
    ax.plot(steps, mean_weight_step_distance, linewidth=1.8, color="tab:purple")
    ax.set_title("Mean Weight Step Distance")
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Mean |Δw| over all parameters")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    ax = axes[4, 0]
    for class_id in range(num_classes):
        ax.plot(steps, alignment[:, class_id], linewidth=1.6, label=f"class {class_id}")
    ax.set_title("Weight-Mean Alignment")
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Cosine similarity")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, alpha=0.3)
    if num_classes <= 12:
        ax.legend(fontsize=7, ncol=2)

    ax = axes[4, 1]
    ax.plot(steps, condition_number, linewidth=1.8, label="condition number")
    ax.set_title("Classifier Condition Number")
    ax.set_xlabel("Global Step")
    ax.set_ylabel("kappa(W) (log)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    if tpt_step >= 0:
        for axis in axes.flat:
            axis.axvline(tpt_step, color="black", linestyle="-", linewidth=2.0)

    fig.suptitle("Classifier Diagnostics Dashboard", fontsize=16)
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)