"""metrics_for_multiple_classes.py
=====================================
Fully-vectorized, GPU-friendly metrics for multi-class NC / geometry runs.

This module computes, in a single jitted JAX program per epoch (no Python
loop over classes or class pairs):

  * train_accuracy, test_accuracy               (global, not per-class)
  * classifier condition_number, path_curvature_ratio
  * NC1, NC2 (mean |cos - ETF target| deviation)
  * avg_separation_margin                        (mean over all class pairs)
  * cyl_half_length_mean, cyl_radius_mean         (mean over all classes)
  * bhattacharyya_distance_mean                   (mean over all class pairs)
  * nc1_deflated, nc2_deflated_deviation          (after removing each
    class's own first-PC "cylinder axis" — see projection_PCA_analysis.py
    for the un-vectorized reference implementation this mirrors)
  * nc1_ratio_deflated, nc2_ratio_deflated        (deflated / original)

Vectorization notes
--------------------
- Per-class sums/means: one-hot matmul (`onehot.T @ x`), a GEMM — fast on GPU.
- Per-class covariance (C, D, D): built with a single scatter-add
  (`jnp.zeros(...).at[targets].add(outer)`) instead of a Python loop over
  classes, then a single *batched* `jnp.linalg.eigh` call extracts every
  class's first principal component (cylinder axis) at once.
- All-pairs quantities (NC2 cosine deviation, separation margin,
  Bhattacharyya distance) are computed as full (C, C) matrices and reduced
  with an upper-triangular mask, instead of iterating over `class_pairs`.
  This means this module does NOT need a `class_pairs` list at all — every
  unique pair is covered automatically and exactly once.
- The whole pipeline is wrapped in a single `jax.jit` (static on
  `num_classes`), so one call = one compiled program on the device.

Caveat: the pairwise Bhattacharyya step builds a (C, C, D, D) tensor and
solves C*C batched DxD linear systems. This is very fast for typical
pre-classifier widths (D up to a few hundred) and moderate C (a few dozen to
~100 classes). For very large C * D, consider chunking the class-pair axis;
not needed for the dataset configs in this repo.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

EPS = 1e-12
REG_EPS = 1e-6


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class MultiClassEpochRaw:
    condition_number: float
    path_curvature_ratio: float
    nc1: float
    nc2_deviation: float
    avg_separation_margin: float
    cyl_half_length_mean: float
    cyl_radius_mean: float
    bhattacharyya_distance_mean: float
    nc1_deflated: float
    nc2_deflated_deviation: float
    nc1_ratio_deflated: float
    nc2_ratio_deflated: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flatten_features(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.reshape(x, (x.shape[0], -1))


def _extract_classifier_weight_matrix(params: dict) -> jnp.ndarray:
    """Same convention as classifier_metrics.py: params['classifier']['kernel'],
    transposed to (num_classes, feature_dim)."""
    try:
        kernel = params["classifier"]["kernel"]
    except Exception as exc:  # pragma: no cover - guards mismatched params.
        raise ValueError("params must include params['classifier']['kernel']") from exc
    return jnp.transpose(jnp.asarray(kernel), (1, 0))


def compute_batched_logits(
    model,
    params,
    inputs: np.ndarray,
    targets: np.ndarray,
    eval_batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a batched forward pass to get logits for a (typically held-out)
    dataset, without materializing per-class activations. Returns
    (logits, targets) as numpy arrays, concatenated across batches."""
    if eval_batch_size <= 0:
        raise ValueError("eval_batch_size must be > 0")
    num_samples = int(inputs.shape[0])
    if num_samples == 0:
        raise ValueError("No samples provided to compute_batched_logits.")

    logits_chunks: list[np.ndarray] = []
    for start in range(0, num_samples, eval_batch_size):
        end = min(num_samples, start + eval_batch_size)
        batch_x = np.asarray(inputs[start:end], dtype=np.float32)
        batch_logits = model.apply(params, batch_x, return_intermediates=False)
        logits_chunks.append(np.asarray(batch_logits, dtype=np.float32))

    logits = np.concatenate(logits_chunks, axis=0)
    return logits, np.asarray(targets, dtype=np.int64)


# ---------------------------------------------------------------------------
# Core vectorized computation (single jitted program, no per-class loop)
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=("num_classes",))
def _compute_multiclass_core(
    features: jnp.ndarray,               # (N_train, D)
    targets: jnp.ndarray,                # (N_train,)
    train_logits: jnp.ndarray,           # (N_train, C)
    test_logits: jnp.ndarray,            # (N_test, C)
    test_targets: jnp.ndarray,           # (N_test,)
    weight_matrix: jnp.ndarray,          # (C, D)
    initial_weight_matrix: jnp.ndarray,  # (C, D)
    cumulative_weight_distance: jnp.ndarray,  # scalar
    num_classes: int,
) -> dict[str, jnp.ndarray]:
    C = num_classes
    N, D = features.shape

    # ---- classifier weight geometry ----
    singular_vals = jnp.linalg.svd(weight_matrix, compute_uv=False)
    condition_number = singular_vals[0] / jnp.maximum(singular_vals[-1], EPS)

    weight_delta = weight_matrix - initial_weight_matrix
    weight_delta_norm = jnp.linalg.norm(weight_delta)
    path_curvature_ratio = cumulative_weight_distance / (weight_delta_norm + EPS)

    # ---- per-class sums / means, fully vectorized via one-hot matmul ----
    onehot = jax.nn.one_hot(targets, C, dtype=features.dtype)          # (N, C)
    counts = jnp.sum(onehot, axis=0)                                   # (C,)
    class_sums = onehot.T @ features                                   # (C, D)
    class_means = class_sums / counts[:, None]
    global_mean = jnp.mean(class_means, axis=0)
    centered_means = class_means - global_mean

    deltas = features - class_means[targets]                          # (N, D)
    sq_dev = jnp.sum(deltas * deltas, axis=-1)                        # (N,)

    def _nc1_nc2_margin(centered_means_, sq_dev_, onehot_, counts_):
        s_w = jnp.sum(sq_dev_) / jnp.maximum(N, 1)
        s_b = jnp.sum(centered_means_ * centered_means_) / C
        nc1_ = s_w / (s_b + EPS)

        norms = jnp.linalg.norm(centered_means_, axis=1)
        gram = centered_means_ @ centered_means_.T
        denom = jnp.outer(norms, norms) + EPS
        cos_mat = jnp.clip(gram / denom, -1.0, 1.0)

        pair_mask = jnp.triu(jnp.ones((C, C), dtype=centered_means_.dtype), k=1)
        num_pairs = jnp.sum(pair_mask)
        etf_target = -1.0 / jnp.maximum(C - 1, 1)
        nc2_ = jnp.sum(pair_mask * jnp.abs(cos_mat - etf_target)) / (num_pairs + EPS)

        within_var_sum = onehot_.T @ sq_dev_                          # (C,)
        within_var_by_class = within_var_sum / counts_
        per_class_radius = jnp.sqrt(jnp.clip(within_var_by_class, 0.0, None))
        diff = centered_means_[:, None, :] - centered_means_[None, :, :]
        dist_mat = jnp.sqrt(jnp.clip(jnp.sum(diff * diff, axis=-1), 0.0, None))
        r_sum = per_class_radius[:, None] + per_class_radius[None, :]
        margin_mat = (dist_mat - r_sum) / (r_sum + EPS)
        avg_margin_ = jnp.sum(pair_mask * margin_mat) / (num_pairs + EPS)
        return nc1_, nc2_, avg_margin_

    nc1, nc2, avg_margin = _nc1_nc2_margin(centered_means, sq_dev, onehot, counts)

    # ---- per-class cylinder axis (first PC), batched covariance + eigh ----
    outer = deltas[:, :, None] * deltas[:, None, :]                   # (N, D, D)
    cov_sum = jnp.zeros((C, D, D), dtype=features.dtype).at[targets].add(outer)
    cov = cov_sum / jnp.maximum(counts - 1.0, 1.0)[:, None, None]
    cov = cov + REG_EPS * jnp.eye(D, dtype=features.dtype)[None, :, :]

    eigvals, eigvecs = jnp.linalg.eigh(cov)                            # ascending, batched
    lambda1 = eigvals[:, -1]
    axis = eigvecs[:, :, -1]                                           # (C, D) unit vectors
    half_length = jnp.sqrt(jnp.clip(lambda1, 0.0, None))
    rest_mean = jnp.mean(eigvals[:, :-1], axis=-1)
    radius = jnp.sqrt(jnp.clip(rest_mean, 0.0, None))

    # ---- pairwise Bhattacharyya distance over all (C, C) pairs at once ----
    pair_mask = jnp.triu(jnp.ones((C, C), dtype=features.dtype), k=1)
    num_pairs = jnp.sum(pair_mask)
    cov_avg = 0.5 * (cov[:, None] + cov[None, :])                      # (C, C, D, D)
    delta_mu = class_means[None, :, :] - class_means[:, None, :]       # (C, C, D)

    cov_avg_flat = cov_avg.reshape(C * C, D, D)
    delta_flat = delta_mu.reshape(C * C, D, 1)
    solve_flat = jnp.linalg.solve(cov_avg_flat, delta_flat)            # batched solve
    term1 = (0.125 * jnp.sum(delta_flat[..., 0] * solve_flat[..., 0], axis=-1)).reshape(C, C)

    _, logdet_avg_flat = jnp.linalg.slogdet(cov_avg_flat)
    logdet_avg = logdet_avg_flat.reshape(C, C)
    _, logdet_i = jnp.linalg.slogdet(cov)                              # (C,)
    term2 = 0.5 * (logdet_avg - 0.5 * (logdet_i[:, None] + logdet_i[None, :]))

    db_mat = term1 + term2
    bhattacharyya_mean = jnp.sum(pair_mask * db_mat) / (num_pairs + EPS)

    # ---- cylinder-deflated NC1 / NC2 (remove each sample's own-class axis) ----
    u_per_sample = axis[targets]                                       # (N, D)
    proj_coeff = jnp.sum(features * u_per_sample, axis=-1, keepdims=True)
    deflated = features - proj_coeff * u_per_sample

    class_sums_d = onehot.T @ deflated
    class_means_d = class_sums_d / counts[:, None]
    global_mean_d = jnp.mean(class_means_d, axis=0)
    centered_means_d = class_means_d - global_mean_d
    deltas_d = deflated - class_means_d[targets]
    sq_dev_d = jnp.sum(deltas_d * deltas_d, axis=-1)

    nc1_defl, nc2_defl, _ = _nc1_nc2_margin(centered_means_d, sq_dev_d, onehot, counts)

    nc1_ratio = nc1_defl / (nc1 + EPS)
    nc2_ratio = nc2_defl / (nc2 + EPS)

    return {
        "condition_number": condition_number,
        "path_curvature_ratio": path_curvature_ratio,
        "nc1": nc1,
        "nc2": nc2,
        "avg_margin": avg_margin,
        "half_length_mean": jnp.mean(half_length),
        "radius_mean": jnp.mean(radius),
        "bhattacharyya_mean": bhattacharyya_mean,
        "nc1_deflated": nc1_defl,
        "nc2_deflated": nc2_defl,
        "nc1_ratio_deflated": nc1_ratio,
        "nc2_ratio_deflated": nc2_ratio,
    }


# ---------------------------------------------------------------------------
# Public epoch-level API
# ---------------------------------------------------------------------------

def collect_multiclass_epoch(
    pre_classifier: np.ndarray,
    targets: np.ndarray,
    train_logits: np.ndarray,
    test_logits: np.ndarray,
    test_targets: np.ndarray,
    weight_matrix: np.ndarray,
    initial_weight_matrix: np.ndarray,
    cumulative_weight_distance: float,
    num_classes: int,
) -> MultiClassEpochRaw:
    """Compute all multi-class metrics for one epoch in a single jitted call."""
    targets_np = np.asarray(targets, dtype=np.int64)
    class_counts = np.bincount(targets_np, minlength=num_classes)
    missing = np.where(class_counts <= 0)[0].tolist()
    if missing:
        raise ValueError(f"Cannot compute multi-class metrics for classes with no samples: {missing}")

    test_targets_np = np.asarray(test_targets, dtype=np.int64)
    test_missing = np.where(np.bincount(test_targets_np, minlength=num_classes) <= 0)[0].tolist()
    # Test-set class coverage is not required (test accuracy is global), so
    # we don't raise here — just proceed.

    features = _flatten_features(jnp.asarray(pre_classifier, dtype=jnp.float32))
    targets_arr = jnp.asarray(targets_np, dtype=jnp.int32)
    train_logits_arr = jnp.asarray(train_logits, dtype=jnp.float32)
    test_logits_arr = jnp.asarray(test_logits, dtype=jnp.float32)
    test_targets_arr = jnp.asarray(test_targets_np, dtype=jnp.int32)
    weight_matrix_arr = jnp.asarray(weight_matrix, dtype=jnp.float32)
    initial_weight_matrix_arr = jnp.asarray(initial_weight_matrix, dtype=jnp.float32)
    cumulative_weight_distance_arr = jnp.asarray(cumulative_weight_distance, dtype=jnp.float32)

    result = _compute_multiclass_core(
        features,
        targets_arr,
        train_logits_arr,
        test_logits_arr,
        test_targets_arr,
        weight_matrix_arr,
        initial_weight_matrix_arr,
        cumulative_weight_distance_arr,
        num_classes,
    )
    result = {k: float(np.asarray(v)) for k, v in result.items()}

    return MultiClassEpochRaw(
        condition_number=result["condition_number"],
        path_curvature_ratio=result["path_curvature_ratio"],
        nc1=result["nc1"],
        nc2_deviation=result["nc2"],
        avg_separation_margin=result["avg_margin"],
        cyl_half_length_mean=result["half_length_mean"],
        cyl_radius_mean=result["radius_mean"],
        bhattacharyya_distance_mean=result["bhattacharyya_mean"],
        nc1_deflated=result["nc1_deflated"],
        nc2_deflated_deviation=result["nc2_deflated"],
        nc1_ratio_deflated=result["nc1_ratio_deflated"],
        nc2_ratio_deflated=result["nc2_ratio_deflated"],
    )


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

_CSV_FIELDS: list[str] = [
    "condition_number",
    "path_curvature_ratio",
    "nc1",
    "nc2_deviation",
    "avg_separation_margin",
    "cyl_half_length_mean",
    "cyl_radius_mean",
    "bhattacharyya_distance_mean",
    "nc1_deflated",
    "nc2_deflated_deviation",
    "nc1_ratio_deflated",
    "nc2_ratio_deflated",
]


def initialize_multiclass_csv(csv_path: Path) -> None:
    header = ["epoch", "global_step"] + _CSV_FIELDS
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(header)


def append_multiclass_csv_row(
    csv_path: Path,
    epoch: int,
    global_step: int,
    raw: MultiClassEpochRaw,
) -> None:
    row: list[float | int] = [epoch, global_step]
    row.extend(float(getattr(raw, field)) for field in _CSV_FIELDS)
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(row)


def _load_multiclass_columns(csv_path: Path) -> list[dict[str, float]]:
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


# ---------------------------------------------------------------------------
# Plotting / finalization
# ---------------------------------------------------------------------------

def finalize_multiclass_plots(
    csv_path: Path,
    output_path: Path,
    tpt_step: int = -1,
) -> None:
    rows = _load_multiclass_columns(csv_path)
    if not rows:
        raise ValueError(f"No multi-class metric rows found in {csv_path}")

    steps = np.asarray([int(row["global_step"]) for row in rows], dtype=np.int64)
    cols = {name: np.asarray([row[name] for row in rows], dtype=np.float64) for name in _CSV_FIELDS}

    fig, axes = plt.subplots(4, 2, figsize=(16, 20), sharex=True)

    ax = axes[0, 0]
    ax.plot(steps, cols["condition_number"], linewidth=1.8, label="condition number", color="steelblue")
    ax.set_ylabel("kappa(W) (log)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(steps, cols["path_curvature_ratio"], linewidth=1.4, color="darkorange", label="path curvature ratio")
    ax2.set_ylabel("Path curvature ratio")
    ax.set_title("Classifier Condition Number & Path Curvature")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    ax = axes[0, 1]
    ax.plot(steps, cols["nc1"], linewidth=1.8, label="NC1 original", color="steelblue")
    ax.plot(steps, cols["nc1_deflated"], linewidth=1.8, label="NC1 deflated", color="tomato", linestyle="--")
    ax.set_title("NC1: Original vs Cylinder-Deflated")
    ax.set_ylabel("NC1 (log)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(steps, cols["nc2_deviation"], linewidth=1.8, label="NC2 original", color="steelblue")
    ax.plot(steps, cols["nc2_deflated_deviation"], linewidth=1.8, label="NC2 deflated", color="tomato", linestyle="--")
    ax.set_title("NC2 Deviation: Original vs Cylinder-Deflated")
    ax.set_ylabel("Mean |cos - ETF target|")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(steps, cols["avg_separation_margin"], linewidth=1.8, color="seagreen")
    ax.axhline(0.0, color="red", linestyle="--", linewidth=1.4, label="touching (margin=0)")
    ax.set_title("Average Separation Margin (mean over all class pairs)")
    ax.set_ylabel("Margin")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[2, 0]
    ax.plot(steps, cols["cyl_half_length_mean"], linewidth=1.8, label="mean half-length L")
    ax.plot(steps, cols["cyl_radius_mean"], linewidth=1.8, label="mean radius r", linestyle="--")
    ax.set_title("Cylinder Half-Length & Radius (mean over classes)")
    ax.set_ylabel("Length / Radius")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[2, 1]
    ax.plot(steps, cols["bhattacharyya_distance_mean"], linewidth=1.8, color="purple")
    ax.set_title("Bhattacharyya Distance (mean over all class pairs)")
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Distance")
    ax.grid(True, alpha=0.3)

    ax = axes[3, 0]
    ax.plot(steps, cols["nc1_ratio_deflated"], linewidth=1.8, label="NC1 ratio (deflated/original)")
    ax.plot(steps, cols["nc2_ratio_deflated"], linewidth=1.8, label="NC2 ratio (deflated/original)", linestyle="--")
    ax.axhline(1.0, color="black", linestyle=":", linewidth=1.2, label="no change")
    ax.set_title("Deflation Ratios (< 1 confirms cylinder hypothesis)")
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Ratio")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    if tpt_step >= 0:
        for r in range(4):
            for c in range(2):
                axes[r, c].axvline(tpt_step, color="black", linestyle="-", linewidth=2.0)

    fig.suptitle("Multi-Class Metrics (global accuracy, NC, geometry)", fontsize=16)
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)