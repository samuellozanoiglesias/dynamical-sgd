"""metrics_for_multiple_classes.py
=====================================
Fully-vectorized, GPU-friendly metrics for multi-class NC / geometry runs.

This module computes, in a single jitted JAX program per epoch (no Python
loop over classes or class pairs):

  * NC1, NC2 (mean |cos - ETF target| deviation)                    [k=0]
  * avg_separation_margin                        (mean over all class pairs)
  * cyl_half_length_mean, cyl_radius_mean         (mean over all classes)
  * bhattacharyya_distance_mean                   (mean over all class pairs)
  * pca_alignment_mean                            (mean |cos| between each
    class's first-PC "cylinder axis" and every other class's, over all
    class pairs)
  * participation_ratio_mean                      (effective dimensionality
    of within-class scatter: PR_c = (sum lambda)^2 / sum(lambda^2). PR_c ~ d
    for an isotropic "ball", PR_c ~ 1 for a pure "cylinder".)
  * elongation_mean                                (lambda1 / trace(Sigma_c),
    i.e. fraction of within-class variance carried by the single dominant
    eigenvector. Cheap complement to participation ratio.)
  * axis_between_class_alignment_mean              (mean |cos| between each
    class's dominant elongation axis and its own between-class separation
    direction mu_c - global_mean. Near 0 => elongation is a harmless shared
    nuisance direction; near 1 => the elongation axis actually carries
    discriminative signal, i.e. deflating it could *hurt* separability.)
  * nc1_k{1,2,3}, nc2_k{1,2,3}                     (full deflation curve:
    NC1/NC2 recomputed after removing each class's own top-k within-class
    eigenvectors, for k = 1, 2, 3. k=0 is just the original nc1/nc2.)
  * nc1_ratio_k{1,2,3}, nc2_ratio_k{1,2,3}          (deflated / original,
    i.e. NC(k) / NC(0). Ratio << 1 for any k is evidence of cylinder-like
    rather than ball-like within-class geometry.)

Vectorization notes
--------------------
- Per-class sums/means: one-hot matmul (`onehot.T @ x`), a GEMM -- fast on GPU.
- Per-class covariance (C, D, D): built with a single scatter-add
  (`jnp.zeros(...).at[targets].add(outer)`) instead of a Python loop over
  classes, then a single *batched* `jnp.linalg.eigh` call extracts the full
  eigenvalue spectrum and eigenbasis for every class at once. This single
  eigh call is reused for the cylinder radius/half-length, the participation
  ratio, the elongation coefficient, and the top-K axes used for the
  deflation curve -- no extra eigendecompositions anywhere in this module.
- All-pairs quantities (NC2 cosine deviation, separation margin,
  Bhattacharyya distance, PCA axis alignment) are computed as full (C, C)
  matrices and reduced with an upper-triangular mask, instead of iterating
  over `class_pairs`. This means this module does NOT need a `class_pairs`
  list at all -- every unique pair is covered automatically and exactly once.
- The deflation curve (k=1,2,3) is computed with a single cumulative-sum
  trick: project each sample onto its class's top-K eigenvectors once,
  then `jnp.cumsum` along the K axis gives the reconstruction to subtract
  for every k in one shot, instead of K separate projection passes.
- The whole pipeline is wrapped in a single `jax.jit` (static on
  `num_classes`), so one call = one compiled program on the device.

Caveat: the pairwise Bhattacharyya step builds a (C, C, D, D) tensor and
solves C*C batched DxD linear systems. This is very fast for typical
pre-classifier widths (D up to a few hundred) and moderate C (a few dozen to
~100 classes); it remains the dominant cost in this module, well above the
cost of the new PR/elongation/deflation-curve metrics. For very large C * D,
consider chunking the class-pair axis; not needed for the dataset configs
in this repo.
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

# Max k for the NC(k) deflation curve (k = 1, 2, 3; k = 0 is the original NC).
DEFLATION_MAX_K = 3


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class MultiClassEpochRaw:
    nc1: float
    nc2_deviation: float
    avg_separation_margin: float
    cyl_half_length_mean: float
    cyl_radius_mean: float
    bhattacharyya_distance_mean: float
    pca_alignment_mean: float
    participation_ratio_mean: float
    elongation_mean: float
    axis_between_class_alignment_mean: float
    nc1_k1: float
    nc1_k2: float
    nc1_k3: float
    nc2_k1: float
    nc2_k2: float
    nc2_k3: float
    nc1_ratio_k1: float
    nc1_ratio_k2: float
    nc1_ratio_k3: float
    nc2_ratio_k1: float
    nc2_ratio_k2: float
    nc2_ratio_k3: float
    sensitive_param_fraction: float
    mean_weight_step_distance: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flatten_features(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.reshape(x, (x.shape[0], -1))


# ---------------------------------------------------------------------------
# Core vectorized computation (single jitted program, no per-class loop)
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=("num_classes",))
def _compute_multiclass_core(
    features: jnp.ndarray,               # (N, D)
    targets: jnp.ndarray,                # (N,)
    num_classes: int,
) -> dict[str, jnp.ndarray]:
    C = num_classes
    N, D = features.shape

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

    # ---- per-class within-class covariance + full batched eigendecomposition ----
    # This single eigh call is reused below for: cylinder radius/half-length,
    # participation ratio, elongation coefficient, cross-class PCA axis
    # alignment, the between-class alignment check, and the top-K axes used
    # for the deflation curve. No further eigendecompositions are needed.
    outer = deltas[:, :, None] * deltas[:, None, :]                   # (N, D, D)
    cov_sum = jnp.zeros((C, D, D), dtype=features.dtype).at[targets].add(outer)
    cov = cov_sum / jnp.maximum(counts - 1.0, 1.0)[:, None, None]
    cov = cov + REG_EPS * jnp.eye(D, dtype=features.dtype)[None, :, :]

    eigvals, eigvecs = jnp.linalg.eigh(cov)                            # ascending, batched
    lambda1 = eigvals[:, -1]
    axis = eigvecs[:, :, -1]                                           # (C, D) unit vectors, top PC
    half_length = jnp.sqrt(jnp.clip(lambda1, 0.0, None))
    rest_mean = jnp.mean(eigvals[:, :-1], axis=-1)
    radius = jnp.sqrt(jnp.clip(rest_mean, 0.0, None))

    # ---- participation ratio (effective dimensionality) & elongation ----
    # PR_c = (sum_i lambda_i)^2 / sum_i lambda_i^2. Ball (isotropic, dim d)
    # -> PR_c ~= d; pure cylinder (one dominant direction) -> PR_c ~= 1.
    sum_eigvals = jnp.sum(eigvals, axis=-1)                            # (C,) = trace(Sigma_c)
    sum_eigvals_sq = jnp.sum(eigvals * eigvals, axis=-1)               # (C,)
    participation_ratio = (sum_eigvals * sum_eigvals) / (sum_eigvals_sq + EPS)
    participation_ratio_mean = jnp.mean(participation_ratio)

    # e_c = lambda1 / trace(Sigma_c): fraction of within-class variance
    # carried by the single dominant direction. e_c ~= 1/D for a ball,
    # e_c -> 1 for a pure cylinder. Cheaper, more directly interpretable
    # complement to the participation ratio above.
    elongation = lambda1 / (sum_eigvals + EPS)
    elongation_mean = jnp.mean(elongation)

    # ---- pairwise quantities over all (C, C) pairs at once ----
    pair_mask = jnp.triu(jnp.ones((C, C), dtype=features.dtype), k=1)
    num_pairs = jnp.sum(pair_mask)

    # Mean PCA axis alignment: |cos| between each pair of classes' first
    # principal component ("cylinder axis"), averaged over all class pairs.
    axis_gram = axis @ axis.T                                         # (C, C)
    axis_cos_mat = jnp.clip(axis_gram, -1.0, 1.0)
    pca_alignment_mean = jnp.sum(pair_mask * jnp.abs(axis_cos_mat)) / (num_pairs + EPS)

    # Between-class alignment: |cos| between each class's own dominant
    # elongation axis and its own between-class direction (mu_c - global
    # mean). ~0 => elongation is a harmless shared nuisance direction that
    # rides along independently of class separation (expected for the
    # "cylinder" story); ~1 => the elongation axis itself carries
    # discriminative signal, so deflating it would remove real separation,
    # not just nuisance variance.
    between_norms = jnp.linalg.norm(centered_means, axis=-1)
    axis_between_cos = jnp.sum(axis * centered_means, axis=-1) / (between_norms + EPS)
    axis_between_class_alignment_mean = jnp.mean(jnp.abs(axis_between_cos))

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

    # ---- full deflation curve: NC1(k) / NC2(k) for k = 1, 2, ..., DEFLATION_MAX_K ----
    # Remove each sample's own class's top-k within-class eigenvectors
    # (ranked by eigenvalue, largest first) and recompute NC1/NC2. This
    # generalizes the old single-step (k=1) deflation to a full curve: a
    # true "cylinder" shows a sharp NC1 drop at k=1 then a plateau; a more
    # complex or genuinely ball-like geometry will not show that elbow.
    #
    # `eigvecs` is ascending by eigenvalue, so reversing the last axis and
    # taking the first K columns gives the top-K eigenvectors in descending
    # order in a single slice (no extra eigh, no per-k gather).
    top_axes = eigvecs[:, :, ::-1][:, :, :DEFLATION_MAX_K]             # (C, D, K)
    axes_per_sample = top_axes[targets]                                 # (N, D, K)

    # Project once onto all K axes, then a single cumsum gives the
    # cumulative reconstruction to subtract for every k in one shot.
    coeffs = jnp.einsum("nd,ndk->nk", features, axes_per_sample)       # (N, K)
    contrib = coeffs[:, None, :] * axes_per_sample                     # (N, D, K)
    cum_contrib = jnp.cumsum(contrib, axis=-1)                         # (N, D, K)

    nc1_curve = [nc1]
    nc2_curve = [nc2]
    for k in range(1, DEFLATION_MAX_K + 1):
        deflated_k = features - cum_contrib[:, :, k - 1]               # (N, D)
        class_sums_k = onehot.T @ deflated_k
        class_means_k = class_sums_k / counts[:, None]
        global_mean_k = jnp.mean(class_means_k, axis=0)
        centered_means_k = class_means_k - global_mean_k
        deltas_k = deflated_k - class_means_k[targets]
        sq_dev_k = jnp.sum(deltas_k * deltas_k, axis=-1)

        nc1_k, nc2_k, _ = _nc1_nc2_margin(centered_means_k, sq_dev_k, onehot, counts)
        nc1_curve.append(nc1_k)
        nc2_curve.append(nc2_k)

    out = {
        "nc1": nc1,
        "nc2": nc2,
        "avg_margin": avg_margin,
        "half_length_mean": jnp.mean(half_length),
        "radius_mean": jnp.mean(radius),
        "bhattacharyya_mean": bhattacharyya_mean,
        "pca_alignment_mean": pca_alignment_mean,
        "participation_ratio_mean": participation_ratio_mean,
        "elongation_mean": elongation_mean,
        "axis_between_class_alignment_mean": axis_between_class_alignment_mean,
    }
    for k in range(1, DEFLATION_MAX_K + 1):
        out[f"nc1_k{k}"] = nc1_curve[k]
        out[f"nc2_k{k}"] = nc2_curve[k]
        out[f"nc1_ratio_k{k}"] = nc1_curve[k] / (nc1_curve[0] + EPS)
        out[f"nc2_ratio_k{k}"] = nc2_curve[k] / (nc2_curve[0] + EPS)

    return out


# ---------------------------------------------------------------------------
# Public epoch-level API
# ---------------------------------------------------------------------------

def collect_multiclass_epoch(
    pre_classifier: np.ndarray,
    targets: np.ndarray,
    weight_matrix: np.ndarray,
    num_classes: int,
    previous_weight_matrix: np.ndarray | None = None,
    sensitivity_threshold: float = 1e-3,
) -> MultiClassEpochRaw:
    """Compute all multi-class metrics for one epoch in a single jitted call.

    Note: `weight_matrix` / `previous_weight_matrix` are only used for the
    step-wise weight-movement tracking (`sensitive_param_fraction`,
    `mean_weight_step_distance`), computed outside the jitted core. The
    classifier condition-number / path-curvature metrics that used to also
    consume the weight matrix here have been removed; if your training_runner
    was passing `initial_weight_matrix` / `cumulative_weight_distance` only
    for those, those call-site arguments can be dropped too -- happy to
    patch training_runner if you share it.
    """
    targets_np = np.asarray(targets, dtype=np.int64)
    class_counts = np.bincount(targets_np, minlength=num_classes)
    missing = np.where(class_counts <= 0)[0].tolist()
    if missing:
        raise ValueError(f"Cannot compute multi-class metrics for classes with no samples: {missing}")

    features = _flatten_features(jnp.asarray(pre_classifier, dtype=jnp.float32))
    targets_arr = jnp.asarray(targets_np, dtype=jnp.int32)

    # --- Step-wise weight movement (this step vs. the previous recorded step) ---
    if previous_weight_matrix is None:
        sensitive_param_fraction = float("nan")
        mean_weight_step_distance = float("nan")
    else:
        previous_weights = np.asarray(previous_weight_matrix, dtype=np.float64)
        if previous_weights.shape != weight_matrix.shape or weight_matrix.size == 0:
            sensitive_param_fraction = float("nan")
            mean_weight_step_distance = float("nan")
        else:
            step_abs_delta = np.abs(weight_matrix - previous_weights)
            sensitive_param_fraction = float(np.mean(step_abs_delta > sensitivity_threshold))
            mean_weight_step_distance = float(np.mean(step_abs_delta))

    result = _compute_multiclass_core(features, targets_arr, num_classes)
    result = {k: float(np.asarray(v)) for k, v in result.items()}

    return MultiClassEpochRaw(
        nc1=result["nc1"],
        nc2_deviation=result["nc2"],
        avg_separation_margin=result["avg_margin"],
        cyl_half_length_mean=result["half_length_mean"],
        cyl_radius_mean=result["radius_mean"],
        bhattacharyya_distance_mean=result["bhattacharyya_mean"],
        pca_alignment_mean=result["pca_alignment_mean"],
        participation_ratio_mean=result["participation_ratio_mean"],
        elongation_mean=result["elongation_mean"],
        axis_between_class_alignment_mean=result["axis_between_class_alignment_mean"],
        nc1_k1=result["nc1_k1"],
        nc1_k2=result["nc1_k2"],
        nc1_k3=result["nc1_k3"],
        nc2_k1=result["nc2_k1"],
        nc2_k2=result["nc2_k2"],
        nc2_k3=result["nc2_k3"],
        nc1_ratio_k1=result["nc1_ratio_k1"],
        nc1_ratio_k2=result["nc1_ratio_k2"],
        nc1_ratio_k3=result["nc1_ratio_k3"],
        nc2_ratio_k1=result["nc2_ratio_k1"],
        nc2_ratio_k2=result["nc2_ratio_k2"],
        nc2_ratio_k3=result["nc2_ratio_k3"],
        sensitive_param_fraction=sensitive_param_fraction,
        mean_weight_step_distance=mean_weight_step_distance,
    )


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

_CSV_FIELDS: list[str] = [
    "nc1",
    "nc2_deviation",
    "avg_separation_margin",
    "cyl_half_length_mean",
    "cyl_radius_mean",
    "bhattacharyya_distance_mean",
    "pca_alignment_mean",
    "participation_ratio_mean",
    "elongation_mean",
    "axis_between_class_alignment_mean",
    "nc1_k1",
    "nc1_k2",
    "nc1_k3",
    "nc2_k1",
    "nc2_k2",
    "nc2_k3",
    "nc1_ratio_k1",
    "nc1_ratio_k2",
    "nc1_ratio_k3",
    "nc2_ratio_k1",
    "nc2_ratio_k2",
    "nc2_ratio_k3",
    "sensitive_param_fraction",
    "mean_weight_step_distance",
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
# Plotting / finalization -- general geometry metrics (no NC1/NC2, those
# moved to finalize_shape_metrics_plots below)
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

    fig, axes = plt.subplots(3, 2, figsize=(16, 15), sharex=True)

    ax = axes[0, 0]
    ax.plot(steps, cols["avg_separation_margin"], linewidth=1.8, color="seagreen")
    ax.axhline(0.0, color="red", linestyle="--", linewidth=1.4, label="touching (margin=0)")
    ax.set_title("Average Separation Margin (mean over all class pairs)")
    ax.set_ylabel("Margin")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(steps, cols["cyl_half_length_mean"], linewidth=1.8, label="mean half-length L")
    ax.plot(steps, cols["cyl_radius_mean"], linewidth=1.8, label="mean radius r", linestyle="--")
    ax.set_title("Cylinder Half-Length & Radius (mean over classes)")
    ax.set_ylabel("Length / Radius")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(steps, cols["bhattacharyya_distance_mean"], linewidth=1.8, color="purple")
    ax.set_title("Bhattacharyya Distance (mean over all class pairs)")
    ax.set_ylabel("Distance")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(steps, cols["pca_alignment_mean"], linewidth=1.8, color="darkred")
    ax.set_title("Mean PCA Axis Alignment (mean |cos| over all class pairs)")
    ax.set_ylabel("Mean |cos(axis_i, axis_j)|")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)

    ax = axes[2, 0]
    ax.plot(steps, cols["sensitive_param_fraction"], linewidth=1.8, color="darkorange")
    ax.set_title("Sensitive Parameter Fraction (|delta| > threshold)")
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Fraction of parameters")
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    ax.plot(steps, cols["mean_weight_step_distance"], linewidth=1.8, color="steelblue")
    ax.set_title("Mean Weight Step Distance")
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Distance")
    ax.grid(True, alpha=0.3)

    if tpt_step >= 0:
        for r in range(3):
            for c in range(2):
                axes[r, c].axvline(tpt_step, color="black", linestyle="-", linewidth=2.0)

    fig.suptitle("Multi-Class Metrics (separation geometry & weight dynamics)", fontsize=16)
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plotting / finalization -- NC1/NC2 + cylinder-shape metrics (PR,
# elongation, full deflation curve)
# ---------------------------------------------------------------------------

def finalize_shape_metrics_plots(
    csv_path: Path,
    output_path: Path,
    tpt_step: int = -1,
) -> None:
    """Plots NC1/NC2 (original + full k=0..3 deflation curve), the
    deflation ratio curves, participation ratio, elongation coefficient,
    and the between-class alignment check -- everything needed to
    distinguish "ball-like" from "cylinder-like" within-class collapse."""
    rows = _load_multiclass_columns(csv_path)
    if not rows:
        raise ValueError(f"No multi-class metric rows found in {csv_path}")

    steps = np.asarray([int(row["global_step"]) for row in rows], dtype=np.int64)
    cols = {name: np.asarray([row[name] for row in rows], dtype=np.float64) for name in _CSV_FIELDS}

    k_colors = {0: "steelblue", 1: "tomato", 2: "darkorange", 3: "seagreen"}

    fig, axes = plt.subplots(4, 2, figsize=(16, 20), sharex=True)

    ax = axes[0, 0]
    ax.plot(steps, cols["nc1"], linewidth=1.8, label="k=0 (original)", color=k_colors[0])
    for k in (1, 2, 3):
        ax.plot(steps, cols[f"nc1_k{k}"], linewidth=1.6, linestyle="--", label=f"k={k}", color=k_colors[k])
    ax.set_title("NC1(k): Full Deflation Curve")
    ax.set_ylabel("NC1 (log)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(steps, cols["nc2_deviation"], linewidth=1.8, label="k=0 (original)", color=k_colors[0])
    for k in (1, 2, 3):
        ax.plot(steps, cols[f"nc2_k{k}"], linewidth=1.6, linestyle="--", label=f"k={k}", color=k_colors[k])
    ax.set_title("NC2(k): Full Deflation Curve")
    ax.set_ylabel("Mean |cos - ETF target|")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    for k in (1, 2, 3):
        ax.plot(steps, cols[f"nc1_ratio_k{k}"], linewidth=1.6, label=f"k={k}", color=k_colors[k])
    ax.axhline(1.0, color="black", linestyle=":", linewidth=1.2, label="no change")
    ax.set_title("NC1 Deflation Ratios: NC1(k) / NC1(0)\n(< 1 supports cylinder-like geometry)")
    ax.set_ylabel("Ratio")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    for k in (1, 2, 3):
        ax.plot(steps, cols[f"nc2_ratio_k{k}"], linewidth=1.6, label=f"k={k}", color=k_colors[k])
    ax.axhline(1.0, color="black", linestyle=":", linewidth=1.2, label="no change")
    ax.set_title("NC2 Deflation Ratios: NC2(k) / NC2(0)")
    ax.set_ylabel("Ratio")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[2, 0]
    ax.plot(steps, cols["participation_ratio_mean"], linewidth=1.8, color="mediumpurple")
    ax.set_title("Participation Ratio (mean effective dimensionality)\nball -> ~D, cylinder -> ~1")
    ax.set_ylabel("PR_c (mean over classes)")
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    ax.plot(steps, cols["elongation_mean"], linewidth=1.8, color="crimson")
    ax.set_title("Elongation Coefficient (mean, lambda_1 / trace(Sigma_c))\nball -> ~1/D, cylinder -> ~1")
    ax.set_ylabel("e_c (mean over classes)")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)

    ax = axes[3, 0]
    ax.plot(steps, cols["axis_between_class_alignment_mean"], linewidth=1.8, color="teal")
    ax.set_title("Elongation Axis vs. Between-Class Direction\n(~0: harmless nuisance, ~1: axis carries class signal)")
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Mean |cos(axis_c, mu_c - mean)|")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)

    axes[3, 1].axis("off")

    if tpt_step >= 0:
        for r in range(4):
            for c in range(2):
                if r == 3 and c == 1:
                    continue
                axes[r, c].axvline(tpt_step, color="black", linestyle="-", linewidth=2.0)

    fig.suptitle("Cylinder- vs Ball-Like Collapse: NC(k) Deflation Curve & Shape Metrics", fontsize=16)
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)