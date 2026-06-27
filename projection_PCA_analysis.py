"""projection_PCA_analysis.py
=================================
Cylinder-deflation NC analysis.

Hypothesis
----------
Class clouds in the pre-classifier space are *cylinders*: they are strongly
elongated along one dominant direction (the first per-class PCA component)
and otherwise behave like lower-dimensional "disks" in the orthogonal
complement.  Standard NC metrics are sensitive to this elongation and
therefore give pessimistic NC1 values even when the disk cross-sections are
well-separated.

This module tests that hypothesis by:

1. Computing the **first principal component** of each class cloud
   independently (the "cylinder axis" of that class).
2. **Projecting out** that axis from every sample of every class, leaving
   each point in the (D-1)-dimensional orthogonal complement.  This is
   a separate deflation per class, so each class loses its own dominant
   direction rather than a shared global direction.
3. Re-centering the deflated clouds to remove any residual global drift,
   then computing standard NC1 / NC2 metrics on the resulting D-1
   dimensional representations.

Additionally, three diagnostic quantities are tracked every epoch:

* **cylinder_elongation[c]**: ratio of the first eigenvalue to the mean of
  the remaining eigenvalues for class c.  Large values (>> 1) confirm the
  cylinder shape.
* **axis_alignment[pair]**: |cos θ| between the cylinder axes of each pair
  of classes.  If the axes converge to 1.0 the cylinders become parallel.
* **axis_angle_deg[pair]**: angle in degrees between the cylinder axes (0°
  = parallel, 90° = orthogonal).

  
Add the following flag and paths alongside the other `enable_*` flags
(around line 916-924):

    enable_proj_nc = _analysis_output_enabled(analysis_cfg, ("proj_nc_analysis",))
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

Update the `collect_pre_classifier` flag (around line 924) so that the
pre-classifier activations are also gathered for this analysis:

    collect_pre_classifier = (
        enable_classifier_metrics or enable_pca_analysis
        or enable_geo_metrics or enable_hp_metrics
        or enable_proj_nc          # <-- add this line
    )

Inside the per-epoch block (after the existing pca / geo blocks), add:

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

After the training loop add the finalization call alongside the others:

    if enable_proj_nc:
        finalize_proj_nc_plots(
            csv_path=proj_nc_csv_path,
            output_path=proj_nc_figure_path,
            num_classes=num_classes,
            class_pairs=nc_class_pairs,
            tpt_step=tpt_step if tpt_reached else -1,
        )

Enable the analysis in your YAML / JSON config:

    analysis:
      proj_nc_analysis: true
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

EPS = 1e-12


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class ProjNCEpochRaw:
    """All per-epoch metrics produced by the cylinder-deflation analysis."""

    # Per-class elongation ratio: λ_1 / mean(λ_2 … λ_D).
    # Large values confirm a cylinder shape.
    cylinder_elongation: np.ndarray          # shape (num_classes,)

    # Absolute cosine between cylinder axes for every class pair.
    # 1.0 means the cylinders are perfectly parallel.
    axis_alignment: np.ndarray               # shape (num_pairs,)

    # Angle in degrees between cylinder axes (0° = parallel).
    axis_angle_deg: np.ndarray               # shape (num_pairs,)

    # NC1 on the *deflated* (D-1)-dimensional features
    # (within-class / between-class variability ratio, lower = more collapsed).
    nc1_deflated: float

    # NC2 deviation on the deflated features
    # (mean |cosine - ETF target|, lower = closer to equiangular frame).
    nc2_deflated: float

    # NC1 on the original (un-deflated) full-dimensional features,
    # stored here for a side-by-side comparison without a separate call.
    nc1_original: float

    # NC2 deviation on the original full-dimensional features.
    nc2_original: float


# ---------------------------------------------------------------------------
# Core geometry helpers
# ---------------------------------------------------------------------------

def _flatten(x: np.ndarray) -> np.ndarray:
    """Collapse spatial / channel dims: (N, ...) → (N, D)."""
    return np.reshape(x, (x.shape[0], -1))


def _per_class_first_pc(
    features: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the first principal component and elongation ratio per class.

    Parameters
    ----------
    features : (N, D) float64
    targets  : (N,)  int64
    num_classes : int

    Returns
    -------
    axes : (num_classes, D) – unit vectors, one per class
    elongation : (num_classes,) – λ_1 / mean(λ_2 … λ_D)
    """
    D = features.shape[1]
    axes = np.zeros((num_classes, D), dtype=np.float64)
    elongation = np.zeros(num_classes, dtype=np.float64)

    for c in range(num_classes):
        mask = targets == c
        feats_c = features[mask]                        # (N_c, D)
        n_c = feats_c.shape[0]

        if n_c == 0:
            # No samples: leave axis as zero vector, elongation = 1.
            elongation[c] = 1.0
            continue

        # Center within the class.
        centered_c = feats_c - np.mean(feats_c, axis=0, keepdims=True)

        if n_c == 1 or D == 0:
            # Degenerate: no covariance available.
            elongation[c] = 1.0
            continue

        # Economy SVD: cheaper than full eigdecomp when N_c << D.
        # We only need the first right-singular vector.
        try:
            # Shape of V: (D, D) but we only use the first column.
            _U, sv, Vt = np.linalg.svd(centered_c, full_matrices=False)
            # Vt rows are right-singular vectors; Vt[0] is the dominant one.
            axes[c] = Vt[0]                             # unit vector in R^D

            # Elongation: (λ_1 / mean of the rest), where λ_i = sv_i^2 / (n_c-1).
            # We work with squared singular values directly.
            sv2 = sv * sv
            lambda_1 = float(sv2[0])
            rest = sv2[1:]
            if rest.size > 0 and float(np.mean(rest)) > EPS:
                elongation[c] = lambda_1 / float(np.mean(rest))
            elif rest.size == 0:
                elongation[c] = float("inf")
            else:
                elongation[c] = float("inf")
        except np.linalg.LinAlgError:
            # SVD failed (pathological input); fall back to identity axis.
            axes[c, 0] = 1.0
            elongation[c] = 1.0

    return axes, elongation


def _deflate_per_class(
    features: np.ndarray,
    targets: np.ndarray,
    axes: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Remove each class's cylinder axis from its own samples.

    For every sample x of class c, compute:
        x_deflated = x - (x · u_c) * u_c
    where u_c is the unit cylinder axis of class c.

    The deflated features live in a (D-1)-dimensional affine subspace
    per class but all remain in the ambient R^D space, so they can still
    be compared across classes.

    Parameters
    ----------
    features    : (N, D) float64
    targets     : (N,)   int64
    axes        : (num_classes, D) float64 – unit vectors from _per_class_first_pc
    num_classes : int

    Returns
    -------
    deflated : (N, D) float64
    """
    deflated = features.copy()
    for c in range(num_classes):
        mask = targets == c
        if not np.any(mask):
            continue
        u = axes[c]                                     # (D,)
        u_norm = float(np.dot(u, u))
        if u_norm < EPS:
            continue                                    # zero axis – nothing to remove
        # Project out: x -= (x · u / |u|^2) * u
        # (u is already unit-normalised from SVD, so u_norm ≈ 1, kept for safety)
        projections = deflated[mask] @ u                # (N_c,)
        deflated[mask] -= np.outer(projections / u_norm, u)
    return deflated


def _nc_metrics(
    features: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
    class_pairs: list[tuple[int, int]],
) -> tuple[float, float]:
    """Compute NC1 and NC2 deviation from an (N, D) feature array.

    NC1 = S_W / S_B  (within-class variance / between-class variance)
    NC2 = mean |cos(mu_i, mu_j) - cos_ETF|

    Returns (nc1, nc2_deviation).
    """
    N, D = features.shape

    # ---- class means and global mean ----
    class_counts = np.zeros(num_classes, dtype=np.float64)
    class_sums = np.zeros((num_classes, D), dtype=np.float64)
    for c in range(num_classes):
        mask = targets == c
        class_counts[c] = float(np.sum(mask))
        if class_counts[c] > 0:
            class_sums[c] = np.sum(features[mask], axis=0)

    missing = np.where(class_counts <= 0)[0]
    if missing.size:
        raise ValueError(f"Classes with no samples: {missing.tolist()}")

    class_means = class_sums / class_counts[:, None]        # (C, D)
    global_mean = np.mean(class_means, axis=0)              # (D,)
    centered_means = class_means - global_mean              # (C, D)

    # ---- NC1 ----
    deltas = features - global_mean - centered_means[targets]   # within-class deviations
    s_w = float(np.sum(deltas * deltas)) / float(max(1, N))
    s_b = float(np.sum(centered_means * centered_means)) / float(num_classes)
    nc1 = s_w / (s_b + EPS)

    # ---- NC2 ----
    norms = np.linalg.norm(centered_means, axis=1)              # (C,)
    gram = centered_means @ centered_means.T                    # (C, C)
    denom = np.outer(norms, norms) + EPS
    cos_mat = np.clip(gram / denom, -1.0, 1.0)
    pair_cos = np.array(
        [cos_mat[i, j] for i, j in class_pairs],
        dtype=np.float64,
    )
    etf_target = -1.0 / float(num_classes - 1) if num_classes > 1 else 0.0
    nc2 = float(np.mean(np.abs(pair_cos - etf_target)))

    return nc1, nc2


# ---------------------------------------------------------------------------
# Public epoch-level API
# ---------------------------------------------------------------------------

def collect_proj_nc_epoch(
    pre_classifier: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
    class_pairs: list[tuple[int, int]],
) -> ProjNCEpochRaw:
    """Compute all cylinder-deflation metrics for one epoch.

    Parameters
    ----------
    pre_classifier : (N, D) or (N, ...) float array
        Pre-classifier activations (will be flattened automatically).
    targets : (N,) int array
        Class labels in [0, num_classes).
    num_classes : int
    class_pairs : list of (int, int)
        All pairs to track axis alignment for.

    Returns
    -------
    ProjNCEpochRaw
    """
    features = _flatten(np.asarray(pre_classifier, dtype=np.float64))
    targets_arr = np.asarray(targets, dtype=np.int64)

    if features.shape[0] == 0:
        raise ValueError("No samples provided for projection NC analysis.")
    if features.shape[0] != targets_arr.shape[0]:
        raise ValueError("features and targets must have the same length.")

    # 1. Per-class cylinder axes and elongation diagnostics.
    axes, elongation = _per_class_first_pc(features, targets_arr, num_classes)

    # 2. Axis alignment between every class pair.
    num_pairs = len(class_pairs)
    axis_alignment = np.zeros(num_pairs, dtype=np.float64)
    axis_angle_deg = np.zeros(num_pairs, dtype=np.float64)
    for k, (i, j) in enumerate(class_pairs):
        u_i = axes[i]
        u_j = axes[j]
        norm_i = float(np.linalg.norm(u_i))
        norm_j = float(np.linalg.norm(u_j))
        if norm_i < EPS or norm_j < EPS:
            axis_alignment[k] = 0.0
            axis_angle_deg[k] = 90.0
        else:
            cos_val = float(np.clip(np.dot(u_i, u_j) / (norm_i * norm_j), -1.0, 1.0))
            axis_alignment[k] = abs(cos_val)            # |cos| so direction sign is ignored
            axis_angle_deg[k] = float(np.degrees(np.arccos(abs(cos_val))))

    # 3. NC metrics on the original (full-dimensional) features.
    nc1_orig, nc2_orig = _nc_metrics(features, targets_arr, num_classes, class_pairs)

    # 4. Deflate each class along its own cylinder axis.
    deflated = _deflate_per_class(features, targets_arr, axes, num_classes)

    # 5. NC metrics on the deflated features.
    nc1_defl, nc2_defl = _nc_metrics(deflated, targets_arr, num_classes, class_pairs)

    return ProjNCEpochRaw(
        cylinder_elongation=elongation,
        axis_alignment=axis_alignment,
        axis_angle_deg=axis_angle_deg,
        nc1_deflated=nc1_defl,
        nc2_deflated=nc2_defl,
        nc1_original=nc1_orig,
        nc2_original=nc2_orig,
    )


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

def initialize_proj_nc_csv(
    csv_path: Path,
    num_classes: int,
    class_pairs: list[tuple[int, int]],
) -> None:
    """Write the CSV header row."""
    header = ["epoch", "global_step"]
    # Scalar NC metrics
    header += ["nc1_original", "nc1_deflated", "nc2_original", "nc2_deflated"]
    # Per-class elongation
    header += [f"elongation_{c}" for c in range(num_classes)]
    # Per-pair axis alignment and angle
    header += [f"axis_alignment_{i}_{j}" for i, j in class_pairs]
    header += [f"axis_angle_deg_{i}_{j}" for i, j in class_pairs]

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(header)


def append_proj_nc_csv_row(
    csv_path: Path,
    epoch: int,
    global_step: int,
    raw: ProjNCEpochRaw,
    num_classes: int,
    class_pairs: list[tuple[int, int]],
) -> None:
    """Append one epoch's metrics to the CSV."""
    row: list[float | int] = [epoch, global_step]
    row += [
        float(raw.nc1_original),
        float(raw.nc1_deflated),
        float(raw.nc2_original),
        float(raw.nc2_deflated),
    ]
    row += raw.cylinder_elongation[:num_classes].tolist()
    row += raw.axis_alignment.tolist()
    row += raw.axis_angle_deg.tolist()

    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(row)


def _load_proj_nc_csv(csv_path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            casted: dict[str, float] = {}
            for key, value in row.items():
                if value is None or value == "":
                    casted[key] = float("nan")
                elif key in {"epoch", "global_step"}:
                    casted[key] = float(int(float(value)))
                else:
                    casted[key] = float(value)
            rows.append(casted)
    return rows


# ---------------------------------------------------------------------------
# Plotting / finalization
# ---------------------------------------------------------------------------

def finalize_proj_nc_plots(
    csv_path: Path,
    output_path: Path,
    num_classes: int,
    class_pairs: list[tuple[int, int]],
    tpt_step: int = -1,
) -> None:
    """Read the CSV and produce a multi-panel diagnostic figure.

    Panel layout (5 rows × 2 cols):
    ┌──────────────────────────────────────────┐
    │ [0,0] NC1: original vs deflated          │
    │ [0,1] NC2 deviation: original vs deflated│
    │ [1,0] Cylinder elongation per class      │
    │ [1,1] Axis alignment |cos θ| per pair    │
    │ [2,0] Axis angle (deg) per pair          │
    │ [2,1] NC1 ratio (deflated / original)    │
    │ [3,0] Mean elongation over time          │
    │ [3,1] Mean axis alignment over time      │
    └──────────────────────────────────────────┘
    """
    rows = _load_proj_nc_csv(csv_path)
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")

    steps = np.asarray([int(row["global_step"]) for row in rows], dtype=np.int64)

    nc1_orig = np.asarray([row["nc1_original"] for row in rows], dtype=np.float64)
    nc1_defl = np.asarray([row["nc1_deflated"] for row in rows], dtype=np.float64)
    nc2_orig = np.asarray([row["nc2_original"] for row in rows], dtype=np.float64)
    nc2_defl = np.asarray([row["nc2_deflated"] for row in rows], dtype=np.float64)

    elongation = np.asarray(
        [[row.get(f"elongation_{c}", float("nan")) for c in range(num_classes)] for row in rows],
        dtype=np.float64,
    )
    axis_alignment = np.asarray(
        [[row.get(f"axis_alignment_{i}_{j}", float("nan")) for i, j in class_pairs] for row in rows],
        dtype=np.float64,
    )
    axis_angle_deg = np.asarray(
        [[row.get(f"axis_angle_deg_{i}_{j}", float("nan")) for i, j in class_pairs] for row in rows],
        dtype=np.float64,
    )

    nc1_ratio = nc1_defl / (nc1_orig + EPS)
    mean_elongation = np.nanmean(elongation, axis=1)
    mean_alignment = np.nanmean(axis_alignment, axis=1)

    fig, axes = plt.subplots(4, 2, figsize=(18, 20), sharex=True)

    # --- [0,0] NC1 original vs deflated ---
    ax = axes[0, 0]
    ax.plot(steps, nc1_orig, linewidth=1.8, label="NC1 original", color="steelblue")
    ax.plot(steps, nc1_defl, linewidth=1.8, label="NC1 deflated (−1 PC)", color="tomato", linestyle="--")
    ax.axhline(0.0, color="black", linestyle=":", linewidth=1.2)
    ax.set_title("NC1: Original vs Cylinder-Deflated")
    ax.set_ylabel("NC1 (log scale)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # --- [0,1] NC2 deviation original vs deflated ---
    ax = axes[0, 1]
    ax.plot(steps, nc2_orig, linewidth=1.8, label="NC2 original", color="steelblue")
    ax.plot(steps, nc2_defl, linewidth=1.8, label="NC2 deflated (−1 PC)", color="tomato", linestyle="--")
    ax.axhline(0.0, color="black", linestyle=":", linewidth=1.2, label="perfect ETF")
    ax.set_title("NC2 Deviation: Original vs Deflated")
    ax.set_ylabel("Mean |cos − ETF target|")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # --- [1,0] Elongation per class ---
    ax = axes[1, 0]
    for c in range(num_classes):
        ax.plot(steps, elongation[:, c], linewidth=1.4, label=f"class {c}")
    ax.axhline(1.0, color="black", linestyle=":", linewidth=1.2, label="sphere (ratio=1)")
    ax.set_title("Cylinder Elongation per Class  [λ₁ / mean(λ₂…λ_D)]")
    ax.set_ylabel("Elongation ratio")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    if num_classes <= 16:
        ax.legend(fontsize=7, ncol=2)

    # --- [1,1] Axis alignment |cos θ| per pair ---
    ax = axes[1, 1]
    for k, (i, j) in enumerate(class_pairs):
        ax.plot(steps, axis_alignment[:, k], linewidth=1.2, label=f"{i}–{j}")
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.4, label="|cos|=1 (parallel)")
    ax.axhline(0.0, color="black", linestyle=":", linewidth=1.2, label="|cos|=0 (orthogonal)")
    ax.set_title("Cylinder Axis Alignment  |cos θ| between Class Pairs")
    ax.set_ylabel("|cos θ|")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    if len(class_pairs) <= 16:
        ax.legend(fontsize=7)

    # --- [2,0] Axis angle in degrees per pair ---
    ax = axes[2, 0]
    for k, (i, j) in enumerate(class_pairs):
        ax.plot(steps, axis_angle_deg[:, k], linewidth=1.2, label=f"{i}–{j}")
    ax.axhline(0.0, color="red", linestyle="--", linewidth=1.4, label="0° = parallel")
    ax.axhline(90.0, color="black", linestyle=":", linewidth=1.2, label="90° = orthogonal")
    ax.set_title("Cylinder Axis Angle (degrees) between Class Pairs")
    ax.set_ylabel("Angle (°)")
    ax.set_ylim(-5, 100)
    ax.grid(True, alpha=0.3)
    if len(class_pairs) <= 16:
        ax.legend(fontsize=7)

    # --- [2,1] NC1 ratio (deflated / original) ---
    ax = axes[2, 1]
    ax.plot(steps, nc1_ratio, linewidth=1.8, color="purple")
    ax.axhline(1.0, color="black", linestyle=":", linewidth=1.2, label="no change")
    ax.axhline(0.0, color="green", linestyle="--", linewidth=1.2, label="full collapse after deflation")
    ax.set_title("NC1 Ratio: Deflated / Original  (< 1 confirms cylinder hypothesis)")
    ax.set_ylabel("NC1_deflated / NC1_original")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # --- [3,0] Mean elongation ---
    ax = axes[3, 0]
    ax.plot(steps, mean_elongation, linewidth=1.8, color="darkorange")
    ax.axhline(1.0, color="black", linestyle=":", linewidth=1.2, label="sphere")
    ax.set_title("Mean Cylinder Elongation (averaged over classes)")
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Mean elongation (log)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # --- [3,1] Mean axis alignment ---
    ax = axes[3, 1]
    ax.plot(steps, mean_alignment, linewidth=1.8, color="seagreen")
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.4, label="fully parallel")
    ax.axhline(0.0, color="black", linestyle=":", linewidth=1.2, label="fully orthogonal")
    ax.set_title("Mean Axis Alignment (averaged over pairs)")
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Mean |cos θ|")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Vertical line at TPT
    if tpt_step >= 0:
        for r in range(4):
            for c in range(2):
                axes[r, c].axvline(tpt_step, color="black", linestyle="-", linewidth=2.0)

    fig.suptitle("Cylinder-Deflation NC Analysis\n"
                 "(per-class first PC removed before computing NC metrics)",
                 fontsize=14)
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)