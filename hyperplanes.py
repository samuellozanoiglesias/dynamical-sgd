"""hyperplanes.py
================
Geometric analysis of the classifier hyperplanes during training.

For a softmax classifier with weight matrix W ∈ ℝ^{C×D} and bias b ∈ ℝ^C,
the decision boundary between class i and class j is the set of points where

    (w_i - w_j)·x + (b_i - b_j) = 0

so the *effective normal* of the boundary hyperplane separating class i from
class j is  n_ij = w_i - w_j  (unnormalised).

We also track the individual row vectors w_i as "class hyperplane normals"
since class i is predicted wherever w_i·x + b_i is the largest logit, i.e.
the Voronoi-like region of class i is bounded by normals parallel to w_i.

Four analysis groups are computed at every checkpoint
─────────────────────────────────────────────────────
1. CM-alignment  :  Cosine similarity between each boundary normal n_ij and the
                    displacement vector Δμ_ij = μ_j − μ_i.

2. PCA-alignment :  Cosine similarity between the per-class first PCA eigenvector
                    p₁^(c) and the relevant boundaries (n_ij vs p₁^(i) and p₁^(j)),
                    as well as the raw class weights (w_c vs p₁^(c)).

3. Cell occupancy:  every sample is assigned a sign-tuple
                    s = (sign(n_01·x + b_01), sign(n_12·x + b_12),
                         sign(n_02·x + b_02))
                    We record cell occupancy overall, as well as the point count
                    per class within each of the 2^K Voronoi cells.

4. Hyperplane angles: pairwise angles between the three boundary normals
                      n_01, n_12, n_02 (and also between the raw class
                      weight rows w_0, w_1, w_2).

All numbers are written to hyperplanes.csv and summarised in two figures.
"""

from __future__ import annotations

import csv
import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# ── tiny numerical guard ───────────────────────────────────────────────────
_EPS = 1e-12

# ── helpers ────────────────────────────────────────────────────────────────

def _unit(v: np.ndarray) -> np.ndarray:
    """Return v / ‖v‖, or the zero vector if ‖v‖ ≈ 0."""
    n = float(np.linalg.norm(v))
    return v / n if n > _EPS else np.zeros_like(v)


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Angle in degrees in [0, 90] between vectors a and b (unsigned)."""
    cos = float(np.clip(np.dot(_unit(a), _unit(b)), -1.0, 1.0))
    # we take the absolute value so that the sign of the normal does not
    # matter (the hyperplane has no intrinsic orientation).
    return float(np.degrees(np.arccos(abs(cos))))


# ── boundary normals ───────────────────────────────────────────────────────

def _boundary_normals(
    W: np.ndarray,
    b: Optional[np.ndarray],
    num_classes: int,
) -> Tuple[Dict[Tuple[int, int], np.ndarray], Dict[Tuple[int, int], float]]:
    """
    For a softmax classifier with weight rows W[i] ∈ ℝ^D and biases b[i],
    the decision boundary between class i and j is

        (W[i] - W[j])·x + (b[i] - b[j]) = 0

    Returns
    -------
    normals : dict[(i,j) → np.ndarray shape (D,)]
    biases  : dict[(i,j) → float]   effective bias of the boundary
    """
    pairs = list(itertools.combinations(range(num_classes), 2))
    normals: Dict[Tuple[int, int], np.ndarray] = {}
    biases:  Dict[Tuple[int, int], float]      = {}
    for i, j in pairs:
        normals[(i, j)] = W[i] - W[j]
        if b is not None:
            biases[(i, j)] = float(b[i] - b[j])
        else:
            biases[(i, j)] = 0.0
    return normals, biases


# ── dataclass ─────────────────────────────────────────────────────────────

@dataclass
class HyperplaneEpochRaw:
    # --- CM alignment -------------------------------------------------------
    # cosine similarity (signed −1…1)
    cm_cosine_by_pair: np.ndarray         # shape (num_pairs,)

    # --- PCA alignment ------------------------------------------------------
    # cosine between per-class PCA and boundary/raw normals
    pca_cosine_bound_i: np.ndarray        # shape (num_pairs,) -> cos(n_ij, p1_i)
    pca_cosine_bound_j: np.ndarray        # shape (num_pairs,) -> cos(n_ij, p1_j)
    pca_cosine_raw: np.ndarray            # shape (num_classes,) -> cos(w_c, p1_c)

    # --- Hyperplane–hyperplane angles ---------------------------------------
    # pairwise angles between boundary normals (n_ij vs n_kl)
    hp_angle_by_pairkl: np.ndarray        # shape (num_pair_pairs,)
    hp_cosine_by_pairkl: np.ndarray

    # pairwise angles between raw class weight rows w_i vs w_j
    raw_angle_by_pair: np.ndarray         # shape (num_pairs,)
    raw_cosine_by_pair: np.ndarray

    # --- Cell occupancy -----------------------------------------------------
    num_cells_occupied: int               # how many of the 2^K cells have ≥1 point
    num_cells_total: int                  # = 2^K  where K = num_pairs
    cell_occupancy_fraction: float        # fraction of occupied cells (0…1)
    
    # count per cell (total)
    cell_counts: np.ndarray               # shape (2^K,)
    # count per cell per class
    cell_class_counts: np.ndarray         # shape (2^K, num_classes)

    # --- pair metadata (stored once, used for CSV headers) -----------------
    pair_labels: List[str] = field(default_factory=list)       # e.g. ["0v1","0v2","1v2"]
    pair_pair_labels: List[str] = field(default_factory=list)  # e.g. ["(0v1)v(0v2)", ...]


# ── main collection function ───────────────────────────────────────────────

def collect_hyperplane_epoch(
    pre_classifier: np.ndarray,
    targets: np.ndarray,
    weight_matrix: np.ndarray,
    bias: Optional[np.ndarray],
    num_classes: int,
) -> HyperplaneEpochRaw:
    W  = np.asarray(weight_matrix, dtype=np.float64)        # (C, D)
    X  = np.asarray(pre_classifier, dtype=np.float64)       # (N, D)
    y  = np.asarray(targets, dtype=np.int64)                # (N,)
    b  = np.asarray(bias, dtype=np.float64) if bias is not None else None

    C = int(num_classes)
    pairs = list(itertools.combinations(range(C), 2))
    K = len(pairs)                                           # = C*(C-1)/2

    pair_labels      = [f"{i}v{j}" for i, j in pairs]
    normals, b_eff   = _boundary_normals(W, b, C)

    # ── 1. Class centre of mass ────────────────────────────────────────────
    mu = np.zeros((C, X.shape[1]), dtype=np.float64)
    for c in range(C):
        mask = y == c
        if mask.sum() == 0:
            raise ValueError(f"Class {c} has no samples – cannot compute CM.")
        mu[c] = X[mask].mean(axis=0)

    # ── 2. Per-class PCA – first eigenvector ──────────────────────────────
    p1_by_class = np.zeros((C, X.shape[1]), dtype=np.float64)
    for c in range(C):
        X_c = X[y == c]
        if X_c.shape[0] > 1:
            X_c_centered = X_c - X_c.mean(axis=0)
            try:
                _, _, Vt = np.linalg.svd(X_c_centered, full_matrices=False)
                p1_by_class[c] = Vt[0]
            except np.linalg.LinAlgError:
                cov = (X_c_centered.T @ X_c_centered) / max(1, X_c_centered.shape[0] - 1)
                eigvals, eigvecs = np.linalg.eigh(cov)
                p1_by_class[c] = eigvecs[:, np.argmax(eigvals)]
        else:
            # 0 or 1 samples => no meaningful variance
            p1_by_class[c] = np.zeros(X.shape[1], dtype=np.float64)

    # ── 3. CM-alignment & PCA-alignment ───────────────────────────────────
    cm_cosine          = np.zeros(K, dtype=np.float64)
    pca_cosine_bound_i = np.zeros(K, dtype=np.float64)
    pca_cosine_bound_j = np.zeros(K, dtype=np.float64)

    for idx, (i, j) in enumerate(pairs):
        n_ij   = normals[(i, j)]
        delta  = mu[j] - mu[i]

        cm_cosine[idx]          = float(np.dot(_unit(n_ij), _unit(delta)))
        pca_cosine_bound_i[idx] = float(np.dot(_unit(n_ij), _unit(p1_by_class[i])))
        pca_cosine_bound_j[idx] = float(np.dot(_unit(n_ij), _unit(p1_by_class[j])))

    pca_cosine_raw = np.zeros(C, dtype=np.float64)
    for c in range(C):
        pca_cosine_raw[c] = float(np.dot(_unit(W[c]), _unit(p1_by_class[c])))

    # ── 4. Hyperplane–hyperplane angles ───────────────────────────────────
    pair_pairs       = list(itertools.combinations(range(K), 2))
    pair_pair_labels = [
        f"({pair_labels[a]})v({pair_labels[b_]})"
        for a, b_ in pair_pairs
    ]
    hp_angle  = np.zeros(len(pair_pairs), dtype=np.float64)
    hp_cosine = np.zeros(len(pair_pairs), dtype=np.float64)
    norm_list = [normals[pairs[k]] for k in range(K)]
    
    for idx2, (a, b_) in enumerate(pair_pairs):
        hp_angle[idx2]  = _angle_deg(norm_list[a], norm_list[b_])
        hp_cosine[idx2] = float(np.dot(_unit(norm_list[a]), _unit(norm_list[b_])))

    raw_angle  = np.zeros(K, dtype=np.float64)
    raw_cosine = np.zeros(K, dtype=np.float64)
    for idx, (i, j) in enumerate(pairs):
        raw_angle[idx]  = _angle_deg(W[i], W[j])
        raw_cosine[idx] = float(np.dot(_unit(W[i]), _unit(W[j])))

    # ── 5. Cell occupancy ─────────────────────────────────────────────────
    score = np.zeros((X.shape[0], K), dtype=np.float64)
    for idx, (i, j) in enumerate(pairs):
        score[:, idx] = X @ normals[(i, j)] + b_eff[(i, j)]

    bits = (score > 0).astype(np.int64)
    powers = (2 ** np.arange(K, dtype=np.int64))[::-1]
    cell_ids = bits @ powers

    num_total = 2 ** K
    cell_counts = np.bincount(cell_ids, minlength=num_total).astype(np.int64)
    num_occupied = int((cell_counts > 0).sum())

    # Calculate points per class in each cell
    cell_class_counts = np.zeros((num_total, C), dtype=np.int64)
    for c in range(C):
        mask_c = (y == c)
        if mask_c.sum() > 0:
            counts_c = np.bincount(cell_ids[mask_c], minlength=num_total)
            cell_class_counts[:, c] = counts_c

    return HyperplaneEpochRaw(
        cm_cosine_by_pair=cm_cosine,
        pca_cosine_bound_i=pca_cosine_bound_i,
        pca_cosine_bound_j=pca_cosine_bound_j,
        pca_cosine_raw=pca_cosine_raw,
        hp_angle_by_pairkl=hp_angle,
        hp_cosine_by_pairkl=hp_cosine,
        raw_angle_by_pair=raw_angle,
        raw_cosine_by_pair=raw_cosine,
        num_cells_occupied=num_occupied,
        num_cells_total=num_total,
        cell_occupancy_fraction=num_occupied / max(1, num_total),
        cell_counts=cell_counts,
        cell_class_counts=cell_class_counts,
        pair_labels=pair_labels,
        pair_pair_labels=pair_pair_labels,
    )


# ── CSV I/O ────────────────────────────────────────────────────────────────

def _csv_header(num_classes: int) -> List[str]:
    pairs      = list(itertools.combinations(range(num_classes), 2))
    pair_labels = [f"{i}v{j}" for i, j in pairs]
    pair_pairs  = list(itertools.combinations(range(len(pairs)), 2))
    pp_labels   = [
        f"({pair_labels[a]})v({pair_labels[b_]})"
        for a, b_ in pair_pairs
    ]
    K        = len(pairs)
    num_cell = 2 ** K

    header = ["epoch", "global_step"]

    for lbl in pair_labels:
        header.append(f"cm_cosine_{lbl}")

    for i, j in pairs:
        header.append(f"pca_cosine_{i}v{j}_p1_{i}")
        header.append(f"pca_cosine_{i}v{j}_p1_{j}")

    for c in range(num_classes):
        header.append(f"pca_cosine_raw_{c}_p1_{c}")

    for lbl in pp_labels:
        header.append(f"hp_angle_{lbl}")
    for lbl in pp_labels:
        header.append(f"hp_cosine_{lbl}")

    for lbl in pair_labels:
        header.append(f"raw_angle_{lbl}")
    for lbl in pair_labels:
        header.append(f"raw_cosine_{lbl}")

    header += ["num_cells_occupied", "num_cells_total", "cell_occupancy_fraction"]

    for cell_id in range(num_cell):
        bits_str = format(cell_id, f"0{K}b")
        header.append(f"cell_{bits_str}_count")
        for c in range(num_classes):
            header.append(f"cell_{bits_str}_class_{c}_count")

    return header


def initialize_hyperplane_csv(csv_path: Path, num_classes: int) -> None:
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(_csv_header(num_classes))


def append_hyperplane_csv_row(
    csv_path: Path,
    epoch: int,
    global_step: int,
    raw: HyperplaneEpochRaw,
    num_classes: int,
) -> None:
    row: list = [epoch, global_step]

    for v in raw.cm_cosine_by_pair:  row.append(float(v))
    for v in raw.pca_cosine_bound_i: row.append(float(v))
    for v in raw.pca_cosine_bound_j: row.append(float(v))
    for v in raw.pca_cosine_raw:     row.append(float(v))
    
    for v in raw.hp_angle_by_pairkl:  row.append(float(v))
    for v in raw.hp_cosine_by_pairkl: row.append(float(v))
    for v in raw.raw_angle_by_pair:   row.append(float(v))
    for v in raw.raw_cosine_by_pair:  row.append(float(v))

    row.append(int(raw.num_cells_occupied))
    row.append(int(raw.num_cells_total))
    row.append(float(raw.cell_occupancy_fraction))

    num_cell = 2 ** len(raw.cm_cosine_by_pair)
    for cell_id in range(num_cell):
        row.append(int(raw.cell_counts[cell_id]))
        for c in range(num_classes):
            row.append(int(raw.cell_class_counts[cell_id, c]))

    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def _load_hp_csv(csv_path: Path) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    rows: list[dict] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    steps = np.asarray([int(float(r["global_step"])) for r in rows], dtype=np.int64)
    data: Dict[str, np.ndarray] = {}
    if not rows:
        return steps, data

    for key in rows[0]:
        if key in {"epoch", "global_step"}:
            continue
        vals = []
        for r in rows:
            v = r.get(key, "")
            vals.append(float(v) if v not in ("", None) else float("nan"))
        data[key] = np.asarray(vals, dtype=np.float64)

    return steps, data


# ── plotting ───────────────────────────────────────────────────────────────

def finalize_hyperplane_plots(
    csv_path: Path,
    output_path: Path,
    num_classes: int,
    tpt_step: int = -1,
) -> None:
    steps, data = _load_hp_csv(csv_path)
    if steps.size == 0:
        raise ValueError(f"No rows found in {csv_path}")

    pairs      = list(itertools.combinations(range(num_classes), 2))
    pair_labels = [f"{i}v{j}" for i, j in pairs]
    pair_pairs  = list(itertools.combinations(range(len(pairs)), 2))
    pp_labels   = [
        f"({pair_labels[a]})v({pair_labels[b_]})"
        for a, b_ in pair_pairs
    ]
    K        = len(pairs)
    num_cell = 2 ** K

    def _vline(ax: plt.Axes) -> None:
        if tpt_step >= 0:
            ax.axvline(tpt_step, color="black", lw=2.0, ls="-", label="TPT")

    # ── 1. MAIN HYPERPLANE FIGURE ──────────────────────────────────────────
    nrows, ncols = 3, 2
    fig, axs = plt.subplots(nrows, ncols, figsize=(16, 15), sharex=True)

    # Row 0, Col 0 : CM alignment cosine
    ax = axs[0, 0]
    for lbl in pair_labels:
        key = f"cm_cosine_{lbl}"
        if key in data:
            ax.plot(steps, data[key], lw=1.8, label=lbl)
    ax.axhline( 1, color="grey", ls=":", lw=1)
    ax.axhline(-1, color="grey", ls=":", lw=1)
    ax.axhline( 0, color="grey", ls=":", lw=1)
    ax.set_title("CM-alignment cosine (boundary normal ↔ Δμ)")
    ax.set_ylabel("Cosine similarity")
    ax.set_ylim(-1.1, 1.1)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    _vline(ax)

    # Row 0, Col 1 : PCA alignment boundary cosine vs per-class p1
    ax = axs[0, 1]
    for i, j in pairs:
        key_i = f"pca_cosine_{i}v{j}_p1_{i}"
        key_j = f"pca_cosine_{i}v{j}_p1_{j}"
        if key_i in data:
            ax.plot(steps, data[key_i], lw=1.8, label=f"{i}v{j} (vs p1_{i})")
        if key_j in data:
            ax.plot(steps, data[key_j], lw=1.8, ls="--", label=f"{i}v{j} (vs p1_{j})")
    ax.axhline(0, color="grey", ls=":", lw=1, label="0 (⊥ → p₁ ∥ hyperplane)")
    ax.set_title("PCA-alignment cosine (boundary normal ↔ per-class p₁)")
    ax.set_ylabel("Cosine similarity")
    ax.set_ylim(-1.1, 1.1)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    _vline(ax)

    # Row 1, Col 0 : PCA alignment raw weights vs per-class p1
    ax = axs[1, 0]
    for c in range(num_classes):
        key = f"pca_cosine_raw_{c}_p1_{c}"
        if key in data:
            ax.plot(steps, data[key], lw=1.8, label=f"w_{c} (vs p1_{c})")
    ax.axhline(0, color="grey", ls=":", lw=1)
    ax.set_title("PCA-alignment cosine (raw class weight w_c ↔ p1_c)")
    ax.set_ylabel("Cosine similarity")
    ax.set_ylim(-1.1, 1.1)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    _vline(ax)

    # Row 1, Col 1 : Boundary normal angle
    ax = axs[1, 1]
    for lbl in pp_labels:
        key = f"hp_angle_{lbl}"
        if key in data:
            ax.plot(steps, data[key], lw=1.8, label=lbl)
    ax.axhline(90, color="grey", ls=":", lw=1, label="90° (⊥)")
    ax.axhline(0,  color="grey", ls=":", lw=1, label="0° (∥)")
    ax.set_title("Angle between boundary normals (n_ij ↔ n_kl)")
    ax.set_ylabel("Angle (°)")
    ax.set_ylim(-2, 95)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _vline(ax)

    # Row 2, Col 0 : Raw weight-row angles
    ax = axs[2, 0]
    for lbl in pair_labels:
        key = f"raw_angle_{lbl}"
        if key in data:
            ax.plot(steps, data[key], lw=1.8, label=f"w_{lbl}")
    ax.axhline(90, color="grey", ls=":", lw=1, label="90° (⊥)")
    ax.axhline(0,  color="grey", ls=":", lw=1, label="0° (∥)")
    ax.set_title("Angle between raw class weight rows (w_i ↔ w_j)")
    ax.set_ylabel("Angle (°)")
    ax.set_xlabel("Global step")
    ax.set_ylim(-2, 95)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    _vline(ax)

    # Row 2, Col 1 : Cell occupancy count
    ax = axs[2, 1]
    if "num_cells_occupied" in data:
        ax.plot(steps, data["num_cells_occupied"], lw=2.0, color="tab:blue",
                label="occupied cells")
    ax.axhline(num_cell, color="grey", ls=":", lw=1, label=f"max ({num_cell})")
    ax.set_title("Number of occupied Voronoi cells")
    ax.set_xlabel("Global step")
    ax.set_ylabel("# cells occupied")
    ax.set_ylim(0, num_cell + 1)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    _vline(ax)

    fig.suptitle("Hyperplane Geometry Analysis", fontsize=16, y=1.01)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # ── 2. VORONOI CELL PROPORTIONS FIGURE ────────────────────────────────
    ncols_v = min(num_cell, 4)
    nrows_v = (num_cell + ncols_v - 1) // ncols_v
    fig_v, axs_v = plt.subplots(nrows_v, ncols_v, figsize=(4 * ncols_v, 3 * nrows_v), sharex=True, sharey=True)
    axs_v_flat = np.atleast_1d(axs_v).flatten()

    for cell_id in range(num_cell):
        ax_v = axs_v_flat[cell_id]
        bits_str = format(cell_id, f"0{K}b")
        total_key = f"cell_{bits_str}_count"
        
        total_pts = data.get(total_key, np.zeros_like(steps, dtype=np.float64))

        if np.sum(total_pts) == 0:
            ax_v.text(0.5, 0.5, "NOT USED", ha='center', va='center', 
                      transform=ax_v.transAxes, fontsize=14, color='gray', fontweight='bold')
            ax_v.set_title(f"Cell {bits_str}")
            ax_v.set_xticks([])
            ax_v.set_yticks([])
            continue

        stack_arrays = []
        labels_v = []
        # Prevent division by zero
        safe_total = np.where(total_pts > 0, total_pts, 1.0)
        
        for c in range(num_classes):
            c_key = f"cell_{bits_str}_class_{c}_count"
            c_pts = data.get(c_key, np.zeros_like(steps, dtype=np.float64))
            stack_arrays.append(c_pts / safe_total)
            labels_v.append(f"Class {c}")

        ax_v.stackplot(steps, stack_arrays, labels=labels_v, alpha=0.75)
        ax_v.set_title(f"Cell {bits_str}")
        ax_v.set_ylim(0, 1.0)
        
        if cell_id % ncols_v == 0:
            ax_v.set_ylabel("Class Proportion")
        if cell_id >= len(axs_v_flat) - ncols_v:
            ax_v.set_xlabel("Global step")
        if tpt_step >= 0:
            ax_v.axvline(tpt_step, color="black", ls="-", lw=1.5)

    # Turn off any unused grid axes logically
    for j in range(num_cell, len(axs_v_flat)):
        axs_v_flat[j].axis('off')

    # Assign single legend below grid
    for ax_v in axs_v_flat:
        handles, labels = ax_v.get_legend_handles_labels()
        if handles:
            fig_v.legend(handles, labels, loc='lower center', ncol=num_classes, bbox_to_anchor=(0.5, -0.05))
            break

    fig_v.suptitle("Voronoi Cell Class Proportions", fontsize=16)
    plt.tight_layout()
    voronoi_out_path = output_path.parent / "voronoi_cells.png"
    plt.savefig(voronoi_out_path, dpi=180, bbox_inches="tight")
    plt.close(fig_v)