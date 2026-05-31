from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


EPS = 1e-12
REG_EPS = 1e-6


@dataclass
class GeoEpochRaw:
    cyl_overlap_by_pair: np.ndarray
    ellipsoid_bhattacharyya_by_pair: np.ndarray
    ellipsoid_overlap_by_pair: np.ndarray
    cyl_axis_cosine_by_pair: np.ndarray
    cyl_half_length_by_class: np.ndarray
    cyl_radius_by_class: np.ndarray
    cyl_axis_index_by_class: np.ndarray


def _flatten_features(x: np.ndarray) -> np.ndarray:
    return np.reshape(x, (x.shape[0], -1))


def _compute_class_stats(
    features: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dim = int(features.shape[1])
    means = np.zeros((num_classes, dim), dtype=np.float64)
    axes = np.zeros((num_classes, dim), dtype=np.float64)
    half_lengths = np.zeros(num_classes, dtype=np.float64)
    radii = np.zeros(num_classes, dtype=np.float64)
    axis_indices = np.zeros(num_classes, dtype=np.float64)
    covs = np.zeros((num_classes, dim, dim), dtype=np.float64)

    class_counts = np.bincount(targets, minlength=num_classes).astype(np.int64)
    missing = np.where(class_counts <= 0)[0].tolist()
    if missing:
        raise ValueError(f"Cannot compute geometric metrics for classes with no samples: {missing}")

    for class_id in range(num_classes):
        class_feats = features[targets == class_id]
        mu = np.mean(class_feats, axis=0)
        means[class_id] = mu

        centered = class_feats - mu
        n_c = int(class_feats.shape[0])
        if n_c <= 1:
            cov = np.zeros((dim, dim), dtype=np.float64)
            eigvals = np.zeros(dim, dtype=np.float64)
            eigvecs = np.eye(dim, dtype=np.float64)
        else:
            cov = (centered.T @ centered) / float(n_c - 1)
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = np.argsort(eigvals)[::-1]
            eigvals = eigvals[order]
            eigvecs = eigvecs[:, order]

        eigvals = np.clip(eigvals, 0.0, None)
        if dim > 0:
            axis = eigvecs[:, 0]
            axis_norm = float(np.linalg.norm(axis))
            if axis_norm > 0.0:
                axis = axis / axis_norm
            else:
                axis = np.zeros(dim, dtype=np.float64)
                axis[0] = 1.0
        else:
            axis = np.zeros(0, dtype=np.float64)

        axes[class_id] = axis
        # record the 1-based index of the input dimension with largest absolute
        # loading in the leading eigenvector; this lets us track which original
        # feature dimension dominates the principal direction over time.
        if axis.size > 0:
            axis_idx = int(np.argmax(np.abs(axis))) + 1
        else:
            axis_idx = 1
        axis_indices[class_id] = float(axis_idx)
        if eigvals.size > 0:
            half_lengths[class_id] = float(np.sqrt(eigvals[0]))
        if eigvals.size > 1:
            radii[class_id] = float(np.sqrt(np.mean(eigvals[1:])))
        else:
            radii[class_id] = 0.0

        covs[class_id] = cov + REG_EPS * np.eye(dim, dtype=np.float64)

    return means, axes, half_lengths, radii, covs, axis_indices


def _cylinder_overlap(
    delta: np.ndarray,
    v_i: np.ndarray,
    v_j: np.ndarray,
    l_i: float,
    l_j: float,
    r_i: float,
    r_j: float,
) -> float:
    delta_norm_sq = float(np.dot(delta, delta))
    l_sum = float(l_i + l_j)
    r_sum = float(r_i + r_j)
    l_denom = max(l_sum, EPS)
    r_denom = max(r_sum, EPS)

    a_i = abs(float(np.dot(delta, v_i)))
    a_j = abs(float(np.dot(delta, v_j)))

    p_i_sq = max(delta_norm_sq - a_i * a_i, 0.0)
    p_j_sq = max(delta_norm_sq - a_j * a_j, 0.0)
    p_i = float(np.sqrt(p_i_sq))
    p_j = float(np.sqrt(p_j_sq))

    axial_i = max(0.0, 1.0 - a_i / l_denom)
    axial_j = max(0.0, 1.0 - a_j / l_denom)
    radial_i = max(0.0, 1.0 - p_i / r_denom)
    radial_j = max(0.0, 1.0 - p_j / r_denom)

    overlap = 0.5 * (axial_i * radial_i + axial_j * radial_j)
    return float(np.clip(overlap, 0.0, 1.0))


def _bhattacharyya_distance(
    mu_i: np.ndarray,
    cov_i: np.ndarray,
    mu_j: np.ndarray,
    cov_j: np.ndarray,
) -> float:
    delta = mu_j - mu_i
    cov_avg = 0.5 * (cov_i + cov_j)

    try:
        solve = np.linalg.solve(cov_avg, delta)
    except np.linalg.LinAlgError:
        solve = np.linalg.pinv(cov_avg) @ delta

    term1 = 0.125 * float(np.dot(delta, solve))

    sign_avg, logdet_avg = np.linalg.slogdet(cov_avg)
    sign_i, logdet_i = np.linalg.slogdet(cov_i)
    sign_j, logdet_j = np.linalg.slogdet(cov_j)
    if sign_avg <= 0.0 or sign_i <= 0.0 or sign_j <= 0.0:
        return float("nan")

    term2 = 0.5 * (logdet_avg - 0.5 * (logdet_i + logdet_j))
    return float(term1 + term2)


def collect_geo_epoch(
    pre_classifier: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
    class_pairs: list[tuple[int, int]],
) -> GeoEpochRaw:
    features = _flatten_features(np.asarray(pre_classifier, dtype=np.float64))
    targets_arr = np.asarray(targets, dtype=np.int64)
    if features.shape[0] == 0:
        raise ValueError("No samples provided for geometric overlap metrics.")

    means, axes, half_lengths, radii, covs, axis_indices = _compute_class_stats(
        features=features,
        targets=targets_arr,
        num_classes=num_classes,
    )

    cyl_overlap_by_pair = np.zeros(len(class_pairs), dtype=np.float64)
    ellipsoid_bhattacharyya_by_pair = np.zeros(len(class_pairs), dtype=np.float64)
    ellipsoid_overlap_by_pair = np.zeros(len(class_pairs), dtype=np.float64)
    cyl_axis_cosine_by_pair = np.zeros(len(class_pairs), dtype=np.float64)

    for pair_idx, (left, right) in enumerate(class_pairs):
        delta = means[right] - means[left]
        cyl_overlap_by_pair[pair_idx] = _cylinder_overlap(
            delta=delta,
            v_i=axes[left],
            v_j=axes[right],
            l_i=float(half_lengths[left]),
            l_j=float(half_lengths[right]),
            r_i=float(radii[left]),
            r_j=float(radii[right]),
        )
        axis_cos = float(abs(np.dot(axes[left], axes[right])))
        cyl_axis_cosine_by_pair[pair_idx] = float(np.clip(axis_cos, 0.0, 1.0))

        db = _bhattacharyya_distance(
            mu_i=means[left],
            cov_i=covs[left],
            mu_j=means[right],
            cov_j=covs[right],
        )
        if np.isfinite(db):
            bc = float(np.exp(-db))
            bc = float(np.clip(bc, 0.0, 1.0))
        else:
            bc = float("nan")
        ellipsoid_bhattacharyya_by_pair[pair_idx] = float(db)
        ellipsoid_overlap_by_pair[pair_idx] = float(bc)

    return GeoEpochRaw(
        cyl_overlap_by_pair=cyl_overlap_by_pair.astype(np.float64),
        ellipsoid_bhattacharyya_by_pair=ellipsoid_bhattacharyya_by_pair.astype(np.float64),
        ellipsoid_overlap_by_pair=ellipsoid_overlap_by_pair.astype(np.float64),
        cyl_axis_cosine_by_pair=cyl_axis_cosine_by_pair.astype(np.float64),
        cyl_half_length_by_class=half_lengths.astype(np.float64),
        cyl_radius_by_class=radii.astype(np.float64),
        cyl_axis_index_by_class=axis_indices.astype(np.float64),
    )


def initialize_geo_csv(
    csv_path: Path,
    num_classes: int,
    class_pairs: list[tuple[int, int]],
) -> None:
    header = ["epoch", "global_step"]
    for left, right in class_pairs:
        header.append(f"cyl_overlap_{left}v{right}")
        header.append(f"ellipsoid_bhattacharyya_{left}v{right}")
        header.append(f"ellipsoid_overlap_{left}v{right}")
        header.append(f"cyl_axis_cosine_{left}v{right}")
    header.extend([f"cyl_half_length_class_{class_id}" for class_id in range(num_classes)])
    header.extend([f"cyl_radius_class_{class_id}" for class_id in range(num_classes)])
    header.extend([f"cyl_axis_index_class_{class_id}" for class_id in range(num_classes)])

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)


def append_geo_csv_row(
    csv_path: Path,
    epoch: int,
    global_step: int,
    raw: GeoEpochRaw,
    num_classes: int,
) -> None:
    row: list[float | int] = [epoch, global_step]
    for idx in range(raw.cyl_overlap_by_pair.shape[0]):
        row.append(float(raw.cyl_overlap_by_pair[idx]))
        row.append(float(raw.ellipsoid_bhattacharyya_by_pair[idx]))
        row.append(float(raw.ellipsoid_overlap_by_pair[idx]))
        row.append(float(raw.cyl_axis_cosine_by_pair[idx]))
    row.extend(raw.cyl_half_length_by_class[:num_classes].tolist())
    row.extend(raw.cyl_radius_by_class[:num_classes].tolist())
    row.extend(raw.cyl_axis_index_by_class[:num_classes].tolist())

    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def _load_geo_columns(csv_path: Path) -> list[dict[str, float]]:
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


def finalize_geo_plots(
    csv_path: Path,
    output_path: Path,
    num_classes: int,
    class_pairs: list[tuple[int, int]],
    tpt_step: int = -1,
) -> None:
    rows = _load_geo_columns(csv_path)
    if not rows:
        raise ValueError(f"No geometric overlap rows found in {csv_path}")

    steps = np.asarray([int(row["global_step"]) for row in rows], dtype=np.int64)
    cyl_overlap_by_pair = np.asarray(
        [[row[f"cyl_overlap_{left}v{right}"] for left, right in class_pairs] for row in rows],
        dtype=np.float64,
    )
    ellipsoid_bhattacharyya_by_pair = np.asarray(
        [[row[f"ellipsoid_bhattacharyya_{left}v{right}"] for left, right in class_pairs] for row in rows],
        dtype=np.float64,
    )
    ellipsoid_overlap_by_pair = np.asarray(
        [[row[f"ellipsoid_overlap_{left}v{right}"] for left, right in class_pairs] for row in rows],
        dtype=np.float64,
    )
    cyl_axis_cosine_by_pair = np.asarray(
        [[row.get(f"cyl_axis_cosine_{left}v{right}", float("nan")) for left, right in class_pairs] for row in rows],
        dtype=np.float64,
    )
    cyl_half_length_by_class = np.asarray(
        [[row[f"cyl_half_length_class_{class_id}"] for class_id in range(num_classes)] for row in rows],
        dtype=np.float64,
    )
    cyl_radius_by_class = np.asarray(
        [[row[f"cyl_radius_class_{class_id}"] for class_id in range(num_classes)] for row in rows],
        dtype=np.float64,
    )

    # try to collect axis index history if present
    axis_index_key = f"cyl_axis_index_class_0"
    has_axis_index = axis_index_key in rows[0]
    if has_axis_index:
        cyl_axis_index_by_class = np.asarray(
            [[row.get(f"cyl_axis_index_class_{class_id}", float("nan")) for class_id in range(num_classes)] for row in rows],
            dtype=np.float64,
        )
    else:
        cyl_axis_index_by_class = None

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True)

    ax = axes[0, 0]
    for pair_idx, (left, right) in enumerate(class_pairs):
        ax.plot(
            steps,
            cyl_overlap_by_pair[:, pair_idx],
            linewidth=1.6,
            label=f"{left}v{right}",
        )
    ax.set_title("Cylinder Overlap")
    ax.set_ylabel("Overlap")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

    ax = axes[0, 1]
    colors = plt.rcParams.get("axes.prop_cycle", None)
    color_list = None
    if colors is not None:
        color_list = colors.by_key().get("color")
    if not color_list:
        color_list = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]

    for class_id in range(num_classes):
        color = color_list[class_id % len(color_list)]
        ax.plot(
            steps,
            cyl_half_length_by_class[:, class_id],
            linewidth=1.8,
            color=color,
            label=f"L class {class_id}",
        )
        ax.plot(
            steps,
            cyl_radius_by_class[:, class_id],
            linewidth=1.5,
            linestyle="--",
            color=color,
            label=f"r class {class_id}",
        )
    ax.set_title("Cylinder Half-Lengths and Radii")
    ax.set_ylabel("Length / Radius")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1, 0]
    overlap_plot = np.clip(ellipsoid_overlap_by_pair, EPS, None)
    for pair_idx, (left, right) in enumerate(class_pairs):
        ax.plot(
            steps,
            overlap_plot[:, pair_idx],
            linewidth=1.6,
            label=f"{left}v{right}",
        )
    ax.set_title("Ellipsoid Overlap (Bhattacharyya Coefficient)")
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Overlap")
    ax.set_yscale("log")
    overlap_min = float(np.nanmin(overlap_plot))
    overlap_max = float(np.nanmax(overlap_plot))
    if not np.isfinite(overlap_min) or overlap_min <= 0.0:
        overlap_min = EPS
    if not np.isfinite(overlap_max) or overlap_max <= 0.0:
        overlap_max = 1e-1
    low = float(10.0 ** np.floor(np.log10(overlap_min)))
    high = float(10.0 ** np.ceil(np.log10(overlap_max)))
    if high <= low:
        high = low * 10.0
    ax.set_ylim(low, high)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

    ax = axes[1, 1]
    bhat_plot = np.clip(ellipsoid_bhattacharyya_by_pair, EPS, None)
    for pair_idx, (left, right) in enumerate(class_pairs):
        ax.plot(
            steps,
            bhat_plot[:, pair_idx],
            linewidth=1.6,
            label=f"{left}v{right}",
        )
    ax.set_title("Ellipsoid Bhattacharyya Distance")
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Distance (log)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

    if tpt_step >= 0:
        for axis in axes.flat:
            axis.axvline(tpt_step, color="black", linestyle="-", linewidth=2.0)

    # If axis indices were recorded, also save a separate plot showing index vs step
    if cyl_axis_index_by_class is not None:
        try:
            idx_fig, (idx_ax, ang_ax) = plt.subplots(1, 2, figsize=(14, 4), sharex=True)
            for class_id in range(num_classes):
                idx_ax.plot(steps, cyl_axis_index_by_class[:, class_id], linewidth=1.6, label=f"class {class_id}")
            idx_ax.set_title("Cylinder Principal Axis Index")
            idx_ax.set_xlabel("Global Step")
            idx_ax.set_ylabel("Axis Index (1-based)")
            idx_ax.grid(True, alpha=0.3)
            idx_ax.legend(fontsize=8)

            for pair_idx, (left, right) in enumerate(class_pairs):
                ang_ax.plot(steps, cyl_axis_cosine_by_pair[:, pair_idx], linewidth=1.6, label=f"{left}v{right}")
            ang_ax.set_title("Principal Axis Alignment (|v_i · v_j|)")
            ang_ax.set_xlabel("Global Step")
            ang_ax.set_ylabel("Cosine")
            ang_ax.set_ylim(0.0, 1.02)
            ang_ax.grid(True, alpha=0.3)
            ang_ax.legend(fontsize=8, ncol=2)

            if tpt_step >= 0:
                idx_ax.axvline(tpt_step, color="black", linestyle="-", linewidth=2.0)
                ang_ax.axvline(tpt_step, color="black", linestyle="-", linewidth=2.0)
            idx_out = output_path.parent / "cylinder_length_index.png"
            idx_out.parent.mkdir(parents=True, exist_ok=True)
            idx_fig.tight_layout()
            idx_fig.savefig(idx_out, dpi=180, bbox_inches="tight")
            plt.close(idx_fig)
        except Exception:
            pass

    fig.suptitle("PCA Geometric Overlap Metrics", fontsize=16)
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def finalize_geo_plots_simplified(
    csv_path: Path,
    output_path: Path,
    num_classes: int,
    class_pairs: list[tuple[int, int]],
    tpt_step: int = -1,
) -> None:
    rows = _load_geo_columns(csv_path)
    if not rows:
        raise ValueError(f"No geometric overlap rows found in {csv_path}")

    steps = np.asarray([int(row["global_step"]) for row in rows], dtype=np.int64)
    cyl_overlap_by_pair = np.asarray(
        [
            [row.get(f"cyl_overlap_{left}v{right}", float("nan")) for left, right in class_pairs]
            for row in rows
        ],
        dtype=np.float64,
    )
    ellipsoid_bhattacharyya_by_pair = np.asarray(
        [
            [row.get(f"ellipsoid_bhattacharyya_{left}v{right}", float("nan")) for left, right in class_pairs]
            for row in rows
        ],
        dtype=np.float64,
    )
    ellipsoid_overlap_by_pair = np.asarray(
        [
            [row.get(f"ellipsoid_overlap_{left}v{right}", float("nan")) for left, right in class_pairs]
            for row in rows
        ],
        dtype=np.float64,
    )
    cyl_axis_cosine_by_pair = np.asarray(
        [
            [row.get(f"cyl_axis_cosine_{left}v{right}", float("nan")) for left, right in class_pairs]
            for row in rows
        ],
        dtype=np.float64,
    )
    cyl_half_length_by_class = np.asarray(
        [
            [row.get(f"cyl_half_length_class_{class_id}", float("nan")) for class_id in range(num_classes)]
            for row in rows
        ],
        dtype=np.float64,
    )
    cyl_radius_by_class = np.asarray(
        [
            [row.get(f"cyl_radius_class_{class_id}", float("nan")) for class_id in range(num_classes)]
            for row in rows
        ],
        dtype=np.float64,
    )

    # try to collect axis index history if present
    axis_index_key = f"cyl_axis_index_class_0"
    has_axis_index = axis_index_key in rows[0]
    if has_axis_index:
        cyl_axis_index_by_class = np.asarray(
            [
                [row.get(f"cyl_axis_index_class_{class_id}", float("nan")) for class_id in range(num_classes)]
                for row in rows
            ],
            dtype=np.float64,
        )
    else:
        cyl_axis_index_by_class = None

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True)

    ax = axes[0, 0]
    for pair_idx, (left, right) in enumerate(class_pairs):
        ax.plot(
            steps,
            cyl_overlap_by_pair[:, pair_idx],
            linewidth=1.6,
            label=f"{left}v{right}",
        )
    ax.set_title("Cylinder Overlap")
    ax.set_ylabel("Overlap")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

    ax = axes[0, 1]
    colors = plt.rcParams.get("axes.prop_cycle", None)
    color_list = None
    if colors is not None:
        color_list = colors.by_key().get("color")
    if not color_list:
        color_list = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]

    for class_id in range(num_classes):
        color = color_list[class_id % len(color_list)]
        ax.plot(
            steps,
            cyl_half_length_by_class[:, class_id],
            linewidth=1.8,
            color=color,
            label=f"L class {class_id}",
        )
        ax.plot(
            steps,
            cyl_radius_by_class[:, class_id],
            linewidth=1.5,
            linestyle="--",
            color=color,
            label=f"r class {class_id}",
        )
    ax.set_title("Cylinder Half-Lengths and Radii")
    ax.set_ylabel("Length / Radius")
    ax.set_ylim(0.0, 20.0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1, 0]
    overlap_plot = np.clip(ellipsoid_overlap_by_pair, EPS, None)
    for pair_idx, (left, right) in enumerate(class_pairs):
        ax.plot(
            steps,
            overlap_plot[:, pair_idx],
            linewidth=1.6,
            label=f"{left}v{right}",
        )
    ax.set_title("Ellipsoid Overlap (Bhattacharyya Coefficient)")
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Overlap")
    ax.set_yscale("log")
    ax.set_ylim(1e-12, 1e-1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

    ax = axes[1, 1]
    for pair_idx, (left, right) in enumerate(class_pairs):
        ax.plot(
            steps,
            ellipsoid_bhattacharyya_by_pair[:, pair_idx],
            linewidth=1.6,
            label=f"{left}v{right}",
        )
    ax.set_title("Ellipsoid Bhattacharyya Distance")
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Distance")
    ax.set_ylim(0.0, 30.0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

    if tpt_step >= 0:
        for axis in axes.flat:
            axis.axvline(tpt_step, color="black", linestyle="-", linewidth=2.0)

    if cyl_axis_index_by_class is not None:
        try:
            idx_fig, (idx_ax, ang_ax) = plt.subplots(1, 2, figsize=(14, 4), sharex=True)
            for class_id in range(num_classes):
                idx_ax.plot(steps, cyl_axis_index_by_class[:, class_id], linewidth=1.6, label=f"class {class_id}")
            idx_ax.set_title("Cylinder Principal Axis Index")
            idx_ax.set_xlabel("Global Step")
            idx_ax.set_ylabel("Axis Index (1-based)")
            idx_ax.grid(True, alpha=0.3)
            idx_ax.legend(fontsize=8)

            for pair_idx, (left, right) in enumerate(class_pairs):
                ang_ax.plot(steps, cyl_axis_cosine_by_pair[:, pair_idx], linewidth=1.6, label=f"{left}v{right}")
            ang_ax.set_title("Principal Axis Alignment (|v_i · v_j|)")
            ang_ax.set_xlabel("Global Step")
            ang_ax.set_ylabel("Cosine")
            ang_ax.set_ylim(0.0, 1.02)
            ang_ax.grid(True, alpha=0.3)
            ang_ax.legend(fontsize=8, ncol=2)

            if tpt_step >= 0:
                idx_ax.axvline(tpt_step, color="black", linestyle="-", linewidth=2.0)
                ang_ax.axvline(tpt_step, color="black", linestyle="-", linewidth=2.0)
            idx_out = output_path.parent / "cylinder_length_index.png"
            idx_out.parent.mkdir(parents=True, exist_ok=True)
            idx_fig.tight_layout()
            idx_fig.savefig(idx_out, dpi=180, bbox_inches="tight")
            plt.close(idx_fig)
        except Exception:
            pass

    fig.suptitle("PCA Geometric Overlap Metrics", fontsize=16)
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
