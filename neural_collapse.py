from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from model import JAXModel, ParamTree


EPS = 1e-12

# CSV schema v2: angles in degrees, within-class variance, NC1, NC2 columns.


@dataclass
class NCEpochRaw:
    pre_classifier_mu_sqnorm: np.ndarray
    pre_classifier_pair_angle_deg: np.ndarray
    pre_classifier_within_var_by_class: np.ndarray
    nc1: float
    nc2_deviation: float
    pair_separation_margin: np.ndarray


def build_nc_class_pairs(num_classes: int) -> list[tuple[int, int]]:
    # Keep adjacency-first ordering: (0,1), (1,2), ... then remaining pairs.
    pairs: list[tuple[int, int]] = []
    for left in range(num_classes - 1):
        pairs.append((left, left + 1))
    for left in range(num_classes):
        for right in range(left + 2, num_classes):
            pairs.append((left, right))
    return pairs


def _flatten_features(x: np.ndarray) -> np.ndarray:
    return np.reshape(x, (x.shape[0], -1))


def collect_nc_raw_epoch(
    model: JAXModel,
    params: ParamTree,
    inputs: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
    class_pairs: list[tuple[int, int]],
    eval_batch_size: int,
) -> NCEpochRaw:
    if eval_batch_size <= 0:
        raise ValueError("eval_batch_size must be > 0")
    class_counts = np.zeros(num_classes, dtype=np.float64)
    pre_classifier_sums: np.ndarray | None = None
    all_feats: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    num_samples = int(inputs.shape[0])
    if num_samples == 0:
        raise ValueError("No samples provided for neural collapse metrics.")

    for start in range(0, num_samples, eval_batch_size):
        end = min(num_samples, start + eval_batch_size)
        batch_x = np.asarray(inputs[start:end], dtype=np.float32)
        batch_y = np.asarray(targets[start:end], dtype=np.int64)

        _logits, intermediates = model.apply(params, batch_x, return_intermediates=True)
        feat = intermediates.get("pre_classifier")
        if feat is None:
            raise RuntimeError("Failed to capture activations for layer 'pre_classifier'.")
        features = _flatten_features(np.asarray(feat, dtype=np.float64))

        all_feats.append(features)
        all_labels.append(batch_y)

        if pre_classifier_sums is None:
            pre_classifier_sums = np.zeros((num_classes, features.shape[1]), dtype=np.float64)

        class_counts += np.bincount(batch_y, minlength=num_classes).astype(np.float64)
        for class_id in range(num_classes):
            mask = batch_y == class_id
            if np.any(mask):
                class_feats = features[mask]
                pre_classifier_sums[class_id] += np.sum(class_feats, axis=0)

    missing = np.where(class_counts <= 0)[0].tolist()
    if missing:
        raise ValueError(f"Cannot compute class means for classes with no samples: {missing}")

    if pre_classifier_sums is None:
        raise ValueError("No activations found for layer 'pre_classifier'.")

    if not all_feats:
        raise ValueError("No activations cached for layer 'pre_classifier'.")

    all_features = np.concatenate(all_feats, axis=0)
    all_targets = np.concatenate(all_labels, axis=0)
    if all_features.shape[0] != num_samples:
        raise ValueError("Cached activations do not match expected sample count.")

    pre_classifier_means = pre_classifier_sums / class_counts[:, None]
    global_mean = np.mean(pre_classifier_means, axis=0)
    centered_means = pre_classifier_means - global_mean

    pre_classifier_mu_sqnorm = np.sum(centered_means * centered_means, axis=1)

    norms = np.linalg.norm(centered_means, axis=1)
    gram = centered_means @ centered_means.T
    denom = np.outer(norms, norms) + EPS
    cos_mat = np.clip(gram / denom, -1.0, 1.0)
    pair_cos_vals = np.asarray(
        [cos_mat[left, right] for left, right in class_pairs],
        dtype=np.float64,
    )
    pre_classifier_pair_angle_deg = np.degrees(np.arccos(pair_cos_vals)).astype(np.float64)

    all_features_centered = all_features - global_mean
    deltas = all_features_centered - centered_means[all_targets]
    squared_radii = np.sum(deltas * deltas, axis=1)
    pre_classifier_within_var_sum = np.bincount(
        all_targets,
        weights=squared_radii,
        minlength=num_classes,
    ).astype(np.float64)
    pre_classifier_within_var_by_class = pre_classifier_within_var_sum / class_counts

    s_w = float(np.sum(squared_radii)) / float(max(1, num_samples))
    s_b = float(np.sum(centered_means * centered_means)) / float(num_classes)
    nc1 = float(s_w / (s_b + EPS))

    etf_target_cos = -1.0 / float(num_classes - 1)
    nc2_deviation = float(np.mean(np.abs(pair_cos_vals - etf_target_cos)))

    per_class_radius = np.sqrt(np.clip(pre_classifier_within_var_by_class, 0.0, None))
    pair_separation_margin = np.zeros(len(class_pairs), dtype=np.float64)
    for pair_idx, (left, right) in enumerate(class_pairs):
        diff = centered_means[left] - centered_means[right]
        dist_ij = float(np.sqrt(np.dot(diff, diff)))
        r_i = float(per_class_radius[left])
        r_j = float(per_class_radius[right])
        denom = r_i + r_j
        pair_separation_margin[pair_idx] = (dist_ij - denom) / (denom + EPS)

    return NCEpochRaw(
        pre_classifier_mu_sqnorm=pre_classifier_mu_sqnorm.astype(np.float64),
        pre_classifier_pair_angle_deg=pre_classifier_pair_angle_deg.astype(np.float64),
        pre_classifier_within_var_by_class=pre_classifier_within_var_by_class.astype(np.float64),
        nc1=nc1,
        nc2_deviation=nc2_deviation,
        pair_separation_margin=pair_separation_margin.astype(np.float64),
    )


def initialize_nc_csv(
    nc_csv_path: Path,
    num_classes: int,
    class_pairs: list[tuple[int, int]],
) -> None:
    header = ["epoch", "global_step"]
    header.extend([f"pre_classifier_mu_sqnorm_{class_id}" for class_id in range(num_classes)])
    header.extend([f"pre_classifier_angle_deg_{left}_{right}" for left, right in class_pairs])
    header.extend([f"pre_classifier_within_var_{class_id}" for class_id in range(num_classes)])
    header.extend([f"separation_margin_{left}_{right}" for left, right in class_pairs])
    header.append("nc1")
    header.append("nc2_deviation")

    with open(nc_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)


def append_nc_csv_row(
    nc_csv_path: Path,
    epoch: int,
    global_step: int,
    raw: NCEpochRaw,
    num_classes: int,
) -> None:
    row: list[float | int] = [epoch, global_step]
    row.extend(raw.pre_classifier_mu_sqnorm[:num_classes].tolist())
    row.extend(raw.pre_classifier_pair_angle_deg.tolist())
    row.extend(raw.pre_classifier_within_var_by_class[:num_classes].tolist())
    row.extend(raw.pair_separation_margin.tolist())
    row.append(float(raw.nc1))
    row.append(float(raw.nc2_deviation))

    with open(nc_csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def _load_nc_columns(nc_csv_path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with open(nc_csv_path, "r", encoding="utf-8", newline="") as f:
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


def _plot_nc_dashboard(
    steps: np.ndarray,
    pair_dist: np.ndarray,
    mean_radius_by_class: np.ndarray,
    avg_pair_dist: np.ndarray,
    avg_radius: np.ndarray,
    separation_ratio: np.ndarray,
    pair_cos: np.ndarray,
    pair_angle_deg: np.ndarray,
    separation_margin: np.ndarray,
    nc2_deviation: np.ndarray,
    nc1: np.ndarray,
    etf_target_deg: float,
    class_pairs: list[tuple[int, int]],
    output_path: Path,
    tpt_step: int,
) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(18, 18), sharex=True)

    ax = axes[0, 0]
    for pair_idx, (left, right) in enumerate(class_pairs):
        ax.plot(steps, pair_dist[:, pair_idx], linewidth=1.6, label=f"d(mu_{left}, mu_{right})")
    ax.set_title("Center Distance by Class Pair")
    ax.set_ylabel("Distance")
    ax.grid(True, alpha=0.3)
    if len(class_pairs) <= 16:
        ax.legend(fontsize=7)

    ax = axes[0, 1]
    for class_id in range(mean_radius_by_class.shape[1]):
        ax.plot(steps, mean_radius_by_class[:, class_id], linewidth=1.6, label=f"class {class_id}")
    ax.set_title("Mean Radius by Class")
    ax.set_ylabel("Mean radius")
    ax.grid(True, alpha=0.3)
    if mean_radius_by_class.shape[1] <= 16:
        ax.legend(fontsize=7, ncol=2)

    ax = axes[1, 0]
    for pair_idx, (left, right) in enumerate(class_pairs):
        ax.plot(steps, pair_cos[:, pair_idx], linewidth=1.6, label=f"cos(mu_{left}, mu_{right})")
    ax.set_title("Pairwise Cosine Between Class Centers")
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Cosine")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, alpha=0.3)
    if len(class_pairs) <= 16:
        ax.legend(fontsize=7)

    ax = axes[1, 1]
    for pair_idx, (left, right) in enumerate(class_pairs):
        ax.plot(steps, separation_margin[:, pair_idx], linewidth=1.6, label=f"{left}-{right}")
    ax.axhline(0.0, color="red", linestyle="--", linewidth=1.5, label="touching (margin=0)")
    ax.set_title("Pairwise Separation Margin  [(d_ij - r_i - r_j) / (r_i + r_j)]")
    ax.set_ylabel("Margin (+ = separated, - = overlap)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[2, 0]
    for pair_idx, (left, right) in enumerate(class_pairs):
        ax.plot(steps, pair_angle_deg[:, pair_idx], linewidth=1.6, label=f"angle {left}-{right}")
    ax.axhline(
        etf_target_deg,
        color="red",
        linestyle="--",
        linewidth=2.0,
        label=f"ETF target ({etf_target_deg:.1f}°)",
    )
    ax.set_title("Pairwise Angle Between Class Means (Degrees)")
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Angle (degrees)")
    ax.set_ylim(0, 200)
    ax.grid(True, alpha=0.3)
    if len(class_pairs) <= 16:
        ax.legend(fontsize=7)

    ax = axes[2, 1]
    ax.plot(steps, nc2_deviation, linewidth=1.8, label="NC2 deviation")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.5, label="perfect ETF")
    ax.set_title("Mean Cosine Deviation from ETF (NC2)")
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Mean |cos - ETF target|")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[3, 0]
    ax.plot(steps, nc1, linewidth=1.8, label="NC1")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.5, label="perfect collapse")
    ax.set_title("NC1: Within/Between Class Variability")
    ax.set_xlabel("Global Step")
    ax.set_ylabel("NC1 (log scale)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[3, 1]
    avg_margin = np.mean(separation_margin, axis=1)
    ax.plot(steps, avg_margin, linewidth=1.8, label="avg separation margin")
    ax.set_title("Average Separation Margin")
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Mean margin")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    if tpt_step >= 0:
        for axis in [axes[r, c] for r in range(4) for c in range(2)]:
            axis.axvline(tpt_step, color="black", linestyle="-", linewidth=2.0)

    fig.suptitle("Neural Collapse Metrics", fontsize=16)
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def finalize_nc_metrics(
    nc_csv_path: Path,
    output_path: Path,
    num_classes: int,
    class_pairs: list[tuple[int, int]],
    tpt_step: int = -1,
) -> None:
    rows = _load_nc_columns(nc_csv_path)
    if not rows:
        raise ValueError(f"No NC rows found in {nc_csv_path}")

    steps = np.asarray([int(row["global_step"]) for row in rows], dtype=np.int64)

    pair_left = np.asarray([left for left, _ in class_pairs], dtype=np.int64)
    pair_right = np.asarray([right for _, right in class_pairs], dtype=np.int64)
    mu_sqnorm = np.asarray(
        [[row[f"pre_classifier_mu_sqnorm_{class_id}"] for class_id in range(num_classes)] for row in rows],
        dtype=np.float64,
    )
    pair_angle_deg = np.asarray(
        [[row[f"pre_classifier_angle_deg_{left}_{right}"] for left, right in class_pairs] for row in rows],
        dtype=np.float64,
    )
    within_var_by_class = np.asarray(
        [[row[f"pre_classifier_within_var_{class_id}"] for class_id in range(num_classes)] for row in rows],
        dtype=np.float64,
    )
    separation_margin = np.asarray(
        [[row[f"separation_margin_{left}_{right}"] for left, right in class_pairs] for row in rows],
        dtype=np.float64,
    )
    separation_margin = np.nan_to_num(separation_margin, nan=0.0)
    nc1 = np.asarray([row["nc1"] for row in rows], dtype=np.float64)
    nc2_deviation = np.asarray([row["nc2_deviation"] for row in rows], dtype=np.float64)

    pair_cos = np.cos(np.deg2rad(pair_angle_deg))
    norm_i = np.sqrt(np.clip(mu_sqnorm[:, pair_left], 0.0, None))
    norm_j = np.sqrt(np.clip(mu_sqnorm[:, pair_right], 0.0, None))
    pair_dist = np.sqrt(
        np.clip(
            mu_sqnorm[:, pair_left]
            + mu_sqnorm[:, pair_right]
            - 2.0 * norm_i * norm_j * pair_cos,
            0.0,
            None,
        )
    )
    pair_cos = np.nan_to_num(pair_cos, nan=0.0)
    pair_dist = np.nan_to_num(pair_dist, nan=0.0)

    mean_radius_by_class = np.sqrt(np.clip(within_var_by_class, 0.0, None))

    avg_pair_dist = np.mean(pair_dist, axis=1)
    avg_radius = np.mean(mean_radius_by_class, axis=1)
    separation_ratio = avg_pair_dist / (avg_radius + EPS)

    etf_target_deg = float(np.degrees(np.arccos(-1.0 / float(num_classes - 1))))

    _plot_nc_dashboard(
        steps=steps,
        pair_dist=pair_dist,
        mean_radius_by_class=mean_radius_by_class,
        avg_pair_dist=avg_pair_dist,
        avg_radius=avg_radius,
        separation_ratio=separation_ratio,
        pair_cos=pair_cos,
        pair_angle_deg=pair_angle_deg,
        separation_margin=separation_margin,
        nc2_deviation=nc2_deviation,
        nc1=nc1,
        etf_target_deg=etf_target_deg,
        class_pairs=class_pairs,
        output_path=output_path,
        tpt_step=tpt_step,
    )