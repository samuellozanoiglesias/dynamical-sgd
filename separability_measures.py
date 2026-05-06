from __future__ import annotations

import csv
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from model import JAXModel, ParamTree


EPS = 1e-12


@dataclass
class SepEpochRaw:
    participation_ratio_by_class: np.ndarray
    avg_participation_ratio: float
    knn_accuracy: float
    knn_accuracy_by_class: np.ndarray
    scale_factor: float
    scale_norm_pair_dist: np.ndarray
    scale_norm_radius: np.ndarray
    bhattacharyya_by_pair: np.ndarray


def _flatten_features(x: np.ndarray) -> np.ndarray:
    return np.reshape(x, (x.shape[0], -1))


def collect_sep_raw_epoch(
    model: JAXModel,
    params: ParamTree,
    inputs: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
    class_pairs: list[tuple[int, int]],
    eval_batch_size: int,
) -> SepEpochRaw:
    if eval_batch_size <= 0:
        raise ValueError("eval_batch_size must be > 0")

    class_counts = np.zeros(num_classes, dtype=np.float64)
    pre_classifier_sums: np.ndarray | None = None
    all_feats: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    num_samples = int(inputs.shape[0])
    if num_samples == 0:
        raise ValueError("No samples provided for separability metrics.")

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
                pre_classifier_sums[class_id] += np.sum(features[mask], axis=0)

    missing = np.where(class_counts <= 0)[0].tolist()
    if missing:
        raise ValueError(f"Cannot compute separability metrics for classes with no samples: {missing}")

    if pre_classifier_sums is None:
        raise ValueError("No activations found for layer 'pre_classifier'.")

    if not all_feats:
        raise ValueError("No activations cached for layer 'pre_classifier'.")

    all_features = np.concatenate(all_feats, axis=0)
    all_targets = np.concatenate(all_labels, axis=0)
    if all_features.shape[0] != num_samples:
        raise ValueError("Cached activations do not match expected sample count.")

    class_means = pre_classifier_sums / class_counts[:, None]

    participation_ratio_by_class = np.zeros(num_classes, dtype=np.float64)
    for class_id in range(num_classes):
        class_mask = all_targets == class_id
        class_feats = all_features[class_mask]
        n_c, feat_dim = class_feats.shape
        if n_c == 1:
            participation_ratio_by_class[class_id] = 1.0
            continue
        centered = class_feats - class_means[class_id]
        if n_c < feat_dim:
            gram = (centered @ centered.T) / float(n_c - 1)
            eigvals = np.linalg.eigvalsh(gram)
        else:
            cov = (centered.T @ centered) / float(n_c - 1)
            eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.clip(eigvals, 0.0, None)
        sum_eigs = float(np.sum(eigvals))
        sum_sq = float(np.sum(eigvals * eigvals))
        participation_ratio_by_class[class_id] = (sum_eigs * sum_eigs) / (sum_sq + EPS)

    avg_participation_ratio = float(np.mean(participation_ratio_by_class))

    num_total = int(all_features.shape[0])
    if num_total > 2000:
        warnings.warn(
            f"k-NN distance matrix is {num_total}x{num_total}; this may be slow.",
            RuntimeWarning,
        )

    if num_total <= 1:
        knn_accuracy = float("nan")
        knn_accuracy_by_class = np.full(num_classes, np.nan, dtype=np.float64)
    else:
        k = min(5, num_total - 1)
        norms = np.sum(all_features * all_features, axis=1, keepdims=True)
        dists = norms + norms.T - 2.0 * (all_features @ all_features.T)
        dists = np.maximum(dists, 0.0)
        np.fill_diagonal(dists, np.inf)

        neighbor_idx = np.argsort(dists, axis=1)[:, :k]
        preds = np.empty(num_total, dtype=np.int64)
        for idx in range(num_total):
            labels = all_targets[neighbor_idx[idx]]
            counts = np.bincount(labels, minlength=num_classes)
            preds[idx] = int(np.argmax(counts))

        correct = preds == all_targets
        knn_accuracy = float(np.mean(correct))
        knn_accuracy_by_class = np.full(num_classes, np.nan, dtype=np.float64)
        for class_id in range(num_classes):
            mask = all_targets == class_id
            if np.any(mask):
                knn_accuracy_by_class[class_id] = float(np.mean(correct[mask]))

    feature_norms = np.linalg.norm(all_features, axis=1)
    scale_factor = float(np.mean(feature_norms))

    centered = all_features - class_means[all_targets]
    squared_radii = np.sum(centered * centered, axis=1)
    within_var_sum = np.bincount(all_targets, weights=squared_radii, minlength=num_classes).astype(np.float64)
    mean_sq_radius = within_var_sum / class_counts
    radius_by_class = np.sqrt(np.clip(mean_sq_radius, 0.0, None))
    scale_norm_radius = radius_by_class / (scale_factor + EPS)

    scale_norm_pair_dist = np.zeros(len(class_pairs), dtype=np.float64)
    for pair_idx, (left, right) in enumerate(class_pairs):
        diff = class_means[left] - class_means[right]
        dist = float(np.sqrt(np.dot(diff, diff)))
        scale_norm_pair_dist[pair_idx] = dist / (scale_factor + EPS)

    bhattacharyya_by_pair = np.zeros(len(class_pairs), dtype=np.float64)
    for pair_idx, (left, right) in enumerate(class_pairs):
        diff = class_means[left] - class_means[right]
        norm = float(np.sqrt(np.dot(diff, diff)))
        if norm <= EPS:
            bhattacharyya_by_pair[pair_idx] = 1.0
            continue
        direction = diff / (norm + EPS)
        projections = all_features @ direction
        proj_min = float(np.min(projections))
        proj_max = float(np.max(projections))
        if (proj_max - proj_min) <= EPS:
            bhattacharyya_by_pair[pair_idx] = 1.0
            continue
        bins = np.linspace(proj_min, proj_max, 51)
        proj_left = projections[all_targets == left]
        proj_right = projections[all_targets == right]
        hist_left, _ = np.histogram(proj_left, bins=bins)
        hist_right, _ = np.histogram(proj_right, bins=bins)
        sum_left = float(np.sum(hist_left))
        sum_right = float(np.sum(hist_right))
        if sum_left <= 0.0 or sum_right <= 0.0:
            bhattacharyya_by_pair[pair_idx] = float("nan")
            continue
        h_left = hist_left / sum_left
        h_right = hist_right / sum_right
        bhattacharyya_by_pair[pair_idx] = float(np.sum(np.sqrt(h_left * h_right)))

    return SepEpochRaw(
        participation_ratio_by_class=participation_ratio_by_class.astype(np.float64),
        avg_participation_ratio=avg_participation_ratio,
        knn_accuracy=knn_accuracy,
        knn_accuracy_by_class=knn_accuracy_by_class.astype(np.float64),
        scale_factor=scale_factor,
        scale_norm_pair_dist=scale_norm_pair_dist.astype(np.float64),
        scale_norm_radius=scale_norm_radius.astype(np.float64),
        bhattacharyya_by_pair=bhattacharyya_by_pair.astype(np.float64),
    )


def initialize_sep_csv(
    sep_csv_path: Path,
    num_classes: int,
    class_pairs: list[tuple[int, int]],
) -> None:
    header = ["epoch", "global_step"]
    header.extend([f"pr_class_{class_id}" for class_id in range(num_classes)])
    header.append("avg_pr")
    header.append("knn_acc")
    header.extend([f"knn_acc_class_{class_id}" for class_id in range(num_classes)])
    header.append("scale_factor")
    header.extend([f"scale_norm_dist_{left}_{right}" for left, right in class_pairs])
    header.extend([f"scale_norm_radius_{class_id}" for class_id in range(num_classes)])
    header.extend([f"bhattacharyya_{left}_{right}" for left, right in class_pairs])

    with open(sep_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)


def append_sep_csv_row(
    sep_csv_path: Path,
    epoch: int,
    global_step: int,
    raw: SepEpochRaw,
    num_classes: int,
) -> None:
    row: list[float | int] = [epoch, global_step]
    row.extend(raw.participation_ratio_by_class[:num_classes].tolist())
    row.append(float(raw.avg_participation_ratio))
    row.append(float(raw.knn_accuracy))
    row.extend(raw.knn_accuracy_by_class[:num_classes].tolist())
    row.append(float(raw.scale_factor))
    row.extend(raw.scale_norm_pair_dist.tolist())
    row.extend(raw.scale_norm_radius[:num_classes].tolist())
    row.extend(raw.bhattacharyya_by_pair.tolist())

    with open(sep_csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def _load_sep_columns(sep_csv_path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with open(sep_csv_path, "r", encoding="utf-8", newline="") as f:
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


def _plot_sep_dashboard(
    steps: np.ndarray,
    participation_ratio_by_class: np.ndarray,
    avg_participation_ratio: np.ndarray,
    knn_accuracy: np.ndarray,
    knn_accuracy_by_class: np.ndarray,
    scale_norm_pair_dist: np.ndarray,
    scale_norm_radius: np.ndarray,
    bhattacharyya_by_pair: np.ndarray,
    num_classes: int,
    class_pairs: list[tuple[int, int]],
    output_path: Path,
    tpt_step: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True)

    ax = axes[0, 0]
    for class_id in range(participation_ratio_by_class.shape[1]):
        ax.plot(steps, participation_ratio_by_class[:, class_id], linewidth=1.6, label=f"class {class_id}")
    ax.plot(
        steps,
        avg_participation_ratio,
        linewidth=2.6,
        linestyle="--",
        color="black",
        label="avg PR",
    )
    ax.set_title("Participation Ratio")
    ax.set_ylabel("PR (1=line, d=ball)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

    ax = axes[0, 1]
    for class_id in range(knn_accuracy_by_class.shape[1]):
        ax.plot(steps, knn_accuracy_by_class[:, class_id], linewidth=1.6, label=f"class {class_id}")
    ax.plot(steps, knn_accuracy, linewidth=2.4, color="black", label="overall")
    ax.axhline(
        1.0 / float(num_classes),
        color="gray",
        linestyle="--",
        linewidth=1.6,
        label="random",
    )
    ax.set_title("k-NN Accuracy")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

    ax = axes[1, 0]
    for pair_idx, (left, right) in enumerate(class_pairs):
        ax.plot(
            steps,
            scale_norm_pair_dist[:, pair_idx],
            linewidth=1.6,
            label=f"pair {left}-{right}",
        )
    for class_id in range(scale_norm_radius.shape[1]):
        ax.plot(
            steps,
            scale_norm_radius[:, class_id],
            linewidth=1.6,
            linestyle="--",
            label=f"radius {class_id}",
        )
    ax.set_title("Scale-Normalized Distances and Radii")
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Normalized value")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1, 1]
    for pair_idx, (left, right) in enumerate(class_pairs):
        ax.plot(steps, bhattacharyya_by_pair[:, pair_idx], linewidth=1.6, label=f"{left}-{right}")
    ax.axhline(0.0, color="green", linestyle="--", linewidth=1.6, label="perfect separation")
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.6, label="total overlap")
    ax.set_title("Bhattacharyya Coefficient")
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Bhattacharyya coefficient")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)

    if tpt_step >= 0:
        for axis in [axes[r, c] for r in range(2) for c in range(2)]:
            axis.axvline(tpt_step, color="black", linestyle="-", linewidth=2.0)

    fig.suptitle("Separability Measures", fontsize=16)
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def finalize_sep_metrics(
    sep_csv_path: Path,
    output_path: Path,
    num_classes: int,
    class_pairs: list[tuple[int, int]],
    tpt_step: int = -1,
) -> None:
    rows = _load_sep_columns(sep_csv_path)
    if not rows:
        raise ValueError(f"No separability rows found in {sep_csv_path}")

    steps = np.asarray([int(row["global_step"]) for row in rows], dtype=np.int64)

    participation_ratio_by_class = np.asarray(
        [[row[f"pr_class_{class_id}"] for class_id in range(num_classes)] for row in rows],
        dtype=np.float64,
    )
    avg_participation_ratio = np.asarray([row["avg_pr"] for row in rows], dtype=np.float64)
    knn_accuracy = np.asarray([row["knn_acc"] for row in rows], dtype=np.float64)
    knn_accuracy_by_class = np.asarray(
        [[row[f"knn_acc_class_{class_id}"] for class_id in range(num_classes)] for row in rows],
        dtype=np.float64,
    )
    scale_norm_pair_dist = np.asarray(
        [[row[f"scale_norm_dist_{left}_{right}"] for left, right in class_pairs] for row in rows],
        dtype=np.float64,
    )
    scale_norm_radius = np.asarray(
        [[row[f"scale_norm_radius_{class_id}"] for class_id in range(num_classes)] for row in rows],
        dtype=np.float64,
    )
    bhattacharyya_by_pair = np.asarray(
        [[row[f"bhattacharyya_{left}_{right}"] for left, right in class_pairs] for row in rows],
        dtype=np.float64,
    )

    _plot_sep_dashboard(
        steps=steps,
        participation_ratio_by_class=participation_ratio_by_class,
        avg_participation_ratio=avg_participation_ratio,
        knn_accuracy=knn_accuracy,
        knn_accuracy_by_class=knn_accuracy_by_class,
        scale_norm_pair_dist=scale_norm_pair_dist,
        scale_norm_radius=scale_norm_radius,
        bhattacharyya_by_pair=bhattacharyya_by_pair,
        num_classes=num_classes,
        class_pairs=class_pairs,
        output_path=output_path,
        tpt_step=tpt_step,
    )
