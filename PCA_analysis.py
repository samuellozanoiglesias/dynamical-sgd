from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np

from neural_collapse import collect_nc_raw_epoch
from separability_measures import collect_sep_raw_epoch


EPS = 1e-12


@dataclass
class ProjectedMetrics:
    nc1: float
    nc2_deviation: float
    knn_accuracy: float
    bhattacharyya_mean: float


@dataclass
class PCAEpochRaw:
    explained_variance_ratio: np.ndarray
    k95: int
    k99: int
    projected: dict[int, ProjectedMetrics]


def _flatten_features(x: np.ndarray) -> np.ndarray:
    return np.reshape(x, (x.shape[0], -1))


def _compute_pca(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    centered = features - np.mean(features, axis=0, keepdims=True)
    num_samples = int(centered.shape[0])
    if num_samples <= 1:
        dim = centered.shape[1]
        explained_ratio = np.zeros(dim, dtype=np.float64)
        components = np.eye(dim, dtype=np.float64)
        return centered, components, explained_ratio, np.cumsum(explained_ratio)

    cov = (centered.T @ centered) / float(max(1, num_samples - 1))
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.clip(eigvals[order], 0.0, None)
    eigvecs = eigvecs[:, order]

    total_var = float(np.sum(eigvals))
    if total_var <= EPS:
        explained_ratio = np.zeros_like(eigvals, dtype=np.float64)
    else:
        explained_ratio = eigvals / (total_var + EPS)

    components = eigvecs.T
    cum_ratio = np.cumsum(explained_ratio)
    return centered, components, explained_ratio.astype(np.float64), cum_ratio.astype(np.float64)


def _intrinsic_dim(cum_ratio: np.ndarray, threshold: float) -> int:
    if cum_ratio.size == 0:
        return 0
    idx = int(np.searchsorted(cum_ratio, threshold, side="left"))
    return int(min(idx + 1, cum_ratio.size))


def normalize_projected_dims(projected_dims: Sequence[int] | int | None, max_dim: int) -> list[int]:
    if max_dim <= 0:
        return [1]

    if projected_dims is None:
        dims: Iterable[int] = [3, 5]
    elif isinstance(projected_dims, (int, np.integer)):
        dims = [int(projected_dims)]
    else:
        dims = [int(v) for v in projected_dims]

    cleaned = [int(v) for v in dims if int(v) > 0]
    if not cleaned:
        cleaned = [min(3, max_dim)]

    cleaned = [min(int(v), max_dim) for v in cleaned]
    return sorted(set(cleaned))


class _ProjectedFeatureModel:
    def apply(self, params, x: np.ndarray, *, return_intermediates: bool = False):
        features = np.asarray(x, dtype=np.float64)
        if not return_intermediates:
            return features
        logits = np.zeros((features.shape[0], 1), dtype=np.float32)
        return logits, {"pre_classifier": features}


def collect_pca_epoch(
    pre_classifier: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
    class_pairs: list[tuple[int, int]],
    eval_batch_size: int,
    projected_dims: Sequence[int] | int | None = None,
) -> PCAEpochRaw:
    features = _flatten_features(np.asarray(pre_classifier, dtype=np.float64))
    targets_arr = np.asarray(targets, dtype=np.int64)
    if features.shape[0] == 0:
        raise ValueError("No samples provided for PCA analysis.")

    centered, components, explained_ratio, cum_ratio = _compute_pca(features)
    k95 = _intrinsic_dim(cum_ratio, 0.95)
    k99 = _intrinsic_dim(cum_ratio, 0.99)

    max_dim = int(features.shape[1])
    proj_dims = normalize_projected_dims(projected_dims, max_dim)

    projected_metrics: dict[int, ProjectedMetrics] = {}
    proxy_model = _ProjectedFeatureModel()
    for k in proj_dims:
        k = int(min(max(1, k), max_dim))
        projected = centered @ components[:k].T
        nc_raw = collect_nc_raw_epoch(
            model=proxy_model,
            params={},
            inputs=projected,
            targets=targets_arr,
            num_classes=num_classes,
            class_pairs=class_pairs,
            eval_batch_size=eval_batch_size,
        )
        sep_raw = collect_sep_raw_epoch(
            model=proxy_model,
            params={},
            inputs=projected,
            targets=targets_arr,
            num_classes=num_classes,
            class_pairs=class_pairs,
            eval_batch_size=eval_batch_size,
        )
        projected_metrics[k] = ProjectedMetrics(
            nc1=float(nc_raw.nc1),
            nc2_deviation=float(nc_raw.nc2_deviation),
            knn_accuracy=float(sep_raw.knn_accuracy),
            bhattacharyya_mean=float(np.nanmean(sep_raw.bhattacharyya_by_pair)),
        )

    return PCAEpochRaw(
        explained_variance_ratio=explained_ratio.astype(np.float64),
        k95=int(k95),
        k99=int(k99),
        projected=projected_metrics,
    )


def initialize_pca_csv(
    csv_path: Path,
    feature_dim: int,
    projected_dims: Sequence[int] | int | None,
) -> None:
    dims = normalize_projected_dims(projected_dims, feature_dim)
    header = ["epoch", "global_step", "pca_k95", "pca_k99"]
    header.extend([f"pca_var_ratio_{idx + 1}" for idx in range(feature_dim)])
    for k in dims:
        header.extend(
            [
                f"proj_{k}_nc1",
                f"proj_{k}_nc2_deviation",
                f"proj_{k}_knn_acc",
                f"proj_{k}_bhattacharyya_mean",
            ]
        )

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)


def append_pca_csv_row(
    csv_path: Path,
    epoch: int,
    global_step: int,
    raw: PCAEpochRaw,
    feature_dim: int,
    projected_dims: Sequence[int] | int | None,
) -> None:
    dims = normalize_projected_dims(projected_dims, feature_dim)
    ratio = raw.explained_variance_ratio
    if ratio.size < feature_dim:
        padded = np.full(feature_dim, np.nan, dtype=np.float64)
        padded[: ratio.size] = ratio
        ratio = padded

    row: list[float | int] = [epoch, global_step, raw.k95, raw.k99]
    row.extend(ratio[:feature_dim].tolist())
    for k in dims:
        metrics = raw.projected.get(k)
        if metrics is None:
            row.extend([float("nan"), float("nan"), float("nan"), float("nan")])
        else:
            row.extend([
                float(metrics.nc1),
                float(metrics.nc2_deviation),
                float(metrics.knn_accuracy),
                float(metrics.bhattacharyya_mean),
            ])

    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def _load_pca_columns(csv_path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            casted: dict[str, float] = {}
            for key, value in row.items():
                if value is None or value == "":
                    casted[key] = float("nan")
                    continue
                if key in {"epoch", "global_step", "pca_k95", "pca_k99"}:
                    casted[key] = float(int(float(value)))
                else:
                    casted[key] = float(value)
            rows.append(casted)
    return rows


def finalize_pca_analysis(
    csv_path: Path,
    variance_output_path: Path,
    projected_output_path: Path,
    feature_dim: int,
    projected_dims: Sequence[int] | int | None,
    tpt_step: int = -1,
) -> None:
    rows = _load_pca_columns(csv_path)
    if not rows:
        raise ValueError(f"No PCA rows found in {csv_path}")

    dims = normalize_projected_dims(projected_dims, feature_dim)
    steps = np.asarray([int(row["global_step"]) for row in rows], dtype=np.int64)
    k95 = np.asarray([int(row["pca_k95"]) for row in rows], dtype=np.int64)
    k99 = np.asarray([int(row["pca_k99"]) for row in rows], dtype=np.int64)

    var_ratio = np.asarray(
        [[row.get(f"pca_var_ratio_{idx + 1}", float("nan")) for idx in range(feature_dim)] for row in rows],
        dtype=np.float64,
    )

    last_ratio = var_ratio[-1]
    last_ratio = np.nan_to_num(last_ratio, nan=0.0)
    cum_ratio = np.cumsum(last_ratio)

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    ax = axes[0]
    ax.plot(np.arange(1, cum_ratio.size + 1), cum_ratio, linewidth=2.0)
    ax.axhline(0.95, color="red", linestyle="--", linewidth=1.4, label="95%")
    ax.axhline(0.99, color="black", linestyle="--", linewidth=1.4, label="99%")
    if k95[-1] > 0:
        ax.axvline(k95[-1], color="red", linestyle=":", linewidth=1.6)
    if k99[-1] > 0:
        ax.axvline(k99[-1], color="black", linestyle=":", linewidth=1.6)
    ax.set_title("Cumulative Explained Variance (last epoch)")
    ax.set_ylabel("Cumulative variance")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(steps, k95, linewidth=1.8, label="95% components")
    ax.plot(steps, k99, linewidth=1.8, label="99% components")
    ax.set_title("Intrinsic Dimensionality Over Time")
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Components")
    ax.grid(True, alpha=0.3)
    if tpt_step >= 0:
        ax.axvline(tpt_step, color="black", linestyle="-", linewidth=2.0)
    ax.legend(fontsize=8)

    plt.tight_layout()
    variance_output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(variance_output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    metrics: dict[int, dict[str, np.ndarray]] = {}
    for k in dims:
        metrics[k] = {
            "nc1": np.asarray([row.get(f"proj_{k}_nc1", float("nan")) for row in rows], dtype=np.float64),
            "nc2": np.asarray([row.get(f"proj_{k}_nc2_deviation", float("nan")) for row in rows], dtype=np.float64),
            "knn": np.asarray([row.get(f"proj_{k}_knn_acc", float("nan")) for row in rows], dtype=np.float64),
            "bhat": np.asarray(
                [row.get(f"proj_{k}_bhattacharyya_mean", float("nan")) for row in rows], dtype=np.float64
            ),
        }

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True)

    ax = axes[0, 0]
    for k in dims:
        ax.plot(steps, metrics[k]["nc1"], linewidth=1.8, label=f"PCA={k}")
    ax.set_title("Projected NC1")
    ax.set_ylabel("NC1 (log)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for k in dims:
        ax.plot(steps, metrics[k]["nc2"], linewidth=1.8, label=f"PCA={k}")
    ax.set_title("Projected NC2 Deviation")
    ax.set_ylabel("Mean |cos - ETF|")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    for k in dims:
        ax.plot(steps, metrics[k]["knn"], linewidth=1.8, label=f"PCA={k}")
    ax.set_title("Projected k-NN Accuracy")
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    for k in dims:
        ax.plot(steps, metrics[k]["bhat"], linewidth=1.8, label=f"PCA={k}")
    ax.set_title("Projected Bhattacharyya (mean)")
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Coefficient")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    if tpt_step >= 0:
        for axis in axes.flat:
            axis.axvline(tpt_step, color="black", linestyle="-", linewidth=2.0)

    fig.suptitle("Projected NC/Sep Metrics", fontsize=16)
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    projected_output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(projected_output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
