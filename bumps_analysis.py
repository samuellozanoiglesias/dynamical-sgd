"""
USE EXAMPLE:

nohup python bumps_analysis.py --bumps_type=with_bumps_before_tpt --training_config=spiral_mlp_relu_1layer_bumps_large > bumps_analysis.log 2>&1 &
"""


from __future__ import annotations

import argparse
import logging
import math
from itertools import product
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


TPT_THRESHOLD = 1.00
PCA_EXPLAINED_COMPONENTS = (1, 2, 3, 5)
PCA_PROJECTED_COMPONENTS = (1, 2, 3, 5)
PCA_PROJECTED_METRICS = ("nc1", "nc2_deviation", "knn_acc", "bhattacharyya_mean")
BASE_ROOT = Path("/data/samuel_lozano/dynamical-sgd")

COLUMN_ORDER = {
    "pca": [],
    "nc": [],
    "sep": [],
    "classifier": [],
}
COLUMN_SETS = {key: set() for key in COLUMN_ORDER}

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze bumps grid experiments.")
    parser.add_argument("--bumps_type", required=True, help="Bumps experiment folder name.")
    parser.add_argument("--training_config", required=True, help="Training config name prefix.")
    return parser.parse_args()


def _all_permutations() -> list[str]:
    return [f"{a}{b}{c}" for a, b, c in product(range(3), repeat=3)]


def discover_experiments(bumps_type: str, training_config: str) -> dict[str, Path]:
    base_path = BASE_ROOT / bumps_type
    if not base_path.exists():
        LOGGER.error("Base path does not exist: %s", base_path)
        return {}

    experiments: dict[str, Path] = {}
    for perm in _all_permutations():
        exp_dir = base_path / f"{perm}_{training_config}"
        if not exp_dir.exists():
            LOGGER.info("Missing experiment folder: %s", exp_dir)
            continue

        runs = [
            child
            for child in exp_dir.iterdir()
            if child.is_dir() and child.name.startswith("training_")
        ]
        if not runs:
            LOGGER.info("No training runs found in: %s", exp_dir)
            continue

        runs_sorted = sorted(runs, key=lambda p: p.name)
        latest_run = runs_sorted[-1]
        experiments[perm] = latest_run
        LOGGER.info("Using latest run for %s: %s", perm, latest_run)

    return experiments


def _update_column_order(category: str, columns: list[str]) -> None:
    seen = COLUMN_SETS[category]
    order = COLUMN_ORDER[category]
    for col in columns:
        if col not in seen:
            seen.add(col)
            order.append(col)


def _read_last_numeric_row(csv_path: Path) -> tuple[dict[str, float], list[str]]:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("CSV is empty")
    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    numeric_cols = [col for col in numeric_df.columns if numeric_df[col].notna().any()]
    if not numeric_cols:
        return {}, []
    last_row = numeric_df.iloc[-1]
    data = {col: float(last_row[col]) for col in numeric_cols}
    return data, numeric_cols


def _infer_class_ids(columns: list[str], prefix: str) -> list[int]:
    ids: set[int] = set()
    for col in columns:
        if col.startswith(prefix):
            tail = col[len(prefix) :]
            if tail.isdigit():
                ids.add(int(tail))
    return sorted(ids)


def _infer_pairs(columns: list[str], prefix: str) -> list[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for col in columns:
        if not col.startswith(prefix):
            continue
        tail = col[len(prefix) :]
        parts = tail.split("_")
        if len(parts) != 2:
            continue
        left, right = parts
        if left.isdigit() and right.isdigit():
            pairs.add((int(left), int(right)))
    return sorted(pairs)


def _build_class_pairs(num_classes: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for left in range(max(0, num_classes - 1)):
        pairs.append((left, left + 1))
    for left in range(num_classes):
        for right in range(left + 2, num_classes):
            pairs.append((left, right))
    return pairs


def _order_pairs(pairs: list[tuple[int, int]], num_classes: int) -> list[tuple[int, int]]:
    if not pairs:
        return []
    preferred = _build_class_pairs(num_classes)
    ordered = [pair for pair in preferred if pair in pairs]
    if ordered:
        return ordered
    return sorted(pairs)


def _augment_pca_metrics(metrics: dict[str, float]) -> None:
    for k in PCA_EXPLAINED_COMPONENTS:
        ratios: list[float] = []
        for idx in range(1, k + 1):
            value = metrics.get(f"pca_var_ratio_{idx}")
            if value is None or not np.isfinite(value):
                ratios.append(float("nan"))
            else:
                ratios.append(float(value))
        if ratios and np.all(np.isfinite(ratios)):
            metrics[f"pca_cum_ratio_{k}"] = float(np.sum(ratios))
        else:
            metrics[f"pca_cum_ratio_{k}"] = float("nan")


def _augment_nc_metrics(
    metrics: dict[str, float],
    class_ids: list[int],
    pairs: list[tuple[int, int]],
) -> None:
    if not class_ids:
        return

    for class_id in class_ids:
        within = metrics.get(f"pre_classifier_within_var_{class_id}")
        if within is None or not np.isfinite(within):
            metrics[f"nc_mean_radius_class_{class_id}"] = float("nan")
        else:
            metrics[f"nc_mean_radius_class_{class_id}"] = float(np.sqrt(max(0.0, float(within))))

    for left, right in pairs:
        angle = metrics.get(f"pre_classifier_angle_deg_{left}_{right}")
        mu_left = metrics.get(f"pre_classifier_mu_sqnorm_{left}")
        mu_right = metrics.get(f"pre_classifier_mu_sqnorm_{right}")
        if angle is None or mu_left is None or mu_right is None:
            metrics[f"nc_pair_cos_{left}_{right}"] = float("nan")
            metrics[f"nc_pair_dist_{left}_{right}"] = float("nan")
            continue
        if not (np.isfinite(angle) and np.isfinite(mu_left) and np.isfinite(mu_right)):
            metrics[f"nc_pair_cos_{left}_{right}"] = float("nan")
            metrics[f"nc_pair_dist_{left}_{right}"] = float("nan")
            continue
        cos_val = float(np.cos(np.deg2rad(float(angle))))
        norm_left = float(np.sqrt(max(0.0, float(mu_left))))
        norm_right = float(np.sqrt(max(0.0, float(mu_right))))
        dist_sq = float(mu_left) + float(mu_right) - 2.0 * norm_left * norm_right * cos_val
        dist = float(np.sqrt(max(dist_sq, 0.0)))
        metrics[f"nc_pair_cos_{left}_{right}"] = cos_val
        metrics[f"nc_pair_dist_{left}_{right}"] = dist

    margins: list[float] = []
    for left, right in pairs:
        margin = metrics.get(f"separation_margin_{left}_{right}")
        if margin is None or not np.isfinite(margin):
            continue
        margins.append(float(margin))
    if margins:
        metrics["nc_avg_separation_margin"] = float(np.mean(margins))
    else:
        metrics["nc_avg_separation_margin"] = float("nan")


def _augment_derived_metrics(
    all_metrics: dict[str, dict[str, float]],
    nc_class_ids: list[int],
    nc_pairs: list[tuple[int, int]],
) -> None:
    for metrics in all_metrics.values():
        _augment_pca_metrics(metrics)
        _augment_nc_metrics(metrics, nc_class_ids, nc_pairs)


def _extract_tpt_episode(numeric_df: pd.DataFrame) -> float:
    if "train_accuracy" not in numeric_df.columns:
        return float("nan")

    train_acc = pd.to_numeric(numeric_df["train_accuracy"], errors="coerce").to_numpy()
    idx = np.where(train_acc >= TPT_THRESHOLD)[0]
    if idx.size == 0:
        return 0.0

    idx0 = int(idx[0])
    for col in ("episode", "epoch", "global_step", "step"):
        if col in numeric_df.columns:
            value = numeric_df.iloc[idx0][col]
            if pd.notna(value):
                return float(int(value))
    return float(idx0 + 1)


def _read_training_metrics(csv_path: Path) -> dict[str, float]:
    metrics = {
        "train_accuracy": float("nan"),
        "test_accuracy": float("nan"),
        "tpt_episode": float("nan"),
    }

    if not csv_path.exists():
        LOGGER.warning("Missing training metrics: %s", csv_path)
        return metrics

    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            raise ValueError("training_metrics.csv is empty")
        numeric_df = df.apply(pd.to_numeric, errors="coerce")
        last_row = numeric_df.iloc[-1]
        metrics["train_accuracy"] = float(last_row.get("train_accuracy", np.nan))
        metrics["test_accuracy"] = float(last_row.get("test_accuracy", np.nan))
        if "tpt_step" in numeric_df.columns:
            tpt_value = last_row.get("tpt_step", np.nan)
            if pd.isna(tpt_value):
                metrics["tpt_episode"] = float("nan")
            elif float(tpt_value) < 0:
                metrics["tpt_episode"] = 0.0
            else:
                metrics["tpt_episode"] = float(tpt_value)
        else:
            metrics["tpt_episode"] = _extract_tpt_episode(numeric_df)
        LOGGER.info("Loaded training metrics: %s", csv_path)
    except Exception as exc:
        LOGGER.warning("Failed to read training metrics %s: %s", csv_path, exc)
    return metrics


def read_experiment(path: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}

    metrics.update(_read_training_metrics(path / "training_metrics.csv"))

    csv_specs = {
        "pca": "pca_analysis.csv",
        "nc": "nc_metrics.csv",
        "sep": "separability_metrics.csv",
        "classifier": "classifier_metrics.csv",
    }

    for category, filename in csv_specs.items():
        csv_path = path / filename
        if not csv_path.exists():
            LOGGER.warning("Missing %s: %s", filename, csv_path)
            continue
        try:
            data, columns = _read_last_numeric_row(csv_path)
            if not columns:
                LOGGER.warning("No numeric columns in %s", csv_path)
                continue
            metrics.update(data)
            _update_column_order(category, columns)
            LOGGER.info("Loaded %s (%d columns)", csv_path, len(columns))
        except Exception as exc:
            LOGGER.warning("Failed to read %s: %s", csv_path, exc)

    return metrics


def build_grid(all_metrics: dict[str, dict[str, float]], metric_name: str) -> np.ndarray:
    grid = np.full((3, 3, 3), np.nan, dtype=np.float64)
    for key, metrics in all_metrics.items():
        if len(key) != 3 or not key.isdigit():
            continue
        d0, d1, d2 = (int(key[0]), int(key[1]), int(key[2]))
        value = metrics.get(metric_name, np.nan)
        try:
            grid[d0, d1, d2] = float(value)
        except Exception:
            grid[d0, d1, d2] = np.nan
    return grid


def _grid_dims(num_items: int) -> tuple[int, int]:
    if num_items <= 0:
        return 1, 1
    cols = int(math.ceil(math.sqrt(num_items)))
    rows = int(math.ceil(num_items / cols))
    return rows, cols


def _normalize_axes(axes, rows: int, cols: int) -> np.ndarray:
    axes_arr = np.array(axes)
    if axes_arr.ndim == 0:
        axes_arr = axes_arr.reshape((1, 1))
    elif axes_arr.ndim == 1:
        axes_arr = axes_arr.reshape((rows, cols))
    return axes_arr


def plot_heatmap_grid(
    data_3d: list[np.ndarray],
    title: str,
    subplot_titles: list[str],
    output_path: Path,
    panel_rows: int | None = None,
    panel_cols: int | None = None,
    cmap: str = "viridis",
    fmt: str = ".3g",
    cbar_labels: list[str] | None = None,
    metric_limits: list[tuple[float | None, float | None] | None] | None = None,
    panel_title_pad: float = 0.01,
    layout_top: float = 0.92,
    panel_width: float = 4.2,
    panel_height: float = 3.2,
    dpi: int = 180,
    annot_kws: dict[str, float] | None = None,
    cbar_mode: str = "each",
) -> None:
    num_metrics = len(data_3d)
    if num_metrics == 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No metrics to plot", ha="center", va="center")
        ax.axis("off")
        fig.suptitle(title)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return

    if panel_rows is None or panel_cols is None:
        panel_rows, panel_cols = _grid_dims(num_metrics)

    total_rows = panel_rows * 3
    fig_width = max(8.0, panel_cols * panel_width)
    fig_height = max(6.0, total_rows * panel_height)
    fig, axes = plt.subplots(total_rows, panel_cols, figsize=(fig_width, fig_height))
    axes = _normalize_axes(axes, total_rows, panel_cols)

    total_slots = panel_rows * panel_cols
    for d0 in range(3):
        for metric_idx in range(total_slots):
            row = d0 * panel_rows + (metric_idx // panel_cols)
            col = metric_idx % panel_cols
            ax = axes[row, col]
            if metric_idx >= num_metrics:
                ax.axis("off")
                continue
            data = data_3d[metric_idx][d0]
            mask = np.isnan(data)
            show_cbar = True
            if cbar_mode == "per_metric":
                show_cbar = d0 == 0
            elif cbar_mode == "first":
                show_cbar = d0 == 0 and metric_idx == 0
            cbar_kws = None
            if show_cbar and cbar_labels and metric_idx < len(cbar_labels) and cbar_labels[metric_idx]:
                cbar_kws = {"label": cbar_labels[metric_idx]}
            vmin = None
            vmax = None
            if metric_limits and metric_idx < len(metric_limits):
                limits = metric_limits[metric_idx]
                if limits is not None:
                    vmin, vmax = limits
            sns.heatmap(
                data,
                ax=ax,
                annot=True,
                fmt=fmt,
                annot_kws=annot_kws,
                cmap=cmap,
                mask=mask,
                vmin=vmin,
                vmax=vmax,
                cbar=show_cbar,
                cbar_kws=cbar_kws,
                linewidths=0.5,
                linecolor="white",
                xticklabels=[0, 1, 2],
                yticklabels=[0, 1, 2],
            )
            ax.set_xlabel("digit_2")
            ax.set_ylabel("digit_1")
            ax.set_title(subplot_titles[metric_idx], fontsize=9)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0.03, 0.03, 1.0, layout_top])

    for d0 in range(3):
        first_row = d0 * panel_rows
        last_row = first_row + panel_rows - 1
        left = axes[first_row, 0].get_position().x0
        right = axes[first_row, panel_cols - 1].get_position().x1
        top = axes[first_row, 0].get_position().y1
        x_center = (left + right) / 2.0
        y_pos = top + panel_title_pad
        if y_pos > 0.95:
            y_pos = top - 0.02
        fig.text(
            x_center,
            y_pos,
            f"digit_0 = {d0}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _figure_style(num_metrics: int) -> dict[str, object]:
    if num_metrics >= 40:
        return {
            "panel_width": 2.6,
            "panel_height": 2.0,
            "dpi": 120,
            "annot_kws": {"fontsize": 6.0},
        }
    if num_metrics >= 20:
        return {
            "panel_width": 3.0,
            "panel_height": 2.4,
            "dpi": 140,
            "annot_kws": {"fontsize": 7.0},
        }
    return {
        "panel_width": 3.8,
        "panel_height": 3.0,
        "dpi": 180,
        "annot_kws": {"fontsize": 8.0},
    }


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    LOGGER.info("Starting bumps analysis for %s / %s", args.bumps_type, args.training_config)
    experiments = discover_experiments(args.bumps_type, args.training_config)

    all_metrics: dict[str, dict[str, float]] = {}
    success_count = 0
    for key in _all_permutations():
        run_path = experiments.get(key)
        if run_path is None:
            all_metrics[key] = {}
            continue
        try:
            metrics = read_experiment(run_path)
            all_metrics[key] = metrics
            if any(np.isfinite(value) for value in metrics.values()):
                success_count += 1
        except Exception as exc:
            LOGGER.error("Failed to read experiment %s: %s", key, exc)
            all_metrics[key] = {}

    plots_dir = BASE_ROOT / args.bumps_type / "bumps_analysis_plots" / args.training_config
    plots_dir.mkdir(parents=True, exist_ok=True)

    nc_class_ids = _infer_class_ids(COLUMN_ORDER["nc"], "pre_classifier_mu_sqnorm_")
    if not nc_class_ids:
        nc_class_ids = [0, 1, 2]
    nc_pairs = _infer_pairs(COLUMN_ORDER["nc"], "pre_classifier_angle_deg_")
    nc_pairs = _order_pairs(nc_pairs, max(nc_class_ids) + 1 if nc_class_ids else 0)
    _augment_derived_metrics(all_metrics, nc_class_ids, nc_pairs)

    training_metrics = ["train_accuracy", "test_accuracy", "tpt_episode"]
    training_titles = [
        "final train_accuracy",
        "final test_accuracy",
        "tpt_episode (0 = not reached)",
    ]
    training_cbar = ["train_accuracy", "test_accuracy", "tpt_episode (0 = not reached)"]
    training_data = [build_grid(all_metrics, metric) for metric in training_metrics]
    train_test_values = np.concatenate([training_data[0].ravel(), training_data[1].ravel()])
    finite_train_test = train_test_values[np.isfinite(train_test_values)]
    if finite_train_test.size > 0:
        shared_min = float(np.min(finite_train_test))
        shared_max = float(np.max(finite_train_test))
    else:
        shared_min = None
        shared_max = None
    tpt_values = training_data[2].ravel()
    finite_tpt = tpt_values[np.isfinite(tpt_values)]
    if finite_tpt.size > 0:
        tpt_min = 0.0
        tpt_max = float(np.max(finite_tpt))
    else:
        tpt_min = None
        tpt_max = None
    training_limits = [
        (shared_min, shared_max),
        (shared_min, shared_max),
        (tpt_min, tpt_max),
    ]
    LOGGER.info("Plotting fig1 (%d metrics)", len(training_metrics))
    plot_heatmap_grid(
        training_data,
        "Training Summary",
        training_titles,
        plots_dir / "fig1_training_summary.png",
        panel_rows=1,
        panel_cols=3,
        cbar_labels=training_cbar,
        metric_limits=training_limits,
        panel_title_pad=0.05,
        layout_top=0.84,
    )
    LOGGER.info("Saved fig1 to %s", plots_dir / "fig1_training_summary.png")

    pca_variance_cols = [
        *(f"pca_cum_ratio_{k}" for k in PCA_EXPLAINED_COMPONENTS),
        "pca_k95",
        "pca_k99",
    ]
    pca_variance_titles = [
        *(f"cum var (k={k})" for k in PCA_EXPLAINED_COMPONENTS),
        "k95 components",
        "k99 components",
    ]
    pca_projected_cols = [
        f"proj_{k}_{metric}"
        for metric in PCA_PROJECTED_METRICS
        for k in PCA_PROJECTED_COMPONENTS
    ]
    pca_projected_titles = [
        f"PCA={k} {metric.replace('_', ' ')}"
        for metric in PCA_PROJECTED_METRICS
        for k in PCA_PROJECTED_COMPONENTS
    ]

    variance_style = _figure_style(len(pca_variance_cols))
    LOGGER.info("Plotting fig2 (%d metrics)", len(pca_variance_cols))
    plot_heatmap_grid(
        [build_grid(all_metrics, col) for col in pca_variance_cols],
        "PCA Explained Variance",
        pca_variance_titles,
        plots_dir / "fig2_pca_explained_variance.png",
        cbar_mode="per_metric",
        **variance_style,
    )
    LOGGER.info("Saved fig2 to %s", plots_dir / "fig2_pca_explained_variance.png")

    projected_style = _figure_style(len(pca_projected_cols))
    LOGGER.info("Plotting fig3 (%d metrics)", len(pca_projected_cols))
    plot_heatmap_grid(
        [build_grid(all_metrics, col) for col in pca_projected_cols],
        "PCA Projected Metrics",
        pca_projected_titles,
        plots_dir / "fig3_pca_projected_metrics.png",
        cbar_mode="per_metric",
        **projected_style,
    )
    LOGGER.info("Saved fig3 to %s", plots_dir / "fig3_pca_projected_metrics.png")

    sep_class_ids = _infer_class_ids(COLUMN_ORDER["sep"], "pr_class_")
    if not sep_class_ids:
        sep_class_ids = [0, 1, 2]
    sep_pairs = _infer_pairs(COLUMN_ORDER["sep"], "scale_norm_dist_")
    sep_pairs = _order_pairs(sep_pairs, max(sep_class_ids) + 1 if sep_class_ids else 0)
    sep_columns = [
        *(f"pr_class_{class_id}" for class_id in sep_class_ids),
        "avg_pr",
        *(f"knn_acc_class_{class_id}" for class_id in sep_class_ids),
        "knn_acc",
        *(f"scale_norm_dist_{left}_{right}" for left, right in sep_pairs),
        *(f"scale_norm_radius_{class_id}" for class_id in sep_class_ids),
        *(f"bhattacharyya_{left}_{right}" for left, right in sep_pairs),
    ]
    sep_titles = [
        *(f"PR class {class_id}" for class_id in sep_class_ids),
        "PR avg",
        *(f"kNN class {class_id}" for class_id in sep_class_ids),
        "kNN avg",
        *(f"scale dist {left}-{right}" for left, right in sep_pairs),
        *(f"scale radius {class_id}" for class_id in sep_class_ids),
        *(f"bhattacharyya {left}-{right}" for left, right in sep_pairs),
    ]
    sep_style = _figure_style(len(sep_columns))
    LOGGER.info("Plotting fig4 (%d metrics)", len(sep_columns))
    plot_heatmap_grid(
        [build_grid(all_metrics, col) for col in sep_columns],
        "Separability Measures",
        sep_titles,
        plots_dir / "fig4_separability_measures.png",
        cbar_mode="per_metric",
        **sep_style,
    )
    LOGGER.info("Saved fig4 to %s", plots_dir / "fig4_separability_measures.png")

    nc_columns = [
        *(f"nc_pair_dist_{left}_{right}" for left, right in nc_pairs),
        *(f"nc_mean_radius_class_{class_id}" for class_id in nc_class_ids),
        *(f"nc_pair_cos_{left}_{right}" for left, right in nc_pairs),
        *(f"separation_margin_{left}_{right}" for left, right in nc_pairs),
        *(f"pre_classifier_angle_deg_{left}_{right}" for left, right in nc_pairs),
        "nc2_deviation",
        "nc1",
        "nc_avg_separation_margin",
    ]
    nc_titles = [
        *(f"pair dist {left}-{right}" for left, right in nc_pairs),
        *(f"mean radius {class_id}" for class_id in nc_class_ids),
        *(f"pair cos {left}-{right}" for left, right in nc_pairs),
        *(f"sep margin {left}-{right}" for left, right in nc_pairs),
        *(f"angle {left}-{right}" for left, right in nc_pairs),
        "NC2 deviation",
        "NC1",
        "avg sep margin",
    ]
    nc_style = _figure_style(len(nc_columns))
    LOGGER.info("Plotting fig5 (%d metrics)", len(nc_columns))
    plot_heatmap_grid(
        [build_grid(all_metrics, col) for col in nc_columns],
        "Neural Collapse Metrics",
        nc_titles,
        plots_dir / "fig5_neural_collapse.png",
        cbar_mode="per_metric",
        **nc_style,
    )
    LOGGER.info("Saved fig5 to %s", plots_dir / "fig5_neural_collapse.png")

    classifier_class_ids = _infer_class_ids(COLUMN_ORDER["classifier"], "weight_norm_class_")
    if not classifier_class_ids:
        classifier_class_ids = [0, 1, 2]
    classifier_columns = [
        *(f"logit_correct_mean_class_{class_id}" for class_id in classifier_class_ids),
        *(f"logit_max_wrong_mean_class_{class_id}" for class_id in classifier_class_ids),
        "weight_orthogonality",
        "stable_rank",
        "path_curvature_ratio",
        *(f"gsnr_class_{class_id}" for class_id in classifier_class_ids),
        *(f"weight_norm_class_{class_id}" for class_id in classifier_class_ids),
        *(f"weight_mean_alignment_class_{class_id}" for class_id in classifier_class_ids),
        "condition_number",
    ]
    classifier_titles = [
        *(f"logit correct {class_id}" for class_id in classifier_class_ids),
        *(f"logit max wrong {class_id}" for class_id in classifier_class_ids),
        "weight orthogonality",
        "stable rank",
        "path curvature",
        *(f"GSNR {class_id}" for class_id in classifier_class_ids),
        *(f"weight norm {class_id}" for class_id in classifier_class_ids),
        *(f"alignment {class_id}" for class_id in classifier_class_ids),
        "condition number",
    ]
    classifier_style = _figure_style(len(classifier_columns))
    LOGGER.info("Plotting fig6 (%d metrics)", len(classifier_columns))
    plot_heatmap_grid(
        [build_grid(all_metrics, col) for col in classifier_columns],
        "Classifier Metrics",
        classifier_titles,
        plots_dir / "fig6_classifier_metrics.png",
        cbar_mode="per_metric",
        **classifier_style,
    )
    LOGGER.info("Saved fig6 to %s", plots_dir / "fig6_classifier_metrics.png")

    LOGGER.info("Loaded %d/%d experiments successfully", success_count, len(_all_permutations()))


if __name__ == "__main__":
    main()
