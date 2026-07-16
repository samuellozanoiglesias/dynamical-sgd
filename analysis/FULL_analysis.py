#!/usr/bin/env python3
"""
FULL_analysis.py

Processes toy datasets (Blobs, Spiral, Dartboard, Rings, Checkerboard) AND
real datasets (mnist_mlp_small, mnist_mlp, cifar10_mlp, cifar100_mlp,
mnist_resnet, mnist_resnet18, mnist_resnet18_2, cifar10_resnet,
cifar100_resnet) to compare training metrics with and without oscillations.

Toy datasets store metrics across three CSVs per run:
  training_metrics.csv, classifier_metrics.csv, PCA_geometric.csv

Real datasets store metrics across two CSVs per run:
  training_metrics.csv (accuracy, averaged over classes)
  metrics_for_multiple_classes.csv (path_curvature_ratio,
      bhattacharyya_distance_mean, already averaged/vectorized -
      see metrics_for_multiple_classes.py)

For each of the 4 metrics (train accuracy, test accuracy, Bhattacharyya
distance, path curvature ratio) this script produces THREE versions of the
difference plot (With - Without oscillations):
  - all datasets together (toy + real)
  - toy datasets only
  - real datasets only

The path-curvature plots additionally draw a constant horizontal reference
line: the mean (across the datasets shown in that plot) of the cumulative
(step-integrated) path-curvature-ratio difference. This is a single scalar
per plot, not a per-step curve, hence "constant".

Outputs (13 PNGs):
  plots_difference/all/train_accuracy_diff.png
  plots_difference/all/test_accuracy_diff.png
  plots_difference/all/bhattacharyya_diff.png
  plots_difference/all/path_curvature_diff.png
  plots_difference/toy/...            (same 4)
  plots_difference/real/...           (same 4)
  plots_difference/legend.png

Usage
-----
nohup python FULL_analysis.py > log_diff.out 2>&1 &
"""

import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
ROOT_DIR = "/data/samuel_lozano/dynamical-sgd"

# (label, directory_key) pairs. label is used for legends/titles; directory_key
# is the token used inside "<key>-no_bumps" / "<key>-always_bumps" folder names.
TOY_DATASET_SPECS = [
    ("Blobs", "blobs"),
    ("Spiral", "spiral"),
    ("Dartboard", "dartboard"),
    ("Rings", "rings"),
    ("Checkerboard", "checkerboard"),
]

REAL_DATASET_SPECS = [
    ("mnist_mlp_small", "mnist_mlp_small"),
    ("mnist_mlp", "mnist_mlp"),
    ("cifar10_mlp", "cifar10_mlp"),
    ("cifar100_mlp", "cifar100_mlp"),
    ("mnist_resnet", "mnist_resnet"),
    ("mnist_resnet18", "mnist_resnet18"),
    ("mnist_resnet18_2", "mnist_resnet18_2"),
    ("cifar10_resnet", "cifar10_resnet"),
    ("cifar100_resnet", "cifar100_resnet"),
]

# label -> path, label -> kind ("toy" / "real")
DATASETS_WITHOUT = {}
DATASETS_WITH = {}
DATASET_KIND = {}

for label, key in TOY_DATASET_SPECS:
    DATASETS_WITHOUT[label] = f"{ROOT_DIR}/without_bumps/{key}-no_bumps/"
    DATASETS_WITH[label] = f"{ROOT_DIR}/with_bumps/{key}-always_bumps/"
    DATASET_KIND[label] = "toy"

for label, key in REAL_DATASET_SPECS:
    DATASETS_WITHOUT[label] = f"{ROOT_DIR}/without_bumps/{key}-no_bumps/"
    DATASETS_WITH[label] = f"{ROOT_DIR}/with_bumps/{key}-always_bumps/"
    DATASET_KIND[label] = "real"

SMOOTH_WINDOW = 1

# CSV files to look for, per dataset kind.
CSV_FILES_BY_KIND = {
    "toy": {
        "training": "training_metrics.csv",
        "classifier": "classifier_metrics.csv",
        "pca_geom": "PCA_geometric.csv",
    },
    "real": {
        "training": "training_metrics.csv",
        "multiclass": "metrics_for_multiple_classes.csv",
    },
}

# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def find_file(run_dir, filename):
    matches = glob.glob(os.path.join(run_dir, "**", filename), recursive=True)
    if not matches:
        return None
    return matches[0]


def discover_runs_for_label(label, path, kind):
    runs = []
    if not os.path.isdir(path):
        print(f"[WARN] path for dataset '{label}' does not exist: {path}")
        return runs

    clean_path = path.rstrip(os.sep)
    if os.path.basename(clean_path).startswith("training_"):
        train_dirs = [path]
    else:
        train_dirs = sorted(glob.glob(os.path.join(path, "training_*")))
        if not train_dirs:
            train_dirs = [path]

    csv_files = CSV_FILES_BY_KIND[kind]

    for d in train_dirs:
        if not os.path.isdir(d):
            continue

        dfs = {}
        for csv_key, fname in csv_files.items():
            fpath = find_file(d, fname)
            if fpath is None:
                continue
            try:
                dfs[csv_key] = pd.read_csv(fpath)
            except Exception:
                pass

        if not dfs:
            continue
        runs.append({"dir": d, "label": label, "kind": kind, "dfs": dfs})

    return runs


def discover_all(datasets, kind_map):
    runs = []
    for label, path in datasets.items():
        runs.extend(discover_runs_for_label(label, path, kind_map[label]))
    return runs


def get_grouped(runs):
    groups = {}
    for r in runs:
        groups.setdefault(r["label"], []).append(r)
    return groups

# --------------------------------------------------------------------------- #
# Helpers & metric definitions
# --------------------------------------------------------------------------- #
def detect_pairs(df, prefix):
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)v(\d+)$")
    pairs = []
    for c in df.columns:
        m = pattern.match(c)
        if m:
            pairs.append((int(m.group(1)), int(m.group(2)), c))
    pairs.sort(key=lambda t: (t[0], t[1]))
    return pairs


def mean_over_pairs(df, prefix):
    pairs = detect_pairs(df, prefix)
    if not pairs:
        return pd.Series(np.nan, index=df.index)
    cols = [c for (_, _, c) in pairs]
    return df[cols].mean(axis=1)


# Per-metric config: for each metric, which csv_key + value_fn to use for
# each dataset kind ("toy" / "real"). This is the single place that encodes
# the mapping between the old three-CSV toy format and the new
# metrics_for_multiple_classes.csv format for real datasets.
METRIC_CONFIG = {
    "train_accuracy": {
        "toy": ("training", lambda df: df["train_accuracy"]),
        "real": ("training", lambda df: df["train_accuracy"]),
        "ylabel": "Train accuracy diff (With - Without)",
        "title": "Train accuracy difference\n(With vs Without Oscillations)",
        "filename": "train_accuracy_diff.png",
    },
    "test_accuracy": {
        "toy": ("training", lambda df: df["test_accuracy"]),
        "real": ("training", lambda df: df["test_accuracy"]),
        "ylabel": "Test accuracy diff (With - Without)",
        "title": "Test accuracy difference\n(With vs Without Oscillations)",
        "filename": "test_accuracy_diff.png",
    },
    "bhattacharyya": {
        "toy": ("pca_geom", lambda df: mean_over_pairs(df, "ellipsoid_bhattacharyya")),
        "real": ("multiclass", lambda df: df["bhattacharyya_distance_mean"]),
        "ylabel": "Mean Bhattacharyya distance diff",
        "title": "Bhattacharyya distance difference\n(With vs Without Oscillations)",
        "filename": "bhattacharyya_diff.png",
    },
    "path_curvature_ratio": {
        "toy": ("classifier", lambda df: df["path_curvature_ratio"]),
        "real": ("multiclass", lambda df: df["path_curvature_ratio"]),
        "ylabel": "Path curvature ratio diff",
        "title": "Path curvature ratio difference\n(With vs Without Oscillations)",
        "filename": "path_curvature_diff.png",
    },
}

# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def get_curve(group_runs, csv_key, value_fn):
    series_list = []
    for r in group_runs:
        df = r["dfs"].get(csv_key)
        if df is None or "global_step" not in df.columns:
            continue
        try:
            vals = value_fn(df)
        except Exception:
            continue
        if vals is None:
            continue
        s = pd.Series(np.asarray(vals, dtype=float), index=df["global_step"].values)
        s = s[~s.index.duplicated(keep="first")]
        series_list.append(s)

    if not series_list:
        return None, None

    combined = pd.concat(series_list, axis=1)
    mean_curve = combined.mean(axis=1, skipna=True).sort_index()
    return mean_curve.index.values, mean_curve.values


def get_metric_curve_for_group(group_runs, kind, metric_name):
    """Resolve the correct csv_key/value_fn for this metric+kind, then get_curve."""
    csv_key, value_fn = METRIC_CONFIG[metric_name][kind]
    return get_curve(group_runs, csv_key, value_fn)


def smooth(values, window):
    if window is None or window <= 1:
        return values
    s = pd.Series(values)
    return s.rolling(window=window, min_periods=1, center=True).mean().values


def extend_flat(idx, vals, x_min, x_max):
    """Extend a (idx, vals) curve so it starts at x_min and ends at x_max,
    by holding the first/last value flat where the curve doesn't already
    reach that far. This makes curves with different native step ranges
    visually comparable on a shared x-axis instead of stopping early."""
    idx = np.asarray(idx, dtype=float)
    vals = np.asarray(vals, dtype=float)
    if len(idx) == 0:
        return idx, vals

    new_idx = idx
    new_vals = vals
    if idx[0] > x_min:
        new_idx = np.concatenate(([x_min], new_idx))
        new_vals = np.concatenate(([vals[0]], new_vals))
    if idx[-1] < x_max:
        new_idx = np.concatenate((new_idx, [x_max]))
        new_vals = np.concatenate((new_vals, [vals[-1]]))
    return new_idx, new_vals


def common_x_range(plot_data):
    """Global (min, max) step across all curves in plot_data."""
    mins, maxs = [], []
    for idx, _ in plot_data.values():
        if len(idx) == 0:
            continue
        mins.append(np.min(idx))
        maxs.append(np.max(idx))
    if not mins:
        return None, None
    return min(mins), max(maxs)


def extend_all_to_common_range(plot_data):
    """Apply extend_flat to every curve in plot_data so they all share the
    same start/end x-value. Returns a new dict in the same shape."""
    x_min, x_max = common_x_range(plot_data)
    if x_min is None:
        return plot_data
    extended = {}
    for label, (idx, vals) in plot_data.items():
        extended[label] = extend_flat(idx, vals, x_min, x_max)
    return extended

# --------------------------------------------------------------------------- #
# Plotting logic
# --------------------------------------------------------------------------- #
def compute_all_diffs(groups_with, groups_without, metric_name, labels, smooth_window):
    """Compute the (with - without) diff curve for each label for a metric.
    Returns dict: label -> (index_array, diff_values_array)."""
    plot_data = {}
    for label in labels:
        if label not in groups_with or label not in groups_without:
            continue
        kind = DATASET_KIND[label]

        step_w, vals_w = get_metric_curve_for_group(groups_with[label], kind, metric_name)
        step_wo, vals_wo = get_metric_curve_for_group(groups_without[label], kind, metric_name)

        if step_w is None or step_wo is None or len(step_w) == 0 or len(step_wo) == 0:
            continue

        s_w = pd.Series(vals_w, index=step_w)
        s_wo = pd.Series(vals_wo, index=step_wo)

        df_align = pd.concat([s_w, s_wo], axis=1, join="inner")
        if df_align.empty:
            continue

        diff = df_align.iloc[:, 0] - df_align.iloc[:, 1]
        vals_diff = smooth(diff.values, smooth_window)

        plot_data[label] = (df_align.index.values, vals_diff)

    return plot_data


def running_cumulative_curve(steps, vals):
    """Running (step-integrated) trapezoidal cumulative sum of vals over
    steps, starting at 0. Returns an array the same length as vals."""
    steps = np.asarray(steps, dtype=float)
    vals = np.asarray(vals, dtype=float)
    if len(vals) < 2:
        return np.zeros_like(vals)
    valid = ~np.isnan(vals)
    increments = np.zeros(len(vals))
    for i in range(1, len(vals)):
        if valid[i] and valid[i - 1]:
            increments[i] = 0.5 * (vals[i] + vals[i - 1]) * (steps[i] - steps[i - 1])
    cum = np.cumsum(increments)
    return cum


def plot_difference(groups_with, groups_without, metric_name, labels, color_map,
                     output_path, smooth_window, yscale=None):
    """Plots (with_bumps - without_bumps) for each dataset, extended so all
    curves share a common x-axis start/end."""
    cfg = METRIC_CONFIG[metric_name]
    fig, ax = plt.subplots(figsize=(8, 6))

    plot_data = compute_all_diffs(groups_with, groups_without, metric_name, labels, smooth_window)
    plot_data = extend_all_to_common_range(plot_data)
    all_diffs = [vals for (_, vals) in plot_data.values()]

    for label, (idx, vals_diff) in plot_data.items():
        ax.plot(idx, vals_diff, color=color_map[label], linewidth=1.6, alpha=0.9)

    if all_diffs:
        x_min, x_max = common_x_range(plot_data)
        ax.set_xlim(x_min, x_max)

        if yscale == "symlog":
            abs_vals = np.concatenate([np.abs(d[~np.isnan(d)]) for d in all_diffs])
            nonzero = abs_vals[abs_vals > 0]
            linthresh = max(float(np.percentile(nonzero, 1)), 1e-8) if nonzero.size else 1e-6
            ax.set_yscale("symlog", linthresh=linthresh)
        else:
            y_min = min(np.nanmin(d) for d in all_diffs)
            y_max = max(np.nanmax(d) for d in all_diffs)
            margin = max((y_max - y_min) * 0.05, 1e-4)
            ax.set_ylim(y_min - margin, y_max + margin)
    else:
        ax.text(0.5, 0.5, "no overlapping data found", ha="center", va="center", transform=ax.transAxes)

    ax.set_xlabel("Global step")
    ax.set_ylabel(cfg["ylabel"])
    ax.set_title(cfg["title"], fontsize=12)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_path_curvature_difference(groups_with, groups_without, labels, color_map,
                                    output_path, smooth_window):
    """Path curvature ratio diff plot with a second, twin y-axis showing the
    per-dataset running cumulative curvature diff (one dashed line per
    dataset, same color as its solid path-curvature line)."""
    cfg = METRIC_CONFIG["path_curvature_ratio"]
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax2 = ax.twinx()

    plot_data = compute_all_diffs(groups_with, groups_without, "path_curvature_ratio", labels, smooth_window)

    # Cumulative curves computed from the real (non-extended) data first,
    # then extended flat too so both axes share the same x range.
    cum_data = {
        label: (idx, running_cumulative_curve(idx, vals))
        for label, (idx, vals) in plot_data.items()
    }

    plot_data_ext = extend_all_to_common_range(plot_data)
    cum_data_ext = extend_all_to_common_range(cum_data)

    for label, (idx, vals) in plot_data_ext.items():
        ax.plot(idx, vals, color=color_map[label], linewidth=1.6, alpha=0.9, linestyle="-")

    for label, (idx, vals) in cum_data_ext.items():
        ax2.plot(idx, vals, color=color_map[label], linewidth=1.4, alpha=0.85, linestyle="--")

    all_primary = [vals for (_, vals) in plot_data_ext.values()]
    all_cum = [vals for (_, vals) in cum_data_ext.values()]

    if all_primary:
        x_min, x_max = common_x_range(plot_data_ext)
        ax.set_xlim(x_min, x_max)

        y_min = min(np.nanmin(d) for d in all_primary)
        y_max = max(np.nanmax(d) for d in all_primary)
        margin = max((y_max - y_min) * 0.05, 1e-4)
        ax.set_ylim(y_min - margin, y_max + margin)
    else:
        ax.text(0.5, 0.5, "no overlapping data found", ha="center", va="center", transform=ax.transAxes)

    if all_cum:
        yc_min = min(np.nanmin(d) for d in all_cum)
        yc_max = max(np.nanmax(d) for d in all_cum)
        margin_c = max((yc_max - yc_min) * 0.05, 1e-4)
        ax2.set_ylim(yc_min - margin_c, yc_max + margin_c)

    ax.set_xlabel("Global step")
    ax.set_ylabel(cfg["ylabel"] + " (solid)")
    ax2.set_ylabel("Cumulative curvature diff, running integral (dashed)")
    ax.set_title(cfg["title"], fontsize=12)
    ax.grid(alpha=0.3)

    style_handles = [
        Line2D([0], [0], color="black", lw=1.8, linestyle="-", label="Path curvature ratio diff"),
        Line2D([0], [0], color="black", lw=1.8, linestyle="--", label="Cumulative curvature diff"),
    ]
    ax.legend(handles=style_handles, fontsize=9, loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_legend_only(color_map, output_path, labels=None, title="Dataset"):
    """Outputs ONLY the legend in a transparent/borderless figure."""
    if labels is None:
        labels = list(color_map.keys())
    fig = plt.figure(figsize=(4, 3))
    handles = [
        Line2D([0], [0], color=color_map[label], lw=3.0, label=str(label))
        for label in labels
    ]
    fig.legend(handles=handles, loc="center", fontsize=10, title=title, framealpha=0.0)
    plt.axis("off")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")

# --------------------------------------------------------------------------- #
# Final-value ("last episode") comparison figure
# --------------------------------------------------------------------------- #
def last_valid_value(steps, vals):
    """Value of the last non-NaN entry in vals (assumed sorted by steps)."""
    vals = np.asarray(vals, dtype=float)
    valid = ~np.isnan(vals)
    if not np.any(valid):
        return None
    return float(vals[valid][-1])


def get_raw_final_value(group_runs, kind, metric_name):
    """Mean raw curve for this dataset/condition, then its last valid value."""
    steps, vals = get_metric_curve_for_group(group_runs, kind, metric_name)
    if steps is None or len(steps) == 0:
        return None
    return last_valid_value(steps, vals)


def get_raw_final_cumulative(group_runs, kind, metric_name):
    """Running cumulative integral of the mean raw curve, final (last) value."""
    steps, vals = get_metric_curve_for_group(group_runs, kind, metric_name)
    if steps is None or len(steps) == 0:
        return None
    cum = running_cumulative_curve(steps, vals)
    return last_valid_value(steps, cum)


def choose_axis_scale(values):
    """Pick 'log', 'symlog', or 'linear' for a set of values so wildly
    different magnitudes across datasets stay readable."""
    values = np.asarray([v for v in values if v is not None], dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return "linear"
    if np.any(values <= 0):
        nonzero = np.abs(values[values != 0])
        linthresh = max(float(np.percentile(nonzero, 5)), 1e-8) if nonzero.size else 1e-6
        span = (np.nanmax(np.abs(values)) / linthresh) if linthresh > 0 else 1
        return "symlog" if span > 15 else "linear"
    ratio = np.nanmax(values) / max(np.nanmin(values), 1e-12)
    return "log" if ratio > 15 else "linear"


def draw_point_pair(ax, x, y_without, y_with, color, linestyle="-", dot_size=55, diamond_size=70):
    """Dot at y_without, diamond at y_with, arrow pointing without -> with."""
    if y_without is None or y_with is None:
        return
    ax.scatter([x], [y_without], marker="o", s=dot_size, color=color, zorder=3, edgecolor="black", linewidth=0.5)
    ax.scatter([x], [y_with], marker="D", s=diamond_size, color=color, zorder=3, edgecolor="black", linewidth=0.5)
    ax.annotate(
        "", xy=(x, y_with), xytext=(x, y_without),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6, linestyle=linestyle, shrinkA=6, shrinkB=6),
        zorder=2,
    )


def plot_final_value_comparison(groups_with, groups_without, labels, color_map, output_path):
    """One figure, 4 subplots (train acc, test acc, bhattacharyya,
    path-curvature+cumulative), showing only the LAST-episode value per
    dataset: a dot for 'without oscillations', a diamond for 'with
    oscillations', and an arrow from dot to diamond. X-axis = dataset names.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    x_positions = np.arange(len(labels))

    def setup_categorical_x(ax):
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_xlim(-0.6, len(labels) - 0.4)
        ax.grid(alpha=0.3, axis="y")

    # --- simple metrics: train_accuracy, test_accuracy, bhattacharyya ---
    simple_metrics = [
        (axes[0, 0], "train_accuracy", "Train accuracy (last episode)"),
        (axes[0, 1], "test_accuracy", "Test accuracy (last episode)"),
        (axes[1, 0], "bhattacharyya", "Bhattacharyya distance (last episode)"),
    ]

    for ax, metric_name, title in simple_metrics:
        without_vals, with_vals = [], []
        for label in labels:
            kind = DATASET_KIND[label]
            v_without = get_raw_final_value(groups_without.get(label, []), kind, metric_name) if label in groups_without else None
            v_with = get_raw_final_value(groups_with.get(label, []), kind, metric_name) if label in groups_with else None
            without_vals.append(v_without)
            with_vals.append(v_with)

        for x, label, v_wo, v_w in zip(x_positions, labels, without_vals, with_vals):
            draw_point_pair(ax, x, v_wo, v_w, color_map[label])

        scale = choose_axis_scale(without_vals + with_vals)
        if scale == "log":
            ax.set_yscale("log")
        elif scale == "symlog":
            all_vals = np.asarray([v for v in without_vals + with_vals if v is not None], dtype=float)
            nonzero = np.abs(all_vals[all_vals != 0])
            linthresh = max(float(np.percentile(nonzero, 5)), 1e-8) if nonzero.size else 1e-6
            ax.set_yscale("symlog", linthresh=linthresh)

        setup_categorical_x(ax)
        ax.set_ylabel(METRIC_CONFIG[metric_name]["ylabel"].split(" diff")[0])
        ax.set_title(title, fontsize=11)

    # --- path curvature ratio + cumulative curvature (twin axis) ---
    ax4 = axes[1, 1]
    ax4b = ax4.twinx()

    curv_without_vals, curv_with_vals = [], []
    cum_without_vals, cum_with_vals = [], []

    for label in labels:
        kind = DATASET_KIND[label]
        gw = groups_with.get(label, [])
        gwo = groups_without.get(label, [])

        v_wo = get_raw_final_value(gwo, kind, "path_curvature_ratio") if label in groups_without else None
        v_w = get_raw_final_value(gw, kind, "path_curvature_ratio") if label in groups_with else None
        c_wo = get_raw_final_cumulative(gwo, kind, "path_curvature_ratio") if label in groups_without else None
        c_w = get_raw_final_cumulative(gw, kind, "path_curvature_ratio") if label in groups_with else None

        curv_without_vals.append(v_wo)
        curv_with_vals.append(v_w)
        cum_without_vals.append(c_wo)
        cum_with_vals.append(c_w)

    x_offset = 0.15
    for i, label in enumerate(labels):
        draw_point_pair(ax4, x_positions[i] - x_offset, curv_without_vals[i], curv_with_vals[i],
                         color_map[label], linestyle="-")
        draw_point_pair(ax4b, x_positions[i] + x_offset, cum_without_vals[i], cum_with_vals[i],
                         color_map[label], linestyle="--")

    scale4 = choose_axis_scale(curv_without_vals + curv_with_vals)
    if scale4 == "log":
        ax4.set_yscale("log")
    elif scale4 == "symlog":
        all_vals = np.asarray([v for v in curv_without_vals + curv_with_vals if v is not None], dtype=float)
        nonzero = np.abs(all_vals[all_vals != 0])
        linthresh = max(float(np.percentile(nonzero, 5)), 1e-8) if nonzero.size else 1e-6
        ax4.set_yscale("symlog", linthresh=linthresh)

    scale4b = choose_axis_scale(cum_without_vals + cum_with_vals)
    if scale4b == "log":
        ax4b.set_yscale("log")
    elif scale4b == "symlog":
        all_vals = np.asarray([v for v in cum_without_vals + cum_with_vals if v is not None], dtype=float)
        nonzero = np.abs(all_vals[all_vals != 0])
        linthresh = max(float(np.percentile(nonzero, 5)), 1e-8) if nonzero.size else 1e-6
        ax4b.set_yscale("symlog", linthresh=linthresh)

    setup_categorical_x(ax4)
    ax4.set_ylabel("Path curvature ratio (solid)")
    ax4b.set_ylabel("Cumulative curvature, running integral (dashed)")
    ax4.set_title("Path curvature ratio & cumulative (last episode)", fontsize=11)

    style_handles = [
        Line2D([0], [0], marker="o", color="gray", linestyle="None", markersize=7, label="Without oscillations"),
        Line2D([0], [0], marker="D", color="gray", linestyle="None", markersize=7, label="With oscillations"),
        Line2D([0], [0], color="gray", lw=1.6, linestyle="-", label="Path curvature ratio"),
        Line2D([0], [0], color="gray", lw=1.6, linestyle="--", label="Cumulative curvature"),
    ]
    ax4.legend(handles=style_handles, fontsize=8, loc="best")

    fig.suptitle("Last-episode values: With vs Without oscillations", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")


def shade_zero_background(ax, top_color="green", bottom_color="red", alpha=0.06):
    """Very light green above y=0, very light red below y=0, plus a thick
    black dashed line at y=0. Call AFTER points are plotted so ylim reflects
    the data; ylim is restored afterwards so axhspan doesn't distort it."""
    ylim = ax.get_ylim()
    if ylim[1] > 0:
        ax.axhspan(0, ylim[1], facecolor=top_color, alpha=alpha, zorder=0)
    if ylim[0] < 0:
        ax.axhspan(ylim[0], 0, facecolor=bottom_color, alpha=alpha, zorder=0)
    ax.axhline(0, color="black", linestyle="--", linewidth=2.5, zorder=1)
    ax.set_ylim(ylim)


def choose_diff_axis_scale(values):
    """Symlog if diffs (which can be negative) span a wide range, else linear."""
    values = np.asarray([v for v in values if v is not None], dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return "linear"
    nonzero = np.abs(values[values != 0])
    if nonzero.size == 0:
        return "linear"
    linthresh = max(float(np.percentile(nonzero, 5)), 1e-8)
    span = np.nanmax(np.abs(values)) / linthresh
    return "symlog" if span > 15 else "linear"


def plot_final_diff_comparison(groups_with, groups_without, labels, color_map, output_path):
    """Same 4-subplot layout as plot_final_value_comparison, but each dataset
    is a single star at (with - without), last episode only. y=0 is marked
    with a thick black dashed line; the region above is tinted very light
    green, below very light red."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    x_positions = np.arange(len(labels))

    def setup_categorical_x(ax):
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_xlim(-0.6, len(labels) - 0.4)
        ax.grid(alpha=0.3, axis="y")

    simple_metrics = [
        (axes[0, 0], "train_accuracy", "Train accuracy diff (last episode)"),
        (axes[0, 1], "test_accuracy", "Test accuracy diff (last episode)"),
        (axes[1, 0], "bhattacharyya", "Bhattacharyya distance diff (last episode)"),
    ]

    for ax, metric_name, title in simple_metrics:
        diffs = []
        for label in labels:
            kind = DATASET_KIND[label]
            v_without = get_raw_final_value(groups_without.get(label, []), kind, metric_name) if label in groups_without else None
            v_with = get_raw_final_value(groups_with.get(label, []), kind, metric_name) if label in groups_with else None
            diffs.append(v_with - v_without if (v_with is not None and v_without is not None) else None)

        for x, label, d in zip(x_positions, labels, diffs):
            if d is None:
                continue
            ax.scatter([x], [d], marker="*", s=220, color=color_map[label], zorder=3, edgecolor="black", linewidth=0.6)

        setup_categorical_x(ax)
        scale = choose_diff_axis_scale(diffs)
        if scale == "symlog":
            valid = np.asarray([d for d in diffs if d is not None], dtype=float)
            nonzero = np.abs(valid[valid != 0])
            linthresh = max(float(np.percentile(nonzero, 5)), 1e-8) if nonzero.size else 1e-6
            ax.set_yscale("symlog", linthresh=linthresh)

        shade_zero_background(ax)
        ax.set_ylabel(METRIC_CONFIG[metric_name]["ylabel"])
        ax.set_title(title, fontsize=11)

    # --- path curvature ratio diff + cumulative curvature diff (twin axis) ---
    ax4 = axes[1, 1]
    ax4b = ax4.twinx()

    curv_diffs, cum_diffs = [], []
    for label in labels:
        kind = DATASET_KIND[label]
        gw = groups_with.get(label, [])
        gwo = groups_without.get(label, [])

        v_wo = get_raw_final_value(gwo, kind, "path_curvature_ratio") if label in groups_without else None
        v_w = get_raw_final_value(gw, kind, "path_curvature_ratio") if label in groups_with else None
        c_wo = get_raw_final_cumulative(gwo, kind, "path_curvature_ratio") if label in groups_without else None
        c_w = get_raw_final_cumulative(gw, kind, "path_curvature_ratio") if label in groups_with else None

        curv_diffs.append(v_w - v_wo if (v_w is not None and v_wo is not None) else None)
        cum_diffs.append(c_w - c_wo if (c_w is not None and c_wo is not None) else None)

    x_offset = 0.15
    for i, label in enumerate(labels):
        if curv_diffs[i] is not None:
            ax4.scatter([x_positions[i] - x_offset], [curv_diffs[i]], marker="*", s=200,
                        color=color_map[label], zorder=3, edgecolor="black", linewidth=0.6)
        if cum_diffs[i] is not None:
            ax4b.scatter([x_positions[i] + x_offset], [cum_diffs[i]], marker="*", s=260,
                         facecolors="none", edgecolors=color_map[label], linewidths=1.8, zorder=3)

    setup_categorical_x(ax4)

    scale4 = choose_diff_axis_scale(curv_diffs)
    if scale4 == "symlog":
        valid = np.asarray([d for d in curv_diffs if d is not None], dtype=float)
        nonzero = np.abs(valid[valid != 0])
        linthresh = max(float(np.percentile(nonzero, 5)), 1e-8) if nonzero.size else 1e-6
        ax4.set_yscale("symlog", linthresh=linthresh)

    scale4b = choose_diff_axis_scale(cum_diffs)
    if scale4b == "symlog":
        valid = np.asarray([d for d in cum_diffs if d is not None], dtype=float)
        nonzero = np.abs(valid[valid != 0])
        linthresh = max(float(np.percentile(nonzero, 5)), 1e-8) if nonzero.size else 1e-6
        ax4b.set_yscale("symlog", linthresh=linthresh)

    # Background/zero-line shading is drawn using the primary (path
    # curvature) axis; the twin axis shares the same x-range so the
    # green/red bands still line up with each dataset's stars.
    # Flipped vs. the other subplots: here a NEGATIVE curvature diff is the
    # "good" direction, so green sits below 0 and red sits above 0.
    shade_zero_background(ax4, top_color="red", bottom_color="green")
    ax4b.axhline(0, color="black", linestyle="--", linewidth=2.5, zorder=1)

    ax4.set_ylabel("Path curvature ratio diff (filled star)")
    ax4b.set_ylabel("Cumulative curvature diff (open star)")
    ax4.set_title("Path curvature ratio & cumulative diff (last episode)", fontsize=11)

    style_handles = [
        Line2D([0], [0], marker="*", color="gray", linestyle="None", markersize=14,
               markeredgecolor="black", label="Path curvature ratio diff"),
        Line2D([0], [0], marker="*", color="none", markeredgecolor="gray", linestyle="None",
               markersize=16, markeredgewidth=1.8, label="Cumulative curvature diff"),
    ]
    ax4.legend(handles=style_handles, fontsize=8, loc="best")

    fig.suptitle("Last-episode difference (With − Without oscillations)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    print("Discovering 'without bumps' runs...")
    runs_without = discover_all(DATASETS_WITHOUT, DATASET_KIND)
    groups_without = get_grouped(runs_without)

    print("Discovering 'with bumps' runs...")
    runs_with = discover_all(DATASETS_WITH, DATASET_KIND)
    groups_with = get_grouped(runs_with)

    all_labels = [lbl for lbl in groups_with if lbl in groups_without]
    if not all_labels:
        print("No paired datasets found across with/without bumps, aborting.")
        return

    toy_labels = [lbl for lbl in all_labels if DATASET_KIND[lbl] == "toy"]
    real_labels = [lbl for lbl in all_labels if DATASET_KIND[lbl] == "real"]

    print(f"\nAll matching datasets:  {all_labels}")
    print(f"Toy datasets found:     {toy_labels}")
    print(f"Real datasets found:    {real_labels}\n")

    cmap = plt.get_cmap("tab20")
    color_map = {label: cmap(i % 20) for i, label in enumerate(all_labels)}

    output_root = os.path.join(os.getcwd(), f"{ROOT_DIR}/plots_difference")
    subdirs = {
        "all": os.path.join(output_root, "all"),
        "toy": os.path.join(output_root, "toy"),
        "real": os.path.join(output_root, "real"),
    }
    for d in subdirs.values():
        os.makedirs(d, exist_ok=True)

    label_sets = {
        "all": all_labels,
        "toy": toy_labels,
        "real": real_labels,
    }

    # yscale=None -> linear (auto ylim); "symlog" -> log-like magnitude with
    # a small linear region around 0 so negative diffs still show.
    metrics_to_plot = [
        ("train_accuracy", None),
        ("test_accuracy", None),
        ("bhattacharyya", "symlog"),
    ]

    for scope, labels in label_sets.items():
        if not labels:
            print(f"[WARN] no datasets available for scope '{scope}', skipping its plots.")
            continue
        for metric_name, yscale in metrics_to_plot:
            cfg = METRIC_CONFIG[metric_name]
            out_path = os.path.join(subdirs[scope], cfg["filename"])
            plot_difference(
                groups_with, groups_without, metric_name, labels, color_map,
                out_path, SMOOTH_WINDOW, yscale=yscale,
            )

        # Path curvature: dedicated twin-axis plot (solid = ratio diff,
        # dashed = per-dataset running cumulative curvature diff).
        cfg = METRIC_CONFIG["path_curvature_ratio"]
        out_path = os.path.join(subdirs[scope], cfg["filename"])
        plot_path_curvature_difference(
            groups_with, groups_without, labels, color_map, out_path, SMOOTH_WINDOW
        )

        # Last-episode dot/diamond/arrow comparison, 4 subplots in one figure.
        final_out_path = os.path.join(subdirs[scope], "final_value_comparison.png")
        plot_final_value_comparison(groups_with, groups_without, labels, color_map, final_out_path)

        # Same layout, but as a single star per dataset showing the
        # with-minus-without difference, with a 0-line and green/red shading.
        final_diff_out_path = os.path.join(subdirs[scope], "final_diff_comparison.png")
        plot_final_diff_comparison(groups_with, groups_without, labels, color_map, final_diff_out_path)

    # Legends: one for everything, plus toy-only / real-only for convenience.
    plot_legend_only(color_map, os.path.join(output_root, "legend_all.png"), labels=all_labels, title="Dataset")
    if toy_labels:
        plot_legend_only(color_map, os.path.join(output_root, "legend_toy.png"), labels=toy_labels, title="Toy datasets")
    if real_labels:
        plot_legend_only(color_map, os.path.join(output_root, "legend_real.png"), labels=real_labels, title="Real datasets")

    print("\nDone.")


if __name__ == "__main__":
    main()