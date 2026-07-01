#!/usr/bin/env python3
"""
DATASETS_analysis.py

Same four multi-panel figures as plot_num_modes.py, but instead of overlaying
one line per `num_modes` value (within a single dataset), this version
overlays one line per *dataset* (spiral, dartboard, rings, checkerboard, ...),
each pointing at its own hardcoded folder.

Configure your datasets in the DATASETS dict below:

    DATASETS = {
        "Spiral":       "/data/samuel_lozano/dynamical-sgd/without_bumps/spiral-no_bumps/",
        "Dartboard":    "/data/samuel_lozano/dynamical-sgd/without_bumps/dartboard-no_bumps/",
        "Rings":        "/data/samuel_lozano/dynamical-sgd/without_bumps/rings-no_bumps/",
        "Checkerboard": "/data/samuel_lozano/dynamical-sgd/without_bumps/checkerboard-no_bumps/",
    }

Each path can point either to:
  (a) a parent directory containing one or more `training_<timestamp>/` runs
      (if several runs are found, they are averaged together, aligned on
      `global_step`, into a single curve per dataset -- handy if you have several
      seeds per dataset), or
  (b) directly to a single `training_<timestamp>/` run folder.

Each run folder is expected to contain (searched recursively, so subfolders
are fine too):
    config.yaml (or .yml)        -- used only to sanity-check `data.dataset_name`
                                     against the label you gave it in DATASETS;
                                     a mismatch only prints a warning, it does
                                     not stop the script.
    training_metrics.csv
    classifier_metrics.csv
    hyperplanes.csv
    nc_metrics.csv
    pca_analysis.csv
    PCA_geometric.csv
    proj_nc_metrics.csv          -- produced by projection_PCA_analysis.py (optional;
                                     a warning is printed if missing and the
                                     projected_nc.png panels show "no data found")

Output
------
Five PNGs are written to OUTPUT_DIR (one line per dataset in every panel):
    accuracies.png                       (1x2)
    neural_collapse.png                  (2x2)
    high_dimensional_classification.png  (2x2)
    cylinder_hyperplanes.png             (3x2)
    projected_nc.png                     (2x2)  <-- new

Usage
-----

nohup python DATASETS_analysis.py --training_type without_bumps > log.out 2>&1 &

Column-mapping assumptions are identical to plot_num_modes.py -- see that
script's docstring / the comments below if you need to tweak a mapping
(e.g. "Principal axis alignment" currently uses `cyl_axis_cosine_<i>v<j>`
from PCA_geometric.csv).
"""

import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.lines import Line2D

# --------------------------------------------------------------------------- #
# Configuration -- EDIT THIS DICT with your dataset name -> folder mapping
# --------------------------------------------------------------------------- #
ROOT_DIR="/data/samuel_lozano/dynamical-sgd/"
TRAINING_TYPE="without_bumps"  # or "with_bumps" if you want to analyze the bumps runs instead

SMOOTH_WINDOW = 1  # rolling-mean window (in global_steps); 1 disables smoothing
CONFIG_FILENAMES = ["config.yaml", "config.yml"]

CSV_FILES = {
    "training": "training_metrics.csv",
    "classifier": "classifier_metrics.csv",
    "hyperplanes": "hyperplanes.csv",
    "nc": "nc_metrics.csv",
    "pca": "pca_analysis.csv",
    "pca_geom": "PCA_geometric.csv",
    "proj_nc": "proj_nc_metrics.csv",
}

LINESTYLES = ["-", "--", ":", "-."]

# --------------------------------------------------------------------------- #
# Discovery: walk each dataset's folder, sanity-check config.yaml, read csvs
# --------------------------------------------------------------------------- #
def find_file(run_dir, filename):
    """Find `filename` anywhere under run_dir (handles nested subfolders)."""
    matches = glob.glob(os.path.join(run_dir, "**", filename), recursive=True)
    if not matches:
        return None
    if len(matches) > 1:
        print(f"[WARN] multiple '{filename}' found under {run_dir}, using {matches[0]}")
    return matches[0]
 
 
def find_key(cfg, key):
    """Recursively search a (possibly nested) yaml-loaded dict/list for `key`."""
 
    def search(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if str(k).strip().lower() == key.lower():
                    return v
                found = search(v)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for item in node:
                found = search(item)
                if found is not None:
                    return found
        return None
 
    return search(cfg)
 
 
def discover_runs_for_label(label, path):
    """
    Discover one or more run folders for a given dataset label.
    `path` may be a parent dir containing training_* folders, or a single run dir.
    """
    runs = []
    if not os.path.isdir(path):
        print(f"[WARN] path for dataset '{label}' does not exist: {path}")
        return runs
 
    # Check if the path itself is a training run directory
    clean_path = path.rstrip(os.sep)
    if os.path.basename(clean_path).startswith("training_"):
        train_dirs = [path]
    else:
        train_dirs = sorted(glob.glob(os.path.join(path, "training_*")))
        if not train_dirs:
            # `path` itself is presumably a single run directory
            train_dirs = [path]
 
    for d in train_dirs:
        if not os.path.isdir(d):
            continue
 
        config_path = None
        for cfg_name in CONFIG_FILENAMES:
            config_path = find_file(d, cfg_name)
            if config_path:
                break
 
        if config_path:
            try:
                with open(config_path, "r") as f:
                    cfg = yaml.safe_load(f)
                cfg_dataset_name = find_key(cfg, "dataset_name")
                if cfg_dataset_name is not None and str(cfg_dataset_name).strip().lower() != label.strip().lower():
                    print(
                        f"[WARN] {d}: config data.dataset_name='{cfg_dataset_name}' "
                        f"doesn't match expected label '{label}' -- double check the DATASETS dict."
                    )
            except Exception as e:
                print(f"[WARN] Could not read {config_path}: {e}")
        else:
            print(f"[WARN] No config.yaml/.yml found in {d} (continuing without the sanity check).")
 
        dfs = {}
        missing = []
        for key, fname in CSV_FILES.items():
            fpath = find_file(d, fname)
            if fpath is None:
                missing.append(fname)
                continue
            try:
                dfs[key] = pd.read_csv(fpath)
            except Exception as e:
                print(f"[WARN] Failed to read {fpath}: {e}")
        if missing:
            print(f"[WARN] {label} / {os.path.basename(d)}: missing csv(s) {missing}")
        if not dfs:
            print(f"[WARN] {label} / {os.path.basename(d)}: no CSVs read successfully, skipping run.")
            continue
 
        runs.append({"dir": d, "label": label, "dfs": dfs})
        print(f"[OK]   {label} / {os.path.basename(d)}: csvs={sorted(dfs.keys())}")
 
    return runs
 
 
def discover_all(datasets):
    runs = []
    for label, path in datasets.items():
        runs.extend(discover_runs_for_label(label, path))
    return runs
 
 
def get_grouped(runs):
    """Group runs by dataset label -> {label: [run, run, ...]}, preserving DATASETS order."""
    groups = {}
    for r in runs:
        groups.setdefault(r["label"], []).append(r)
    return groups
 
 
# --------------------------------------------------------------------------- #
# Column auto-detection helpers (no hardcoded class count)
# --------------------------------------------------------------------------- #
def detect_classes(df, prefix):
    """Return sorted class indices for columns named '<prefix>_<i>'."""
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    idxs = sorted(int(m.group(1)) for c in df.columns for m in [pattern.match(c)] if m)
    return idxs
 
 
def detect_pairs(df, prefix):
    """Return sorted (i, j, column_name) for columns named '<prefix>_<i>v<j>'."""
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)v(\d+)$")
    pairs = []
    for c in df.columns:
        m = pattern.match(c)
        if m:
            pairs.append((int(m.group(1)), int(m.group(2)), c))
    pairs.sort(key=lambda t: (t[0], t[1]))
    return pairs
 
 
def safe_div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        out = a / b
    return out.replace([np.inf, -np.inf], np.nan)
 
 
# --------------------------------------------------------------------------- #
# Metric definitions (each takes a single dataframe, returns a pd.Series)
# --------------------------------------------------------------------------- #
def mean_over_pairs(df, prefix):
    pairs = detect_pairs(df, prefix)
    if not pairs:
        return pd.Series(np.nan, index=df.index)
    cols = [c for (_, _, c) in pairs]
    return df[cols].mean(axis=1)
 
 
def cyl_half_over_radius(df):
    """Mean over classes of cyl_half_length_class_i / cyl_radius_class_i."""
    classes = detect_classes(df, "cyl_half_length_class")
    ratios = []
    for c in classes:
        h_col, r_col = f"cyl_half_length_class_{c}", f"cyl_radius_class_{c}"
        if h_col in df.columns and r_col in df.columns:
            ratios.append(safe_div(df[h_col], df[r_col]))
    if not ratios:
        return pd.Series(np.nan, index=df.index)
    return pd.concat(ratios, axis=1).mean(axis=1)
 
 
def logit_decomposition(df):
    """Mean over classes of logit_correct_mean_class_i / logit_max_wrong_mean_class_i."""
    classes = detect_classes(df, "logit_correct_mean_class")
    ratios = []
    for c in classes:
        correct_col = f"logit_correct_mean_class_{c}"
        wrong_col = f"logit_max_wrong_mean_class_{c}"
        if correct_col in df.columns and wrong_col in df.columns:
            ratios.append(safe_div(df[correct_col], df[wrong_col]))
    if not ratios:
        return pd.Series(np.nan, index=df.index)
    return pd.concat(ratios, axis=1).mean(axis=1)
 
 
def pca_alignment_cosine(df):
    """Mean of the boundary-normal vs per-class-PC1 cosines: pca_cosine_<i>v<j>_p1_<k>."""
    pattern = re.compile(r"^pca_cosine_(\d+)v(\d+)_p1_(\d+)$")
    cols = sorted(c for c in df.columns if pattern.match(c))
    if not cols:
        return pd.Series(np.nan, index=df.index)
    return df[cols].mean(axis=1)
 
 
def detect_hp_angle_cols(df):
    """All hp_angle_(...)v(...) columns -- angles between pairwise boundary normals."""
    pattern = re.compile(r"^hp_angle_\(.+\)v\(.+\)$")
    return sorted(c for c in df.columns if pattern.match(c))
 
 
def cyl_axis_cosine_cols(df):
    """All cyl_axis_cosine_<i>v<j> columns -- cosine between class PCA principal axes."""
    return [c for (_, _, c) in detect_pairs(df, "cyl_axis_cosine")]
 
 
def std_hp_angle(df):
    """Std (population, ddof=0) of the 3 hp_angle_(...)v(...) columns.
    Their mean is a fixed 60 degrees, so this equals the RMS deviation
    from 60 degrees -- i.e. how unevenly the 3 pairwise angles are spread."""
    cols = detect_hp_angle_cols(df)
    if not cols:
        return pd.Series(np.nan, index=df.index)
    return df[cols].std(axis=1, ddof=0)
 
 
def mean_cyl_axis_cosine(df):
    """Mean of the 3 cyl_axis_cosine_<i>v<j> columns -- average principal-axis alignment."""
    cols = cyl_axis_cosine_cols(df)
    if not cols:
        return pd.Series(np.nan, index=df.index)
    return df[cols].mean(axis=1)
 
 
# --------------------------------------------------------------------------- #
# proj_nc_metrics.csv helpers
# Column naming from projection_PCA_analysis.py:
#   elongation_<c>           -- per-class elongation ratio (λ₁ / mean(λ₂…λ_D))
#   axis_alignment_<i>_<j>  -- |cos θ| between cylinder axes of class i and j
#   axis_angle_deg_<i>_<j>  -- angle in degrees between cylinder axes
#   nc1_original / nc1_deflated
#   nc2_original / nc2_deflated
# --------------------------------------------------------------------------- #
def detect_classes_proj(df, prefix):
    """Return sorted class indices for columns named '<prefix>_<i>' (single index)."""
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    idxs = sorted(int(m.group(1)) for c in df.columns for m in [pattern.match(c)] if m)
    return idxs


def detect_pairs_proj(df, prefix):
    """Return sorted (i, j, col_name) for columns named '<prefix>_<i>_<j>' (underscore sep)."""
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)_(\d+)$")
    pairs = []
    for c in df.columns:
        m = pattern.match(c)
        if m:
            pairs.append((int(m.group(1)), int(m.group(2)), c))
    pairs.sort(key=lambda t: (t[0], t[1]))
    return pairs


def mean_elongation(df):
    """Mean cylinder elongation averaged over all classes."""
    classes = detect_classes_proj(df, "elongation")
    cols = [f"elongation_{c}" for c in classes if f"elongation_{c}" in df.columns]
    if not cols:
        return pd.Series(np.nan, index=df.index)
    return df[cols].mean(axis=1)


def mean_axis_alignment(df):
    """Mean |cos θ| between cylinder axes averaged over all class pairs."""
    pairs = detect_pairs_proj(df, "axis_alignment")
    cols = [c for (_, _, c) in pairs]
    if not cols:
        return pd.Series(np.nan, index=df.index)
    return df[cols].mean(axis=1)


def nc1_ratio(df):
    """NC1 deflated / NC1 original -- < 1 confirms the cylinder hypothesis."""
    if "nc1_deflated" not in df.columns or "nc1_original" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return safe_div(df["nc1_deflated"], df["nc1_original"])


def nc2_ratio(df):
    """NC2 deviation deflated / NC2 deviation original."""
    if "nc2_deflated" not in df.columns or "nc2_original" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return safe_div(df["nc2_deflated"], df["nc2_original"])


def axis_alignment_cols(df):
    """All axis_alignment_<i>_<j> columns (for plot_multi_metric)."""
    return [c for (_, _, c) in detect_pairs_proj(df, "axis_alignment")]


# --------------------------------------------------------------------------- #
# Aggregation across runs sharing the same dataset label, aligned on global_step
# --------------------------------------------------------------------------- #
def get_curve(group_runs, csv_key, value_fn):
    """
    value_fn(df) -> pd.Series of the metric, indexed like df.
    Returns (global_step_array, mean_value_array) averaged across all runs in
    group_runs that have csv_key available (handles multiple seeds sharing
    the same dataset label).
    """
    series_list = []
    for r in group_runs:
        df = r["dfs"].get(csv_key)
        if df is None or "global_step" not in df.columns:
            continue
        try:
            vals = value_fn(df)
        except Exception as e:
            print(f"[WARN] metric failed for {r['dir']} / {csv_key}: {e}")
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
 
 
def smooth(values, window):
    if window is None or window <= 1:
        return values
    s = pd.Series(values)
    return s.rolling(window=window, min_periods=1, center=True).mean().values
 
 
# --------------------------------------------------------------------------- #
# Plotting primitives
# --------------------------------------------------------------------------- #
def plot_single_metric(ax, groups, csv_key, value_fn, ylabel, title, color_map, smooth_window):
    """One line per dataset, color encodes dataset."""
    any_data = False
    for label, group_runs in groups.items():
        global_step, vals = get_curve(group_runs, csv_key, value_fn)
        if global_step is None or len(global_step) == 0:
            continue
        vals = smooth(vals, smooth_window)
        ax.plot(global_step, vals, color=color_map[label], linewidth=1.6, alpha=0.9, label=str(label))
        any_data = True
    if not any_data:
        ax.text(0.5, 0.5, "no data found", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Global step")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
 
 
def plot_multi_metric(ax, groups, csv_key, columns_fn, ylabel, title, color_map, smooth_window, transform=None):
    """
    Several lines per dataset (e.g. one per class-pair). Color encodes
    dataset, linestyle encodes which column. Returns the canonical column
    list used (for building a separate linestyle legend), or [] if no data.
    """
    canonical_cols = None
    for label, group_runs in groups.items():
        for r in group_runs:
            df = r["dfs"].get(csv_key)
            if df is not None:
                cols = columns_fn(df)
                if cols:
                    canonical_cols = cols
                    break
        if canonical_cols:
            break
 
    if not canonical_cols:
        ax.text(0.5, 0.5, "no data found", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("Global step")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        return []
 
    for label, group_runs in groups.items():
        color = color_map[label]
        for i, col in enumerate(canonical_cols):
            ls = LINESTYLES[i % len(LINESTYLES)]

            def val_fn(df, c=col):
                if c not in df.columns: return None
                val = df[c]
                if transform: return transform(val)
                return val
            
            global_step, vals = get_curve(group_runs, csv_key, val_fn)
            if global_step is None or len(global_step) == 0:
                continue
            vals = smooth(vals, smooth_window)
            ax.plot(global_step, vals, color=color, linestyle=ls, linewidth=1.3, alpha=0.85)
 
    ax.set_xlabel("Global step")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    return canonical_cols
 
 
def add_linestyle_legend(ax, cols, strip_prefix=None, loc="best"):
    """Neutral-color legend explaining what each linestyle means (column name)."""
    def short(col):
        if strip_prefix and col.startswith(strip_prefix):
            return col[len(strip_prefix):]
        return col
 
    handles = [
        Line2D([0], [0], color="black", linestyle=LINESTYLES[i % len(LINESTYLES)], label=short(col))
        for i, col in enumerate(cols)
    ]
    ax.legend(handles=handles, fontsize=7, loc=loc, framealpha=0.7, handlelength=2.2)
 
 
def add_dataset_legend(fig, color_map):
    """Shared figure-level legend mapping color -> dataset name."""
    handles = [
        Line2D([0], [0], color=color_map[label], lw=2.5, label=str(label))
        for label in color_map
    ]
    fig.legend(
        handles=handles, loc="center right", bbox_to_anchor=(0.995, 0.5),
        fontsize=9, title="Dataset", framealpha=0.85,
    )
 
 
# --------------------------------------------------------------------------- #
# The four figures
# --------------------------------------------------------------------------- #
def plot_accuracies(groups, output_dir, color_map, smooth_window):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
 
    plot_single_metric(axes[0], groups, "training", lambda df: df["train_accuracy"],
                        "Train accuracy", "Train accuracy", color_map, smooth_window)
    plot_single_metric(axes[1], groups, "training", lambda df: df["test_accuracy"],
                        "Test accuracy", "Test accuracy", color_map, smooth_window)
    axes[0].set_ylim(-0.05, 1.05) # Fixed limits [0, 1] for train accuracy
    axes[1].set_ylim(-0.05, 1.05) # Fixed limits [0, 1] for test accuracy
 
    fig.suptitle("Train / test accuracy across datasets", fontsize=14)
    add_dataset_legend(fig, color_map)
    fig.tight_layout(rect=[0, 0, 0.93, 0.95])
 
    path = os.path.join(output_dir, "accuracies.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")
 
 
def plot_neural_collapse(groups, output_dir, color_map, smooth_window):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.flatten()
 
    plot_single_metric(axes[0], groups, "nc", lambda df: df["nc1"],
                        "NC1", "NC1 (within / between-class variance collapse)", color_map, smooth_window)
    axes[0].set_yscale('log') # Log Y axis for NC1
    
    plot_single_metric(axes[1], groups, "nc", lambda df: df["nc2_deviation"],
                        "NC2 deviation", "NC2 (equinormality / equiangularity deviation)", color_map, smooth_window)
    plot_single_metric(axes[2], groups, "pca", lambda df: df["pca_k95"],
                        "# PCA components", "PCA dimensions needed for 95% variance", color_map, smooth_window)
    plot_single_metric(axes[3], groups, "pca_geom", cyl_half_over_radius,
                        "Mean half-length / radius", "Mean cylinder half-length / radius (over classes)",
                        color_map, smooth_window)
    axes[3].set_yscale('log') # Log Y axis for cylinder half-length/radius
 
    fig.suptitle("Neural collapse diagnostics across datasets", fontsize=14)
    add_dataset_legend(fig, color_map)
    fig.tight_layout(rect=[0, 0, 0.93, 0.95])
 
    path = os.path.join(output_dir, "neural_collapse.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")
 
 
def plot_high_dim_classification(groups, output_dir, color_map, smooth_window):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.flatten()
 
    plot_single_metric(axes[0], groups, "pca_geom", lambda df: mean_over_pairs(df, "ellipsoid_bhattacharyya"),
                        "Mean Bhattacharyya distance", "Mean ellipsoid Bhattacharyya distance (over class pairs)",
                        color_map, smooth_window)
    axes[0].set_yscale('log') # Log Y axis for Bhattacharyya distance

    plot_single_metric(axes[1], groups, "pca_geom", lambda df: mean_over_pairs(df, "cyl_overlap"),
                        "Mean cylinder overlap", "Mean cylinder overlap (over class pairs)",
                        color_map, smooth_window)
    axes[1].set_ylim(-0.05, 1.05) # Fixed limits [0, 1] for cylinder overlap

    plot_single_metric(axes[2], groups, "classifier", logit_decomposition,
                        "Mean correct / max-wrong logit", "Mean logit decomposition: correct / max-wrong (over classes)",
                        color_map, smooth_window)
    axes[2].set_yscale('log') # Log Y axis for logit decomposition
    
    plot_single_metric(axes[3], groups, "classifier", lambda df: df["path_curvature_ratio"],
                        "Path curvature ratio", "Path curvature ratio", color_map, smooth_window)
 
    fig.suptitle("High-dimensional classification geometry across datasets", fontsize=14)
    add_dataset_legend(fig, color_map)
    fig.tight_layout(rect=[0, 0, 0.93, 0.95])
 
    path = os.path.join(output_dir, "high_dimensional_classification.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")
 
 
def plot_cylinder_hyperplanes(groups, output_dir, color_map, smooth_window):
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    axes = axes.flatten()
 
    plot_single_metric(axes[0], groups, "hyperplanes", 
                        lambda df: np.degrees(np.arcsin(np.clip(np.abs(mean_over_pairs(df, "cm_cosine")), 0, 1))),
                        "Angle to boundary (degrees)", "CM-alignment angle (boundary vs $\\Delta\\mu$)",
                        color_map, smooth_window)
    axes[0].set_ylim(-2, 92) # Fixed limits [0, 90] for angles
 
    plot_single_metric(axes[1], groups, "hyperplanes", 
                        lambda df: np.degrees(np.arcsin(np.clip(np.abs(pca_alignment_cosine(df)), 0, 1))),
                        "Mean PCA-alignment angle (degrees)",
                        "PCA-alignment angle (boundary vs class PC1, mean of 6)",
                        color_map, smooth_window)
    axes[1].set_ylim(-2, 92) # Fixed limits [0, 90] for angles

    cols = plot_multi_metric(axes[2], groups, "hyperplanes", detect_hp_angle_cols,
                              "Angle (degrees)", "Angle between boundary normals",
                              color_map, smooth_window)
    if cols:
        add_linestyle_legend(axes[2], cols, strip_prefix="hp_angle_")
 
    cols2 = plot_multi_metric(axes[3], groups, "pca_geom", cyl_axis_cosine_cols,
                               "Angle (degrees)", "Principal axis alignment (cosine between class PCA axes)",
                               color_map, smooth_window,
                               transform=lambda x: np.degrees(np.arccos(np.clip(np.abs(x), 0, 1))))
    if cols2:
        add_linestyle_legend(axes[3], cols2, strip_prefix="cyl_axis_cosine_", loc="lower left")
    axes[2].set_ylim(-2, 92) # Fixed limits [0, 90] for angles
    axes[3].set_ylim(-2, 92) # Fixed limits [0, 90] for angles
 
    # --- mean-across-the-3-pairs companions to the two panels above ---
    plot_single_metric(axes[4], groups, "hyperplanes", std_hp_angle,
                        "Std angle (degrees)", "Std angle (mean=60°) between boundary normals (avg. of 3 pairs)",
                        color_map, smooth_window)
 
    plot_single_metric(axes[5], groups, "pca_geom", 
                        lambda df: np.degrees(np.arccos(np.clip(np.abs(mean_cyl_axis_cosine(df)), 0, 1))),
                        "Mean angle (degrees)", "Mean principal axis alignment (avg. of 3 pairs)",
                        color_map, smooth_window)
    axes[5].set_ylim(-2, 92) # Fixed limits [0, 90] for angles
 
    fig.suptitle("Cylinder / hyperplane alignment geometry across datasets", fontsize=14)
    add_dataset_legend(fig, color_map)
    fig.tight_layout(rect=[0, 0, 0.93, 0.95])
 
    path = os.path.join(output_dir, "cylinder_hyperplanes.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")
 
 
def plot_projected_nc(groups, output_dir, color_map, smooth_window):
    """2×2 figure with the four cylinder-deflation summary metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.flatten()

    # [0] Mean cylinder elongation (log scale -- should be >> 1 for cylinders)
    plot_single_metric(
        axes[0], groups, "proj_nc", mean_elongation,
        "Elongation ratio (log)", "Mean cylinder elongation per class  [λ₁ / mean(λ₂…λ_D)]",
        color_map, smooth_window,
    )
    axes[0].set_yscale("log")
    axes[0].axhline(1.0, color="black", linestyle=":", linewidth=1.2)

    # [1] Mean axis alignment |cos θ| over pairs -- 1 = parallel cylinders
    plot_single_metric(
        axes[1], groups, "proj_nc", mean_axis_alignment,
        "Mean |cos θ|", "Mean cylinder-axis alignment over pairs  (1 = parallel)",
        color_map, smooth_window,
    )
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].axhline(1.0, color="red",   linestyle="--", linewidth=1.2)
    axes[1].axhline(0.0, color="black", linestyle=":",  linewidth=1.2)

    # [2] NC1 ratio -- the smoking-gun panel
    plot_single_metric(
        axes[2], groups, "proj_nc", nc1_ratio,
        "NC1 deflated / original", "NC1 ratio (deflated / original)  [< 1 confirms cylinder hypothesis]",
        color_map, smooth_window,
    )
    axes[2].axhline(1.0, color="black", linestyle=":",  linewidth=1.2, label="no change")
    axes[2].axhline(0.0, color="green", linestyle="--", linewidth=1.2, label="full collapse after deflation")
    axes[2].legend(fontsize=7)

    # [3] NC2 ratio -- does deflation also improve equiangularity?
    plot_single_metric(
        axes[3], groups, "proj_nc", nc2_ratio,
        "NC2 dev. deflated / original", "NC2 deviation ratio (deflated / original)",
        color_map, smooth_window,
    )
    axes[3].axhline(1.0, color="black", linestyle=":",  linewidth=1.2, label="no change")
    axes[3].axhline(0.0, color="green", linestyle="--", linewidth=1.2, label="perfect ETF after deflation")
    axes[3].legend(fontsize=7)

    fig.suptitle("Projected NC analysis — cylinder-deflation diagnostics", fontsize=14)
    add_dataset_legend(fig, color_map)
    fig.tight_layout(rect=[0, 0, 0.93, 0.95])

    path = os.path.join(output_dir, "projected_nc.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_NN_comparison(groups, output_dir, color_map, smooth_window):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
 
    # Subplot 1: Bhattacharyya distance
    plot_single_metric(axes[0], groups, "pca_geom", lambda df: mean_over_pairs(df, "ellipsoid_bhattacharyya"),
                        "Mean Bhattacharyya distance", "Mean ellipsoid Bhattacharyya distance (over class pairs)",
                        color_map, smooth_window)
    axes[0].set_yscale('log') # Log Y axis for Bhattacharyya distance
    axes[0].set_ylim(0.99, 80) # Fixed limits for Bhattacharyya distance
 
    # Subplot 2: Path curvature ratio
    plot_single_metric(axes[1], groups, "classifier", lambda df: df["path_curvature_ratio"],
                        "Path curvature ratio", "Path curvature ratio", color_map, smooth_window)
    axes[1].set_yscale('log') # Log Y axis for Path curvature ratio
    axes[1].set_ylim(0.99, 60) # Fixed limits for Path curvature ratio
 
    fig.suptitle("Neural Network Comparison: Representation and Curvature Geometry", fontsize=14)
    add_dataset_legend(fig, color_map)
    fig.tight_layout(rect=[0, 0, 0.93, 0.95])
 
    path = os.path.join(output_dir, "NN_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Plot CSV metrics across datasets (one line per dataset).")
    parser.add_argument("--training_type", default="without_bumps", help="Type of training data to analyze.")
    parser.add_argument("--output", default=None, help="Directory to save output PNGs.")
    parser.add_argument("--smooth", type=int, default=SMOOTH_WINDOW,
                         help="Rolling-mean smoothing window in global_steps (1 = no smoothing).")
    args = parser.parse_args()

    root_dir = f"{ROOT_DIR}/{args.training_type}"
    if args.training_type == "without_bumps":
        extension = "no_bumps"
    elif args.training_type == "with_bumps":
        extension = "always_bumps"
    elif args.training_type == "with_bumps_before_tpt":
        extension = "bumps"
    else:
        error_msg = f"Unknown training_type '{args.training_type}'. Valid options are: 'without_bumps', 'with_bumps', 'with_bumps_before_tpt'."
        raise ValueError(error_msg)
    
    DATASETS = {
    "Blobs": f"{root_dir}/blobs-{extension}/",
    "Spiral": f"{root_dir}/spiral-{extension}/",
    "Dartboard": f"{root_dir}/dartboard-{extension}/",
    "Rings": f"{root_dir}/rings-{extension}/",
    "Checkerboard": f"{root_dir}/checkerboard-{extension}/",
    # add more datasets here, e.g.:
    # "Blobs": f"{root_dir}/blobs-{args.training_type}/training_2026..._.../",
}
 
    runs = discover_all(DATASETS)
    if not runs:
        print("No valid runs discovered, aborting.")
        return
 
    groups = get_grouped(runs)
    labels = list(groups.keys())
    print(f"\nFound datasets: {labels}\n")
 
    cmap = plt.get_cmap("tab10")
    color_map = {label: cmap(i % 10) for i, label in enumerate(labels)}
    
    if args.output is None:
        output_dir = os.path.join(os.getcwd(), f"{root_dir}/plots_across_datasets")
    else:
        output_dir = args.output

    os.makedirs(output_dir, exist_ok=True)
 
    plot_accuracies(groups, output_dir, color_map, args.smooth)
    plot_neural_collapse(groups, output_dir, color_map, args.smooth)
    plot_high_dim_classification(groups, output_dir, color_map, args.smooth)
    plot_cylinder_hyperplanes(groups, output_dir, color_map, args.smooth)
    plot_NN_comparison(groups, output_dir, color_map, args.smooth)
    plot_projected_nc(groups, output_dir, color_map, args.smooth)
 
    print("\nDone.")
 
 
if __name__ == "__main__":
    main()