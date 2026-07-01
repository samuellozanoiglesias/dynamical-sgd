#!/usr/bin/env python3
"""
BLOBS_analysis.py

Reads a batch of training runs (one per `num_modes` configuration) and produces
four multi-panel figures where every panel overlays one line per `num_modes`
value (color-coded with a continuous colormap + colorbar).

Expected layout on disk
------------------------
ROOT_DIR/
    training_<timestamp_1>/
        config.yaml                 # must contain a `num_modes` key somewhere
        training_metrics.csv
        classifier_metrics.csv
        hyperplanes.csv
        nc_metrics.csv
        pca_analysis.csv
        PCA_geometric.csv
    training_<timestamp_2>/
        ...
    ...

If two `training_*` folders happen to share the same `num_modes` (e.g. several
seeds), their curves are averaged together (aligned on `global_step`) before
plotting, so you still get exactly one line (or one set of lines, for the
multi-line panels) per `num_modes` value.

Output
------
Four PNGs are written to OUTPUT_DIR:
    accuracies.png                  (1x2)
    neural_collapse.png             (2x2)
    high_dimensional_classification.png (2x2)
    cylinder_hyperplanes.png        (2x2)

Usage
-----
    nohup python BLOBS_analysis.py --root /data/samuel_lozano/dynamical-sgd/without_bumps/blobs-no_bumps/ > log.out 2>&1 &
    python BLOBS_analysis.py --smooth 25     # rolling-mean smoothing window in global_steps (1 = off)

Column-mapping assumptions (please sanity-check these against your pipeline)
------------------------------------------------------------------------------
- "mean ... over classes/pairs" metrics are computed by auto-detecting all
  columns matching the relevant `<prefix>_<i>` (per-class) or `<prefix>_<i>v<j>`
  (per-pair) pattern and averaging across them row-wise. This means the code
  does NOT hardcode "3 classes" anywhere; it adapts to however many class /
  pair columns are actually present in each CSV.
- "CM-alignment cosine"        -> mean of `cm_cosine_<i>v<j>`            (hyperplanes.csv)
- "PCA-alignment cosine"       -> mean of `pca_cosine_<i>v<j>_p1_<k>`    (hyperplanes.csv, the 6 boundary-normal vs per-class-PC1 cosines)
- "Angle between boundary normals" -> all `hp_angle_(...)v(...)` columns plotted as separate lines (one linestyle per pair-of-pairs), 3 lines for the 3-class case (hyperplanes.csv)
- "Principal axis alignment"  -> `cyl_axis_cosine_<i>v<j>` columns (PCA_geometric.csv), i.e. cosine between the PCA-derived principal axes of each class pair, plotted as separate lines.
  (If you intended a different column for "principal axis alignment", e.g.
  `raw_cosine_<i>v<j>` from hyperplanes.csv, just swap the column lookup in
  `plot_cylinder_hyperplanes()` below — it's a one-line change.)
"""

import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.lines import Line2D

# --------------------------------------------------------------------------- #
# Configuration (override via CLI flags --root / --output / --smooth)
# --------------------------------------------------------------------------- #
ROOT_DIR = "/data/samuel_lozano/dynamical-sgd/without_bumps/blobs-no_bumps/"
SMOOTH_WINDOW = 1  # rolling-mean window (in global_steps); 1 disables smoothing
CONFIG_FILENAMES = ["config.yaml", "config.yml"]

CSV_FILES = {
    "training": "training_metrics.csv",
    "classifier": "classifier_metrics.csv",
    "hyperplanes": "hyperplanes.csv",
    "nc": "nc_metrics.csv",
    "pca": "pca_analysis.csv",
    "pca_geom": "PCA_geometric.csv",
}

LINESTYLES = ["-", "--", ":", "-."]


# --------------------------------------------------------------------------- #
# Discovery: walk training_* folders, read config.yaml + csvs
# --------------------------------------------------------------------------- #
def find_file(run_dir, filename):
    """Find `filename` anywhere under run_dir (handles nested subfolders)."""
    matches = glob.glob(os.path.join(run_dir, "**", filename), recursive=True)
    if not matches:
        return None
    if len(matches) > 1:
        print(f"[WARN] multiple '{filename}' found under {run_dir}, using {matches[0]}")
    return matches[0]
 
 
def find_num_modes(config_path):
    """Recursively search a (possibly nested) yaml config for a `num_modes` key."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
 
    def search(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if str(k).strip().lower() == "num_modes":
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
 
    nm = search(cfg)
    if nm is None:
        raise ValueError(f"key 'num_modes' not found in {config_path}")
    return nm
 
 
def discover_runs(root_dir):
    runs = []
    train_dirs = sorted(glob.glob(os.path.join(root_dir, "training_*")))
    if not train_dirs:
        print(f"[WARN] No 'training_*' directories found under {root_dir}")
 
    for d in train_dirs:
        if not os.path.isdir(d):
            continue
 
        config_path = None
        for cfg_name in CONFIG_FILENAMES:
            config_path = find_file(d, cfg_name)
            if config_path:
                break
        if config_path is None:
            print(f"[WARN] No config.yaml/.yml found in {d}, skipping.")
            continue
 
        try:
            num_modes = find_num_modes(config_path)
        except Exception as e:
            print(f"[WARN] Could not extract num_modes from {config_path}: {e}")
            continue
 
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
            print(f"[WARN] {os.path.basename(d)}: missing csv(s) {missing}")
        if not dfs:
            print(f"[WARN] {os.path.basename(d)}: no CSVs read successfully, skipping run.")
            continue
 
        runs.append({"dir": d, "num_modes": num_modes, "dfs": dfs})
        print(f"[OK]   {os.path.basename(d)}: num_modes={num_modes}, csvs={sorted(dfs.keys())}")
 
    return runs
 
 
def get_grouped(runs):
    """Group runs by num_modes -> {num_modes: [run, run, ...]} (sorted by num_modes)."""
    groups = {}
    for r in runs:
        groups.setdefault(r["num_modes"], []).append(r)
    return dict(sorted(groups.items()))
 
 
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
# Aggregation across runs sharing the same num_modes, aligned on global_step
# --------------------------------------------------------------------------- #
def get_curve(group_runs, csv_key, value_fn):
    """
    value_fn(df) -> pd.Series of the metric, indexed like df.
    Returns (global_step_array, mean_value_array) averaged across all runs in
    group_runs that have csv_key available (handles multiple seeds sharing
    the same num_modes).
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
def plot_single_metric(ax, groups, csv_key, value_fn, ylabel, title, cmap, norm, smooth_window):
    """One line per num_modes, color encodes num_modes."""
    any_data = False
    for nm in sorted(groups.keys()):
        global_step, vals = get_curve(groups[nm], csv_key, value_fn)
        if global_step is None or len(global_step) == 0:
            continue
        vals = smooth(vals, smooth_window)
        ax.plot(global_step, vals, color=cmap(norm(nm)), linewidth=1.4, alpha=0.9)
        any_data = True
    if not any_data:
        ax.text(0.5, 0.5, "no data found", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Global step")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
 
 
def plot_multi_metric(ax, groups, csv_key, columns_fn, ylabel, title, cmap, norm, smooth_window, transform=None):
    """
    Several lines per num_modes (e.g. one per class-pair). Color encodes
    num_modes, linestyle encodes which column. Returns the canonical column
    list used (for building a separate linestyle legend), or [] if no data.
    """
    canonical_cols = None
    for nm in sorted(groups.keys()):
        for r in groups[nm]:
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
 
    for nm in sorted(groups.keys()):
        color = cmap(norm(nm))
        for i, col in enumerate(canonical_cols):
            ls = LINESTYLES[i % len(LINESTYLES)]
            
            def val_fn(df, c=col):
                if c not in df.columns: return None
                val = df[c]
                if transform: return transform(val)
                return val

            global_step, vals = get_curve(groups[nm], csv_key, val_fn)
            if global_step is None or len(global_step) == 0:
                continue
            vals = smooth(vals, smooth_window)
            ax.plot(global_step, vals, color=color, linestyle=ls, linewidth=1.2, alpha=0.85)
 
    ax.set_xlabel("Global step")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    return canonical_cols
 
 
def add_linestyle_legend(ax, cols, strip_prefix=None, loc="best"):
    """Neutral-color legend explaining what each linestyle means (column name).
    `strip_prefix`, if given, is removed from each label to keep the legend compact."""
    def short(col):
        if strip_prefix and col.startswith(strip_prefix):
            return col[len(strip_prefix):]
        return col
 
    handles = [
        Line2D([0], [0], color="black", linestyle=LINESTYLES[i % len(LINESTYLES)], label=short(col))
        for i, col in enumerate(cols)
    ]
    ax.legend(handles=handles, fontsize=7, loc=loc, framealpha=0.7, handlelength=2.2)
 
 
def make_colorbar(fig, axes, cmap, norm, modes):
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, label="num_modes", pad=0.02, fraction=0.03)
    if len(modes) <= 12:
        cbar.set_ticks(sorted(modes))
    return cbar
 
 
# --------------------------------------------------------------------------- #
# The four figures
# --------------------------------------------------------------------------- #
def plot_accuracies(groups, output_dir, cmap, norm, smooth_window):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
 
    plot_single_metric(axes[0], groups, "training", lambda df: df["train_accuracy"],
                        "Train accuracy", "Train accuracy", cmap, norm, smooth_window)
    plot_single_metric(axes[1], groups, "training", lambda df: df["test_accuracy"],
                        "Test accuracy", "Test accuracy", cmap, norm, smooth_window)
    axes[0].set_ylim(-0.05, 1.05) # Fixed limits [0, 1] for train accuracy
    axes[1].set_ylim(-0.05, 1.05) # Fixed limits [0, 1] for test accuracy
 
    fig.suptitle("Train / test accuracy across num_modes", fontsize=14)
    make_colorbar(fig, axes.tolist(), cmap, norm, groups.keys())
    fig.tight_layout(rect=[0, 0, 0.93, 0.95])
 
    path = os.path.join(output_dir, "accuracies.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")
 
 
def plot_neural_collapse(groups, output_dir, cmap, norm, smooth_window):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.flatten()
 
    plot_single_metric(axes[0], groups, "nc", lambda df: df["nc1"],
                        "NC1", "NC1 (within / between-class variance collapse)", cmap, norm, smooth_window)
    axes[0].set_yscale('log') # Log Y axis for NC1

    plot_single_metric(axes[1], groups, "nc", lambda df: df["nc2_deviation"],
                        "NC2 deviation", "NC2 (equinormality / equiangularity deviation)", cmap, norm, smooth_window)
    plot_single_metric(axes[2], groups, "pca", lambda df: df["pca_k95"],
                        "# PCA components", "PCA dimensions needed for 95% variance", cmap, norm, smooth_window)
    plot_single_metric(axes[3], groups, "pca_geom", cyl_half_over_radius,
                        "Mean half-length / radius", "Mean cylinder half-length / radius (over classes)",
                        cmap, norm, smooth_window)
    axes[3].set_yscale('log') # Log Y axis for cylinder half-length/radius
 
    fig.suptitle("Neural collapse diagnostics across num_modes", fontsize=14)
    make_colorbar(fig, axes.tolist(), cmap, norm, groups.keys())
    fig.tight_layout(rect=[0, 0, 0.93, 0.95])
 
    path = os.path.join(output_dir, "neural_collapse.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")
 
 
def plot_high_dim_classification(groups, output_dir, cmap, norm, smooth_window):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.flatten()
 
    plot_single_metric(axes[0], groups, "pca_geom", lambda df: mean_over_pairs(df, "ellipsoid_bhattacharyya"),
                        "Mean Bhattacharyya distance", "Mean ellipsoid Bhattacharyya distance (over class pairs)",
                        cmap, norm, smooth_window)
    axes[0].set_yscale('log') # Log Y axis for Bhattacharyya distance

    plot_single_metric(axes[1], groups, "pca_geom", lambda df: mean_over_pairs(df, "cyl_overlap"),
                        "Mean cylinder overlap", "Mean cylinder overlap (over class pairs)",
                        cmap, norm, smooth_window)
    axes[1].set_ylim(-0.05, 1.05) # Fixed limits [0, 1] for cylinder overlap

    plot_single_metric(axes[2], groups, "classifier", logit_decomposition,
                        "Mean correct / max-wrong logit", "Mean logit decomposition: correct / max-wrong (over classes)",
                        cmap, norm, smooth_window)
    axes[2].set_yscale('log') # Log Y axis for logit decomposition

    plot_single_metric(axes[3], groups, "classifier", lambda df: df["path_curvature_ratio"],
                        "Path curvature ratio", "Path curvature ratio", cmap, norm, smooth_window)
 
    fig.suptitle("High-dimensional classification geometry across num_modes", fontsize=14)
    make_colorbar(fig, axes.tolist(), cmap, norm, groups.keys())
    fig.tight_layout(rect=[0, 0, 0.93, 0.95])
 
    path = os.path.join(output_dir, "high_dimensional_classification.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")
 
 
def plot_cylinder_hyperplanes(groups, output_dir, cmap, norm, smooth_window):
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    axes = axes.flatten()
 
    # arcsin(abs(cos(normal))) to calculate the complementary angle (with the hyperplane boundary itself)
    plot_single_metric(axes[0], groups, "hyperplanes", 
                        lambda df: np.degrees(np.arcsin(np.clip(np.abs(mean_over_pairs(df, "cm_cosine")), 0, 1))),
                        "Angle to boundary (degrees)", "CM-alignment angle (boundary vs $\\Delta\\mu$)",
                        cmap, norm, smooth_window)
    axes[0].set_ylim(-2, 92) # Fixed limits [0, 90] for angles
 
    plot_single_metric(axes[1], groups, "hyperplanes", 
                        lambda df: np.degrees(np.arcsin(np.clip(np.abs(pca_alignment_cosine(df)), 0, 1))),
                        "Angle to boundary (degrees)",
                        "PCA-alignment angle (boundary vs class PC1, mean of 6)",
                        cmap, norm, smooth_window)
    axes[1].set_ylim(-2, 92) # Fixed limits [0, 90] for angles

    cols = plot_multi_metric(axes[2], groups, "hyperplanes", detect_hp_angle_cols,
                              "Angle (degrees)", "Angle between boundary normals",
                              cmap, norm, smooth_window)
    if cols:
        add_linestyle_legend(axes[2], cols, strip_prefix="hp_angle_")
 
    # Convert cosine to angle using arccos(abs(cosine))
    cols2 = plot_multi_metric(axes[3], groups, "pca_geom", cyl_axis_cosine_cols,
                               "Angle (degrees)", "Principal axis alignment (angle between class PCA axes)",
                               cmap, norm, smooth_window,
                               transform=lambda x: np.degrees(np.arccos(np.clip(np.abs(x), 0, 1))))
    if cols2:
        add_linestyle_legend(axes[3], cols2, strip_prefix="cyl_axis_cosine_", loc="lower left")
    
    axes[2].set_ylim(-2, 92) # Fixed limits [0, 90] for angles
    axes[3].set_ylim(-2, 92) # Fixed limits [0, 90] for angles

    # --- mean-across-the-3-pairs companions to the two panels above ---
    plot_single_metric(axes[4], groups, "hyperplanes", std_hp_angle,
                        "Std angle (degrees)", "Std angle (mean=60°) between boundary normals (avg. of 3 pairs)",
                        cmap, norm, smooth_window)
 
    plot_single_metric(axes[5], groups, "pca_geom", 
                        lambda df: np.degrees(np.arccos(np.clip(np.abs(mean_cyl_axis_cosine(df)), 0, 1))),
                        "Mean angle (degrees)", "Mean principal axis alignment (avg. of 3 pairs)",
                        cmap, norm, smooth_window)
    axes[5].set_ylim(-2, 92) # Fixed limits [0, 90] for angles

    fig.suptitle("Cylinder / hyperplane alignment geometry across num_modes", fontsize=14)
    make_colorbar(fig, axes.tolist(), cmap, norm, groups.keys())
    fig.tight_layout(rect=[0, 0, 0.93, 0.95])
 
    path = os.path.join(output_dir, "cylinder_hyperplanes.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")
 
 
# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Plot CSV metrics across num_modes configurations.")
    parser.add_argument("--root", default=ROOT_DIR, help="Root directory containing training_* folders.")
    parser.add_argument("--output", default=None, help="Directory to save output PNGs.")
    parser.add_argument("--smooth", type=int, default=SMOOTH_WINDOW,
                         help="Rolling-mean smoothing window in global_steps (1 = no smoothing).")
    args = parser.parse_args()
    
    root_dir = args.root
    runs = discover_runs(root_dir)
    if not runs:
        print("No valid runs discovered, aborting.")
        return
 
    groups = get_grouped(runs)
    modes = sorted(groups.keys())
    print(f"\nFound num_modes values: {modes}\n")
 
    cmap = plt.get_cmap("viridis")
    vmin, vmax = min(modes), max(modes)
    if vmin == vmax:
        vmin, vmax = vmin - 0.5, vmax + 0.5
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    
    output_dir = f"{root_dir}/plots_num_modes" if not args.output else args.output
    os.makedirs(output_dir, exist_ok=True)
 
    plot_accuracies(groups, output_dir, cmap, norm, args.smooth)
    plot_neural_collapse(groups, output_dir, cmap, norm, args.smooth)
    plot_high_dim_classification(groups, output_dir, cmap, norm, args.smooth)
    plot_cylinder_hyperplanes(groups, output_dir, cmap, norm, args.smooth)
 
    print("\nDone.")
 
 
if __name__ == "__main__":
    main()