#!/usr/bin/env python3
"""
grid_analysis.py
=================
Builds gridmaps (heatmaps) for every metric produced by
`metrics_for_multiple_classes.py`, comparing the two swept hyperparameters
from the `gridsearch_wmaxs_periods.sbatch` sweep:

    x-axis -> dynamics.w_max        ("w", bump width)
    y-axis -> dynamics.period_length ("T", bump period)

It also adds a fake "w=1" column (leftmost, since real widths start at 5)
that is filled with the metrics of a separate `without_bumps` run, repeated
identically for every T row (there is no period dependency when there are
no bumps at all).

Directory layout expected (matches launch.py + the sbatch script):

    BASE_OUTPUT_DIR/
      w{W}_p{P}/                      <- output.output_dir for that combo
        {experiment_name}/            <- inferred by launch.py from
                                          dynamics.bumps_before_tpt / bumps_at_tpt
          {config_name}/              <- config file stem
            training_<timestamp>/     <- actual run dir (may have _NN suffix)
              *.csv                   <- metrics CSV written by the training run

    WITHOUT_BUMPS_DIR/                <- output.output_dir of the separate
      {experiment_name}/                 without_bumps run (same layout)
        {config_name}/
          training_<timestamp>/
            *.csv

`experiment_name` is auto-detected (it is the same for every combo in a
single sweep, since the sbatch script never overrides the bump flags), but
can be forced with --experiment-name / --without-bumps-experiment-name.

The metrics CSV filename is not fixed anywhere in this repo's public API
(it's written by training_runner.py), so by default this script
auto-discovers it: it looks at every *.csv file in the run directory and
picks the one whose header is a superset of the known metric field names.
Use --csv-name to force a specific filename instead.

train_accuracy / test_accuracy live in a *separate* CSV, training_metrics.csv
by default (--accuracy-csv-name to override), which is looked up in each
run dir the same way and merged into the same gridmaps / summary CSV.

Usage
-----
    python grid_analysis.py \\
        --base-dir /path/to/gridsearch_wmaxs_periods \\
        --config-name cifar10_resnet10-always_bumps \\
        --without-bumps-dir /path/to/without_bumps_run_output_dir \\
        --wmaxs 5 10 20 50 70 100 150 250 \\
        --periods 5 10 20 50 100 200 400 800 \\
        --output-dir /path/to/gridsearch_wmaxs_periods/grid_analysis

Example:
nohup python grid_analysis.py \
    --base-dir /mnt/lustre/home/samuloza/data/samuel_lozano/dynamical-sgd/gridsearch_wmaxs_periods \
    --config-name cifar10_resnet10-always_bumps \
    > log_grid_analysis.out 2>&1 &
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # headless / cluster-safe
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Metric field names, kept in sync with metrics_for_multiple_classes.py
# (duplicated here, rather than imported, so this script has no dependency
# on where that module lives relative to this file / on sys.path).
# ---------------------------------------------------------------------------

METRIC_FIELDS: list[str] = [
    "condition_number",
    "path_curvature_ratio",
    "weight_path_distance",
    "nc1",
    "nc2_deviation",
    "avg_separation_margin",
    "cyl_half_length_mean",
    "cyl_radius_mean",
    "bhattacharyya_distance_mean",
    "pca_alignment_mean",
    "nc1_deflated",
    "nc2_deflated_deviation",
    "nc1_ratio_deflated",
    "nc2_ratio_deflated",
]

# train_accuracy / test_accuracy live in a separate CSV (training_metrics.csv
# by default) rather than in the multi-class metrics CSV above.
ACCURACY_FIELDS: list[str] = [
    "train_accuracy",
    "test_accuracy",
]

# Everything that ends up as one gridmap + one row in the summary CSV.
ALL_METRIC_FIELDS: list[str] = ACCURACY_FIELDS + METRIC_FIELDS

DEFAULT_TRAINING_METRICS_CSV = "training_metrics.csv"

# Pretty axis / title labels for the metrics that benefit from a log scale
# colorbar (mirrors finalize_multiclass_plots' choice of log axes).
LOG_SCALE_METRICS = {"condition_number", "nc1", "nc1_deflated"}

WITHOUT_BUMPS_LABEL = "1\n(without_bumps)"


# ---------------------------------------------------------------------------
# Directory / file discovery
# ---------------------------------------------------------------------------

def resolve_training_dir(
    root: Path,
    config_name: Optional[str] = None,
    experiment_name: Optional[str] = None,
) -> tuple[Optional[Path], Optional[str]]:
    """Locate a training_<timestamp> dir under `root`.

    Unlike a blind recursive search, this explicitly narrows down to the
    `experiment_name` and/or `config_name` subfolder *by name* before looking
    for training_* dirs. This matters because sibling folders under `root`
    (e.g. other model variants sharing the same widths/periods sweep dir)
    can contain their own training_* dirs with a different config_name — if
    the requested config_name folder isn't found, this returns "not found"
    rather than silently falling back to a sibling folder's run.

    If `experiment_name` is given, `root` (or root/**/) must contain a
    directory with exactly that name; the search is then restricted to that
    subtree. Same for `config_name` (with the extra allowance that `root`
    itself may already *be* the config_name folder, e.g. when the caller
    passes a path like '.../without_bumps/cifar10_resnet10-no_bumps/').

    Among training_* matches, the earliest one (lexicographically first,
    which equals chronologically first given the training_YYYY_MM_DD-HH_MM_SS
    naming) is returned, since that's the original run rather than a re-run.

    Returns (training_dir, experiment_name_used) or (None, None) if not found.
    """
    if not root.is_dir():
        return None, None

    search_root = root
    exp_used: Optional[str] = None

    if experiment_name:
        exp_dirs = sorted(d for d in root.rglob(experiment_name) if d.is_dir())
        if root.name == experiment_name:
            exp_dirs = [root] + exp_dirs
        if not exp_dirs:
            return None, None  # do NOT fall back to some other subfolder
        if len(exp_dirs) > 1:
            print(
                f"  [warn] multiple '{experiment_name}' dirs found under {root}: "
                f"{[str(d) for d in exp_dirs]} -> using {exp_dirs[0]}",
                file=sys.stderr,
            )
        search_root = exp_dirs[0]
        exp_used = experiment_name

    if config_name:
        cfg_dirs = sorted(d for d in search_root.rglob(config_name) if d.is_dir())
        if search_root.name == config_name:
            cfg_dirs = [search_root] + cfg_dirs
        if not cfg_dirs:
            return None, None  # do NOT fall back to a sibling config folder
        if len(cfg_dirs) > 1:
            print(
                f"  [warn] multiple '{config_name}' dirs found under {search_root}: "
                f"{[str(d) for d in cfg_dirs]} -> using {cfg_dirs[0]}",
                file=sys.stderr,
            )
        search_root = cfg_dirs[0]

    matches = sorted(d for d in search_root.rglob("training_*") if d.is_dir())
    if not matches:
        return None, None

    if len(matches) > 1:
        print(
            f"  [warn] multiple training_* dirs matched under {search_root}: "
            f"{[str(m) for m in matches]} -> using the earliest one",
            file=sys.stderr,
        )

    chosen = matches[0]
    if exp_used is None:
        exp_used = chosen.parent.parent.name if len(chosen.parents) >= 2 else None
    return chosen, exp_used


def find_metrics_csv(run_dir: Path, csv_name: Optional[str]) -> Optional[Path]:
    if csv_name is not None:
        candidate = run_dir / csv_name
        return candidate if candidate.is_file() else None

    best: Optional[Path] = None
    for csv_path in sorted(run_dir.glob("*.csv")):
        try:
            with open(csv_path, "r", encoding="utf-8", newline="") as f:
                header = next(csv.reader(f), [])
        except (OSError, StopIteration):
            continue
        if set(METRIC_FIELDS).issubset(set(header)):
            best = csv_path
            break
    return best


def find_training_metrics_csv(run_dir: Path, csv_name: Optional[str]) -> Optional[Path]:
    """Locate the CSV holding train_accuracy / test_accuracy.

    Defaults to an exact filename match (training_metrics.csv), falling back
    to auto-discovery (any *.csv whose header contains both accuracy fields)
    if that exact file isn't present.
    """
    name = csv_name or DEFAULT_TRAINING_METRICS_CSV
    candidate = run_dir / name
    if candidate.is_file():
        return candidate

    for csv_path in sorted(run_dir.glob("*.csv")):
        try:
            with open(csv_path, "r", encoding="utf-8", newline="") as f:
                header = next(csv.reader(f), [])
        except (OSError, StopIteration):
            continue
        if set(ACCURACY_FIELDS).issubset(set(header)):
            return csv_path
    return None


def load_final_row(csv_path: Path, fields: list[str]) -> dict[str, float]:
    """Return the requested field values from the last row (highest
    global_step / epoch) of a metrics CSV."""
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}

    def _sort_key(row: dict[str, str]) -> float:
        for key in ("global_step", "epoch"):
            value = row.get(key)
            if value not in (None, ""):
                try:
                    return float(value)
                except ValueError:
                    pass
        return 0.0

    last_row = max(rows, key=_sort_key)
    result: dict[str, float] = {}
    for field_name in fields:
        raw = last_row.get(field_name)
        try:
            result[field_name] = float(raw) if raw not in (None, "") else float("nan")
        except ValueError:
            result[field_name] = float("nan")
    return result


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------

def _collect_run_values(
    run_dir: Path,
    csv_name: Optional[str],
    accuracy_csv_name: Optional[str],
    label: str,
    missing: list[str],
) -> tuple[dict[str, float], Optional[Path], Optional[Path]]:
    """Load both the multi-class metrics CSV and the training_metrics.csv
    (train_accuracy / test_accuracy) for a single run dir, merging the
    results into one dict keyed by ALL_METRIC_FIELDS."""
    values: dict[str, float] = {}

    csv_path = find_metrics_csv(run_dir, csv_name)
    if csv_path is None:
        missing.append(f"{label}: metrics CSV not found in {run_dir}")
    else:
        values.update(load_final_row(csv_path, METRIC_FIELDS))

    accuracy_csv_path = find_training_metrics_csv(run_dir, accuracy_csv_name)
    if accuracy_csv_path is None:
        missing.append(f"{label}: training metrics CSV (train/test accuracy) not found in {run_dir}")
    else:
        values.update(load_final_row(accuracy_csv_path, ACCURACY_FIELDS))

    return values, csv_path, accuracy_csv_path


@dataclass
class GridResult:
    widths: list[int]
    periods: list[int]
    x_labels: list[str]
    grids: dict[str, np.ndarray]  # metric -> (n_periods, n_widths_plus_one)
    missing: list[str] = field(default_factory=list)
    n_found: int = 0
    n_total: int = 0


def build_grids(
    base_dir: Path,
    config_name: str,
    widths: list[int],
    periods: list[int],
    experiment_name: Optional[str],
    csv_name: Optional[str],
    accuracy_csv_name: Optional[str],
    without_bumps_dir: Optional[Path],
    without_bumps_config_name: Optional[str],
    without_bumps_experiment_name: Optional[str],
) -> GridResult:
    sorted_widths = sorted(widths)
    sorted_periods = sorted(periods)
    n_rows = len(sorted_periods)
    n_cols = len(sorted_widths) + 1  # +1 for the fake without_bumps column

    grids = {m: np.full((n_rows, n_cols), np.nan) for m in ALL_METRIC_FIELDS}
    missing: list[str] = []

    # ---- without_bumps fake column (index 0), replicated for every T ----
    if without_bumps_dir is not None:
        run_dir, used_exp = resolve_training_dir(
            without_bumps_dir, without_bumps_config_name, without_bumps_experiment_name
        )
        if run_dir is None:
            missing.append(f"without_bumps run not found under {without_bumps_dir}")
        else:
            values, csv_path, accuracy_csv_path = _collect_run_values(
                run_dir, csv_name, accuracy_csv_name, "without_bumps", missing
            )
            print(
                f"without_bumps -> {run_dir} (experiment='{used_exp}', "
                f"metrics_csv={csv_path.name if csv_path else 'MISSING'}, "
                f"accuracy_csv={accuracy_csv_path.name if accuracy_csv_path else 'MISSING'})"
            )
            for metric in ALL_METRIC_FIELDS:
                grids[metric][:, 0] = values.get(metric, np.nan)
    else:
        missing.append("--without-bumps-dir not provided; w=1 column left empty")

    # ---- swept combos ----
    seen_experiment_names: set[str] = set()
    n_total = len(sorted_widths) * len(sorted_periods)
    n_found = 0
    for j, w in enumerate(sorted_widths, start=1):
        for i, p in enumerate(sorted_periods):
            combo_dir = base_dir / f"w{w}_p{p}"
            run_dir, used_exp = resolve_training_dir(combo_dir, config_name, experiment_name)
            if run_dir is None:
                missing.append(f"w={w}, T={p}: run not found under {combo_dir}")
                continue
            if used_exp:
                seen_experiment_names.add(used_exp)
            values, csv_path, accuracy_csv_path = _collect_run_values(
                run_dir, csv_name, accuracy_csv_name, f"w={w}, T={p}", missing
            )
            if csv_path is None and accuracy_csv_path is None:
                continue
            for metric in ALL_METRIC_FIELDS:
                grids[metric][i, j] = values.get(metric, np.nan)
            n_found += 1

    if len(seen_experiment_names) > 1:
        print(
            f"[warn] combos resolved to different experiment names: {seen_experiment_names} "
            "(expected a single consistent name across the sweep)",
            file=sys.stderr,
        )

    x_labels = [WITHOUT_BUMPS_LABEL] + [str(w) for w in sorted_widths]
    return GridResult(
        widths=sorted_widths,
        periods=sorted_periods,
        n_found=n_found,
        n_total=n_total,
        x_labels=x_labels,
        grids=grids,
        missing=missing,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_single_heatmap(ax, grid: np.ndarray, x_labels: list[str], periods: list[int],
                          title: str, log_scale: bool, annotate: bool) -> None:
    plot_grid = grid.copy()
    norm = None
    if log_scale:
        plot_grid = np.where(plot_grid > 0, plot_grid, np.nan)
        if np.any(~np.isnan(plot_grid)):
            norm = matplotlib.colors.LogNorm(
                vmin=np.nanmin(plot_grid), vmax=np.nanmax(plot_grid)
            )

    im = ax.imshow(plot_grid, origin="lower", aspect="auto", cmap="viridis", norm=norm)
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_yticks(range(len(periods)))
    ax.set_yticklabels([str(p) for p in periods], fontsize=8)
    ax.axvline(0.5, color="white", linewidth=1.5, linestyle="--")
    ax.set_xlabel("w_max (bump width)")
    ax.set_ylabel("period_length (T)")
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.85)

    if annotate:
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                val = grid[i, j]
                if np.isnan(val):
                    continue
                ax.text(
                    j, i, f"{val:.2g}", ha="center", va="center",
                    fontsize=6, color="white",
                )


def save_individual_heatmaps(result: GridResult, output_dir: Path, annotate: bool) -> None:
    out = output_dir / "gridmaps"
    out.mkdir(parents=True, exist_ok=True)
    for metric, grid in result.grids.items():
        fig, ax = plt.subplots(figsize=(1.0 + 0.8 * len(result.x_labels), 1.0 + 0.6 * len(result.periods)))
        _plot_single_heatmap(
            ax, grid, result.x_labels, result.periods,
            title=metric, log_scale=metric in LOG_SCALE_METRICS, annotate=annotate,
        )
        fig.tight_layout()
        fig.savefig(out / f"{metric}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def save_overview(result: GridResult, output_dir: Path, annotate: bool) -> None:
    n_metrics = len(ALL_METRIC_FIELDS)
    n_cols = 3
    n_rows = int(np.ceil(n_metrics / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows))
    axes = np.atleast_2d(axes)

    for idx, metric in enumerate(ALL_METRIC_FIELDS):
        ax = axes[idx // n_cols, idx % n_cols]
        _plot_single_heatmap(
            ax, result.grids[metric], result.x_labels, result.periods,
            title=metric, log_scale=metric in LOG_SCALE_METRICS, annotate=annotate,
        )

    for idx in range(n_metrics, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].axis("off")

    fig.suptitle("Widths x Periods gridsearch — final-step metrics", fontsize=16)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "grid_analysis_overview.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_summary_csv(result: GridResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "grid_analysis_summary.csv"
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "w_max", "period_length", "value"])
        for metric, grid in result.grids.items():
            for j, w_label in enumerate(["1_without_bumps"] + [str(w) for w in result.widths]):
                for i, p in enumerate(result.periods):
                    writer.writerow([metric, w_label, p, grid[i, j]])
    print(f"Summary CSV written to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build w vs T gridmaps for every metric of a dynamical-sgd widths/periods sweep."
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="/mnt/lustre/home/samuloza/data/samuel_lozano/dynamical-sgd/gridsearch_wmaxs_periods",
        help="BASE_OUTPUT_DIR used by the sbatch script (contains one w{W}_p{P} subfolder per combo).",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="cifar10_resnet_narrow-always_bumps",
        help="Stem of the config file used for the sweep (output.config_name).",
    )
    parser.add_argument(
        "--wmaxs", type=int, nargs="+",
        default=[5, 10, 20, 50, 70, 100, 120, 150, 180, 200, 250, 300, 350, 400, 450, 500],
        help="WMAXS array from the sbatch script (dynamics.w_max values).",
    )
    parser.add_argument(
        "--periods", type=int, nargs="+",
        default=[5, 10, 20, 50, 100, 200, 400, 800, 1000, 1200, 1500, 1800, 2000],
        help="PERIODS array from the sbatch script (dynamics.period_length values).",
    )
    parser.add_argument(
        "--experiment-name", type=str, default=None,
        help="Force the experiment_name subfolder for swept combos instead of auto-detecting it.",
    )
    parser.add_argument(
        "--without-bumps-dir", type=str, default=None,
        help="output.output_dir of a separate without_bumps run (same folder layout). "
             "Its final-step metrics are used to fill the fake w=1 column for every T.",
    )
    parser.add_argument(
        "--without-bumps-experiment-name", type=str, default=None,
        help="If the without_bumps run dir has an ancestor folder with this exact name, "
             "use it to disambiguate between multiple training_* matches. Leave unset "
             "if --without-bumps-dir already points close to (or at) the run itself "
             "(e.g. '.../without_bumps/cifar10_resnet10-no_bumps/').",
    )
    parser.add_argument(
        "--without-bumps-config-name", type=str, default=None,
        help="Config file stem used by the without_bumps run, if different from "
             "--config-name (e.g. a dedicated 'cifar10_resnet10-no_bumps' config). "
             "Used only to disambiguate if --without-bumps-dir contains more than one "
             "training_* dir. Leave unset to skip this filter.",
    )
    parser.add_argument(
        "--csv-name", type=str, default=None,
        help="Force a specific metrics CSV filename instead of auto-discovering it.",
    )
    parser.add_argument(
        "--accuracy-csv-name", type=str, default=None,
        help=f"Filename of the CSV holding train_accuracy / test_accuracy "
             f"(default: '{DEFAULT_TRAINING_METRICS_CSV}'; falls back to "
             "auto-discovery if that exact file isn't found).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Where to write the gridmaps. Defaults to <base-dir>/grid_analysis.",
    )
    parser.add_argument(
        "--no-annotate", action="store_true",
        help="Disable printing the numeric value inside each grid cell.",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else base_dir / "grid_analysis" / args.config_name.split("-")[0]
    )
    without_bumps_dir = (
        Path(args.without_bumps_dir).expanduser().resolve() if args.without_bumps_dir else None
    )
    if without_bumps_dir is None:
        new_config_name = f"{args.config_name.split('-')[0]}-no_bumps"
        without_bumps_dir = Path(f"/mnt/lustre/home/samuloza/data/samuel_lozano/dynamical-sgd/without_bumps/{new_config_name}").expanduser().resolve()

    without_bumps_experiment_name = args.without_bumps_experiment_name or None
    without_bumps_config_name = args.without_bumps_config_name or None

    print(f"Base dir:            {base_dir}")
    print(f"Config name:         {args.config_name}")
    print(f"Widths:              {sorted(args.widths)}")
    print(f"Periods:             {sorted(args.periods)}")
    print(f"Without-bumps dir:   {without_bumps_dir}")
    print(f"Output dir:          {output_dir}")
    print("-" * 90)

    result = build_grids(
        base_dir=base_dir,
        config_name=args.config_name,
        widths=args.widths,
        periods=args.periods,
        experiment_name=args.experiment_name,
        csv_name=args.csv_name,
        accuracy_csv_name=args.accuracy_csv_name,
        without_bumps_dir=without_bumps_dir,
        without_bumps_config_name=without_bumps_config_name,
        without_bumps_experiment_name=without_bumps_experiment_name,
    )

    print(
        f"\nConfigs found: {result.n_found}/{result.n_total}   "
        f"missing: {result.n_total - result.n_found}/{result.n_total}"
    )

    if result.missing:
        print("\nMissing / unresolved runs:")
        for line in result.missing:
            print(f"  - {line}")
        missing_path = output_dir
        missing_path.mkdir(parents=True, exist_ok=True)
        with open(missing_path / "missing_runs.txt", "w", encoding="utf-8") as f:
            f.write(
                f"Configs found: {result.n_found}/{result.n_total}   "
                f"missing: {result.n_total - result.n_found}/{result.n_total}\n\n"
            )
            f.write("\n".join(result.missing) + "\n")

    annotate = not args.no_annotate
    save_individual_heatmaps(result, output_dir, annotate=annotate)
    save_overview(result, output_dir, annotate=annotate)
    save_summary_csv(result, output_dir)

    print(f"\nDone. Gridmaps written under {output_dir}")


if __name__ == "__main__":
    main()