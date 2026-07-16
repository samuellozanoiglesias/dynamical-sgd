"""
USAGE:
    python reobtain_plots.py /path/to/run_or_parent

EXAMPLE:
    nohup python reobtain_plots.py /data/samuel_lozano/dynamical-sgd/without_bumps/glorot_50-no_bumps_LONG/training_2026_05_18-16_36_44/ > reobtain.log 2>&1 &
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


CLASS_ID_RE = re.compile(r"class_(\d+)")
GEO_CLASS_ID_RE = re.compile(r"cyl_(?:half_length|radius)_class_(\d+)")
AXIS_INDEX_RE = re.compile(r"cyl_axis_index_class_(\d+)")
GEO_PAIR_RE = re.compile(r"^cyl_overlap_(\d+)v(\d+)$")
GEO_PAIR_FALLBACK_RE = re.compile(r"^ellipsoid_overlap_(\d+)v(\d+)$")
EPS = 1e-140


def _read_csv_header(csv_path: Path) -> list[str]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
    if not header:
        raise ValueError(f"CSV header missing in {csv_path}")
    return header


def _load_csv_rows(csv_path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
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


def _infer_num_classes_from_classifier_header(header: Iterable[str]) -> int:
    class_ids: list[int] = []
    for name in header:
        match = CLASS_ID_RE.search(name)
        if match:
            class_ids.append(int(match.group(1)))
    if not class_ids:
        raise ValueError("No class columns found in classifier metrics CSV.")
    return max(class_ids) + 1


def _infer_num_classes_from_geo_header(header: Iterable[str]) -> int:
    class_ids: list[int] = []
    for name in header:
        match = GEO_CLASS_ID_RE.search(name)
        if match:
            class_ids.append(int(match.group(1)))
    if not class_ids:
        for name in header:
            match = CLASS_ID_RE.search(name)
            if match:
                class_ids.append(int(match.group(1)))
    if not class_ids:
        raise ValueError("No class columns found in PCA geometric CSV.")
    return max(class_ids) + 1


def _has_axis_index(header: Iterable[str]) -> bool:
    for name in header:
        if AXIS_INDEX_RE.search(name):
            return True
    return False


def _infer_class_pairs_from_geo_header(header: Iterable[str]) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for name in header:
        match = GEO_PAIR_RE.match(name)
        if match:
            pairs.append((int(match.group(1)), int(match.group(2))))
    if not pairs:
        for name in header:
            match = GEO_PAIR_FALLBACK_RE.match(name)
            if match:
                pairs.append((int(match.group(1)), int(match.group(2))))
    if not pairs:
        raise ValueError("No class pair columns found in PCA geometric CSV.")
    return pairs


def _infer_tpt_step(run_dir: Path) -> int:
    metrics_path = run_dir / "training_metrics.csv"
    if not metrics_path.is_file():
        return -1

    tpt_step = -1
    tpt_found = False
    with metrics_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_reached = row.get("tpt_reached")
            if raw_reached is None:
                continue
            try:
                reached = int(float(raw_reached))
            except ValueError:
                continue
            if reached:
                tpt_found = True
                raw_step = row.get("tpt_step", "-1")
                try:
                    tpt_step = int(float(raw_step))
                except ValueError:
                    tpt_step = -1
    return tpt_step if tpt_found else -1


def _plot_reobtained_classifier_metrics(
    csv_path: Path,
    output_path: Path,
    num_classes: int,
    tpt_step: int = -1,
) -> None:
    rows = _load_csv_rows(csv_path)
    if not rows:
        raise ValueError(f"No classifier rows found in {csv_path}")

    steps = np.asarray([int(row["global_step"]) for row in rows], dtype=np.int64)
    correct_logits = np.asarray(
        [[row.get(f"logit_correct_mean_class_{class_id}", float("nan")) for class_id in range(num_classes)] for row in rows],
        dtype=np.float64,
    )
    max_wrong_logits = np.asarray(
        [[row.get(f"logit_max_wrong_mean_class_{class_id}", float("nan")) for class_id in range(num_classes)] for row in rows],
        dtype=np.float64,
    )
    path_curvature_ratio = np.asarray(
        [row.get("path_curvature_ratio", float("nan")) for row in rows],
        dtype=np.float64,
    )
    weight_norms = np.asarray(
        [[row.get(f"weight_norm_class_{class_id}", float("nan")) for class_id in range(num_classes)] for row in rows],
        dtype=np.float64,
    )
    condition_number = np.asarray(
        [row.get("condition_number", float("nan")) for row in rows],
        dtype=np.float64,
    )
    
    # New Extractions for Sensitive Params and Weight Distance
    sensitive_param_fraction = np.asarray(
        [row.get("sensitive_param_fraction", float("nan")) for row in rows],
        dtype=np.float64,
    )
    mean_weight_step_distance = np.asarray(
        [row.get("mean_weight_step_distance", float("nan")) for row in rows],
        dtype=np.float64,
    )

    fig, axes = plt.subplots(3, 2, figsize=(16, 18), sharex=True)

    ax = axes[0, 0]
    for class_id in range(num_classes):
        color = None
        if num_classes <= 10:
            color = plt.get_cmap("tab10")(class_id % 10)
        ax.plot(steps, correct_logits[:, class_id], linewidth=1.6, color=color, label=f"class {class_id} correct")
        ax.plot(
            steps,
            max_wrong_logits[:, class_id],
            linewidth=1.2,
            linestyle="--",
            alpha=0.8,
            color=color,
            label=f"class {class_id} max wrong",
        )
    ax.set_title("Logit Decomposition")
    ax.set_ylabel("Logit")
    ax.set_ylim(-13000.0, 25.0)
    ax.grid(True, alpha=0.3)
    if num_classes <= 6:
        ax.legend(fontsize=7, ncol=2)

    ax = axes[0, 1]
    ax.plot(steps, path_curvature_ratio, linewidth=1.8)
    ax.set_title("Path Curvature Ratio")
    ax.set_ylabel("cumulative / ||W - W0||")
    ax.set_ylim(0.0, 25.0)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    for class_id in range(num_classes):
        ax.plot(steps, weight_norms[:, class_id], linewidth=1.6, label=f"class {class_id}")
    ax.set_title("Classifier Weight Norms")
    ax.set_xlabel("Global Step")
    ax.set_ylabel("L2 norm")
    ax.set_ylim(0.0, 300.0)
    ax.grid(True, alpha=0.3)
    if num_classes <= 12:
        ax.legend(fontsize=7, ncol=2)

    ax = axes[1, 1]
    ax.plot(steps, condition_number, linewidth=1.8, label="condition number")
    ax.set_title("Classifier Condition Number")
    ax.set_xlabel("Global Step")
    ax.set_ylabel("kappa(W)")
    ax.set_ylim(0.0, 15.0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Plot Sensitive Parameter Fraction
    ax = axes[2, 0]
    ax.plot(steps, sensitive_param_fraction, linewidth=1.8, color="tab:red")
    ax.set_title("Sensitive Parameter Fraction")
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Fraction with |Δw| > threshold")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)

    # Plot Mean Weight Step Distance
    ax = axes[2, 1]
    ax.plot(steps, mean_weight_step_distance, linewidth=1.8, color="tab:purple")
    ax.set_title("Mean Weight Step Distance")
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Mean |Δw| over all parameters")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    if tpt_step >= 0:
        for axis in axes.flat:
            axis.axvline(tpt_step, color="black", linestyle="-", linewidth=2.0)

    fig.suptitle("Classifier Metrics", fontsize=16)
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_reobtained_logit_margin(
    csv_path: Path,
    output_path: Path,
    num_classes: int,
    tpt_step: int = -1,
) -> None:
    rows = _load_csv_rows(csv_path)
    if not rows:
        raise ValueError(f"No classifier rows found in {csv_path}")

    steps = np.asarray([int(row["global_step"]) for row in rows], dtype=np.int64)
    margin_mean = np.asarray(
        [[row.get(f"logit_margin_mean_class_{class_id}", float("nan")) for class_id in range(num_classes)] for row in rows],
        dtype=np.float64,
    )
    margin_var = np.asarray(
        [[row.get(f"logit_margin_var_class_{class_id}", float("nan")) for class_id in range(num_classes)] for row in rows],
        dtype=np.float64,
    )
    margin_std = np.sqrt(np.clip(margin_var, 0.0, None))

    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    for class_id in range(num_classes):
        color = None
        if num_classes <= 10:
            color = plt.get_cmap("tab10")(class_id % 10)
        ax.plot(
            steps,
            margin_mean[:, class_id],
            linewidth=1.8,
            color=color,
            label=f"class {class_id}",
        )
        ax.fill_between(
            steps,
            margin_mean[:, class_id] - margin_std[:, class_id],
            margin_mean[:, class_id] + margin_std[:, class_id],
            color=color,
            alpha=0.18,
            linewidth=0.0,
        )

    ax.set_title("Logit Margin (Mean ± sqrt(var))")
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Logit Margin")
    ax.set_ylim(-10.0, 35.0)
    ax.grid(True, alpha=0.3)
    if num_classes <= 12:
        ax.legend(fontsize=8, ncol=2)

    if tpt_step >= 0:
        ax.axvline(tpt_step, color="black", linestyle="-", linewidth=2.0)

    fig.suptitle("Logit Margin by Class", fontsize=16)
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_reobtained_geo_metrics(
    csv_path: Path,
    output_path: Path,
    num_classes: int,
    class_pairs: list[tuple[int, int]],
    tpt_step: int = -1,
) -> None:
    rows = _load_csv_rows(csv_path)
    if not rows:
        raise ValueError(f"No geometric overlap rows found in {csv_path}")

    steps = np.asarray([int(row["global_step"]) for row in rows], dtype=np.int64)
    cyl_overlap_by_pair = np.asarray(
        [[row.get(f"cyl_overlap_{left}v{right}", float("nan")) for left, right in class_pairs] for row in rows],
        dtype=np.float64,
    )
    ellipsoid_bhattacharyya_by_pair = np.asarray(
        [[row.get(f"ellipsoid_bhattacharyya_{left}v{right}", float("nan")) for left, right in class_pairs] for row in rows],
        dtype=np.float64,
    )
    ellipsoid_overlap_by_pair = np.asarray(
        [[row.get(f"ellipsoid_overlap_{left}v{right}", float("nan")) for left, right in class_pairs] for row in rows],
        dtype=np.float64,
    )
    cyl_half_length_by_class = np.asarray(
        [[row.get(f"cyl_half_length_class_{class_id}", float("nan")) for class_id in range(num_classes)] for row in rows],
        dtype=np.float64,
    )
    cyl_radius_by_class = np.asarray(
        [[row.get(f"cyl_radius_class_{class_id}", float("nan")) for class_id in range(num_classes)] for row in rows],
        dtype=np.float64,
    )

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
    ax.set_ylim(1e-80, 1e-1)
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
    ax.set_ylim(0.0, 185.0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

    if tpt_step >= 0:
        for axis in axes.flat:
            axis.axvline(tpt_step, color="black", linestyle="-", linewidth=2.0)

    fig.suptitle("PCA Geometric Overlap Metrics", fontsize=16)
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_reobtained_geo_axis_index(
    csv_path: Path,
    output_path: Path,
    num_classes: int,
    tpt_step: int = -1,
) -> None:
    rows = _load_csv_rows(csv_path)
    if not rows:
        raise ValueError(f"No geometric overlap rows found in {csv_path}")

    steps = np.asarray([int(row["global_step"]) for row in rows], dtype=np.int64)
    axis_index_by_class = np.asarray(
        [[row.get(f"cyl_axis_index_class_{class_id}", float("nan")) for class_id in range(num_classes)] for row in rows],
        dtype=np.float64,
    )

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    for class_id in range(num_classes):
        ax.plot(steps, axis_index_by_class[:, class_id], linewidth=1.6, label=f"class {class_id}")
    ax.set_title("Cylinder Principal Axis Index over Training")
    ax.set_xlabel("Global Step")
    ax.set_ylabel("Axis Index (1-based)")
    ax.grid(True, alpha=0.3)
    if num_classes <= 12:
        ax.legend(fontsize=8, ncol=2)

    if tpt_step >= 0:
        ax.axvline(tpt_step, color="black", linestyle="-", linewidth=2.0)

    fig.suptitle("PCA Geometric Axis Index", fontsize=16)
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _find_run_dirs(root: Path) -> list[Path]:
    if root.is_file():
        return [root.parent]
    if not root.exists():
        return []
    if root.is_dir():
        if root.name.lower().startswith("training_"):
            return [root]

        training_dirs: set[Path] = set()
        for pattern in ("training_*", "Training_*"):
            for path in root.rglob(pattern):
                if path.is_dir():
                    training_dirs.add(path)
        if training_dirs:
            return sorted(training_dirs, key=lambda p: str(p))

        local_classifier = (root / "classifier_metrics.csv").is_file()
        local_geo = (root / "PCA_geometric.csv").is_file()
        if local_classifier or local_geo:
            return [root]
        candidates: set[Path] = set()
        for name in ("classifier_metrics.csv", "PCA_geometric.csv"):
            for csv_path in root.rglob(name):
                candidates.add(csv_path.parent)
        return sorted(candidates, key=lambda p: str(p))
    return []


def reobtain_plots(run_dir: Path) -> list[Path]:
    outputs: list[Path] = []
    tpt_step = _infer_tpt_step(run_dir)

    classifier_csv = run_dir / "classifier_metrics.csv"
    if classifier_csv.is_file():
        header = _read_csv_header(classifier_csv)
        num_classes = _infer_num_classes_from_classifier_header(header)
        output_path = run_dir / "REOBTAINED_classifier_metrics.png"
        _plot_reobtained_classifier_metrics(
            csv_path=classifier_csv,
            output_path=output_path,
            num_classes=num_classes,
            tpt_step=tpt_step,
        )
        outputs.append(output_path)
        logit_output_path = run_dir / "REOBTAINED_logit.png"
        _plot_reobtained_logit_margin(
            csv_path=classifier_csv,
            output_path=logit_output_path,
            num_classes=num_classes,
            tpt_step=tpt_step,
        )
        outputs.append(logit_output_path)
    else:
        print(f"[reobtain] Missing classifier_metrics.csv in {run_dir}")

    geo_csv = run_dir / "PCA_geometric.csv"
    if geo_csv.is_file():
        header = _read_csv_header(geo_csv)
        num_classes = _infer_num_classes_from_geo_header(header)
        class_pairs = _infer_class_pairs_from_geo_header(header)
        output_path = run_dir / "REOBTAINED_PCA_geometric_overlapping.png"
        _plot_reobtained_geo_metrics(
            csv_path=geo_csv,
            output_path=output_path,
            num_classes=num_classes,
            class_pairs=class_pairs,
            tpt_step=tpt_step,
        )
        outputs.append(output_path)
        if _has_axis_index(header):
            axis_output_path = run_dir / "REOBTAINED_cylinder_length_index.png"
            _plot_reobtained_geo_axis_index(
                csv_path=geo_csv,
                output_path=axis_output_path,
                num_classes=num_classes,
                tpt_step=tpt_step,
            )
            outputs.append(axis_output_path)
    else:
        print(f"[reobtain] Missing PCA_geometric.csv in {run_dir}")

    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reobtain classifier and PCA geometric plots from CSVs.",
    )
    parser.add_argument(
        "path",
        help="Run directory (or parent directory to scan) containing the CSVs.",
    )
    args = parser.parse_args(argv)

    root = Path(args.path).expanduser()
    run_dirs = _find_run_dirs(root)
    if not run_dirs:
        print(f"No run directories with CSVs found under {root}", file=sys.stderr)
        return 1

    exit_code = 0
    for run_dir in run_dirs:
        try:
            outputs = reobtain_plots(run_dir)
        except Exception as exc:
            print(f"[reobtain] Failed for {run_dir}: {exc}", file=sys.stderr)
            exit_code = 1
            continue
        if outputs:
            output_list = ", ".join(str(path) for path in outputs)
            print(f"[reobtain] Saved {output_list}")
        else:
            print(f"[reobtain] No plots generated for {run_dir}")
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())