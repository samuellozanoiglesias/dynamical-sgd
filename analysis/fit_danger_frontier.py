#!/usr/bin/env python3
"""
fit_danger_frontier.py
=======================
Fits the "amplitude ceiling + log-period resonance" model to the
test_accuracy gridmap of a w_max x period_length sweep, and draws the
predicted danger frontier T0(w_max) (plus a +/-1 sigma risk band) on top of
the test_accuracy heatmap.

Unlike a hand-copied grid of numbers, T_list / w_list / grid are obtained
straight from the experiment directories via grid_analysis.build_grids(),
so this script is meant to be run with (a subset of) the exact same CLI
arguments you already pass to grid_analysis.py for the same sweep.

The model
---------
    eps_max(w, c) = (w - 1) / (w + c - 1)

    T0(w)     = T0base * w ** p                        <- danger-frontier line
    A_ceil(w) = A0 - beta * eps_max(w, c)               <- slow amplitude-only decay
    Delta(w)  = delta0 * eps_max(w, c)                  <- depth of the dip
    L(T, w)   = exp( -(ln T - ln T0(w))**2 / (2*sigma**2) )   <- resonance shape

    Acc(T, w) = A_ceil(w) - Delta(w) * L(T, w)

Fit is least-squares over all (T, w_max) cells with w_max > 1 and a
non-NaN test_accuracy (the w_max=1 / without_bumps column is a fixed
control, not part of what the model explains, so it's excluded from the
fit the same way the original hand-written script excluded it).

Usage
-----
nohup python fit_danger_frontier.py > log_fit_danger_frontier.out 2>&1 &
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# Reuse grid_analysis.py's directory-discovery / grid-building code instead
# of re-implementing (and risking drifting out of sync with) it. Assumes
# grid_analysis.py lives next to this file; use --grid-analysis-path to
# point elsewhere.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import grid_analysis as ga


# ---------------------------------------------------------------------------
# 1. Model
# ---------------------------------------------------------------------------

PARAM_NAMES = ["A0", "beta", "c", "delta0", "p", "T0base", "sigma"]
PARAM_X0 = [0.80, 0.08, 10.0, 0.65, 0.5, 5.0, 1.0]
PARAM_LB = [0.50, 0.00, 1.0, 0.00, 0.0, 0.5, 0.2]
PARAM_UB = [0.85, 0.30, 300.0, 1.00, 1.5, None, 3.0]  # T0base upper bound set at runtime


def eps_max(w: np.ndarray, c: float) -> np.ndarray:
    return (w - 1.0) / (w + c - 1.0)


def model(params: np.ndarray, T: np.ndarray, w: np.ndarray) -> np.ndarray:
    A0, beta, c, delta0, p, T0base, sigma = params
    e = eps_max(w, c)
    A_ceiling = A0 - beta * e
    Delta = delta0 * e
    T0 = T0base * np.power(w, p)
    L = np.exp(-(np.log(T) - np.log(T0)) ** 2 / (2 * sigma ** 2))
    return A_ceiling - Delta * L


def fit_model(T_flat: np.ndarray, w_flat: np.ndarray, acc_flat: np.ndarray):
    def resid(params):
        return model(params, T_flat, w_flat) - acc_flat

    ub = list(PARAM_UB)
    ub[5] = max(50.0, float(T_flat.max()) * 2)  # let T0base range far enough for wide sweeps

    res = least_squares(resid, PARAM_X0, bounds=(PARAM_LB, ub), max_nfev=20000)
    pred = model(res.x, T_flat, w_flat)
    rmse = float(np.sqrt(np.mean((pred - acc_flat) ** 2)))
    ss_res = float(np.sum((pred - acc_flat) ** 2))
    ss_tot = float(np.sum((acc_flat - acc_flat.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return res, rmse, r2


# ---------------------------------------------------------------------------
# 2. Data <-> pixel mapping for the categorical imshow axes
#
# grid_analysis.py's heatmaps use imshow on a purely categorical grid (each
# tick is one index, regardless of how far apart the real T / w_max values
# are). To draw a *continuous* curve like T0(w_max) on top of that grid we
# have to run it through the same index mapping, interpolating between
# known ticks and linearly extrapolating beyond the swept range.
# ---------------------------------------------------------------------------

def interp_extrap(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """np.interp, but linearly extrapolated (using the boundary segment's
    slope) instead of clamped outside [xp[0], xp[-1]]."""
    x = np.asarray(x, dtype=float)
    y = np.interp(x, xp, fp)

    below = x < xp[0]
    if np.any(below):
        slope = (fp[1] - fp[0]) / (xp[1] - xp[0])
        y[below] = fp[0] + slope * (x[below] - xp[0])

    above = x > xp[-1]
    if np.any(above):
        slope = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
        y[above] = fp[-1] + slope * (x[above] - xp[-1])

    return y


# ---------------------------------------------------------------------------
# 3. Plotting
# ---------------------------------------------------------------------------

def plot_frontier(
    grid: np.ndarray,
    periods: list[int],
    widths_with_control: list[float],  # [1, w1, w2, ...] matching grid columns
    x_labels: list[str],
    fit_params: np.ndarray,
    output_path: Path,
    n_sigma_band: float = 1.0,
) -> None:
    periods_arr = np.array(periods, dtype=float)
    real_widths = np.array(widths_with_control[1:], dtype=float)  # drop the w=1 control

    # pixel-index lookup tables (grid_analysis.py convention: imshow origin
    # is "lower", row index i <-> periods[i] ascending, col index j <-> the
    # j-th entry of x_labels, where col 0 is the w=1 control column)
    row_ticks = np.arange(len(periods))
    col_ticks = np.arange(1, len(widths_with_control))  # skip the w=1 control column

    A0, beta, c, delta0, p, T0base, sigma = fit_params

    # dense w range spanning (a bit beyond) the swept widths, in log-space
    w_dense = np.geomspace(real_widths.min() * 0.8, real_widths.max() * 1.2, 200)
    T0_dense = T0base * np.power(w_dense, p)
    T_lo = T0_dense * np.exp(-n_sigma_band * sigma)
    T_hi = T0_dense * np.exp(n_sigma_band * sigma)

    x_pix = interp_extrap(w_dense, real_widths, col_ticks)
    y_center = interp_extrap(T0_dense, periods_arr, row_ticks)
    y_lo = interp_extrap(T_lo, periods_arr, row_ticks)
    y_hi = interp_extrap(T_hi, periods_arr, row_ticks)

    fig, ax = plt.subplots(
        figsize=(1.0 + 0.8 * len(x_labels), 1.0 + 0.6 * len(periods))
    )
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_yticks(range(len(periods)))
    ax.set_yticklabels([str(p) for p in periods], fontsize=8)
    ax.axvline(0.5, color="white", linewidth=1.5, linestyle="--")
    ax.set_xlabel("w_max (bump width)")
    ax.set_ylabel("period_length (T)")
    ax.set_title("test_accuracy  +  fitted danger frontier", fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.85, label="test_accuracy")

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            val = grid[i, j]
            if np.isnan(val):
                continue
            ax.text(j, i, f"{val:.2g}", ha="center", va="center", fontsize=6, color="white")

    ax.fill_between(x_pix, y_lo, y_hi, color="red", alpha=0.15,
                     label=f"predicted danger band (+/-{n_sigma_band:g} sigma)")
    ax.plot(x_pix, y_center, color="red", linewidth=2,
             label=r"predicted worst-T:  $T_0(w)=%.3g \cdot w^{%.2f}$" % (T0base, p))

    ax.set_xlim(-0.5, len(x_labels) - 0.5)
    ax.set_ylim(-0.5, len(periods) - 0.5)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.85)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit the resonance/adiabatic model to a test_accuracy sweep "
                     "and plot the predicted danger frontier on the gridmap."
    )
    parser.add_argument("--base_dir", type=str, default="/mnt/lustre/home/samuloza/data/samuel_lozano/dynamical-sgd/gridsearch_wmaxs_periods/",
                         help="BASE_OUTPUT_DIR used by the sbatch script (contains w{W}_p{P} subfolders).")
    parser.add_argument("--config_name", type=str, default="cifar10_resnet_narrow-always_bumps",
                         help="Stem of the config file used for the sweep (output.config_name).")
    parser.add_argument("--wmaxs", type=int, nargs="+", default=[5, 10, 20, 50, 70, 100, 120, 150, 180, 250, 300, 350, 400, 450, 500],
                         help="dynamics.w_max values that were swept (excluding the without_bumps control).")
    parser.add_argument("--periods", type=int, nargs="+", default=[5, 10, 20, 50, 100, 200, 400, 800, 1000, 1200, 1500, 1800, 2000],
                         help="dynamics.period_length values that were swept.")
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--without_bumps_dir", type=str, default="/mnt/lustre/home/samuloza/data/samuel_lozano/dynamical-sgd/without_bumps/cifar10_resnet_narrow-no_bumps",
                         help="output.output_dir of the separate without_bumps run "
                              "(used only as a reference column, excluded from the fit).")
    parser.add_argument("--without_bumps_experiment_name", type=str, default=None)
    parser.add_argument("--without_bumps_config_name", type=str, default=None)
    parser.add_argument("--csv_name", type=str, default=None)
    parser.add_argument("--accuracy_csv_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None,
                         help="Where to write danger_frontier.png and fitted_params.txt. "
                              "Defaults to <base-dir>/grid_analysis.")
    parser.add_argument("--n_sigma_band", type=float, default=1.0,
                         help="Width (in ln-T sigmas) of the shaded risk band around T0(w). Default 1.0.")
    parser.add_argument("--grid-analysis-path", type=str, default=None,
                         help="Directory containing grid_analysis.py, if not alongside this script.")
    args = parser.parse_args()

    if args.grid_analysis_path:
        sys.path.insert(0, str(Path(args.grid_analysis_path).expanduser().resolve()))
        global ga
        import grid_analysis as ga  # noqa: F811  (re-import from the requested path)

    base_dir = Path(args.base_dir).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve() if args.output_dir else base_dir / "grid_analysis"
    )
    without_bumps_dir = (
        Path(args.without_bumps_dir).expanduser().resolve() if args.without_bumps_dir else None
    )

    print(f"Base dir:          {base_dir}")
    print(f"Config name:       {args.config_name}")
    print(f"Widths swept:      {sorted(args.wmaxs)}")
    print(f"Periods swept:     {sorted(args.periods)}")
    print(f"Without-bumps dir: {without_bumps_dir}")
    print(f"Output dir:        {output_dir}")
    print("-" * 90)

    # ---- pull T_list, w_list, grid straight from the experiment tree ----
    result = ga.build_grids(
        base_dir=base_dir,
        config_name=args.config_name,
        widths=args.wmaxs,
        periods=args.periods,
        experiment_name=args.experiment_name,
        csv_name=args.csv_name,
        accuracy_csv_name=args.accuracy_csv_name,
        without_bumps_dir=without_bumps_dir,
        without_bumps_config_name=args.without_bumps_config_name,
        without_bumps_experiment_name=args.without_bumps_experiment_name,
    )

    print(f"\nConfigs found: {result.n_found}/{result.n_total}")
    if result.missing:
        print(f"({len(result.missing)} missing/unresolved run(s), see below)")
        for line in result.missing:
            print(f"  - {line}")

    grid = result.grids["test_accuracy"]  # shape (n_periods, n_widths + 1), col 0 = w=1 control
    T_list = result.periods               # ascending
    w_list_full = [1] + result.widths     # ascending, includes the w=1 control at index 0

    if np.all(np.isnan(grid)):
        raise SystemExit("No test_accuracy values found anywhere in the sweep - check the paths/args above.")

    # ---- flatten for fitting, dropping the w=1 control column and NaNs ----
    Ts = np.array(T_list, dtype=float)
    Ws = np.array(w_list_full, dtype=float)
    TT, WW = np.meshgrid(Ts, Ws, indexing="ij")
    TT = TT.ravel()
    WW = WW.ravel()
    ACC = grid.ravel()

    mask = (WW > 1) & ~np.isnan(ACC)
    n_dropped_nan = int(((WW > 1) & np.isnan(ACC)).sum())
    if n_dropped_nan:
        print(f"\n[note] {n_dropped_nan} swept (T, w_max) cell(s) have no test_accuracy "
              "(missing run or missing CSV) and were excluded from the fit.")

    if mask.sum() < len(PARAM_NAMES):
        raise SystemExit(
            f"Only {mask.sum()} usable (T, w_max) points found - not enough to fit "
            f"a {len(PARAM_NAMES)}-parameter model. Check --wmaxs / --periods / paths."
        )

    res, rmse, r2 = fit_model(TT[mask], WW[mask], ACC[mask])

    print("\nFitted parameters:")
    for name, value in zip(PARAM_NAMES, res.x):
        print(f"  {name:10s} = {value:.4f}")
    print(f"\nRMSE = {rmse:.4f}   R^2 = {r2:.4f}   (accuracy units, 0-1 scale)")
    p_fit = res.x[PARAM_NAMES.index("p")]
    print(f"\n==> Danger-frontier exponent p = {p_fit:.3f}  (T0(w_max) ~ w_max^p)")
    print("    p=1 -> linear/diagonal,  p=0.5 -> sqrt,  p->0 -> log-like")

    A0, beta, c, delta0, p, T0base, sigma = res.x
    print("\nw_max -> fitted resonance period T0(w) [steps]  (predicted worst-T)")
    for w in result.widths:
        T0 = T0base * (w ** p)
        print(f"  w_max={w:5d}  ->  T0 = {T0:8.1f}")

    output_dir.mkdir(parents=True, exist_ok=True)

    params_path = output_dir / "danger_frontier_fit.txt"
    with open(params_path, "w", encoding="utf-8") as f:
        f.write(f"RMSE={rmse:.6f}  R2={r2:.6f}\n")
        for name, value in zip(PARAM_NAMES, res.x):
            f.write(f"{name}={value:.6f}\n")
    print(f"\nFit report written to {params_path}")

    plot_path = output_dir / "test_accuracy_with_danger_frontier.png"
    plot_frontier(
        grid=grid,
        periods=T_list,
        widths_with_control=w_list_full,
        x_labels=result.x_labels,
        fit_params=res.x,
        output_path=plot_path,
        n_sigma_band=args.n_sigma_band,
    )
    print(f"Gridmap with danger frontier written to {plot_path}")


if __name__ == "__main__":
    main()
