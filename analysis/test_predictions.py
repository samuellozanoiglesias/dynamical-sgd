#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

# ============================================================
# Configuration
# ============================================================

ROOT = Path(
    "/mnt/lustre/home/samuloza/data/samuel_lozano/dynamical-sgd/gridsearch_wmaxs_periods"
)

W_MAXS = [5, 10, 20, 50, 70, 100, 150, 250]
PERIODS = [5, 10, 20, 50, 100, 200, 400, 800, 1000, 1200, 1500, 1800, 2000]

# Change if desired
K = 10.0

OUTPUT_DOSE = "accuracy_vs_dose.png"
OUTPUT_BURST = "accuracy_vs_burst_length.png"

# ============================================================
# Dose formula
# ============================================================

def dose(w_max, T, K):
    w_max = np.asarray(w_max, dtype=np.float64)
    eps_bar = (
        1.0
        - (K / (w_max - 1.0))
        * np.log1p((w_max - 1.0) / K)
    )
    return T * eps_bar

def wmax_critical(batch_size, K):
    return batch_size - K + 1

def danger_fraction(w_max, batch_size, K):
    if w_max <= wmax_critical(batch_size, K):
        return 0.0
    r = (batch_size - K) / (2.0 * (w_max - 1))
    return max(0.0, 1.0 - 2.0 * r)

def burst_length(w_max, T, batch_size, K):
    return T * danger_fraction(w_max, batch_size, K)

# ============================================================
# Helpers
# ============================================================

def find_training_csv(base_dir):
    """
    Returns the csv inside the first training_* directory.
    """
    training_dirs = sorted(base_dir.glob("training_*"))

    if len(training_dirs) == 0:
        return None

    csv = training_dirs[0] / "training_metrics.csv"

    if csv.exists():
        return csv

    return None


def extract_final_test_accuracy(csv_file):
    """
    Reads the csv and returns the final test accuracy.
    """
    df = pd.read_csv(csv_file)

    candidates = [
        "test_accuracy",
        "test_acc",
        "accuracy_test",
        "eval_accuracy",
        "eval_acc",
        "accuracy",
    ]

    for c in candidates:
        if c in df.columns:
            return float(df[c].iloc[-1])

    raise RuntimeError(
        f"No test accuracy column found in {csv_file}\nColumns:\n{list(df.columns)}"
    )


# ============================================================
# Main
# ============================================================

records = []

for w in W_MAXS:

    for p in PERIODS:

        exp_dir = (
            ROOT
            / f"w{w}_p{p}"
            / "with_bumps"
            / "cifar10_resnet_narrow-always_bumps"
        )

        csv = find_training_csv(exp_dir)

        if csv is None:
            print(f"Missing: {exp_dir}")
            continue

        try:
            acc = extract_final_test_accuracy(csv)

            D = dose(w, p, K)
            B = burst_length(w, p, batch_size=128, K=10)

            records.append(
                {
                    "width": w,
                    "period": p,
                    "dose": D,
                    "burst_length": B,
                    "accuracy": acc,
                }
            )

        except Exception as e:
            print(csv)
            print(e)

df = pd.DataFrame(records)

print(df)

# ============================================================
# Plot
# ============================================================

fig, ax = plt.subplots(figsize=(8, 6))

sc = ax.scatter(
    df["dose"],
    df["accuracy"],
    c=df["width"],
    cmap="viridis",
    s=70,
)

for _, r in df.iterrows():
    ax.text(
        r["dose"],
        r["accuracy"],
        f"({int(r['width'])},{int(r['period'])})",
        fontsize=7,
        alpha=0.7,
    )

cbar = plt.colorbar(sc)
cbar.set_label("W_max")

ax.set_xlabel("Dose")
ax.set_ylabel("Final test accuracy")
ax.set_title(f"Accuracy vs Dose (K={K})")

plt.tight_layout()
plt.savefig(OUTPUT_DOSE, dpi=300)
plt.close()

print(f"\nSaved figure to {OUTPUT_DOSE}")

fig, ax = plt.subplots(figsize=(8, 6))

sc = ax.scatter(
    df["burst_length"],
    df["accuracy"],
    c=df["width"],
    cmap="viridis",
    s=70,
)

for _, r in df.iterrows():
    ax.text(
        r["burst_length"],
        r["accuracy"],
        f"({int(r['width'])},{int(r['period'])})",
        fontsize=7,
        alpha=0.7,
    )

cbar = plt.colorbar(sc)
cbar.set_label("W_max")

ax.set_xlabel("Burst Length")
ax.set_ylabel("Final test accuracy")
ax.set_title(f"Accuracy vs Burst Length (K={K})")

plt.tight_layout()
plt.savefig(OUTPUT_BURST, dpi=300)
plt.close()

print(f"\nSaved figure to {OUTPUT_BURST}")