"""
Plot summary comparison of original vs optimized parameters.

Reads parameter_optimization_results.csv and produces comparison plots
for cd, cr, and eta across all trucks.

Usage:
    python source/plot_parameter_summary.py
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

RESULTS_PATH = Path("parameter_optimization_results.csv")
OUTPUT_DIR = Path("plots_messy")
OUTPUT_NAME = "parameter_summary_comparison.png"


def plot_parameter_comparison(df):
    trucks = df["Truck"].tolist()
    x = np.arange(len(trucks))
    width = 0.35

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    # Drag coefficient (cd)
    axes[0].bar(x - width / 2, df["cd_initial"], width=width, label="Original", edgecolor="C0", facecolor="none", linewidth=2)
    axes[0].bar(x + width / 2, df["cd_optimal"], width=width, label="Optimized", edgecolor="C1", facecolor="none", linewidth=2)
    axes[0].set_ylabel("Drag Coefficient (cd)")
    axes[0].set_title("Original vs Optimized Parameters", fontweight="bold")
    axes[0].legend(loc="upper right")
    axes[0].grid(axis="y", alpha=0.3)

    # Rolling resistance (cr)
    axes[1].bar(x - width / 2, df["cr_initial"], width=width, label="Original", edgecolor="C0", facecolor="none", linewidth=2)
    axes[1].bar(x + width / 2, df["cr_optimal"], width=width, label="Optimized", edgecolor="C1", facecolor="none", linewidth=2)
    axes[1].set_ylabel("Rolling Resistance (cr)")
    axes[1].grid(axis="y", alpha=0.3)

    # Combined efficiency (eta)
    axes[2].bar(x - width / 2, df["eta_initial"], width=width, label="Original", edgecolor="C0", facecolor="none", linewidth=2)
    axes[2].bar(x + width / 2, df["eta_optimal"], width=width, label="Optimized", edgecolor="C1", facecolor="none", linewidth=2)
    axes[2].set_ylabel("Inverter×Motor Efficiency (η)")
    axes[2].set_xlabel("Truck")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(trucks)
    axes[2].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    return fig


def main():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"Results file not found: {RESULTS_PATH}")

    df = pd.read_csv(RESULTS_PATH)
    fig = plot_parameter_comparison(df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / OUTPUT_NAME
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved parameter comparison plot to {output_path}")


if __name__ == "__main__":
    main()
