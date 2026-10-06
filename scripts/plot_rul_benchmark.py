"""Render measured FD001 benchmark evidence from the saved report."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "docs/benchmarks/cmapss_fd001.json")
    parser.add_argument("--output", type=Path, default=ROOT / "assets/rul_benchmark.png")
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    engines = report["engines"]
    true = np.array([row["true_rul"] for row in engines])
    predicted = np.array([row["prediction"] for row in engines])
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), layout="constrained")
    fig.suptitle("NASA C-MAPSS FD001 | held-out engine RUL evaluation", fontsize=17, weight="bold")
    mae = [report["baseline"]["mae_cycles"], report["test_metrics"]["mae_cycles"]]
    bars = axes[0].bar(["Training median", "LightGBM"], mae, color=["#919da6", "#008075"], width=.55)
    axes[0].bar_label(bars, fmt="%.2f", padding=5)
    axes[0].set(ylabel="MAE (cycles)", ylim=(0, max(mae) * 1.25), title="Baseline comparison")
    axes[1].scatter(true, predicted, s=24, alpha=.75, color="#008075")
    limit = max(true.max(), predicted.max()) + 10
    axes[1].plot([0, limit], [0, limit], "--", color="#ba4a32", linewidth=1.3)
    axes[1].set(xlabel="True RUL (cycles)", ylabel="Predicted RUL (cycles)", title="100 official test endpoints",
                xlim=(0, limit), ylim=(0, limit))
    indices = np.linspace(0, len(engines) - 1, 15, dtype=int)
    endpoints = [engines[i]["history"][-1] for i in indices]
    lower = np.array([row["lower"] for row in endpoints])
    upper = np.array([row["upper"] for row in endpoints])
    axes[2].errorbar(np.arange(15), predicted[indices],
                     yerr=[predicted[indices] - lower, upper - predicted[indices]],
                     fmt="o", color="#008075", alpha=.8, capsize=3, label="Prediction interval")
    axes[2].scatter(np.arange(15), true[indices], marker="x", color="#ba4a32", label="True RUL", zorder=3)
    axes[2].set(xticks=np.arange(15), xticklabels=[engines[i]["unit"] for i in indices],
                xlabel="Test engine ID (uniformly sampled)", ylabel="RUL (cycles)",
                title=f"90% target interval | {report['test_metrics']['interval_coverage']:.0%} coverage")
    axes[2].legend(frameon=False, fontsize=9)
    for axis in axes:
        axis.grid(axis="y", alpha=.15)
        axis.set_axisbelow(True)
    fig.supxlabel("Simulated turbine degradation, not physical AGV data. Cycles are not robot minutes.", fontsize=10, color="#53616b")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, facecolor="white")
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
