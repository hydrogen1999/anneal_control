#!/usr/bin/env python3
"""Plot complete, matched finite-budget curves from a frozen study report."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    report = json.loads(Path(args.report).read_text())
    points = [p for p in report["curves"] if p["quality_queries_pareto"] is not None]
    if not points:
        raise ValueError("no complete common-population curves to plot")
    methods = sorted({p["method"] for p in points})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for method in methods:
        subset = sorted([p for p in points if p["method"] == method], key=lambda p: p["budget"])
        axes[0].plot([p["mean_online_queries"] for p in subset], [p["parent_mean_loss"] for p in subset],
                     marker="o", markersize=3, label=method)
        timed = [p for p in subset if p["parent_mean_online_seconds"] is not None]
        if timed:
            axes[1].plot([p["parent_mean_online_seconds"] for p in timed], [p["parent_mean_loss"] for p in timed],
                         marker="o", markersize=3, label=method)
    axes[0].set_xlabel("Online objective queries")
    axes[1].set_xlabel("Measured online wall time (seconds)")
    for ax in axes:
        ax.set_ylabel("Equal-parent mean loss (lower is better)")
        ax.grid(alpha=.2)
    axes[0].set_xscale("symlog", linthresh=1)
    axes[1].set_xscale("symlog", linthresh=.001)
    axes[1].legend(fontsize=6, loc="best")
    fig.suptitle("Finite-budget control search; descriptive means, fixed checkpoint")
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
