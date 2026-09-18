#!/usr/bin/env python3
"""Plot archived matched contrasts; descriptive intervals, no significance stars."""
import argparse
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="Controlled acquisition study")
    args = parser.parse_args()
    source = Path(args.summary)
    raw = gzip.decompress(source.read_bytes()) if source.suffix == ".gz" else source.read_bytes()
    data = json.loads(raw)
    target = Path(args.output)
    if target.exists():
        raise FileExistsError(target)
    mechanisms = data.get("mechanism_contrasts", {})
    fig, axes = plt.subplots(1, 3 if mechanisms else 2,
                             figsize=(15, 4.8) if mechanisms else (10, 4.2), layout="constrained")
    panels = [
        (axes[0], [(name.replace("policy_minus_", "POLICY − ").upper(), row)
                   for name, row in data["acquisition_contrasts"].items()],
         "Acquisition effect on direct loss"),
        (axes[1], [(name.upper(), value["direct_minus_bank"])
                   for name, value in data["deployment_contrasts"].items()
                   if not name.startswith("mechanism_")],
         "Direct minus bank loss"),
    ]
    if mechanisms:
        panels.append((axes[2], [(f"{scope.upper()}: {mode}", contrast)
                                 for scope, modes in mechanisms.items() for mode, contrast in modes.items()],
                       "Frozen-backbone label effect\nAcquired minus original"))
    for ax, entries, title in panels:
        for index, (label, row) in enumerate(entries):
            mean = row["mean_difference"]
            ax.scatter([mean], [index], color="#1f618d", s=35, zorder=3)
            if row["ci_low"] is not None:
                ax.hlines(index, row["ci_low"], row["ci_high"], color="#1f618d", lw=2)
        ax.axvline(0, color="#68737d", lw=1, ls="--")
        ax.set_yticks(range(len(entries)), [label for label, _ in entries], fontsize=9)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Paired loss difference (lower is better)")
        ax.grid(axis="x", alpha=.15)
        ax.spines[["top", "right"]].set_visible(False)
    first = next(iter(data["acquisition_contrasts"].values()))
    fig.suptitle(f"{args.title}\n{first['n_parents']} test parents × {first['n_seeds']} training seeds; "
                 "descriptive 95% crossed-bootstrap intervals", fontsize=12)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target)
    if target.suffix == ".svg":
        target.write_text("\n".join(line.rstrip() for line in target.read_text().splitlines()) + "\n")
    plt.close(fig)


if __name__ == "__main__":
    main()
