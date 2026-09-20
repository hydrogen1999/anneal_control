"""Per-training-seed stability of bank selection against direct generation.

Pooling over seeds hides sign changes. On the 48-parent Pegasus arm the
direct policy beats a linear ramp on one training seed and loses on two,
which a pooled interval reports only as "crosses zero" -- true, but it does
not say that the disagreement is between seeds rather than between parents.
Bank selection on the identical records shows no such instability.

Both modes are paired on the logical parent and measured against the same
matched linear ramp, so the two columns are directly comparable.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def parent_differences(payload, mode):
    """Parent-mean (method loss − linear loss); negative favours the method."""
    linear = {str(r["record_id"]): float(r["linear_loss"]) for r in payload["records"]}
    if mode == "bank":
        pairs = [(r["parent_id"], float(r["selected_loss"]) - linear[str(r["record_id"])])
                 for r in payload["records"]]
    else:
        pairs = []
        for r in payload["direct_policy"]["records"]:
            key = str(r["record_id"])
            if key not in linear:
                raise ValueError(f"direct record {key} has no linear reference")
            pairs.append((r["parent_id"], float(r["loss"]) - linear[key]))
    buckets = defaultdict(list)
    for parent, value in pairs:
        buckets[str(parent)].append(value)
    return {p: float(np.mean(v)) for p, v in buckets.items()}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluations", required=True)
    parser.add_argument("--method", default="summary")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    paths = sorted(glob.glob(str(Path(args.evaluations) / f"{args.method}__seed_*.json")))
    if not paths:
        parser.error(f"no evaluations for {args.method!r}")

    result = {"schema_version": 1, "method": args.method,
              "evaluations": str(args.evaluations), "modes": {},
              "difference_definition": "parent-mean (method loss - linear loss); "
                                       "negative favours the method",
              "scope": ("per training seed, paired on the logical parent; a sign change "
                        "between seeds is training instability, not parent disagreement")}
    for mode in ("bank", "direct"):
        per_seed, tables = {}, []
        for path in paths:
            payload = json.loads(Path(path).read_text())
            seed = str(payload.get("training_seed", Path(path).stem.split("seed_")[-1]))
            table = parent_differences(payload, mode)
            tables.append(table)
            values = np.array([table[p] for p in sorted(table)])
            per_seed[seed] = {"mean_difference": float(values.mean()),
                              "parents_beating_linear": int((values < 0).sum()),
                              "n_parents": int(values.size)}
        parents = sorted(tables[0])
        if any(sorted(t) != parents for t in tables):
            raise ValueError("training seeds cover different parents")
        pooled = np.array([np.mean([t[p] for t in tables]) for p in parents])
        signs = {np.sign(block["mean_difference"]) for block in per_seed.values()}
        result["modes"][mode] = {
            "per_seed": per_seed,
            "pooled_mean_difference": float(pooled.mean()),
            "pooled_parents_beating_linear": int((pooled < 0).sum()),
            "n_parents": len(parents),
            "n_seeds": len(per_seed),
            "sign_changes_across_seeds": len(signs) > 1,
        }

    Path(args.output).write_text(json.dumps(result, indent=2))
    for mode, block in result["modes"].items():
        flag = "  <-- SIGN CHANGES ACROSS SEEDS" if block["sign_changes_across_seeds"] else ""
        print(f"{mode}: pooled {block['pooled_mean_difference']:+.5f}, "
              f"{block['pooled_parents_beating_linear']}/{block['n_parents']} parents{flag}")
        for seed, s in sorted(block["per_seed"].items()):
            print(f"   seed {seed}: {s['mean_difference']:+.5f}  "
                  f"{s['parents_beating_linear']}/{s['n_parents']} parents")
    return 0


if __name__ == "__main__":
    sys.exit(main())
