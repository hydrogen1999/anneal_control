"""Report the effect in three units, against a reference that is named and costed.

The published effect-size table was assembled by hand, and that is how a
257-call frontier denominator and a 64-candidate bank-oracle denominator ended
up in one column. This script exists so the table has a generator: the
reference is a required argument, its per-instance cost is read from the
artifacts, and a frontier reference that does not cover the evaluated parents
is refused rather than averaged.

Seeds are training seeds of the same method. Each is reported separately and
then averaged, because the parent set is identical across them and the spread
across seeds is training noise, not instance variation.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

from annealctrl.effect_size import effect_size_report, pooled_across_seeds


def load_frontier(dirs):
    rows = []
    for d in dirs:
        paths = sorted(glob.glob(str(Path(d) / "rows.jsonl"))) or \
                sorted(glob.glob(str(Path(d) / "shard_*" / "rows.jsonl")))
        if not paths:
            raise ValueError(f"{d} contains no rows.jsonl (nor shard_*/rows.jsonl)")
        for path in paths:
            with open(path) as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    if entry.get("status") == "ok":
                        rows.append(entry["result"])
    if not rows:
        raise ValueError("no successful frontier rows found")
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluations", required=True, help="directory of <method>__seed_*.json")
    parser.add_argument("--method", default="summary")
    parser.add_argument("--reference", required=True, choices=["bank_oracle", "frontier"],
                        help="the denominator; there is deliberately no default")
    parser.add_argument("--frontier-sweep", nargs="*", default=None,
                        help="control-sweep directories (or their parent, for shards)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    frontier = load_frontier(args.frontier_sweep) if args.reference == "frontier" else None
    if args.reference == "frontier" and not args.frontier_sweep:
        parser.error("--reference frontier requires --frontier-sweep")

    paths = sorted(glob.glob(str(Path(args.evaluations) / f"{args.method}__seed_*.json")))
    if not paths:
        parser.error(f"no evaluations for method {args.method!r} in {args.evaluations}")

    per_seed = {}
    for path in paths:
        payload = json.loads(Path(path).read_text())
        seed = payload.get("training_seed", Path(path).stem.split("seed_")[-1])
        per_seed[str(seed)] = effect_size_report(
            payload["records"], reference=args.reference, frontier_rows=frontier,
            bootstrap_resamples=args.bootstrap_resamples, seed=args.seed)

    shares = [r["share_of_headroom"] for r in per_seed.values() if r["share_of_headroom"] is not None]
    gains = [r["gain_vs_linear"]["mean"] for r in per_seed.values()]
    ds = [r["gain_vs_linear"]["cohens_d"] for r in per_seed.values()
          if r["gain_vs_linear"]["cohens_d"] is not None]
    first = next(iter(per_seed.values()))
    pooled = pooled_across_seeds(list(per_seed.values()))

    summary = {
        "schema_version": 1,
        "method": args.method,
        "evaluations": str(args.evaluations),
        "reference": first["reference"],
        "n_parents": first["n_parents"],
        "n_training_seeds": len(per_seed),
        "mean_gain_vs_linear": float(np.mean(gains)),
        "mean_headroom": float(np.mean([r["headroom"]["mean"] for r in per_seed.values()])),
        "mean_share_of_headroom": float(np.mean(shares)) if shares else None,
        "mean_cohens_d": float(np.mean(ds)) if ds else None,
        "min_parents_won": min(r["gain_vs_linear"]["parents_won"] for r in per_seed.values()),
        "pooled_across_seeds": pooled,
        "per_seed": per_seed,
        "scope": ("the share is against the named reference only; shares against different "
                  "references are not comparable and must not be tabulated together"),
    }
    Path(args.output).write_text(json.dumps(summary, indent=2))
    ref = summary["reference"]
    print(f"{args.method}: {summary['n_parents']} parents, {summary['n_training_seeds']} seeds")
    print(f"  reference      {ref['name']} @ {ref['objective_calls_per_instance']} calls/instance")
    print(f"  gain           {summary['mean_gain_vs_linear']:+.5f}")
    print(f"  headroom       {summary['mean_headroom']:.5f}")
    share = summary["mean_share_of_headroom"]
    print(f"  share          {share*100:.1f} %" if share is not None else "  share          n/a")
    print(f"  Cohen's d      {pooled['cohens_d']:.2f} pooled over seeds "
          f"({summary['mean_cohens_d']:.2f} mean of per-seed)")
    print(f"  parents won    {pooled['parents_won']} / {summary['n_parents']} pooled; "
          f"worst single seed {pooled['worst_seed_parents_won']} / {summary['n_parents']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
