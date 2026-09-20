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

from annealctrl.effect_size import (decompose_share, effect_size_report,
                                    pooled_across_seeds)


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
    # Sweep directories are merged by the caller, and nothing downstream dedupes:
    # aggregate_frontier averages a plain list. One record present twice -- from
    # overlapping --record-ids runs, or a shard rerun into a new directory --
    # would silently reweight the denominator. Refuse rather than average it.
    seen = {}
    for row in rows:
        key = str(row["record_id"])
        if key in seen:
            raise ValueError(
                f"record {key} appears in more than one sweep directory. Merged shards must "
                "be disjoint; a repeated record would be counted twice in the headroom mean.")
        seen[key] = row
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluations", required=True, help="directory of <method>__seed_*.json")
    parser.add_argument("--method", default="summary")
    parser.add_argument("--reference", required=True,
                        choices=["bank_oracle", "frontier", "both"],
                        help="the denominator; there is deliberately no default. 'both' also "
                             "factorises the share into selector efficiency x bank coverage")
    parser.add_argument("--frontier-sweep", nargs="*", default=None,
                        help="control-sweep directories (or their parent, for shards)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    needs_frontier = args.reference in ("frontier", "both")
    if needs_frontier and not args.frontier_sweep:
        parser.error(f"--reference {args.reference} requires --frontier-sweep")
    frontier = load_frontier(args.frontier_sweep) if needs_frontier else None

    paths = sorted(glob.glob(str(Path(args.evaluations) / f"{args.method}__seed_*.json")))
    if not paths:
        parser.error(f"no evaluations for method {args.method!r} in {args.evaluations}")

    references = ["bank_oracle", "frontier"] if args.reference == "both" else [args.reference]
    by_reference = {name: {} for name in references}
    for path in paths:
        payload = json.loads(Path(path).read_text())
        seed = str(payload.get("training_seed", Path(path).stem.split("seed_")[-1]))
        for name in references:
            by_reference[name][seed] = effect_size_report(
                payload["records"], reference=name,
                frontier_rows=frontier if name == "frontier" else None,
                bootstrap_resamples=args.bootstrap_resamples, seed=args.seed)
    per_seed = by_reference[references[0]]

    blocks = {}
    for name in references:
        seeds = by_reference[name]
        shares = [r["share_of_headroom"] for r in seeds.values()
                  if r["share_of_headroom"] is not None]
        ds = [r["gain_vs_linear"]["cohens_d"] for r in seeds.values()
              if r["gain_vs_linear"]["cohens_d"] is not None]
        pooled = pooled_across_seeds(list(seeds.values()))
        first = next(iter(seeds.values()))
        blocks[name] = {
            "reference": first["reference"],
            "n_parents": first["n_parents"],
            "n_training_seeds": len(seeds),
            "mean_gain_vs_linear": float(np.mean([r["gain_vs_linear"]["mean"]
                                                  for r in seeds.values()])),
            "mean_headroom": float(np.mean([r["headroom"]["mean"] for r in seeds.values()])),
            "mean_share_of_headroom": float(np.mean(shares)) if shares else None,
            "mean_cohens_d": float(np.mean(ds)) if ds else None,
            "pooled_across_seeds": pooled,
            "per_seed": seeds,
        }

    summary = {
        "schema_version": 2,
        "method": args.method,
        "evaluations": str(args.evaluations),
        "by_reference": blocks,
        "scope": ("a share is against its named reference only; shares against different "
                  "references are not comparable and must not be tabulated together"),
    }
    if len(references) == 2:
        # Decompose the SEED-AVERAGED gain, not one seed's. Headroom is a property
        # of the records, so pooling leaves both denominators untouched and the
        # factorisation reproduces the seed-averaged share exactly.
        summary["decomposition"] = decompose_share(
            blocks["bank_oracle"]["pooled_across_seeds"],
            blocks["frontier"]["pooled_across_seeds"])
    Path(args.output).write_text(json.dumps(summary, indent=2))

    for name, block in blocks.items():
        ref = block["reference"]
        pooled = block["pooled_across_seeds"]
        print(f"{args.method}: {block['n_parents']} parents, "
              f"{block['n_training_seeds']} seeds")
        print(f"  reference      {ref['name']} @ "
              f"{ref['objective_calls_per_instance']} calls/instance")
        print(f"  gain           {block['mean_gain_vs_linear']:+.5f}")
        print(f"  headroom       {block['mean_headroom']:.5f}")
        share = block["mean_share_of_headroom"]
        print(f"  share          {share*100:.1f} %" if share is not None
              else "  share          n/a")
        print(f"  Cohen's d      {pooled['cohens_d']:.2f} pooled over seeds "
              f"({block['mean_cohens_d']:.2f} mean of per-seed)")
        print(f"  parents won    {pooled['parents_won']} / {block['n_parents']} pooled; "
              f"worst single seed {pooled['worst_seed_parents_won']} / {block['n_parents']}")
    if "decomposition" in summary:
        d = summary["decomposition"]
        print("  --- share of findable = selector efficiency x bank coverage ---")
        print(f"  selector eff   {d['selector_efficiency']*100:.1f} %  "
              f"(critic against its own {d['bank_calls_per_instance']}-candidate menu)")
        print(f"  bank coverage  {d['bank_coverage']*100:.1f} %  "
              f"(menu against a {d['frontier_calls_per_instance']}-call search)")
        print(f"  share findable {d['share_of_findable']*100:.1f} %  "
              f"binding factor: {d['binding_factor']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
