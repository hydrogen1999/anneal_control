"""Price one forward pass in instance-specific simulator calls.

"Captures X% of what a 257-call search finds" has an arbitrary denominator.
On the synthetic set the same search had already found 96.7% of its final
headroom by 129 calls, so the 257 is a budget someone chose, and X moves when
that choice moves. A call count does not have that problem: "one forward pass
is worth about N calls" is priced in the unit a practitioner spends.

The curve is built from the retained trials of a control-sweep. Each family
is searched under its own equal budget, so the incumbent after m evaluations
per tunable family corresponds to 1 + (#tunable) x m total calls -- the linear
reference is the single free evaluation. Headroom is per record and averaged
within a parent before anything else, because the logical parent is this
project's unit of independence.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from annealctrl.effect_size import search_call_equivalent


def parent_curves(benchmark_dirs):
    """parent -> {total_calls: mean headroom over that parent's records}."""
    per_record = defaultdict(dict)
    parent_of = {}
    tunable = None
    for root in benchmark_dirs:
        paths = sorted(glob.glob(str(Path(root) / "benchmarks" / "*.json"))) or \
                sorted(glob.glob(str(Path(root) / "shard_*" / "benchmarks" / "*.json")))
        for path in paths:
            payload = json.loads(Path(path).read_text())
            families = payload["families"]
            if "linear" not in families:
                raise ValueError(f"{path} has no linear reference to measure headroom against")
            linear = float(families["linear"]["best_loss"])
            traces = {name: [float(r["loss"]) for r in block["records"]]
                      for name, block in families.items()
                      if name != "linear" and block.get("records")}
            if not traces:
                raise ValueError(f"{path} retains no trials; re-run with retain_full_trials")
            names = tuple(sorted(traces))
            if tunable is None:
                tunable = names
            elif names != tunable:
                raise ValueError(f"{path} searched {names}, others searched {tunable}; "
                                 "a shared budget axis needs the same families everywhere")
            depth = min(len(v) for v in traces.values())
            running = {k: float("inf") for k in traces}
            record = str(payload["record_id"])
            parent_of[record] = str(payload["parent_id"])
            for m in range(1, depth + 1):
                for k, v in traces.items():
                    running[k] = min(running[k], v[m - 1])
                best = min([linear, *running.values()])
                per_record[record][1 + len(traces) * m] = linear - best
    if not per_record:
        raise ValueError("no benchmark files found")

    grouped = defaultdict(lambda: defaultdict(list))
    for record, curve in per_record.items():
        for budget, headroom in curve.items():
            grouped[parent_of[record]][budget].append(headroom)
    return {parent: {b: float(np.mean(v)) for b, v in sorted(budgets.items())}
            for parent, budgets in grouped.items()}


TARGETS = {
    # what the learned selector actually achieved, one forward pass
    "learned_gain": lambda row: float(row["linear_loss"]) - float(row["selected_loss"]),
    # the ceiling on its own 64-candidate menu: what a perfect ranker would achieve
    "bank_ceiling": lambda row: float(row["linear_loss"]) - float(row["bank_best_loss"]),
}


def direct_rows(payload):
    """Direct-policy losses joined to the linear reference they are measured against.

    The direct block stores its own loss per record but not the matched linear
    ramp, which lives on the bank-selection rows. The join is by record_id and
    must be total: a direct record with no linear counterpart cannot be turned
    into a gain, and silently dropping it would change the population.
    """
    linear = {str(r["record_id"]): float(r["linear_loss"]) for r in payload["records"]}
    rows = []
    for record in payload["direct_policy"]["records"]:
        key = str(record["record_id"])
        if key not in linear:
            raise ValueError(f"direct record {key} has no linear reference in the same "
                             "evaluation; the two blocks describe different populations")
        rows.append({"parent_id": record["parent_id"], "record_id": key,
                     "linear_loss": linear[key], "selected_loss": float(record["loss"])})
    return rows


def parent_gains(evaluations, method, target="learned_gain", mode="bank"):
    paths = sorted(glob.glob(str(Path(evaluations) / f"{method}__seed_*.json")))
    if not paths:
        raise ValueError(f"no evaluations for {method!r} in {evaluations}")
    if mode == "direct" and target != "learned_gain":
        raise ValueError("the direct policy has no candidate bank, so bank_ceiling is "
                         "undefined for it")
    extract = TARGETS[target]
    per_seed = []
    for path in paths:
        payload = json.loads(Path(path).read_text())
        rows = direct_rows(payload) if mode == "direct" else payload["records"]
        buckets = defaultdict(list)
        for row in rows:
            buckets[str(row["parent_id"])].append(extract(row))
        per_seed.append({p: float(np.mean(v)) for p, v in buckets.items()})
    parents = set(per_seed[0])
    if any(set(s) != parents for s in per_seed):
        raise ValueError("training seeds cover different parents")
    return {p: float(np.mean([s[p] for s in per_seed])) for p in parents}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluations", required=True)
    parser.add_argument("--frontier-sweep", nargs="+", required=True)
    parser.add_argument("--method", default="summary")
    parser.add_argument("--mode", default="bank", choices=["bank", "direct"],
                        help="bank selection from the stored candidates, or the policy's "
                             "own generated control")
    parser.add_argument("--target", default="learned_gain", choices=sorted(TARGETS),
                        help="learned_gain prices what the selector achieved; bank_ceiling "
                             "prices what a perfect ranker on the same menu would achieve")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    curves = parent_curves(args.frontier_sweep)
    gains = parent_gains(args.evaluations, args.method, args.target, args.mode)
    shared = sorted(set(curves) & set(gains))
    if not shared:
        parser.error("the frontier sweep and the evaluation share no parents")
    missing = sorted(set(gains) - set(curves))
    if missing:
        parser.error(f"the frontier does not cover {len(missing)} evaluated parents "
                     f"({missing[:5]}...); it cannot price them")

    curves = {p: curves[p] for p in shared}
    gains = {p: gains[p] for p in shared}
    censor_at = max(max(c) for c in curves.values())
    result = search_call_equivalent(curves, gains, censor_at=censor_at,
                                    bootstrap_resamples=args.bootstrap_resamples,
                                    seed=args.seed)
    result["method"] = args.method
    result["target"] = args.target
    result["mode"] = args.mode
    result["evaluations"] = str(args.evaluations)
    Path(args.output).write_text(json.dumps(result, indent=2))

    ci = result["parent_bootstrap_ci"]
    print(f"{args.method} / {args.mode} / {args.target}: "
          f"against an instance-specific search")
    print(f"  parents           {result['n_parents']} "
          f"({result['n_resolved']} resolved, {result['n_censored']} censored, "
          f"{result['n_no_gain']} with no gain over linear)")
    if result["median_calls"] is None:
        print("  equivalent        undefined: no parent showed a positive gain")
        return 0
    print(f"  median equivalent {result['median_calls']} calls")
    print(f"  mean equivalent   {result['mean_calls']:.1f} calls "
          f"[{ci['low']:.1f}, {ci['high']:.1f}]  "
          f"(over the {result['n_resolved']} parents it won)")
    print(f"  search budget     {censor_at} calls (censoring point)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
