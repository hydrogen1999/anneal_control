"""Run a search, then ask what each budget's answer is worth under noise.

One search per record against the noiseless simulator, incumbents taken at a
ladder of budgets, and every incumbent re-evaluated under a Lindblad channel.
The closed curve falls by construction; the open curve is the measurement.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

BUDGETS = (1, 2, 4, 8, 16, 32, 64)
FAMILIES = ("one_window", "two_window", "eight_bin")


def _run_record(args):
    (data_dir, split, record_id, budgets, rate, seed, family,
     tolerance, max_steps, max_qubits) = args
    import numpy as np

    from annealctrl.benchmarking import score_schedule
    from annealctrl.open_system import open_loss
    from annealctrl.overoptimisation import incumbent_at_budgets
    from annealctrl.pipeline import load_records
    from annealctrl.search import optimize_control_family

    record = next(r for r in load_records(data_dir, split)
                  if str(np.asarray(r["record_id"]).item()) == record_id)

    def loss_fn(schedule):
        return float(score_schedule(record, schedule, tolerance=tolerance,
                                    max_steps=max_steps, max_ds_dtau=1e9)["loss"])

    result = optimize_control_family(loss_fn, family, budget=max(budgets), seed=seed,
                                     split=split)
    trace = [r.loss for r in result.records]
    picks = incumbent_at_budgets(trace, budgets)

    closed, open_, seen = {}, {}, {}
    for budget, index in zip(budgets, picks):
        closed[str(budget)] = float(trace[index])
        if index not in seen:
            seen[index] = float(open_loss(record, result.records[index].candidate.schedule,
                                          dephasing_rate=rate, max_qubits=max_qubits))
        open_[str(budget)] = seen[index]
    return {"record_id": record_id,
            "parent_id": str(np.asarray(record["parent_id"]).item()),
            "family": family, "closed": closed, "open": open_,
            "distinct_incumbents": len(seen)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--family", default="eight_bin", choices=FAMILIES)
    parser.add_argument("--rate", type=float, default=0.1)
    parser.add_argument("--budgets", type=int, nargs="+", default=list(BUDGETS))
    parser.add_argument("--records", type=int, default=40)
    parser.add_argument("--max-qubits", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--tolerance", type=float, default=5e-4)
    parser.add_argument("--max-steps", type=int, default=8192)
    args = parser.parse_args(argv)

    from annealctrl.overoptimisation import overoptimisation_curve
    from annealctrl.pipeline import load_records

    records = [r for r in load_records(args.data, args.split)
               if len(np.asarray(r["physical_h"])) <= args.max_qubits]
    seen, chosen = set(), []
    for r in sorted(records, key=lambda r: str(np.asarray(r["record_id"]).item())):
        pid = str(np.asarray(r["parent_id"]).item())
        if pid in seen:
            continue
        seen.add(pid)
        chosen.append(str(np.asarray(r["record_id"]).item()))
        if len(chosen) >= args.records:
            break
    print(f"{len(chosen)} records, one per parent, <= {args.max_qubits} qubits, "
          f"family {args.family}, budgets {args.budgets}", flush=True)

    jobs = [(args.data, args.split, rid, tuple(args.budgets), args.rate, args.seed,
             args.family, args.tolerance, args.max_steps, args.max_qubits) for rid in chosen]
    rows = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
        for row in pool.map(_run_record, jobs):
            rows.append(row)
            print(f"  {row['record_id']}: {row['distinct_incumbents']} distinct incumbents", flush=True)

    curve = overoptimisation_curve(rows, budgets=args.budgets)
    payload = {"schema_version": 1, "data": args.data, "split": args.split,
               "family": args.family, "rate": args.rate, "seed": args.seed,
               "curve": curve, "rows": rows}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2, default=float))

    print(f"\n{'budget':>7s} {'closed':>10s} {'open':>10s}")
    for b, c, o in zip(args.budgets, curve["closed_curve"], curve["open_curve"]):
        print(f"{b:>7d} {c:>10.5f} {o:>10.5f}")
    best = curve["budget_minimising_open_loss"]
    cost = curve["cost_of_the_largest_budget_against_the_best"]
    ci = cost["parent_bootstrap_ci"]
    print(f"\nbest budget under noise: {best} of {max(args.budgets)}")
    print(f"cost of spending the largest budget instead: {cost['mean']:+.5f} "
          f"[{ci['low']:+.5f},{ci['high']:+.5f}], worse in "
          f"{cost['worse_in_parents']}/{cost['n_parents']} parents")
    print(f"open loss monotone in budget: {curve['open_loss_is_monotone_in_budget']}")
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
