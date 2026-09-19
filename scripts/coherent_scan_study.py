"""Run the design document's coherent-failure test on the instances that qualify.

Multi-crossing records come from the crossing census; an equal number of
monotone-gap records from the same census form a control group. Without that
control, a scan showing no interference proves nothing — it could mean the grid
is too coarse rather than that the instances are smooth.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np


def _one(args):
    (data_dir, split, record_id, runtimes, max_slope, max_qubits,
     tolerance, max_steps, budget, group, allow_test) = args
    import numpy as np

    from annealctrl.benchmarking import record_physics, score_schedule
    from annealctrl.coherent_scan import scan_runtimes
    from annealctrl.physics_baselines import exact_teacher_baseline
    from annealctrl.pipeline import load_records
    from annealctrl.schedules import Schedule
    from annealctrl.search import optimize_control_family

    record = next(r for r in load_records(data_dir, split)
                  if str(np.asarray(r["record_id"]).item()) == record_id)
    reference = float(np.asarray(record["runtime"]).item())
    terms, path, _ = record_physics(record)

    schedules = {"linear": Schedule.linear()}

    # The teacher is runtime-aware, so it is rebuilt at each runtime rather than
    # transplanted from the reference -- that is how it would be deployed. The
    # builder bounds ds/dt while the scorer bounds ds/dtau, so it must be handed
    # max_ds_dtau / runtime; passing the scorer's number straight through gives
    # ds/dtau = 15 at runtime 4 against a cap of 4, which the scorer rejects.
    cache: dict[float, Any] = {}

    def teacher_at(runtime):
        if runtime not in cache:
            built = exact_teacher_baseline(terms, "d2", runtime=float(runtime),
                                           max_slope=max_slope / float(runtime),
                                           path=path, max_qubits=max_qubits)
            cache[runtime] = built.schedule
        return cache[runtime]

    reference_teacher = teacher_at(reference)
    if reference_teacher is not None:
        schedules["d2"] = teacher_at

    # A searched control at the reference runtime, so the scan has something
    # that was optimised for one point and is then asked about the others.
    def loss_fn(schedule):
        return float(score_schedule(record, schedule, tolerance=tolerance,
                                    max_steps=max_steps, max_ds_dtau=max_slope * (1 + 1e-6))["loss"])

    best = None
    for family in ("one_window", "two_window"):
        result = optimize_control_family(loss_fn, family, budget=budget, seed=0,
                                         split=split, allow_test_adaptation=allow_test)
        if best is None or result.best.loss < best.loss:
            best = result.best
    if best is not None:
        schedules["searched"] = best.candidate.schedule

    row = scan_runtimes(record, schedules, runtimes=runtimes, tolerance=tolerance,
                        max_steps=max_steps, max_slope=max_slope)
    row["group"] = group
    row["teacher_resolved_at_reference"] = reference_teacher is not None
    row["reference_runtime"] = reference
    return row


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census", nargs="+", required=True,
                        help="crossing-census JSON files; their `data` field names the dataset")
    parser.add_argument("--output", required=True)
    parser.add_argument("--points", type=int, default=64)
    parser.add_argument("--runtime-min", type=float, default=1.0)
    parser.add_argument("--runtime-max", type=float, default=17.0)
    parser.add_argument("--controls", type=int, default=4,
                        help="monotone-gap records per census, as a control group")
    parser.add_argument("--budget", type=int, default=32)
    parser.add_argument("--max-slope", type=float, default=4.0)
    parser.add_argument("--max-qubits", type=int, default=10)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--tolerance", type=float, default=5e-4)
    parser.add_argument("--max-steps", type=int, default=16384)
    parser.add_argument("--allow-test-adaptation", action="store_true",
                        help="the searched arm is a within-record reference control, not a "
                             "held-out method; required when a census covers the test split")
    args = parser.parse_args(argv)

    from annealctrl.coherent_scan import summarise_scans

    runtimes = list(np.linspace(args.runtime_min, args.runtime_max, args.points))
    jobs = []
    for path in args.census:
        census = json.loads(Path(path).read_text())
        data, split = census["data"], census["split"]
        multi = [r for r in census["rows"] if r["interior_minima"] >= 2]
        flat = [r for r in census["rows"] if r["interior_minima"] == 0][:args.controls]
        print(f"{path}: {len(multi)} multi-crossing, {len(flat)} monotone controls", flush=True)
        for group, subset in (("multi_crossing", multi), ("monotone_control", flat)):
            for r in subset:
                jobs.append((data, split, r["record_id"], runtimes, args.max_slope,
                             args.max_qubits, args.tolerance, args.max_steps,
                             args.budget, group, args.allow_test_adaptation))
    if not jobs:
        raise SystemExit("no qualifying records in the supplied censuses")
    print(f"{len(jobs)} records, {args.points} runtimes over "
          f"[{args.runtime_min}, {args.runtime_max}]", flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
        for row in pool.map(_one, jobs):
            rows.append(row)
            print(f"  {row['record_id']:26s} {row['group']:16s} "
                  f"linear turns={row['linear_turning_points']:2d} "
                  f"range={row['linear_range']:.4f}", flush=True)

    payload = {"schema_version": 1, "censuses": list(args.census),
               "runtimes": runtimes, "n_records": len(rows), "rows": rows, "groups": {},
               "test_adaptation": {
                   "allowed": bool(args.allow_test_adaptation),
                   "why": ("the searched arm is a within-record reference control tuned at the "
                           "reference runtime and then asked about the others; no learned model "
                           "is involved and no held-out generalisation claim is made from it"),
                   "splits": sorted({j[1] for j in jobs})}}
    for group in ("multi_crossing", "monotone_control"):
        subset = [r for r in rows if r["group"] == group]
        if not subset:
            continue
        block = {"n_records": len(subset)}
        for method in ("d2", "searched"):
            try:
                block[method] = summarise_scans(subset, method=method)
            except ValueError as error:
                block[method] = {"status": "unavailable", "reason": str(error)}
        payload["groups"][group] = block

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2, default=float))

    print()
    for group, block in payload["groups"].items():
        print(f"{group} ({block['n_records']} records):")
        for method in ("d2", "searched"):
            b = block.get(method, {})
            if b.get("status") == "unavailable":
                print(f"  {method:10s} unavailable: {b['reason'][:70]}")
                continue
            print(f"  {method:10s} nonmonotone linear curves "
                  f"{b['records_with_nonmonotone_linear_curve']}/{b['n_records']}, "
                  f"mean turns {b['mean_linear_turning_points']:.2f}; advantage changes sign in "
                  f"{b['records_where_the_advantage_changes_sign']}/{b['n_records']}, "
                  f"mean {b['mean_advantage']:+.5f}")
    print(f"\nwrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
