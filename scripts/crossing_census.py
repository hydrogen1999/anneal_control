"""How many transition regions do these instances actually have?

The design document's falsification suite reserves its coherent-failure test
for "paths with two or more transition regions", scanned densely in runtime,
and warns that a scalar profile can succeed after averaging while failing on a
sharply specified coherent experiment.

Whether that test is even runnable on a dataset is a property of the dataset,
and nobody had checked. A transition region appears as an interior local
minimum of the first gap along s, so counting them is cheap and settles the
scope of every representation claim made on this evidence base.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


def _one(args):
    data_dir, split, record_id, points, max_qubits = args
    import numpy as np

    from annealctrl.benchmarking import record_physics
    from annealctrl.pipeline import load_records
    from annealctrl.spectral import spectral_profile

    record = next(r for r in load_records(data_dir, split)
                  if str(np.asarray(r["record_id"]).item()) == record_id)
    terms, path, _ = record_physics(record)
    grid = np.linspace(0.0, 1.0, points)
    profile = spectral_profile(terms, grid, path=path, max_qubits=max_qubits)
    gap = np.array([p.raw_gap for p in profile], dtype=float)
    interior = gap[1:-1]
    is_min = (interior < gap[:-2]) & (interior < gap[2:])
    return {"record_id": record_id,
            "parent_id": str(np.asarray(record["parent_id"]).item()),
            "n_physical": int(np.asarray(record["physical_h"]).size),
            "interior_minima": int(is_min.sum()),
            "minima_locations": [float(x) for x in grid[1:-1][is_min]],
            "min_gap": float(gap.min()),
            "argmin_s": float(grid[int(gap.argmin())]),
            "gap_at_endpoints": [float(gap[0]), float(gap[-1])]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--points", type=int, default=129,
                        help="s-grid resolution; too coarse a grid hides minima")
    parser.add_argument("--records", type=int, default=48)
    parser.add_argument("--max-qubits", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)

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
          f"{args.points}-point s grid", flush=True)

    jobs = [(args.data, args.split, rid, args.points, args.max_qubits) for rid in chosen]
    rows = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
        for row in pool.map(_one, jobs):
            rows.append(row)

    counts = Counter(r["interior_minima"] for r in rows)
    multi = [r for r in rows if r["interior_minima"] >= 2]
    payload = {"schema_version": 1, "data": args.data, "split": args.split,
               "s_grid_points": args.points, "n_records": len(rows),
               "minima_histogram": {int(k): int(v) for k, v in sorted(counts.items())},
               "n_multi_crossing": len(multi),
               "multi_crossing_records": multi, "rows": rows,
               "scope": ("interior local minima of the first gap on a finite s grid; a "
                         "coarser grid hides minima and a degenerate or unresolved gap is "
                         "not filtered here, so this bounds rather than certifies the "
                         "count of transition regions")}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2, default=float))

    print("\ninterior gap minima:")
    for k, v in sorted(counts.items()):
        print(f"  {k} minima: {v:3d} records  ({v/len(rows):5.1%})")
    print(f"\nrecords usable for the coherent-failure test (>=2): "
          f"{len(multi)}/{len(rows)} = {len(multi)/len(rows):.1%}")
    for r in multi[:8]:
        print(f"  {r['record_id']:26s} minima={r['interior_minima']} "
              f"min gap={r['min_gap']:.4f} at s={[round(x,3) for x in r['minima_locations']]}")
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
