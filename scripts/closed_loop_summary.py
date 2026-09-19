"""Decompose the closed-loop gain across search seeds and checkpoints.

Every configuration is run twice -- once with the critic choosing and once with
a seeded random chooser on the identical oversampled proposal stream -- because
the filtered arm walks `oversample` times as much of the Sobol sequence as the
baseline and part of any gain is that wider coverage rather than the model.

The two arms share a baseline that is re-run in each file, so the contrast is
differenced parent by parent against the same reference rather than merely
matched. The script verifies that identity instead of trusting it: if the
baselines differ, the decomposition is meaningless and it says so.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--output")
    args = parser.parse_args(argv)

    from annealctrl.headroom import _bootstrap, _parent_means

    runs: dict[tuple, dict] = {}
    for path in sorted(Path(args.directory).glob("*.json")):
        d = json.loads(path.read_text())
        if "chooser" not in d or "rows" not in d:
            continue
        key = (Path(d["checkpoint"]).parent.name, int(d["seed"]))
        runs.setdefault(key, {})[d["chooser"]] = d

    table, problems = [], []
    for key in sorted(runs):
        pair = runs[key]
        if set(pair) != {"critic", "random"}:
            problems.append(f"{key}: only {sorted(pair)}")
            continue
        C = {r["record_id"]: r for r in pair["critic"]["rows"]}
        R = {r["record_id"]: r for r in pair["random"]["rows"]}
        if set(C) != set(R):
            problems.append(f"{key}: arms cover different records")
            continue
        drift = max(abs(C[k]["baseline_best"] - R[k]["baseline_best"]) for k in C)
        if drift > 1e-12:
            problems.append(f"{key}: baselines differ by {drift:.3e}; decomposition invalid")
            continue
        rows = [{"parent_id": C[k]["parent_id"], "critic": C[k]["improvement"],
                 "random": R[k]["improvement"],
                 "delta": C[k]["improvement"] - R[k]["improvement"]} for k in C]
        entry = {"checkpoint": key[0], "search_seed": key[1], "n_parents": None,
                 "baseline_drift": drift}
        for name in ("critic", "random", "delta"):
            values, parents = _parent_means(rows, lambda r, n=name: r[n])
            ci = _bootstrap(values, n_resamples=8000, seed=0)
            entry[name] = {"mean": float(values.mean()), "low": ci["low"], "high": ci["high"],
                           "positive_parents": int(sum(1 for r in rows if r[name] > 0)),
                           "n_parents": len(parents)}
            entry["n_parents"] = len(parents)
        table.append(entry)

    if not table:
        raise SystemExit("no complete critic/random pair found:\n  " + "\n  ".join(problems))

    print(f"{'checkpoint':12s} {'seed':>4s} {'critic total':>26s} "
          f"{'random control':>26s} {'critic alone (delta)':>28s}")
    for e in table:
        def cell(name, width):
            b = e[name]
            return f"{b['mean']:+.5f}[{b['low']:+.5f},{b['high']:+.5f}] {b['positive_parents']:2d}/{b['n_parents']}".rjust(width)
        print(f"{e['checkpoint']:12s} {e['search_seed']:>4d} {cell('critic',26)} "
              f"{cell('random',26)} {cell('delta',28)}")

    deltas = np.array([e["delta"]["mean"] for e in table])
    criticals = np.array([e["critic"]["mean"] for e in table])
    randoms = np.array([e["random"]["mean"] for e in table])
    print(f"\nacross {len(table)} configurations:")
    print(f"  critic total    {criticals.mean():+.5f}  spread {criticals.min():+.5f} .. {criticals.max():+.5f}")
    print(f"  random control  {randoms.mean():+.5f}  spread {randoms.min():+.5f} .. {randoms.max():+.5f}")
    print(f"  critic alone    {deltas.mean():+.5f}  spread {deltas.min():+.5f} .. {deltas.max():+.5f}")
    print(f"  every configuration's critic-alone interval excludes zero: "
          f"{all(e['delta']['low'] > 0 for e in table)}")
    if problems:
        print("\nincomplete or invalid:\n  " + "\n  ".join(problems))

    if args.output:
        Path(args.output).write_text(json.dumps(
            {"schema_version": 1, "configurations": table, "problems": problems,
             "scope": ("critic-alone is the paired difference between the critic arm and a "
                       "random chooser on the identical oversampled proposal stream, against "
                       "a baseline verified identical in both files")},
            indent=2, default=float))
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
