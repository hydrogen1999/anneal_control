"""Translate held-out losses into reads-to-confidence, with the tail visible.

The design document's amortization section specifies

    n_q = ceil( log(1 - q) / log(1 - p) )

and attaches two conditions that are easy to violate: state the independence
assumption, and never turn a zero-success observation into a finite number.
`evaluation.time_to_solution` implements both. No artifact had ever reported
it, so the project's loss differences had never been expressed in the unit a
practitioner budgets in.

One honesty point governs the whole script. Here p is a **simulated** success
probability from exact propagation, not a fraction of observed reads, so the
binomial Clopper-Pearson interval that `time_to_solution` computes for hardware
counts does not apply and is not used. The uncertainty reported is across
logical parents, which is the unit the rest of this project resamples.

The document also asks for "distributions and failure tails, not only average
improvements". n_q is convex in the loss, so a mean over records is dominated
by its worst records -- which is the point, and why quantiles are reported
beside it.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

# `bank_best_loss` and `best_bank_loss` are the same number -- the ORACLE best
# over the bank, a property of the record, not of the method. The method's own
# selection is recovered as bank_best_loss + bank_regret, and only that is a
# method loss. An earlier version of this script used bank_best_loss directly
# and therefore reported the oracle as if it were the learned selector.
METHOD_KEYS = {"linear": "linear_loss", "global": "global_loss"}
ORACLE_KEY = "bank_best_loss"
REGRET_KEY = "bank_regret"


def reads_to_confidence(loss: float, target: float) -> float:
    """Reads needed for `target` confidence at success probability 1 - loss."""
    p = 1.0 - float(loss)
    if p <= 0:
        return math.inf
    if p >= 1:
        return 1.0
    return float(math.ceil(math.log1p(-target) / math.log1p(-p)))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluations", required=True, help="directory of *__seed_*.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", default="summary")
    parser.add_argument("--target", type=float, default=0.99)
    args = parser.parse_args(argv)

    from annealctrl.headroom import _bootstrap, _parent_means

    files = sorted(glob.glob(f"{args.evaluations}/{args.method}__seed_*.json"))
    if not files:
        raise SystemExit(f"no evaluations for method {args.method!r}")

    # Average each record's loss over training seeds first, so seeds are not
    # treated as independent instances.
    by_record: dict[str, dict] = {}
    for path in files:
        for row in json.loads(Path(path).read_text())["records"]:
            entry = by_record.setdefault(str(row["record_id"]), {
                "parent_id": str(row["parent_id"]), "n": 0,
                "learned_bank": 0.0, "bank_oracle": 0.0,
                **{k: 0.0 for k in METHOD_KEYS}})
            entry["n"] += 1
            for name, key in METHOD_KEYS.items():
                entry[name] += float(row[key])
            entry["bank_oracle"] += float(row[ORACLE_KEY])
            entry["learned_bank"] += float(row[ORACLE_KEY]) + float(row[REGRET_KEY])
    names = list(METHOD_KEYS) + ["learned_bank", "bank_oracle"]
    for entry in by_record.values():
        for name in names:
            entry[name] /= entry["n"]
    # The guard that would have caught the original error. Each evaluation
    # states its own mean_loss; if the reconstruction does not reproduce it,
    # the wrong field is being read and every number below would be wrong.
    stated, rebuilt = [], []
    for path in files:
        payload = json.loads(Path(path).read_text())
        stated.append(float(payload["mean_loss"]))
        rebuilt.append(float(np.mean([float(r[ORACLE_KEY]) + float(r[REGRET_KEY])
                                      for r in payload["records"]])))
    drift = float(np.max(np.abs(np.asarray(stated) - np.asarray(rebuilt))))
    if drift > 1e-9:
        raise SystemExit(
            f"reconstructed method loss disagrees with the evaluation's own mean_loss "
            f"by {drift:.3e}; the wrong field is being read")
    print(f"{len(by_record)} records over {len(files)} seeds, method {args.method}; "
          f"reconstruction matches each evaluation's mean_loss to {drift:.1e}", flush=True)

    rows = []
    for rid, entry in by_record.items():
        row = {"record_id": rid, "parent_id": entry["parent_id"]}
        for name in names:
            row[f"{name}_loss"] = entry[name]
            row[f"{name}_reads"] = reads_to_confidence(entry[name], args.target)
        rows.append(row)

    censored = {name: sum(1 for r in rows if not math.isfinite(r[f"{name}_reads"]))
                for name in names}
    summary = {}
    for name in names:
        finite = [r for r in rows if math.isfinite(r[f"{name}_reads"])]
        values, parents = _parent_means(finite, lambda r, n=name: r[f"{n}_reads"])
        arr = np.array([r[f"{name}_reads"] for r in finite], dtype=float)
        summary[name] = {
            "n_records_finite": len(finite), "n_censored": censored[name],
            "n_parents": len(parents),
            "parent_mean_reads": float(values.mean()),
            "parent_bootstrap_ci": _bootstrap(values, n_resamples=8000, seed=0),
            "record_quantiles": {f"p{q}": float(np.percentile(arr, q))
                                 for q in (10, 25, 50, 75, 90, 99)},
            "max_reads": float(arr.max()),
            "mean_loss": float(np.mean([r[f"{name}_loss"] for r in rows])),
        }

    # The operational headline: paired per-record ratio, not a ratio of means.
    ratios = {}
    for name in ("linear", "global", "bank_oracle"):
        paired = [{"parent_id": r["parent_id"],
                   "v": r[f"{name}_reads"] / r["learned_bank_reads"]}
                  for r in rows
                  if math.isfinite(r[f"{name}_reads"]) and math.isfinite(r["learned_bank_reads"])
                  and r["learned_bank_reads"] > 0]
        values, parents = _parent_means(paired, lambda r: r["v"])
        ratios[f"{name}_reads_over_learned_reads"] = {
            "parent_mean": float(values.mean()),
            "parent_bootstrap_ci": _bootstrap(values, n_resamples=8000, seed=0),
            "n_parents": len(parents),
            "record_median": float(np.median([p["v"] for p in paired])),
            "learned_needs_fewer_in_parents": int((values > 1).sum()),
        }

    payload = {
        "schema_version": 1, "evaluations": args.evaluations, "method": args.method,
        "target_confidence": args.target, "n_records": len(rows), "n_seeds": len(files),
        "summary": summary, "read_ratios": ratios, "rows": rows,
        "field_semantics": ("`bank_best_loss` in the evaluation records is the ORACLE "
                            "best over the bank; the learned selection is recovered as "
                            "bank_best_loss + bank_regret and is reported as learned_bank. "
                            "bank_oracle is reported too, as the ceiling the selector "
                            "aims at, never as the method."),
        "scope": ("p is a simulated success probability from exact propagation, not a "
                  "fraction of observed reads, so the binomial Clopper-Pearson interval "
                  "for hardware counts does not apply and is not used; uncertainty is "
                  "resampled over logical parents. Reads assume independent identically "
                  "distributed trials at fixed per-read cost. This is reads-to-target, "
                  "not a certified optimum, and excludes all offline cost."),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2, default=float))

    print(f"\nreads for {args.target:.0%} confidence, {len(rows)} held-out records\n")
    print(f"{'method':14s} {'mean loss':>10s} {'parent mean':>12s} {'median':>8s} "
          f"{'p90':>8s} {'p99':>9s} {'censored':>9s}")
    for name, b in summary.items():
        q = b["record_quantiles"]
        print(f"{name:14s} {b['mean_loss']:10.4f} {b['parent_mean_reads']:12.1f} "
              f"{q['p50']:8.0f} {q['p90']:8.0f} {q['p99']:9.0f} {b['n_censored']:9d}")
    print()
    for name, b in ratios.items():
        ci = b["parent_bootstrap_ci"]
        print(f"{name}: {b['parent_mean']:.3f}x [{ci['low']:.3f}, {ci['high']:.3f}], "
              f"median {b['record_median']:.3f}x, learned needs fewer in "
              f"{b['learned_needs_fewer_in_parents']}/{b['n_parents']} parents")
    print(f"\nwrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
