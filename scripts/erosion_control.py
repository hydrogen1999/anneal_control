"""Is the erosion result about optimisation, or about two fragile waveforms?

The bank here is shared: all 558 test records at <=6 qubits draw from ONE set of
eight waveforms, and the noiseless winner is candidate 2 in 47.5% of records and
candidate 3 in 32.4%. If those two happen to be noise-fragile, a within-record
correlation between noiseless quality and degradation appears with no general
principle behind it at all.

The control is a candidate fixed effect. Subtract each candidate's own mean
across records from both quantities, leaving only variation that is specific to
the instance -- "for THIS problem, candidate j is unusually good / unusually
fragile". If the correlation survives demeaning, the effect is about which
control is best for a given instance. If it collapses, it was two waveforms.

Reported alongside the raw number, never instead of it.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


def _shard(args):
    data_dir, split, rate, record_ids, max_qubits = args
    import numpy as np

    from annealctrl.open_system import bank_loss_table
    from annealctrl.pipeline import load_records

    wanted = set(record_ids)
    records = [r for r in load_records(data_dir, split)
               if str(np.asarray(r["record_id"]).item()) in wanted]
    # The density solver's own cap is 8; pass the pool's actual size so a study
    # at 7-8 qubits is not silently refused by a default meant for 6.
    return bank_loss_table(records, rates=[0.0, rate],
                           max_qubits=max(max_qubits, 8))["losses"]


def _rho(a, b):
    from scipy.stats import rankdata

    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 3 or np.all(a == a[0]) or np.all(b == b[0]):
        return None
    ra, rb = rankdata(a) - rankdata(a).mean(), rankdata(b) - rankdata(b).mean()
    d = float(np.sqrt((ra**2).sum() * (rb**2).sum()))
    return None if d == 0 else float(ra @ rb / d)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--rate", type=float, default=0.1)
    parser.add_argument("--records", type=int, default=200)
    parser.add_argument("--max-qubits", type=int, default=6)
    parser.add_argument("--min-qubits", type=int, default=0,
                        help="lower bound too, so the effect can be checked at a "
                             "size range rather than only up to a cap")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)

    from annealctrl.pipeline import load_records

    records = [r for r in load_records(args.data, args.split)
               if args.min_qubits <= len(np.asarray(r["physical_h"])) <= args.max_qubits]
    if not records:
        raise SystemExit(f"no record in {args.min_qubits}..{args.max_qubits} qubits")
    ids = sorted(str(np.asarray(r["record_id"]).item()) for r in records)[:args.records]
    banks = {np.asarray(r["candidate_schedules"], float).tobytes() for r in records}
    print(f"{len(ids)} records, {len(banks)} distinct bank(s) in the pool", flush=True)

    chunks = [ids[i::args.workers] for i in range(args.workers)]
    chunks = [c for c in chunks if c]
    losses: dict = {}
    with ProcessPoolExecutor(max_workers=len(chunks)) as pool:
        for part in pool.map(_shard, [(args.data, args.split, args.rate, c, args.max_qubits)
                                      for c in chunks]):
            losses.update(part)
    assert set(losses) == set(ids), "shards did not cover the record set"

    L0 = np.array([losses[i]["0.0"] for i in ids], dtype=float)
    L1 = np.array([losses[i][str(args.rate)] for i in ids], dtype=float)
    DEG = L1 - L0
    FRAC = DEG / np.maximum(1.0 - L0, 1e-12)

    # Candidate fixed effect: each column's own mean across records.
    def demean(M):
        return M - M.mean(axis=0, keepdims=True)

    out = {}
    for name, x, y in (("raw", L0, DEG),
                       ("headroom_normalised", L0, FRAC),
                       ("candidate_demeaned", demean(L0), demean(DEG)),
                       ("candidate_demeaned_headroom", demean(L0), demean(FRAC))):
        vals = [c for c in (_rho(x[i], y[i]) for i in range(len(ids))) if c is not None]
        out[name] = {"mean": float(np.mean(vals)), "median": float(np.median(vals)),
                     "negative": int(sum(1 for c in vals if c < 0)), "n": len(vals)}

    # Per-candidate mean degradation: the fixed effect itself, so a reader can
    # see how much of the story two waveforms could have carried.
    per_candidate = {int(j): {"mean_noiseless_loss": float(L0[:, j].mean()),
                              "mean_degradation": float(DEG[:, j].mean()),
                              "won_noiseless": int((L0.argmin(axis=1) == j).sum())}
                     for j in range(L0.shape[1])}

    payload = {"schema_version": 1, "data": args.data, "split": args.split,
               "rate": args.rate, "n_records": len(ids),
               "qubit_range": [args.min_qubits, args.max_qubits], "n_candidates": int(L0.shape[1]),
               "n_distinct_banks_in_pool": len(banks),
               "correlations": out, "per_candidate": per_candidate,
               "scope": ("candidate_demeaned removes each waveform's own mean across records, "
                         "leaving only instance-specific variation; the shared bank makes that "
                         "control necessary rather than optional")}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2, default=float))

    print(f"\n{'analysis':32s} {'mean rho':>10s} {'median':>9s} {'negative':>10s}")
    for k, v in out.items():
        print(f"{k:32s} {v['mean']:>+10.4f} {v['median']:>+9.4f} {v['negative']:>6d}/{v['n']}")
    print(f"\n{'cand':>4s} {'mean noiseless':>15s} {'mean degradation':>18s} {'won':>6s}")
    for j, v in per_candidate.items():
        print(f"{j:>4d} {v['mean_noiseless_loss']:>15.5f} {v['mean_degradation']:>18.5f} "
              f"{v['won_noiseless']:>6d}")
    print(f"\nwrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
