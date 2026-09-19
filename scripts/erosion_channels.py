"""Is the erosion effect specific to dephasing, and does it grow with rate?

If it appeared only under one hand-chosen perturbation it would be a curiosity
about that perturbation. This runs the same within-record correlation across
two channels and three rates, and reports the headroom-normalised figure beside
the raw one because the [0,1] ceiling inflates the raw one.

Produces the table quoted in reports/erosion_2026-09-18/EROSION.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

CHANNELS = ("dephasing", "relaxation")


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
    parser.add_argument("--rates", type=float, nargs="+", default=[0.02, 0.05, 0.1])
    parser.add_argument("--records", type=int, default=100)
    parser.add_argument("--max-qubits", type=int, default=6)
    args = parser.parse_args(argv)

    from scipy.stats import rankdata

    from annealctrl.open_system import bank_loss_table
    from annealctrl.pipeline import load_records

    records = [r for r in load_records(args.data, args.split)
               if len(np.asarray(r["physical_h"])) <= args.max_qubits]
    records = sorted(records, key=lambda r: str(np.asarray(r["record_id"]).item()))[:args.records]
    if not records:
        raise SystemExit("no record fits the qubit cap")
    print(f"{len(records)} records, "
          f"{np.asarray(records[0]['candidate_schedules']).shape[0]} candidates", flush=True)

    # The noiseless table is solved once and reused by every channel and rate;
    # it is the same physics in all of them and resolving it would only add
    # solver noise to a comparison that is supposed to isolate the channel.
    baseline = bank_loss_table(records, rates=[0.0], max_qubits=args.max_qubits)["losses"]
    rows = []
    for channel in CHANNELS:
        for rate in args.rates:
            if channel == "dephasing":
                t = bank_loss_table(records, rates=[0.0, rate], max_qubits=args.max_qubits)["losses"]
                pick = lambda rid, r=rate: np.asarray(t[rid][str(r)], float)
            else:
                t = bank_loss_table(records, rates=[0.0], relaxation_rate=rate,
                                    max_qubits=args.max_qubits)["losses"]
                pick = lambda rid: np.asarray(t[rid]["0.0"], float)
            raw, frac, ranks = [], [], []
            for record in records:
                rid = str(np.asarray(record["record_id"]).item())
                l0 = np.asarray(baseline[rid]["0.0"], float)
                deg = pick(rid) - l0
                normalised = deg / np.maximum(1.0 - l0, 1e-12)
                for acc, x in ((raw, _rho(l0, deg)), (frac, _rho(l0, normalised))):
                    if x is not None:
                        acc.append(x)
                ranks.append(float(rankdata(normalised)[int(np.argmin(l0))]))
            rows.append({"channel": channel, "rate": rate,
                         "rho_raw": float(np.mean(raw)),
                         "rho_headroom_normalised": float(np.mean(frac)),
                         "negative": int(sum(1 for c in frac if c < 0)), "n": len(frac),
                         "noiseless_best_degradation_rank": float(np.mean(ranks)),
                         "chance_rank": (len(l0) + 1) / 2})
            r = rows[-1]
            print(f"{channel:12s} {rate:>5.2f} raw {r['rho_raw']:+.4f} "
                  f"headroom {r['rho_headroom_normalised']:+.4f} "
                  f"neg {r['negative']}/{r['n']} rank {r['noiseless_best_degradation_rank']:.2f}",
                  flush=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(
        {"schema_version": 1, "data": args.data, "split": args.split,
         "n_records": len(records), "rows": rows,
         "scope": ("uniform single-qubit channels at a declared rate, not a calibrated "
                   "device model; the noiseless table is shared across channels so the "
                   "comparison isolates the channel rather than the solver")},
        indent=2, default=float))
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
