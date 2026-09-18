"""Does a critic trained without an environment still choose well once there is one?

The open-system check already archived asks whether a *fixed* waveform's ranking
survives dephasing. That is not the question a learned method has to answer.
This scores the actual deployed rule -- a critic fit to closed-system labels,
selecting from the bank, never shown a noise rate -- against the same bank
re-solved under a Lindblad master equation.

Three selectors are scored on one shared loss table, which makes the comparison
exact rather than merely matched:

  learned            the trained critic's pick, per method and seed
  closed-system oracle  the best candidate by the stored noiseless label
  noise-aware oracle    the best candidate at that rate

The middle one is the ceiling on any noise-blind rule. The gap between it and
the noise-aware oracle is the intrinsic price of noise-blindness; the gap
between the learned rule and it is the critic's own error. Reporting only the
total would conflate the two and credit or blame the wrong component.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

RATES = (0.0, 0.02, 0.05, 0.1)


def _shard(args):
    data_dir, split, rates, max_qubits, record_ids = args
    from annealctrl.open_system import bank_loss_table
    from annealctrl.pipeline import load_records

    wanted = set(record_ids)
    records = [r for r in load_records(data_dir, split)
               if str(np.asarray(r["record_id"]).item()) in wanted]
    return bank_loss_table(records, rates=rates, max_qubits=max_qubits)["losses"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--experiment", required=True, help="run directory holding models/")
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-qubits", type=int, default=6)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--rates", type=float, nargs="+", default=list(RATES))
    parser.add_argument("--methods", nargs="+", default=["summary"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args(argv)

    from annealctrl.open_system import checkpoint_selector, selection_robustness
    from annealctrl.pipeline import load_records

    everything = load_records(args.data, args.split)
    records = [r for r in everything
               if len(np.asarray(r["physical_h"])) <= args.max_qubits]
    if not records:
        raise SystemExit(f"no record in {args.data}/{args.split} fits {args.max_qubits} qubits")
    ids = [str(np.asarray(r["record_id"]).item()) for r in records]
    print(f"{len(records)}/{len(everything)} records at <= {args.max_qubits} qubits, "
          f"{len({str(np.asarray(r['parent_id']).item()) for r in records})} parents", flush=True)

    # One shared table. Sharded only to spend wall clock, never to change a number:
    # every shard runs the same solver on a disjoint record set.
    chunks = [ids[i::args.workers] for i in range(args.workers)]
    chunks = [c for c in chunks if c]
    losses: dict = {}
    with ProcessPoolExecutor(max_workers=len(chunks)) as pool:
        for part in pool.map(_shard, [(args.data, args.split, args.rates, args.max_qubits, c)
                                      for c in chunks]):
            losses.update(part)
    assert set(losses) == set(ids), "shards did not cover the record set exactly"
    table = {"schema_version": 1, "rates": list(args.rates), "relaxation_rate": 0.0,
             "max_qubits": args.max_qubits, "n_records": len(records), "losses": losses}
    print(f"table solved: {len(losses)} records x {len(args.rates)} rates", flush=True)

    stored = {str(np.asarray(r["record_id"]).item()):
              int(np.argmin(np.asarray(r["candidate_losses"], dtype=float))) for r in records}

    results = {}
    # The ceiling on any rule that cannot see the noise, and the rule that can.
    results["closed_system_oracle"] = selection_robustness(
        records, select=lambda r: stored[str(np.asarray(r["record_id"]).item())],
        rates=args.rates, table=table, max_qubits=args.max_qubits)
    results["noise_aware_oracle"] = selection_robustness(
        records, select=lambda r: None, rates=args.rates, table=table,
        oracle_selects=True, max_qubits=args.max_qubits)

    root = Path(args.experiment)
    for method in args.methods:
        for seed in args.seeds:
            checkpoint = root / "models" / method / f"seed_{seed}" / "best.pt"
            if not checkpoint.exists():
                print(f"skip {method}/seed_{seed}: no {checkpoint}", flush=True)
                continue
            select = checkpoint_selector(checkpoint)
            results[f"{method}/seed_{seed}"] = selection_robustness(
                records, select=select, rates=args.rates, table=table,
                max_qubits=args.max_qubits)
            print(f"scored {method}/seed_{seed}", flush=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({
        "schema_version": 1, "data": args.data, "split": args.split,
        "experiment": str(root), "rates": list(args.rates),
        "max_qubits": args.max_qubits, "n_records": len(records),
        "selectors": results,
        "note": ("one shared Lindblad loss table scores every selector, so the comparison "
                 "between them is exact and not merely matched"),
    }, indent=2, default=float))
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
