"""Diagnostics behind the ranking claims, in one auditable place.

Three numbers quoted in the reports are produced here rather than by throwaway
scripts: the critic's in-distribution rank correlation on its own bank, the
breakdown of ranking quality by system size, and whether the bank is shared
across records at all -- which is what made the erosion result's candidate
fixed effect necessary.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np


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
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="test")
    args = parser.parse_args(argv)

    import torch

    from annealctrl.learning import load_checkpoint
    from annealctrl.models import graph_from_record
    from annealctrl.pipeline import load_records

    records = load_records(args.data, args.split)
    banks = {np.asarray(r["candidate_schedules"], float).tobytes() for r in records}
    winners = collections.Counter(
        int(np.argmin(np.asarray(r["candidate_losses"], float))) for r in records)
    n_cand = int(np.asarray(records[0]["candidate_schedules"]).shape[0])
    print(f"{len(records)} records, {n_cand} candidates, "
          f"{len(banks)} distinct bank(s)", flush=True)
    print("noiseless winner concentration: " +
          ", ".join(f"{k}:{v/len(records):.1%}" for k, v in winners.most_common(4)), flush=True)

    per_checkpoint = {}
    for path in args.checkpoints:
        model, normalizer = load_checkpoint(path, device="cpu")
        model.eval()
        by_size, correlations, top1, tophalf = collections.defaultdict(list), [], 0, 0
        for record in records:
            graph = normalizer.transform(graph_from_record(record, device="cpu"))
            bank = torch.as_tensor(np.asarray(record["candidate_schedules"], float),
                                   dtype=torch.float32)
            with torch.no_grad():
                predicted = model.predict_losses(graph, bank).cpu().numpy()
            true = np.asarray(record["candidate_losses"], float)
            c = _rho(predicted, true)
            if c is not None:
                correlations.append(c)
                by_size[len(np.asarray(record["physical_h"]))].append(c)
            top1 += int(np.argmin(predicted) == np.argmin(true))
            keep = max(1, len(true) // 2)
            tophalf += int(np.argmin(true) in set(np.argsort(predicted)[:keep]))
        name = str(Path(path).parent.parent.name) + "/" + str(Path(path).parent.name)
        per_checkpoint[name] = {
            "mean_rank_correlation": float(np.mean(correlations)),
            "n_records": len(records),
            "picks_true_best": top1 / len(records),
            "true_best_in_top_half": tophalf / len(records),
            "by_physical_qubits": {int(k): {"mean_rho": float(np.mean(v)), "n": len(v)}
                                   for k, v in sorted(by_size.items())}}
        b = per_checkpoint[name]
        print(f"{name}: rho {b['mean_rank_correlation']:+.4f}, picks best "
              f"{b['picks_true_best']:.1%}, true best in top half "
              f"{b['true_best_in_top_half']:.1%}", flush=True)
        for k, v in b["by_physical_qubits"].items():
            print(f"    {k:>2} qubits: {v['mean_rho']:+.4f}  (n={v['n']})", flush=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(
        {"schema_version": 1, "data": args.data, "split": args.split,
         "n_records": len(records), "n_candidates": n_cand,
         "n_distinct_banks": len(banks),
         "noiseless_winner_counts": {int(k): int(v) for k, v in winners.items()},
         "checkpoints": per_checkpoint,
         "scope": ("in-distribution ranking on the bank the critic was trained against; "
                   "the shared-bank count is reported because a shared bank makes a "
                   "candidate fixed effect necessary before any within-record claim")},
        indent=2, default=float))
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
