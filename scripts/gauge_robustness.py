"""Does gauge augmentation buy robustness to a gauge it has not seen?

The in-distribution ablation measures only half of what the design document
asks. Its factor-6 question is whether a stronger inductive bias "improves
robustness without deleting frustration", and the test split stores one
particular gauge, so a model specialised to that gauge is never penalised for
being specialised.

A spin reversal leaves the spectrum invariant, so **every stored candidate loss
remains the correct label under it**. That makes the robustness test cheap: put
each test record into a fresh random gauge, ask each checkpoint to select from
the same bank, and score the selection against the labels the dataset already
holds. No simulation is required.

Reported per checkpoint as the loss in the stored gauge, the loss averaged over
random gauges, and the gap between them -- which is the quantity the claim is
about.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--methods", nargs="+", default=["signed_baseline", "gauge_augmented"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--gauges", type=int, default=8, help="random gauges per record")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    import torch

    from annealctrl.gauge import gauge_record, gauge_signs
    from annealctrl.headroom import _bootstrap, _parent_means
    from annealctrl.learning import load_checkpoint
    from annealctrl.models import graph_from_record
    from annealctrl.pipeline import load_records

    records = load_records(args.data, args.split)
    print(f"{len(records)} records, {args.gauges} random gauges each", flush=True)
    rng = np.random.default_rng(args.seed)
    # One gauge set, shared by every checkpoint, so the arms see identical inputs.
    gauges = {str(np.asarray(r["record_id"]).item()):
              [gauge_signs(int(np.asarray(r["logical_h"]).size), rng) for _ in range(args.gauges)]
              for r in records}

    results = {}
    for method in args.methods:
        for seed in args.seeds:
            path = Path(args.experiment) / "models" / method / f"seed_{seed}" / "best.pt"
            if not path.exists():
                print(f"skip {method}/seed_{seed}", flush=True)
                continue
            model, normalizer = load_checkpoint(path, device="cpu")
            model.eval()
            rows = []
            for record in records:
                rid = str(np.asarray(record["record_id"]).item())
                losses = np.asarray(record["candidate_losses"], dtype=float)
                bank = torch.as_tensor(np.asarray(record["candidate_schedules"], dtype=float),
                                       dtype=torch.float32)

                def pick(source) -> float:
                    graph = normalizer.transform(graph_from_record(source, device="cpu"))
                    with torch.no_grad():
                        return float(losses[int(model.predict_losses(graph, bank).argmin())])

                stored = pick(record)
                gauged = [pick(gauge_record(record, s)) for s in gauges[rid]]
                rows.append({"record_id": rid,
                             "parent_id": str(np.asarray(record["parent_id"]).item()),
                             "stored_gauge_loss": stored,
                             "mean_gauged_loss": float(np.mean(gauged)),
                             "worst_gauged_loss": float(np.max(gauged)),
                             "gauge_penalty": float(np.mean(gauged)) - stored,
                             "selection_unchanged_fraction": float(
                                 np.mean([abs(g - stored) < 1e-12 for g in gauged]))})
            block = {"n_records": len(rows)}
            for key in ("stored_gauge_loss", "mean_gauged_loss", "gauge_penalty",
                        "selection_unchanged_fraction"):
                values, parents = _parent_means(rows, lambda r, k=key: r[k])
                block[key] = {"mean": float(values.mean()),
                              "parent_bootstrap_ci": _bootstrap(values, n_resamples=20000, seed=0),
                              "n_parents": len(parents)}
            block["rows"] = rows
            results[f"{method}/seed_{seed}"] = block
            print(f"{method}/seed_{seed}: stored {block['stored_gauge_loss']['mean']:.5f}  "
                  f"gauged {block['mean_gauged_loss']['mean']:.5f}  "
                  f"penalty {block['gauge_penalty']['mean']:+.5f}  "
                  f"selection unchanged {block['selection_unchanged_fraction']['mean']:.1%}",
                  flush=True)

    payload = {"schema_version": 1, "data": args.data, "split": args.split,
               "experiment": str(args.experiment), "gauges_per_record": args.gauges,
               "selectors": results,
               "scope": ("a spin reversal leaves the spectrum invariant, so the stored "
                         "candidate losses remain the correct labels under it; no simulation "
                         "is needed and none is performed. Every checkpoint sees the same "
                         "random gauges.")}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2, default=float))
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
