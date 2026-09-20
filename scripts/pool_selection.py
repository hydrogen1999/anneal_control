"""The design document's candidate pool: the policy's proposals plus a fallback.

The document specifies three branches -- spectral auxiliaries, a schedule
proposed directly from structural tokens, and a critic that evaluates
"proposed schedules ... alongside the direct proposal and simple baselines"
(main.tex:572). The project implemented the first two and a critic, but the
critic only ever ranked the policy's OWN proposals. The "simple baselines"
half of the pool was never there.

That omission is not cosmetic. On the 48-parent Pegasus arm the direct policy
loses to a linear ramp on 22 of 48 parents and changes sign across training
seeds. Every one of those is a case where a pool containing the linear ramp
could have recovered, IF the critic can tell that its own proposal is the
worse of the two. Whether it can is exactly what this measures.

**Cost class.** The pool here is amortised: the policy's proposals and a fixed
linear ramp, ranked by the critic in one forward pass, no simulation. The
document also puts physics-derived schedules (D2, gap) in the pool, but
main.tex:637 forbids recomputing the spectral teacher at deployment unless the
cost is counted as a separate solver-assisted method. Those belong to the
privileged_spectrum class and are deliberately NOT mixed in here.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--tolerance", type=float, default=2e-4)
    parser.add_argument("--max-steps", type=int, default=16384)
    parser.add_argument("--max-ds-dtau", type=float, default=4.0)
    args = parser.parse_args(argv)

    import torch

    from annealctrl.benchmarking import _numerical_settings, score_schedule
    from annealctrl.learning import load_checkpoint
    from annealctrl.models import graph_from_record
    from annealctrl.pipeline import load_records
    from annealctrl.schedules import Schedule

    _numerical_settings(args.tolerance, 128, args.max_steps, 1e-9, "numpy")
    records = load_records(args.data, args.split)
    model, normalizer = load_checkpoint(args.checkpoint, device="cpu")
    model.eval()

    def scored(record, wave):
        row = np.asarray(wave, dtype=float).copy()
        row[0], row[-1] = 0.0, 1.0
        schedule = Schedule(np.linspace(0.0, 1.0, len(row)), row)
        schedule.validate_slope(runtime=1.0, max_slope=args.max_ds_dtau * (1 + 1e-6))
        return float(score_schedule(record, schedule, backend="numpy", tolerance=args.tolerance,
                                    initial_steps=128, max_steps=args.max_steps,
                                    max_ds_dtau=args.max_ds_dtau * (1 + 1e-6))["loss"])

    rows, infeasible = [], 0
    for record in records:
        ids = [str(x) for x in np.asarray(record["candidate_ids"])]
        if "linear" not in ids:
            raise ValueError(f"{record['record_id']}: bank has no 'linear' entry to fall back to")
        linear_wave = np.asarray(record["candidate_schedules"])[ids.index("linear")]
        linear_loss = float(np.asarray(record["candidate_losses"])[ids.index("linear")])

        graph = normalizer.transform(graph_from_record(record, device="cpu"))
        with torch.no_grad():
            proposals = model(graph)["proposal_schedules"].detach().cpu().double().numpy()
            pool = np.vstack([proposals, linear_wave[None, :]])
            predicted = model.predict_losses(
                graph, torch.as_tensor(pool, dtype=torch.float32)).detach().cpu().numpy()

        fallback = len(pool) - 1
        try:
            # The proposal-only choice is what the project reported as "direct".
            proposal_pick = int(np.argmin(predicted[:fallback]))
            pool_pick = int(np.argmin(predicted))
            proposal_loss = scored(record, pool[proposal_pick])
            pool_loss = proposal_loss if pool_pick == proposal_pick else (
                linear_loss if pool_pick == fallback else scored(record, pool[pool_pick]))
        except ValueError:
            infeasible += 1
            continue

        rows.append({"record_id": str(np.asarray(record["record_id"]).item()),
                     "parent_id": str(np.asarray(record["parent_id"]).item()),
                     "runtime": float(np.asarray(record["runtime"]).item()),
                     "linear_loss": linear_loss,
                     "proposal_only_loss": proposal_loss,
                     "pool_loss": pool_loss,
                     "chose_fallback": bool(pool_pick == fallback)})

    if not rows:
        raise ValueError("no feasible records")

    def parent_mean(key):
        buckets = defaultdict(list)
        for row in rows:
            buckets[row["parent_id"]].append(row[key])
        return np.array([np.mean(buckets[p]) for p in sorted(buckets)])

    linear, proposal, pool_ = parent_mean("linear_loss"), parent_mean("proposal_only_loss"), parent_mean("pool_loss")
    rng = np.random.default_rng(0)

    def ci(diff):
        draws = np.array([diff[rng.integers(len(diff), size=len(diff))].mean() for _ in range(20000)])
        return [float(x) for x in np.quantile(draws, [0.025, 0.975])]

    result = {
        "schema_version": 1, "checkpoint": str(args.checkpoint), "split": args.split,
        "cost_class": "amortised",
        "n_records": len(rows), "n_parents": int(linear.size), "n_infeasible": infeasible,
        "fallback_rate_records": float(np.mean([r["chose_fallback"] for r in rows])),
        "proposal_only_vs_linear": {"mean": float((proposal - linear).mean()),
                                    "ci": ci(proposal - linear),
                                    "parents_beaten": int((proposal < linear).sum())},
        "pool_vs_linear": {"mean": float((pool_ - linear).mean()), "ci": ci(pool_ - linear),
                           "parents_beaten": int((pool_ < linear).sum())},
        "pool_vs_proposal_only": {"mean": float((pool_ - proposal).mean()),
                                  "ci": ci(pool_ - proposal)},
        "parents_rescued": int(((proposal > linear) & (pool_ <= linear)).sum()),
        "parents_harmed": int(((proposal <= linear) & (pool_ > linear)).sum()),
        "rows": rows,
        "scope": ("negative favours the first named method; the pool is the policy's own "
                  "proposals plus the fixed linear ramp, ranked by the critic in one forward "
                  "pass with no simulation. Physics-derived pool members would be "
                  "privileged_spectrum and are excluded, not forgotten."),
    }
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(f"{args.split}: {result['n_parents']} parents, {result['n_records']} records")
    for key in ("proposal_only_vs_linear", "pool_vs_linear"):
        b = result[key]
        print(f"  {key:26s} {b['mean']:+.5f} [{b['ci'][0]:+.5f}, {b['ci'][1]:+.5f}]  "
              f"beats linear on {b['parents_beaten']}/{result['n_parents']}")
    d = result["pool_vs_proposal_only"]
    print(f"  {'pool_vs_proposal_only':26s} {d['mean']:+.5f} [{d['ci'][0]:+.5f}, {d['ci'][1]:+.5f}]")
    print(f"  fallback chosen in {result['fallback_rate_records']*100:.1f}% of records; "
          f"rescued {result['parents_rescued']} parents, harmed {result['parents_harmed']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
