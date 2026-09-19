"""The filter inside the search loop: same simulator budget, spent better?

The offline study established the prerequisite -- the critic ranks
search-generated waveforms at rho ~ 0.93 -- but it re-ranked a completed trace,
so it could not claim a saving. This runs the search twice on the same record
with the same seed and the same number of true evaluations:

    baseline   each simulator call goes to the next proposal
    filtered   each simulator call goes to the best of `oversample` proposals,
               chosen by the critic, which never sees a true loss

The budget is simulator calls and is identical in both arms; a test asserts that
directly. The discarded proposals are free because they are never simulated.

One asymmetry is deliberate and is not a flaw. The critic scores a waveform
re-expressed on its nine-point grid, which differs from the real waveform by a
sup norm of ~0.05, while the true loss is always the simulator on the real
waveform. The surrogate is therefore working from a blurred view. If it helps
anyway, it helps under a handicap.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

FAMILIES = ("one_window", "two_window", "eight_bin")


def _run_record(args):
    (data_dir, split, record_id, checkpoint, budget, oversample, seed,
     tolerance, max_steps, chooser) = args
    import numpy as np
    import torch

    from annealctrl.benchmarking import score_schedule
    from annealctrl.learning import load_checkpoint
    from annealctrl.models import graph_from_record
    from annealctrl.pipeline import load_records
    from annealctrl.search import optimize_control_family

    record = next(r for r in load_records(data_dir, split)
                  if str(np.asarray(r["record_id"]).item()) == record_id)
    model, normalizer = load_checkpoint(checkpoint, device="cpu")
    model.eval()
    points = int(model.schedule_points)
    graph = normalizer.transform(graph_from_record(record, device="cpu"))
    with torch.no_grad():
        encoded = model.encode(graph)
    grid = np.linspace(0.0, 1.0, points)

    calls = {"n": 0}

    def loss_fn(schedule):
        calls["n"] += 1
        return float(score_schedule(record, schedule, tolerance=tolerance,
                                    max_steps=max_steps, max_ds_dtau=1e9)["loss"])

    def critic(schedules):
        waves = np.stack([np.clip(np.interp(grid, s.tau_knots, s.s_knots), 0.0, 1.0)
                          for s in schedules])
        waves[:, 0], waves[:, -1] = 0.0, 1.0
        with torch.no_grad():
            return model.predict_losses(
                graph, torch.as_tensor(waves, dtype=torch.float32), encoded).cpu().numpy()

    # The control that decides whether the model earned the improvement. The
    # filtered arm sees `oversample` times as many proposals, so it walks a
    # wider slice of the Sobol sequence than the baseline does. If picking one
    # of eight AT RANDOM also wins, the gain is the wider slice and not the
    # critic, and the whole claim collapses.
    control_rng = np.random.default_rng(abs(hash((record_id, seed))) % (2**32))

    def random_chooser(schedules):
        return control_rng.random(len(schedules))

    # The third arm separates two explanations of why filtering helps. If the
    # critic only has to avoid the bad tail, rejecting the worst half and then
    # choosing at random should recover most of the gain. If fine ranking among
    # the survivors is what matters, it should not. This is the discriminating
    # experiment for the Pegasus result, where the coarse ranking transfers
    # (rho +0.82) but the top-10% ranking collapses (rho +0.16) and the in-loop
    # gain survives anyway.
    def reject_worst(schedules):
        scores = np.asarray(critic(schedules), dtype=float)
        keep = max(1, len(scores) // 2)
        survivors = np.argsort(scores, kind="stable")[:keep]
        out = np.full(len(scores), np.inf)
        out[survivors] = control_rng.random(keep)
        return out

    surrogate = {"critic": critic, "random": random_chooser,
                 "reject_worst": reject_worst}[chooser]

    out = {"record_id": record_id,
           "parent_id": str(np.asarray(record["parent_id"]).item()), "split": split,
           "families": {}}
    for family in FAMILIES:
        calls["n"] = 0
        base = optimize_control_family(loss_fn, family, budget=budget, seed=seed, split=split)
        base_calls = calls["n"]
        calls["n"] = 0
        filt = optimize_control_family(loss_fn, family, budget=budget, seed=seed, split=split,
                                       surrogate=surrogate, oversample=oversample)
        out["families"][family] = {
            "baseline_best": float(base.best.loss),
            "filtered_best": float(filt.best.loss),
            "improvement": float(base.best.loss - filt.best.loss),
            "baseline_calls": base_calls, "filtered_calls": calls["n"],
            "proposals_considered": int(budget - 1) * oversample + 1}
    best_base = min(v["baseline_best"] for v in out["families"].values())
    best_filt = min(v["filtered_best"] for v in out["families"].values())
    out.update({"baseline_best": best_base, "filtered_best": best_filt,
                "improvement": best_base - best_filt})
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--budget", type=int, default=32)
    parser.add_argument("--oversample", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--records", type=int, default=48)
    parser.add_argument("--max-qubits", type=int, default=8)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--tolerance", type=float, default=5e-4)
    parser.add_argument("--max-steps", type=int, default=8192)
    parser.add_argument("--chooser", choices=("critic", "random", "reject_worst"),
                        default="critic",
                        help="'random' is the control: same oversampled proposal stream, "
                             "chosen without the model. 'reject_worst' keeps the critic's "
                             "better half and then chooses at random among it, which "
                             "separates tail-avoidance from fine ranking.")
    args = parser.parse_args(argv)

    from annealctrl.headroom import _bootstrap, _describe, _parent_means
    from annealctrl.pipeline import load_records

    records = [r for r in load_records(args.data, args.split)
               if len(np.asarray(r["physical_h"])) <= args.max_qubits]
    # One record per parent, so the unit of independence is not diluted by
    # near-duplicate siblings before the bootstrap ever sees them.
    seen, chosen = set(), []
    for r in sorted(records, key=lambda r: str(np.asarray(r["record_id"]).item())):
        pid = str(np.asarray(r["parent_id"]).item())
        if pid in seen:
            continue
        seen.add(pid)
        chosen.append(str(np.asarray(r["record_id"]).item()))
        if len(chosen) >= args.records:
            break
    print(f"{len(chosen)} records, one per parent, <= {args.max_qubits} qubits", flush=True)

    jobs = [(args.data, args.split, rid, args.checkpoint, args.budget, args.oversample,
             args.seed, args.tolerance, args.max_steps, args.chooser) for rid in chosen]
    rows = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
        for row in pool.map(_run_record, jobs):
            rows.append(row)
            print(f"  {row['record_id']}: baseline {row['baseline_best']:.5f} "
                  f"filtered {row['filtered_best']:.5f} "
                  f"improvement {row['improvement']:+.5f}", flush=True)

    mismatched = [r for r in rows for v in r["families"].values()
                  if v["baseline_calls"] != v["filtered_calls"]]
    values, parents = _parent_means(rows, lambda r: r["improvement"])
    block = _describe(values)
    block["parent_bootstrap_ci"] = _bootstrap(values, n_resamples=4000, seed=0)
    per_family = {}
    for family in FAMILIES:
        v, _ = _parent_means(rows, lambda r, f=family: r["families"][f]["improvement"])
        b = _describe(v)
        b["parent_bootstrap_ci"] = _bootstrap(v, n_resamples=4000, seed=0)
        per_family[family] = b

    payload = {
        "schema_version": 1, "data": args.data, "split": args.split,
        "checkpoint": args.checkpoint, "budget": args.budget,
        "oversample": args.oversample, "seed": args.seed, "chooser": args.chooser,
        "n_records": len(rows), "n_parents": len(parents),
        "budget_mismatches": len(mismatched),
        "improvement_best_over_families": block,
        "improvement_per_family": per_family,
        "filtered_wins": sum(1 for r in rows if r["improvement"] > 0),
        "ties": sum(1 for r in rows if r["improvement"] == 0),
        "rows": rows,
        "scope": ("both arms spend exactly `budget` true simulator calls per family and "
                  "share a seed; the surrogate scores a nine-point regridding of each "
                  "proposal while the label is always the simulator on the real waveform"),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2, default=float))
    ci = block["parent_bootstrap_ci"]
    print(f"\nimprovement {block['mean']:+.5f} [{ci['low']:+.5f},{ci['high']:+.5f}] "
          f"over {len(parents)} parents; filtered wins {payload['filtered_wins']}/{len(rows)}, "
          f"ties {payload['ties']}; budget mismatches {len(mismatched)}", flush=True)
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
