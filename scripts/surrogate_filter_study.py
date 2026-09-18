"""Does the critic rank search-generated waveforms, and does that buy budget?

The in-distribution answer is already known and is not the interesting one: on
its own bank the critic reaches rank correlation 0.834 and holds the true best
inside its top half 95.5% of the time. This asks the same question about
waveforms a *search* produced, which is what would turn the model from a picker
into a budget multiplier.

Two hazards are handled rather than assumed away.

The critic needs a uniform tau grid and the search families do not use one.
Re-expressing a search waveform on the bank grid moves it by a sup norm of
0.04-0.07, comparable to the spacing between bank candidates, so the archived
search loss is not a valid label for the regridded waveform. Every candidate is
therefore re-scored with the true simulator after regridding, and the
distribution of representation error is reported.

A checkpoint must not be ranked on the split it was trained on. Record
identities come from the dataset's own validation and test splits; the sweep
only supplies waveforms.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

KEEP = (0.125, 0.25, 0.5)


def _load_trials(sweeps, wanted: set[str]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for sweep in sweeps:
        for path in sorted(Path(sweep).glob("benchmarks/*.json")):
            if path.stem not in wanted:
                continue
            payload = json.loads(path.read_text())
            trials = out.setdefault(path.stem, [])
            for family, block in payload.get("families", {}).items():
                for trial in block.get("records", []):
                    wave = trial.get("waveform") or {}
                    if len(wave.get("tau_knots", [])) >= 2:
                        trials.append({"family": family,
                                       "tau_knots": wave["tau_knots"],
                                       "s_knots": wave["s_knots"],
                                       "archived_loss": trial.get("loss")})
    return {k: v for k, v in out.items() if len(v) >= 4}


def _score_record(args):
    """Re-score every regridded candidate of one record with the true simulator."""
    data_dir, split, record_id, waves, tolerance, max_steps = args
    import numpy as np

    from annealctrl.benchmarking import score_schedule
    from annealctrl.pipeline import load_records
    from annealctrl.schedules import Schedule

    record = next(r for r in load_records(data_dir, split)
                  if str(np.asarray(r["record_id"]).item()) == record_id)
    grid = np.linspace(0.0, 1.0, np.asarray(record["candidate_schedules"]).shape[1])
    losses = []
    for wave in waves:
        w = np.asarray(wave, dtype=float).copy()
        w[0], w[-1] = 0.0, 1.0
        losses.append(float(score_schedule(record, Schedule(grid, w), tolerance=tolerance,
                                           max_steps=max_steps, max_ds_dtau=1e9)["loss"]))
    return record_id, losses


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--sweeps", nargs="+", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--splits", nargs="+", default=["validation", "test"])
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--tolerance", type=float, default=5e-4)
    parser.add_argument("--max-steps", type=int, default=8192)
    args = parser.parse_args(argv)

    import torch

    from annealctrl.learning import load_checkpoint
    from annealctrl.models import graph_from_record
    from annealctrl.pipeline import load_records
    from annealctrl.surrogate_filter import aggregate_filter, filter_curve, regrid

    model, normalizer = load_checkpoint(args.checkpoint, device="cpu")
    model.eval()
    points = int(model.schedule_points)

    results, errors, all_rows = {}, [], []
    for split in args.splits:
        records = {str(np.asarray(r["record_id"]).item()): r for r in load_records(args.data, split)}
        trials = _load_trials(args.sweeps, set(records))
        if not trials:
            print(f"{split}: no overlap with the sweeps", flush=True)
            continue
        print(f"{split}: {len(trials)} records, "
              f"{sum(len(v) for v in trials.values())} archived trials", flush=True)

        regridded = {}
        for rid, entries in trials.items():
            waves, keep = [], []
            for e in entries:
                out = regrid(e["tau_knots"], e["s_knots"], points=points)
                errors.append(out["representation_error"])
                waves.append(out["waveform"])
                keep.append(e)
            regridded[rid] = (np.asarray(waves), keep)

        jobs = [(args.data, split, rid, waves, args.tolerance, args.max_steps)
                for rid, (waves, _) in regridded.items()]
        true_by_record = {}
        with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
            for rid, losses in pool.map(_score_record, jobs):
                true_by_record[rid] = np.asarray(losses, dtype=float)
                print(f"  rescored {rid}: {len(losses)} candidates", flush=True)

        rows = []
        for rid, (waves, entries) in regridded.items():
            record = records[rid]
            graph = normalizer.transform(graph_from_record(record, device="cpu"))
            with torch.no_grad():
                predicted = model.predict_losses(
                    graph, torch.as_tensor(waves, dtype=torch.float32)).cpu().numpy()
            true = true_by_record[rid]
            row = filter_curve(predicted, true, keep_fractions=KEEP)
            row.update({"record_id": rid, "split": split,
                        "parent_id": str(np.asarray(record["parent_id"]).item()),
                        "n_candidates": int(len(true)),
                        "families": sorted({e["family"] for e in entries}),
                        # Persisted so the headline can be re-analysed without
                        # re-simulating: a summary nobody can audit is a claim,
                        # not evidence.
                        "predicted_losses": [float(x) for x in predicted],
                        "true_losses": [float(x) for x in true],
                        "candidate_families": [e["family"] for e in entries]})
            rows.append(row)
        results[split] = aggregate_filter(rows, keep_fractions=KEEP)
        all_rows.extend(rows)

    if not all_rows:
        raise SystemExit("no held-out record overlapped the sweeps")
    errors = np.asarray(errors, dtype=float)
    payload = {
        "schema_version": 1, "data": args.data, "sweeps": list(args.sweeps),
        "checkpoint": args.checkpoint, "schedule_points": points,
        "keep_fractions": list(KEEP),
        "representation_error": {
            "mean": float(errors.mean()), "median": float(np.median(errors)),
            "p90": float(np.percentile(errors, 90)), "max": float(errors.max()),
            "note": ("sup norm between the original piecewise-linear waveform and its "
                     "regridded form; every candidate was re-scored after regridding, so "
                     "prediction and label describe the same object and this error does "
                     "not enter the ranking")},
        "by_split": results,
        "pooled": aggregate_filter(all_rows, keep_fractions=KEEP),
        "rows": all_rows,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2, default=float))
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
