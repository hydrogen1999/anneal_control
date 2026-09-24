"""Emit the actual waveforms a trained model produces, for one instance.

The policy head decodes a schedule directly from structural tokens, and the
critic ranks a pool. Both are real outputs, not descriptions of intent, and
Figure 1(c) needs the waveforms themselves rather than a sketch of them.

For each selected record this writes, on a common uniform tau grid:
the matched linear ramp, every schedule the policy generated, the candidate the
critic picked from the library, and the library's best (an oracle, shown for
scale). Every loss is a true simulator outcome; the selections saw none of them.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--record-ids", nargs="+", required=True)
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
    wanted = set(args.record_ids)
    records = [r for r in load_records(args.data, "test")
               if str(np.asarray(r["record_id"]).item()) in wanted]
    if not records:
        raise SystemExit(f"none of {sorted(wanted)} are in the test split of {args.data}")

    model, normalizer = load_checkpoint(args.checkpoint, device="cpu")
    model.eval()

    def score(record, wave):
        row = np.asarray(wave, dtype=float).copy()
        row[0], row[-1] = 0.0, 1.0
        schedule = Schedule(np.linspace(0.0, 1.0, len(row)), row)
        schedule.validate_slope(runtime=1.0, max_slope=args.max_ds_dtau * (1 + 1e-6))
        return float(score_schedule(record, schedule, backend="numpy",
                                    tolerance=args.tolerance, initial_steps=128,
                                    max_steps=args.max_steps,
                                    max_ds_dtau=args.max_ds_dtau * (1 + 1e-6))["loss"])

    out = []
    for record in records:
        ids = [str(x) for x in np.asarray(record["candidate_ids"])]
        waves = np.asarray(record["candidate_schedules"], dtype=float)
        stored = np.asarray(record["candidate_losses"], dtype=float)
        linear_i = ids.index("linear")

        graph = normalizer.transform(graph_from_record(record, device="cpu"))
        with torch.no_grad():
            proposals = model(graph)["proposal_schedules"].detach().cpu().double().numpy()
            bank_pred = model.predict_losses(
                graph, torch.as_tensor(waves, dtype=torch.float32)).detach().cpu().numpy()
        picked = int(np.argmin(bank_pred))
        oracle = int(np.argmin(stored))

        entry = {
            "record_id": str(np.asarray(record["record_id"]).item()),
            "parent_id": str(np.asarray(record["parent_id"]).item()),
            "runtime": float(np.asarray(record["runtime"]).item()),
            "logical_n": int(np.asarray(record["logical_n"]).item())
            if "logical_n" in record else None,
            "physical_n": int(np.asarray(record["physical_h"]).size),
            "tau_grid": np.linspace(0.0, 1.0, waves.shape[1]).tolist(),
            "schedules": {
                "linear": {"waveform": waves[linear_i].tolist(),
                           "loss": float(stored[linear_i]), "source": "fixed reference"},
                "critic_selected": {"waveform": waves[picked].tolist(),
                                    "loss": float(stored[picked]),
                                    "candidate_id": ids[picked],
                                    "source": "critic argmin over the 64-candidate library"},
                "library_oracle": {"waveform": waves[oracle].tolist(),
                                   "loss": float(stored[oracle]),
                                   "candidate_id": ids[oracle],
                                   "source": "ORACLE: best stored loss, not achievable at "
                                             "deployment; shown for scale only"}},
            "policy_generated": [
                {"waveform": row.tolist(), "loss": score(record, row),
                 "source": "policy head, decoded from structural tokens"}
                for row in proposals],
        }
        entry["notes"] = {
            "feasibility": "every waveform satisfies ds/dtau <= "
                           f"{args.max_ds_dtau} by construction, not by rejection",
            "selection": "the critic saw no outcome; losses were attached afterwards",
        }
        out.append(entry)

    payload = {"schema_version": 1, "checkpoint": str(args.checkpoint),
               "data": str(args.data), "records": out,
               "scope": "illustrative waveforms for one or more instances; the effect "
                        "size is in docs/paper/results/, not here"}
    pathlib.Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")
    for e in out:
        gen = min(p["loss"] for p in e["policy_generated"])
        print(f"{e['record_id']}  phys={e['physical_n']} T={e['runtime']}")
        print(f"   linear            {e['schedules']['linear']['loss']:.5f}")
        print(f"   policy best of {len(e['policy_generated'])}  {gen:.5f}")
        print(f"   critic pick       {e['schedules']['critic_selected']['loss']:.5f}"
              f"  ({e['schedules']['critic_selected']['candidate_id']})")
        print(f"   library oracle    {e['schedules']['library_oracle']['loss']:.5f}"
              f"  ({e['schedules']['library_oracle']['candidate_id']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
