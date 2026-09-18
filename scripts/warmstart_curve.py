"""Does a learned selection make a search cheaper? A budget curve, cold against warm.

The comparison table has two isolated points: an amortised selector at zero
simulator calls and a search at 257. "The learned method loses by 0.039" is the
wrong reading of that, because the deployment question is how many calls a search
needs when it starts from the model's answer.

Equal footing is the whole design. Both arms search the same five families with
the same per-family budget and the same seed. The only difference is the
incumbent: the cold arm starts from linear, the warm arm spends one of its calls
evaluating the model's highest-ranked bank candidate *of that family* and refines
from there. Trial 0 is linear in both, so the linear reference every headroom
number is defined against is untouched.

The model never sees an outcome. It ranks the stored bank, and the hint is
whichever candidate it ranked first.
"""
import json, sys, time
sys.path.insert(0, "/home/nguyencongt/annealctrl_dagger_src")
import numpy as np
import torch

from annealctrl.benchmarking import record_physics, score_schedule
from annealctrl.learning import load_checkpoint
from annealctrl.models import graph_from_record
from annealctrl.pipeline import load_records
from annealctrl.search import (bank_candidate_to_unit_parameters, optimize_control_family,
                               shared_candidate_bank)

DATA = "/home/nguyencongt/runs/research_v1/data"
CHECKPOINT = "/home/nguyencongt/runs/research_v1/models/summary/seed_0/best.pt"
BUDGETS = [8, 16, 32, 64]
FAMILIES = ["linear", "one_window", "two_window", "eight_bin", "pause"]
NUM = dict(backend="numpy", tolerance=2e-4, initial_steps=128, max_steps=16384)
SHARD = int(sys.argv[1]) if len(sys.argv) > 1 else 0
SHARDS = int(sys.argv[2]) if len(sys.argv) > 2 else 1
N_PARENTS = int(sys.argv[3]) if len(sys.argv) > 3 else 24

records = load_records(DATA, "test")
parents = sorted({str(np.asarray(r["parent_id"]).item()) for r in records})[:N_PARENTS]
records = [r for r in records if str(np.asarray(r["parent_id"]).item()) in parents]
records = [r for i, r in enumerate(records) if i % SHARDS == SHARD]
print(f"shard {SHARD}/{SHARDS}: {len(records)} records over <= {N_PARENTS} parents", flush=True)

model, normalizer = load_checkpoint(CHECKPOINT, device="cpu")
model.eval()

rows, began = [], time.time()
for index, record in enumerate(records):
    runtime = float(np.asarray(record["runtime"]).item())
    max_slope = 4.0 / runtime
    context = record_physics(record)

    def loss_fn(schedule, record=record, context=context):
        return float(score_schedule(record, schedule, physics_context=context,
                                    max_ds_dtau=4.0 * (1 + 1e-6), **NUM)["loss"])

    # The model ranks the stored bank; no outcome is consulted.
    graph = graph_from_record(record, device="cpu")
    if normalizer is not None:
        graph = normalizer.transform(graph)
    with torch.no_grad():
        bank_tensor = torch.as_tensor(record["candidate_schedules"], dtype=torch.float32)
        predicted = model.predict_losses(graph, bank_tensor).cpu().numpy()

    bank = shared_candidate_bank(n=len(predicted), n_segments=8, seed=20260916,
                                 runtime=runtime, max_slope=max_slope)
    hints = {}
    for order in np.argsort(predicted):
        mapped = bank_candidate_to_unit_parameters(bank[int(order)], runtime=runtime,
                                                   max_slope=max_slope)
        if mapped is None:
            continue
        family, parameters = mapped
        hints.setdefault(family, parameters)      # first = highest ranked in that family

    entry = {"record_id": str(np.asarray(record["record_id"]).item()),
             "parent_id": str(np.asarray(record["parent_id"]).item()),
             "runtime": runtime, "physical_n": int(np.asarray(record["physical_h"]).size),
             "linear_loss": None, "cold": {}, "warm": {}, "families_hinted": sorted(hints)}

    for budget in BUDGETS:
        for arm in ("cold", "warm"):
            best = None
            for family in FAMILIES:
                hint = hints.get(family) if arm == "warm" else None
                if family == "linear":
                    hint = None
                try:
                    result = optimize_control_family(
                        loss_fn, family, budget=budget, seed=0, runtime=runtime,
                        max_slope=max_slope, split="test", allow_test_adaptation=True,
                        warm_start=hint)
                except Exception as error:                     # noqa: BLE001
                    entry.setdefault("failures", []).append(f"{arm}/{family}/{budget}: {error}")
                    continue
                low = min(r.loss for r in result.records)
                best = low if best is None else min(best, low)
                if family == "linear" and entry["linear_loss"] is None:
                    entry["linear_loss"] = result.records[0].loss
            entry[arm][str(budget)] = best
    rows.append(entry)
    if (index + 1) % 4 == 0:
        print(f"  {index+1}/{len(records)} records, {time.time()-began:.0f}s", flush=True)

out = f"/home/nguyencongt/runs/warmstart_shard_{SHARD}.json"
json.dump({"schema_version": 1, "budgets": BUDGETS, "checkpoint": CHECKPOINT,
           "n_records": len(rows), "wall_seconds": time.time() - began, "rows": rows},
          open(out, "w"), indent=1, default=float)
print(f"wrote {out} ({len(rows)} records, {time.time()-began:.0f}s)", flush=True)
