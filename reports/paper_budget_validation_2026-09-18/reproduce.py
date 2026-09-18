"""Recompute diagnostic budget curves from archived complete raw campaign receipts.

Run from the repository root, with NumPy/SciPy installed:
    PYTHONPATH=src python reports/paper_budget_validation_2026-09-18/reproduce.py
This does not rerun training or simulations. The archive also retains an earlier
interrupted horizon17 request, excluded from all quality calculations.
"""
from pathlib import Path
import hashlib
import json
import gzip
import argparse

import numpy as np

from annealctrl.contrasts import paired_parent_seed_contrast

ROOT = Path(__file__).resolve().parent
RUN = "paper_budget_validation_20260918"
INTERRUPTED = "paper_budget_validation_horizon17_interrupted_20260918"


def summarize(archive):
    evidence = json.loads(gzip.decompress(Path(archive).read_bytes()))
    interrupted = json.loads(gzip.decompress((ROOT / "budget_interrupted_evidence.json.gz").read_bytes()))
    for bundle in (evidence, interrupted):
        for item in bundle["files"].values():
            assert hashlib.sha256(item["utf8"].encode()).hexdigest() == item["sha256"]
    def read(name):
        return json.loads(evidence["files"][name]["utf8"])
    if True:
        campaign = read("campaign.json")
        if campaign["status"] != "complete":
            raise ValueError("complete campaign required")
        curves, rowsets, costs = [], {}, []
        for spec in campaign["config"]["steps"]:
            name = spec["id"]
            method, seed = name.removeprefix("budget_").rsplit("_", 1)
            report, rows = read(f"{name}/report.json"), read(f"{name}/rows.json")
            assert report["n_records"] == 6 and report["n_logical_parents"] == 3
            assert all(r["status"] == "ok" for r in rows)
            for row in rows:
                rowsets.setdefault(method, []).append({"method": row["method"], "mode": f"budget_{row['budget']}",
                    "seed": int(seed), "parent_id": row["logical_fingerprint"],
                    "record_id": f"{row['record_id']}/optimizer_seed_{row['seed']}", "loss": row["selected_loss"]})
            curves.extend({**point, "checkpoint_arm": method, "training_seed": int(seed)} for point in report["curves"])
            costs.extend(report["cost_ledger"])
        aggregate = []
        for arm in sorted(rowsets):
            for method, budget in sorted({(r["method"], r["budget"]) for r in curves if r["checkpoint_arm"] == arm}):
                selected = [r for r in curves if r["checkpoint_arm"] == arm and r["method"] == method and r["budget"] == budget]
                aggregate.append({"checkpoint_arm": arm, "method": method, "budget": budget,
                    "mean_loss": float(np.mean([r["parent_mean_loss"] for r in selected])),
                    "per_training_seed_loss": {str(r["training_seed"]): r["parent_mean_loss"] for r in selected},
                    "mean_online_seconds": float(np.mean([r["parent_mean_online_seconds"] for r in selected]))})
        contrasts = []
        for arm, rows in sorted(rowsets.items()):
            for strategy in ("sobol_local", "bayesian", "finzgar_gp_ucb"):
                for hint in ("bank", "direct"):
                    for comparator in ("cold", "source_global"):
                        result = paired_parent_seed_contrast(rows, f"{strategy}/{comparator}", f"{strategy}/{hint}",
                                                             mode="budget_25", bootstrap_resamples=2000, seed=0)
                        contrasts.append({"checkpoint_arm": arm, **result})
        old_events = []
        for name, item in interrupted["files"].items():
            if name.endswith(".jsonl"):
                old_events.extend(json.loads(line) for line in item["utf8"].splitlines())
        old_calls = sum(e["event"] == "requested" for e in old_events)
        old_done = sum(e["event"] == "finished" for e in old_events)
    return {"schema_version": 1, "source_hash": campaign["source_hash"],
        "scope": "diagnostic follow-up using same three heldout parents; postpilot protocol decision; not independent confirmation",
        "n_parents": 3, "n_records": 6, "n_training_seeds": 3, "n_search_seeds": 2,
        "budgets": [0, 1, 5, 9, 17, 25], "checkpoint_arms": ["control", "policy"],
        "curves": aggregate, "horizon25_contrasts": contrasts,
        "calls": {role: sum(c["attempted_objective_calls"] for c in costs if c["role"] == role)
                  for role in ("online_search", "offline_diagnostic")},
        "failures": sum(c["failed_calls"] for c in costs),
        "interrupted_calls": sum(c["interrupted_calls"] for c in costs),
        "optimizer_adaptation": {"complete_trajectories_per_strategy": 288,
                                 "gp_ucb_adaptive_steps_per_trajectory": 15,
                                 "gp_ei_adaptive_steps_per_trajectory": 8,
                                 "basis": "frozen initial-design rules and all 25 outcomes successful; no model-step claim at lower horizons"},
        "discarded_horizon17_diagnostic_attempt": {"requested_calls": old_calls, "finished_calls": old_done,
            "interrupted_calls": old_calls - old_done, "known_objective_seconds": sum(e.get("objective_seconds", 0.) for e in old_events),
            "quality_results_used": False, "total_cost_is_lower_bound": old_calls != old_done},
        "campaign_observed_seconds": campaign["observed_wall_seconds"],
        "archive_sha256": hashlib.sha256(Path(archive).read_bytes()).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="require the existing exact summary")
    args = parser.parse_args()
    result = summarize(ROOT / "budget_validation_evidence.json.gz")
    destination = ROOT / "summary.json"
    if destination.exists():
        if result != json.loads(destination.read_text()):
            raise SystemExit("ERROR: reproduced summary differs from archive")
        print("Verified exact archived summary reproduction.")
    else:
        if args.check:
            raise SystemExit("Missing archived summary")
        destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print("Wrote", destination)
