"""G5: what a decision costs, and the deployment count at which learning pays.

The protocol (§10) writes the accounting as

    C(M) = (C_data + C_train)/M + C_inference + C_online_search + C_execution
           + C_embedding + C_programming

and attaches two conditions that this module enforces rather than assumes.

**Show the unamortised point.** A curve that starts at M=100 is a choice of
favourable regime. ``compare_amortized`` refuses a deployment list that omits
M=1, so the one-instance cost — where offline work is paid in full and learning
looks worst — is always on the same axes as the amortised one.

**Charge the reference honestly.** The equal-budget family search has no offline
cost to spread: every new instance pays its full 257 propagation-scored controls
again. That is exactly why it is the right comparator for an amortised method,
and why its ``amortisable`` flag is False rather than an offline term of zero
that a reader might mistake for cheapness.

Nothing here is a hardware or energy measurement. These are wall-clock seconds
of closed-system simulation on one declared machine, and they move with the host.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .telemetry import _safe


def _nonnegative(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def _deployments(deployments: Sequence[int]) -> list[int]:
    values = [int(m) for m in deployments]
    if not values or any(m < 1 for m in values):
        raise ValueError("deployment counts must be positive integers")
    return values


def amortization_curve(*, offline_seconds: float, online_seconds: float,
                       deployments: Sequence[int]) -> list[float]:
    """Cost per instance at each deployment count. Never below the online cost."""
    offline = _nonnegative(offline_seconds, "offline_seconds")
    online = _nonnegative(online_seconds, "online_seconds")
    return [offline / m + online for m in _deployments(deployments)]


def crossover_deployments(*, offline_seconds: float, online_seconds: float,
                          reference_online_seconds: float) -> int | None:
    """Smallest M at which the amortised method costs less than the reference.

    ``None`` when the method's per-instance online cost alone already exceeds the
    reference — no amount of amortisation rescues that, and reporting a large M
    would misrepresent it.
    """
    offline = _nonnegative(offline_seconds, "offline_seconds")
    online = _nonnegative(online_seconds, "online_seconds")
    reference = _nonnegative(reference_online_seconds, "reference_online_seconds")
    if reference <= 0:
        raise ValueError("reference_online_seconds must be positive to compare against")
    if online >= reference:
        return None
    if offline == 0:
        return 1
    # offline/M + online < reference  <=>  M > offline / (reference - online)
    return int(np.floor(offline / (reference - online)) + 1)


def experiment_costs(run_dir: str | Path) -> dict:
    """Offline and per-instance online cost of every method in a finished run.

    Offline is the shared dataset generation plus *this method's own* training,
    averaged over its seeds — not the whole campaign's training, which would
    charge each method for its competitors.
    """
    root = Path(run_dir)
    manifest = json.loads((root / "experiment.json").read_text())
    runs = manifest.get("runs") or {}
    if not runs:
        raise ValueError(f"{root} records no training runs")
    unfinished = sorted(rid for rid, state in runs.items() if not state.get("evaluated"))
    if unfinished:
        raise ValueError(f"cost accounting needs a finished campaign; not evaluated: {unfinished}")

    data_seconds = float((manifest.get("stages", {}).get("generate") or {}).get("seconds", 0.0))
    by_method: dict[str, dict[str, Any]] = {}
    incomplete: list[str] = []

    for path in sorted((root / "evaluations").glob("*.json")):
        payload = json.loads(path.read_text())
        method = str(payload.get("method", path.stem.split("__")[0]))
        seed = payload.get("training_seed")
        state = runs.get(f"{method}/seed_{seed}", {})
        costs = payload.get("costs") or {}
        records = int(payload.get("n_records") or 0)
        if records < 1:
            raise ValueError(f"{path} reports no evaluated records")
        # Bank inference is the deployment-time cost: the critic scores the
        # candidate set and picks one. Direct-proposal inference is reported
        # beside it because it is a different deployment mode, not an addition.
        inference = float(costs.get("bank_inference_seconds", payload.get("inference_seconds", 0.0)))
        bucket = by_method.setdefault(method, {
            "training_seconds": [], "inference_seconds_per_instance": [],
            "direct_inference_seconds_per_instance": [], "online_simulator_calls": 0,
            "candidates_scored_by_critic": None, "training_cost_complete": True,
            "n_records": records, "seeds": []})
        bucket["training_seconds"].append(float(state.get("training_seconds",
                                                          state.get("observed_training_seconds", 0.0))))
        bucket["inference_seconds_per_instance"].append(inference / records)
        if costs.get("direct_inference_seconds") is not None:
            bucket["direct_inference_seconds_per_instance"].append(
                float(costs["direct_inference_seconds"]) / records)
        bucket["online_simulator_calls"] = max(
            bucket["online_simulator_calls"], int(costs.get("online_simulator_search_calls", 0)))
        budget = payload.get("read_budget_at_deployment") or {}
        if budget.get("candidates_scored_by_critic") is not None:
            bucket["candidates_scored_by_critic"] = int(budget["candidates_scored_by_critic"])
        bucket["seeds"].append(seed)
        if not state.get("training_cost_complete", True):
            bucket["training_cost_complete"] = False

    methods = {}
    for name, bucket in sorted(by_method.items()):
        training = float(np.mean(bucket["training_seconds"]))
        online = float(np.mean(bucket["inference_seconds_per_instance"]))
        if not bucket["training_cost_complete"]:
            incomplete.append(name)
        methods[name] = {
            "n_seeds": len(bucket["seeds"]),
            "mean_training_seconds": training,
            "std_training_seconds": float(np.std(bucket["training_seconds"], ddof=1))
            if len(bucket["training_seconds"]) > 1 else 0.0,
            "offline_seconds": data_seconds + training,
            "online_seconds_per_instance": online,
            "direct_online_seconds_per_instance": float(np.mean(bucket["direct_inference_seconds_per_instance"]))
            if bucket["direct_inference_seconds_per_instance"] else None,
            "online_simulator_calls": bucket["online_simulator_calls"],
            "candidates_scored_by_critic": bucket["candidates_scored_by_critic"],
            "training_cost_complete": bucket["training_cost_complete"],
            "n_evaluated_records": bucket["n_records"],
        }

    return _safe({
        "schema_version": 1,
        "run_dir": str(root),
        "data_generation_seconds": data_seconds,
        "methods": methods,
        "methods_with_incomplete_training_cost": sorted(incomplete),
        "note": ("offline is shared data generation plus this method's own training, "
                 "averaged over its seeds; a resumed run's training cost can be "
                 "understated and is flagged rather than silently summed"),
        "scope": "wall-clock seconds of closed-system simulation on one declared host",
    })


def search_reference_cost(sweep_dirs: Sequence[str | Path]) -> dict:
    """Per-instance cost of the equal-budget family search, which never amortises."""
    from .sweeps import load_rows

    rows = [row["result"] for part in sweep_dirs for row in load_rows(part)
            if row.get("status") == "ok"]
    if not rows:
        raise ValueError(f"{[str(p) for p in sweep_dirs]} contain no successful sweep rows")
    seconds = [float(row.get("search_seconds", 0.0)) for row in rows]
    calls = [float(row.get("total_objective_calls", 0.0)) for row in rows]
    return _safe({
        "schema_version": 1,
        "n_records": len(rows),
        "offline_seconds": 0.0,
        "online_seconds_per_instance": float(np.mean(seconds)),
        "online_seconds_std": float(np.std(seconds, ddof=1)) if len(seconds) > 1 else 0.0,
        "objective_calls_per_instance": float(np.mean(calls)),
        "amortisable": False,
        "note": ("the equal-budget family search has no offline cost to spread: "
                 "every new instance pays its full objective-call budget again"),
    })


def compare_amortized(costs: Mapping[str, Any], reference: Mapping[str, Any], *,
                      deployments: Sequence[int] = (1, 10, 100, 1000, 10000)) -> dict:
    """C(M) per method against a non-amortisable reference, with the crossover.

    M=1 is required in ``deployments``: starting a curve at M=100 chooses the
    regime that flatters learning, and the protocol asks for the unamortised cost
    to be shown alongside.
    """
    values = _deployments(deployments)
    if 1 not in values:
        raise ValueError("deployments must include M=1 so the unamortised cost is shown; "
                         "a curve that starts above M=1 chooses a favourable regime")
    reference_online = float(reference["online_seconds_per_instance"])

    methods = {}
    for name, block in costs["methods"].items():
        offline = float(block["offline_seconds"])
        online = float(block["online_seconds_per_instance"])
        curve = amortization_curve(offline_seconds=offline, online_seconds=online,
                                   deployments=values)
        methods[name] = {
            "offline_seconds": offline,
            "online_seconds_per_instance": online,
            "unamortised_seconds": offline + online,
            "curve": {str(m): c for m, c in zip(values, curve)},
            "crossover_deployments": crossover_deployments(
                offline_seconds=offline, online_seconds=online,
                reference_online_seconds=reference_online),
            "online_speedup_vs_reference": (reference_online / online) if online > 0 else None,
            "candidates_scored_by_critic": block.get("candidates_scored_by_critic"),
            "online_simulator_calls": block.get("online_simulator_calls"),
            "training_cost_complete": block.get("training_cost_complete", True),
        }

    return _safe({
        "schema_version": 1,
        "deployments": values,
        "methods": methods,
        "reference": {**dict(reference), "curve": {str(m): reference_online for m in values}},
        "shows_unamortised_cost": True,
        "methods_with_incomplete_training_cost": costs.get("methods_with_incomplete_training_cost", []),
        "scope": ("wall-clock seconds on one declared host; no hardware, energy or "
                  "quantum-runtime cost is included, and the reference never amortises"),
    })
