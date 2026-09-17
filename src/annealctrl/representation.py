"""G3, model side: the decision value of physical information, measured causally.

`interventions.py` establishes that a single declared embedding change reverses
which control is preferred. That is a fact about the *instances*. This module
asks the question the representation claim actually rests on: does a model that
can see the embedding make better decisions on exactly those instances than one
that cannot?

The experiment has an unusually clean structure, because one arm of it is forced.
A ``logical`` encoder sees the logical graph, the logical coefficients and the
runtime, and nothing physical — no chain membership, no programmed scale. Both
arms of a single-factor pair share all of those. Its input is therefore **bitwise
identical** on the two arms and it *cannot* propose different controls. On a pair
whose preferred control genuinely swaps, it is wrong on at least one arm, and the
excess loss is the measurable cost of embedding blindness.

That makes the comparison falsifiable in both directions. If a physical-aware
model does not beat the logical one on swap pairs, the representation claim
fails, and it fails on the instances most favourable to it.

Boundaries. Every loss here is a true simulator outcome of a waveform the model
selected **before** the outcome was observed; no bank label is substituted for a
proposal. Excess loss is measured against the pair's finite-budget best-found
reference, so it is not regret against a global optimum.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .headroom import _bootstrap, _describe, _parent_means
from .interventions import InterventionArm, InterventionPair
from .schedules import Schedule
from .sweeps import SweepUnit, run_sweep
from .telemetry import _safe


def logical_overlap(pairs: Sequence[InterventionPair],
                    records: Sequence[Mapping[str, Any]]) -> dict:
    """Do these intervention pairs share a logical instance with these records?

    The model-side test is only a deployment test if the pairs are instances the
    model never trained on. The two are drawn from the same generator with
    different seed streams, which makes collisions unlikely but not impossible,
    so the disjointness is checked on labelled-coefficient fingerprints rather
    than assumed from the seeds.

    A fingerprint match is not a graph-isomorphism or gauge-equivalence
    certificate; it catches identical labelled instances, which is the failure
    that would actually leak.
    """
    from .generation import IsingProblem, logical_fingerprint

    pairs = list(pairs)
    if not pairs:
        raise ValueError("logical_overlap requires a nonempty pair set")
    pair_fingerprints = {logical_fingerprint(pair.arms[0].compiled.logical) for pair in pairs}
    record_fingerprints = {
        logical_fingerprint(IsingProblem(record["logical_h"], record["logical_edges"],
                                         record["logical_J"]))
        for record in records}
    shared = pair_fingerprints & record_fingerprints
    return _safe({
        "n_pair_instances": len(pair_fingerprints),
        "n_record_instances": len(record_fingerprints),
        "n_overlapping": len(shared),
        "disjoint": not shared,
        "basis": "labelled logical coefficient fingerprint",
        "not_a_certificate_of": "graph isomorphism or gauge equivalence",
    })


def arm_record(arm: InterventionArm, runtime: float) -> dict[str, Any]:
    """A ``graph_from_record``-compatible view of one intervention arm.

    Deliberately carries **no** outcome, candidate or spectral label: the model
    must decide from the instance specification alone, exactly as at deployment.
    """
    compiled = arm.compiled
    return {
        "physical_h": np.asarray(compiled.physical.h, dtype=float),
        "physical_edges": np.asarray(compiled.physical.edges, dtype=int),
        "physical_J": np.asarray(compiled.physical.J, dtype=float),
        "problem_J": np.asarray(compiled.problem_J, dtype=float),
        "chain_J": np.asarray(compiled.chain_J, dtype=float),
        "membership": np.asarray(compiled.embedding.membership, dtype=int),
        "logical_h": np.asarray(compiled.logical.h, dtype=float),
        "logical_edges": np.asarray(compiled.logical.edges, dtype=int),
        "logical_J": np.asarray(compiled.logical.J, dtype=float),
        "programmed_scale": float(compiled.programmed_scale),
        "chain_strength": float(compiled.chain_strength),
        "runtime": float(runtime),
        "catalyst_strength": 0.0,
        "energy_scale": 1.0,
    }


# The direct-proposal contract benchmarking.evaluate_checkpoint already uses: the
# waveform is built in float32 and its endpoints are forced to exactly 0 and 1,
# which perturbs the first and last slope at construction level. Scoring therefore
# allows the same relative slack. A violation beyond it is a real infeasible
# proposal and is recorded as one, never repaired and never crashed on.
SLOPE_SLACK = 1.0 + 1e-6


def _select(model, normalizer, record: Mapping[str, Any], *,
            device: str, max_ds_dtau: float) -> tuple[Schedule, int, list[float], bool]:
    """The model's own choice: propose, score with its critic, take the argmin.

    No simulator outcome participates in the selection. Returns a feasibility
    flag rather than raising, so an infeasible policy output becomes a counted
    outcome instead of a dead sweep unit.
    """
    import torch

    from .models import graph_from_record

    graph = normalizer.transform(graph_from_record(record, device=device))
    with torch.no_grad():
        proposals = model(graph)["proposal_schedules"]
        predicted = model.predict_losses(graph, proposals)
        index = int(predicted.argmin())
    wave = proposals[index].cpu().double().numpy().copy()
    wave[0], wave[-1] = 0.0, 1.0
    tau = np.linspace(0.0, 1.0, len(wave))
    schedule = Schedule(tau, wave)
    runtime = float(record["runtime"])
    try:
        schedule.validate_slope(runtime=runtime, max_slope=max_ds_dtau * SLOPE_SLACK / runtime)
        feasible = True
    except ValueError:
        feasible = False
    return schedule, index, [float(x) for x in predicted.cpu().tolist()], feasible


def model_intervention_response(
    pair: InterventionPair, checkpoint: str | Path, *, matrix: Mapping[str, Any] | None = None,
    device: str = "cpu", backend: str = "numpy", tolerance: float = 5e-4,
    initial_steps: int = 128, max_steps: int = 8192, max_ds_dtau: float = 4.0,
    max_qubits: int = 20, method: str | None = None,
) -> dict:
    """Run one checkpoint on both arms of one pair and score its choices truly.

    ``matrix`` is the pair's ``cross_control_matrix``; supplying it adds excess
    loss against each arm's own best-found control, which is the quantity the
    method comparison is built on.
    """
    from .benchmarking import score_schedule
    from .generation import output_observables
    from .learning import load_checkpoint
    from .physics import AnnealPath, HamiltonianTerms

    model, normalizer = load_checkpoint(checkpoint, device=device)
    model.eval()
    variant = str(getattr(model, "encoder_variant", "unknown"))

    chosen, losses, indices, predicted, feasible = {}, {}, {}, {}, {}
    for arm in pair.arms:
        record = arm_record(arm, pair.runtime)
        schedule, index, scores, ok = _select(model, normalizer, record, device=device,
                                              max_ds_dtau=max_ds_dtau)
        chosen[arm.label] = schedule
        indices[arm.label] = index
        predicted[arm.label] = scores
        feasible[arm.label] = ok
        if not ok:
            losses[arm.label] = None
            continue
        physical = arm.compiled.physical
        context = (HamiltonianTerms(physical.n, physical.h, physical.edges, physical.J),
                   AnnealPath(),
                   output_observables(arm.compiled, max_qubits=max_qubits,
                                      ground_energy=arm.validation["logical_ground_energy"]))
        outcome = score_schedule({"runtime": np.array(float(pair.runtime))}, schedule,
                                 physics_context=context, backend=backend, tolerance=tolerance,
                                 initial_steps=initial_steps, max_steps=max_steps,
                                 max_ds_dtau=max_ds_dtau * SLOPE_SLACK)
        losses[arm.label] = float(outcome["loss"])

    wave_a, wave_b = chosen["A"].s_knots, chosen["B"].s_knots
    identical = bool(np.array_equal(wave_a, wave_b))
    distance = float(np.abs(wave_a - wave_b).max()) if wave_a.shape == wave_b.shape else float("inf")
    blind = variant in {"logical", "summary"}

    both_feasible = bool(feasible["A"] and feasible["B"])
    excess = None
    if matrix is not None and both_feasible:
        reference = matrix["loss_matrix"]
        excess = {"A": losses["A"] - float(reference["A_on_A"]),
                  "B": losses["B"] - float(reference["B_on_B"])}

    return _safe({
        "schema_version": 1,
        "pair_id": pair.pair_id, "parent_id": pair.parent_id, "factor": pair.factor,
        "scale_arm": pair.scale_arm, "runtime": pair.runtime,
        "method": method or variant, "encoder_variant": variant,
        "checkpoint": str(checkpoint),
        "model_loss": losses,
        "feasible_proposal": feasible,
        "both_arms_feasible": both_feasible,
        "selected_proposal_index": indices,
        "critic_predicted_losses": predicted,
        "selected_waveform": {"A": chosen["A"].to_dict(), "B": chosen["B"].to_dict()},
        "identical_choice": identical,
        "waveform_distance": distance,
        # A logical/summary encoder receives identical input on both arms, so it
        # cannot differ. Recorded as a claim, and checked in aggregation.
        "embedding_blind_by_construction": blind,
        "excess_loss": excess,
        "mean_excess_loss": None if excess is None else (excess["A"] + excess["B"]) / 2,
        "preferred_control_swapped": None if matrix is None else bool(matrix["preferred_control_swapped"]),
        "resolution_status": None if matrix is None else str(matrix["resolution_status"]),
        "reference_best_found": None if matrix is None else
            {"A": float(matrix["loss_matrix"]["A_on_A"]), "B": float(matrix["loss_matrix"]["B_on_B"])},
        "objective_calls": int(feasible["A"]) + int(feasible["B"]),
        "true_outcome_observed_after_selection": True,
        "reference_status": "best_found_within_evaluated_candidates",
        "causal_scope": "inside the declared closed-system simulator only",
    })


def sweep_model_interventions(
    pairs: Sequence[InterventionPair], checkpoints: Mapping[str, str | Path], *,
    output: str | Path, matrices: Mapping[str, Mapping[str, Any]] | None = None,
    device: str = "cpu", backend: str = "numpy", tolerance: float = 5e-4,
    initial_steps: int = 128, max_steps: int = 8192, max_ds_dtau: float = 4.0,
    max_qubits: int = 20, resume: bool = False, dry_run: bool = False,
    on_error: str = "raise", shard: int = 0, shard_count: int = 1,
) -> dict:
    """Every checkpoint against every pair, one row per (pair, method).

    ``matrices`` maps ``pair_id`` to that pair's ``cross_control_matrix``, so the
    reference the excess loss is measured against is the same one the instance
    side already charged for.
    """
    from .experiments import source_hash

    pairs, checkpoints = list(pairs), dict(checkpoints)
    if not pairs or not checkpoints:
        raise ValueError("sweep_model_interventions requires pairs and checkpoints")
    settings = {"methods": sorted(checkpoints), "device": device, "backend": backend,
                "tolerance": tolerance, "initial_steps": initial_steps, "max_steps": max_steps,
                "max_ds_dtau": max_ds_dtau, "max_qubits": max_qubits,
                "checkpoints": {k: str(v) for k, v in sorted(checkpoints.items())}}
    by_id = {p.pair_id: p for p in pairs}
    if len(by_id) != len(pairs):
        raise ValueError("pair ids must be unique within a sweep")

    units = [SweepUnit(unit_id=f"{pair.pair_id}::{method}",
                       fingerprint=f"{pair.fingerprint}:{method}",
                       payload=(pair.pair_id, method))
             for pair in pairs for method in sorted(checkpoints)]
    if shard_count > 1:
        from .sweeps import select_shard
        units = select_shard(units, shard, shard_count)

    if dry_run:
        return {"dry_run": True, "planned": len(units),
                "requested_objective_calls_total": 2 * len(units), "settings": settings,
                "note": "requested workload only; no control was evaluated and no output written"}

    def worker(unit: SweepUnit) -> dict:
        pair_id, method = unit.payload
        return model_intervention_response(
            by_id[pair_id], checkpoints[method], method=method,
            matrix=None if matrices is None else matrices.get(pair_id),
            device=device, backend=backend, tolerance=tolerance, initial_steps=initial_steps,
            max_steps=max_steps, max_ds_dtau=max_ds_dtau, max_qubits=max_qubits)

    return run_sweep(units, worker, output=output, settings=settings,
                     command="model-intervention-sweep", source_hash=source_hash(),
                     resume=resume, on_error=on_error,
                     extra_manifest={"n_pairs": len(pairs), "methods": sorted(checkpoints),
                                     "scope": "model decisions on paired interventions, "
                                              "scored by the simulator after selection"})


def aggregate_model_interventions(
    rows: Sequence[Mapping[str, Any]], *, baseline: str | None = None,
    swap_pairs_only: bool = False, bootstrap_resamples: int = 2000, seed: int = 0,
) -> dict:
    """Per-method excess loss and, against a declared baseline, the paired difference.

    The contrast is paired on ``pair_id``: every method must have been run on the
    same pairs, or the difference would compare populations rather than methods.
    """
    rows = list(rows)
    if not rows:
        raise ValueError("aggregate_model_interventions requires a nonempty row set")
    if swap_pairs_only:
        swaps = {str(row["pair_id"]) for row in rows if row.get("preferred_control_swapped")}
        rows = [row for row in rows if str(row["pair_id"]) in swaps]
        if not rows:
            raise ValueError("no swap pairs in this row set")

    infeasible = [row for row in rows if row.get("both_arms_feasible") is False]
    methods = sorted({str(row["method"]) for row in rows})
    pairs_by_method = {m: {str(row["pair_id"]) for row in rows if row["method"] == m} for m in methods}
    common = set.intersection(*pairs_by_method.values()) if pairs_by_method else set()

    # A model that claims to be embedding-blind must actually have made the same
    # choice on both arms. Verified, not trusted.
    violations = [row for row in rows
                  if row.get("embedding_blind_by_construction") and not row.get("identical_choice")]

    by_method = {}
    for method in methods:
        subset = [row for row in rows if row["method"] == method]
        values, _ = _parent_means(subset, lambda row: row.get("mean_excess_loss"))
        block = _describe(values)
        block["parent_bootstrap_ci"] = _bootstrap(values, n_resamples=bootstrap_resamples, seed=seed)
        by_method[method] = {
            "mean_excess_loss": block,
            "n_pairs": len(subset),
            "n_infeasible_proposals": sum(1 for r in subset if r.get("both_arms_feasible") is False),
            "identical_choice_fraction": sum(1 for r in subset if r.get("identical_choice")) / len(subset),
            "mean_waveform_distance": float(np.mean([float(r.get("waveform_distance", 0.0))
                                                     for r in subset])),
            "embedding_blind_by_construction": bool(subset[0].get("embedding_blind_by_construction")),
        }

    decision_value = {}
    if baseline is not None:
        if baseline not in methods:
            raise ValueError(f"baseline {baseline!r} is not among the evaluated methods {methods}")
        if any(pairs_by_method[m] != pairs_by_method[baseline] for m in methods):
            raise ValueError("methods were not evaluated on the same pairs; a paired contrast "
                             "across different populations would not compare methods")
        base_rows = {str(row["pair_id"]): row for row in rows if row["method"] == baseline}
        for method in methods:
            if method == baseline:
                continue
            paired, parents = [], []
            for row in rows:
                if row["method"] != method:
                    continue
                other = base_rows.get(str(row["pair_id"]))
                if other is None or row.get("mean_excess_loss") is None or other.get("mean_excess_loss") is None:
                    continue
                paired.append(float(row["mean_excess_loss"]) - float(other["mean_excess_loss"]))
                parents.append(str(row["parent_id"]))
            if not paired:
                continue
            buckets: dict[str, list[float]] = {}
            for parent, value in zip(parents, paired):
                buckets.setdefault(parent, []).append(value)
            per_parent = np.array([float(np.mean(v)) for _, v in sorted(buckets.items())])
            interval = _bootstrap(per_parent, n_resamples=bootstrap_resamples, seed=seed)
            mean = float(per_parent.mean())
            decision_value[method] = {
                "baseline": baseline, "mean_difference": mean,
                "n_parents": int(per_parent.size), "n_pairs": len(paired),
                "parent_bootstrap_ci": interval,
                # Loss: negative favours the method over the baseline.
                "favours_method": bool(mean < 0),
                "interval_excludes_zero": bool(
                    interval["low"] is not None and interval["high"] is not None
                    and (interval["high"] < 0 or interval["low"] > 0)),
                "estimand": "mean over parents of (method excess loss - baseline excess loss)",
            }

    return _safe({
        "schema_version": 1,
        "methods": methods,
        "n_rows": len(rows),
        "n_pairs_per_method": len(common),
        "n_parents": len({str(row["parent_id"]) for row in rows}),
        "restricted_to_swap_pairs": bool(swap_pairs_only),
        "by_method": by_method,
        "decision_value": decision_value,
        "infeasible_proposal_pairs": len(infeasible),
        "infeasible_proposal_fraction": len(infeasible) / len(rows),
        "blindness_violations": len(violations),
        "blindness_violation_methods": sorted({str(r["method"]) for r in violations}),
        "factor_counts": dict(sorted(Counter(str(row.get("factor")) for row in rows).items())),
        "total_objective_calls": int(sum(int(row.get("objective_calls", 0)) for row in rows)),
        "scope": ("model decisions on paired interventions inside the declared closed-system "
                  "simulator; excess loss is against a finite-budget best-found reference, "
                  "not a global optimum"),
    })
