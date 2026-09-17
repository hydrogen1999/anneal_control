"""Held-out acquisition evaluation with frozen-proposal and deployment controls.

The frozen-proposal contrast changes only the critic evaluating a pre-acquisition
model's waveforms. It is diagnostic attribution, not a separately trained critic
ablation. All simulator outcomes are obtained after selections have been frozen.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np

from .benchmarking import (
    _checkpoint_provenance, _identity, _json_safe, _paired, _parent_mean,
    evaluate_checkpoint, probability_tts, record_physics, score_schedule,
)
from .policy_diagnostics import aggregate_proposal_decomposition, decompose_proposals
from .schedules import Schedule


def _synchronize(device):
    import torch
    target = torch.device(device)
    if target.type == "cuda":
        torch.cuda.synchronize(target)
    elif target.type == "mps":
        torch.mps.synchronize()


def benchmark_deployment(model, normalizer, records, *, device="cpu", warmup=2, repeats=5):
    """Time standalone deployment APIs with preprocessing and host result copies.

    Each call constructs and normalizes its own graph. Warmups are excluded;
    the measurement alternates bank/direct order to reduce order bias. Both
    strategies use zero online simulator calls. This measures the current APIs,
    including direct forward's response head and repeated encoding in predict.
    """
    import torch
    from .models import graph_from_record

    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("warmup must be a nonnegative integer")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    records = list(records)
    if not records:
        raise ValueError("deployment benchmark requires records")
    model.eval()

    def deploy(record, mode):
        graph = normalizer.transform(graph_from_record(record, device=device))
        schedules = (torch.as_tensor(record["candidate_schedules"], dtype=torch.float32, device=device)
                     if mode == "bank" else model(graph)["proposal_schedules"])
        predicted = model.predict_losses(graph, schedules)
        if not torch.isfinite(predicted).all() or not torch.isfinite(schedules).all():
            raise ArithmeticError("nonfinite deployment output")
        selected = int(predicted.argmin())
        return schedules[selected].detach().cpu().numpy().copy()

    rows = []
    with torch.inference_mode():
        for record in records:
            for _ in range(warmup):
                for mode in ("bank", "direct"):
                    deploy(record, mode)
                    _synchronize(device)
            samples = {"bank": [], "direct": []}
            for repeat in range(repeats):
                modes = ("bank", "direct") if repeat % 2 == 0 else ("direct", "bank")
                for mode in modes:
                    _synchronize(device)
                    started = perf_counter()
                    deploy(record, mode)
                    _synchronize(device)
                    samples[mode].append(perf_counter() - started)
            rows.append({**_identity(record), "seconds": samples,
                         "median_bank_seconds": float(np.median(samples["bank"])),
                         "median_direct_seconds": float(np.median(samples["direct"])),
                         "bank_candidates": len(record["candidate_schedules"]),
                         "direct_proposals": model.proposals,
                         "bank_waveform_float32_bytes": int(np.asarray(record["candidate_schedules"], dtype=np.float32).nbytes)})
    bank = float(np.median([r["median_bank_seconds"] for r in rows]))
    direct = float(np.median([r["median_direct_seconds"] for r in rows]))
    return {"records": rows, "warmup_per_record_per_method": warmup,
            "repeats_per_record_per_method": repeats, "device": str(device),
            "median_record_bank_seconds": bank, "median_record_direct_seconds": direct,
            "direct_to_bank_latency_ratio": direct / bank,
            "online_simulator_calls": {"bank": 0, "direct": 0},
            "model_parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
            "synchronized_device_timing": True, "preprocessing_included": True,
            "scope": "Current standalone deployment APIs; timing repeats are not independent problem samples."}


def _score_waveforms(record, waveforms, *, context, settings):
    outcomes = []
    for waveform in waveforms:
        wave = np.asarray(waveform, dtype=float).copy()
        wave[0], wave[-1] = 0., 1.
        # No silent record exclusion or interior waveform repair.
        schedule = Schedule(np.linspace(0., 1., len(wave)), wave)
        outcomes.append(score_schedule(record, schedule, physics_context=context, **settings))
    return outcomes


def evaluate_acquisition_arm(data_dir, checkpoint, *, training_records, frozen_checkpoint,
                             seed=0, device="cpu", backend="numpy", tolerance=5e-4,
                             initial_steps=128, max_steps=8192, max_ds_dtau=4.,
                             n_resamples=2000, latency_warmup=2, latency_repeats=5):
    """Evaluate one arm on an untouched common bank and fixed test parents.

    The original checkpoint must match original training data; the new checkpoint
    must match the supplied actual (possibly augmented) training data. Validation
    labels, reference-bank labels, and test parents are never augmented here.
    """
    import torch
    from .learning import load_checkpoint
    from .models import graph_from_record
    from .pipeline import load_records

    started = perf_counter()
    evaluation = evaluate_checkpoint(
        data_dir, checkpoint, device=device, direct=False, seed=seed, backend=backend,
        tolerance=tolerance, initial_steps=initial_steps, max_steps=max_steps,
        max_ds_dtau=max_ds_dtau, n_resamples=n_resamples, training_records=training_records)
    original_training, validation, test = (load_records(data_dir, split)
                                           for split in ("train", "validation", "test"))
    frozen_payload = torch.load(frozen_checkpoint, map_location="cpu", weights_only=True)
    frozen_provenance = _checkpoint_provenance(frozen_payload, original_training, validation, test)
    if not frozen_provenance["training_validation_content_verified"]:
        raise ValueError("frozen checkpoint requires original training content provenance")
    model, normalizer = load_checkpoint(checkpoint, device=device)
    frozen, frozen_normalizer = load_checkpoint(frozen_checkpoint, device=device)
    model.eval()
    frozen.eval()
    for candidate in (model, frozen):
        if not np.isclose(float(candidate.config.get("max_ds_dtau", 4.)), max_ds_dtau):
            raise ValueError("policy slew differs from evaluation max_ds_dtau")
    if model.schedule_points != frozen.schedule_points:
        raise ValueError("frozen and updated critics require the same waveform grid")
    by_id = {row["record_id"]: row for row in evaluation["records"]}
    rows, fixed_rows, direct_rows, scored = [], [], [], []
    context_seconds = 0.
    settings = dict(backend=backend, tolerance=tolerance, initial_steps=initial_steps,
                    max_steps=max_steps, max_ds_dtau=max_ds_dtau * (1 + 1e-6))
    for record in test:
        identity = _identity(record)
        bank_row = by_id[identity["record_id"]]
        graph = normalizer.transform(graph_from_record(record, device=device))
        frozen_graph = frozen_normalizer.transform(graph_from_record(record, device=device))
        # Freeze every reported decision before exposing any fresh true label.
        with torch.inference_mode():
            proposals = model(graph)["proposal_schedules"]
            predicted = model.predict_losses(graph, proposals)
            old_proposals = frozen(frozen_graph)["proposal_schedules"]
            old_predicted = frozen.predict_losses(frozen_graph, old_proposals)
            new_fixed_predicted = model.predict_losses(graph, old_proposals)
            values = (proposals, predicted, old_proposals, old_predicted, new_fixed_predicted)
            if any(not torch.isfinite(value).all() for value in values):
                raise ArithmeticError("nonfinite acquisition evaluation predictions")
            selected = int(predicted.argmin())
            old_selected, new_fixed_selected = int(old_predicted.argmin()), int(new_fixed_predicted.argmin())
            waves, preds, old_waves, old_preds, new_fixed_preds = (
                value.detach().cpu().double().numpy() for value in values)
        context_started = perf_counter()
        context = record_physics(record)
        context_seconds += perf_counter() - context_started
        outcomes = _score_waveforms(record, waves, context=context, settings=settings)
        fixed_outcomes = _score_waveforms(record, old_waves, context=context, settings=settings)
        scored.extend(outcomes + fixed_outcomes)
        losses, fixed_losses = ([outcome["loss"] for outcome in group] for group in (outcomes, fixed_outcomes))
        decompose_args = dict(bank_loss=bank_row["selected_loss"], linear_loss=bank_row["linear_loss"])
        decomposition = decompose_proposals(true_losses=losses, predicted_losses=preds,
                                           selected_index=selected, **decompose_args)
        rows.append({**identity, **decomposition, "selected_proposal": selected,
                     "predicted_losses": preds.tolist(), "outcomes": outcomes})
        old_decomposition = decompose_proposals(true_losses=fixed_losses, predicted_losses=old_preds,
                                               selected_index=old_selected, **decompose_args)
        new_decomposition = decompose_proposals(true_losses=fixed_losses, predicted_losses=new_fixed_preds,
                                               selected_index=new_fixed_selected, **decompose_args)
        fixed_rows.append({**identity, "old_critic": old_decomposition, "new_critic": new_decomposition,
                           "old_selected_proposal": old_selected, "new_selected_proposal": new_fixed_selected,
                           "old_predicted_losses": old_preds.tolist(),
                           "new_predicted_losses": new_fixed_preds.tolist(), "outcomes": fixed_outcomes,
                           "new_minus_old_selected_loss": float(fixed_losses[new_fixed_selected] - fixed_losses[old_selected])})
        outcome = outcomes[selected]
        direct_rows.append({**identity, **outcome, "selected_proposal": selected,
                            "reference_bank_regret": outcome["loss"] - bank_row["bank_best_loss"],
                            "tts": probability_tts(outcome["success"], identity["runtime"])})
    parents = [r["parent_id"] for r in rows]
    direct_losses = [r["loss"] for r in direct_rows]
    evaluation["direct_policy"] = {
        "selection": "critic_selected_among_policy_proposals_then_true_simulator_scored",
        "records": direct_rows, "mean_loss": float(np.mean(direct_losses)),
        "parent_mean_loss": _parent_mean(direct_losses, parents),
        **{f"paired_vs_{label}": _paired(direct_losses, [r[key] for r in evaluation["records"]], parents, seed, n_resamples)
           for label, key in (("linear", "linear_loss"), ("global", "global_loss"), ("bank", "selected_loss"))}}
    new_fixed = [r["new_critic"]["selected_proposal_loss"] for r in fixed_rows]
    old_fixed = [r["old_critic"]["selected_proposal_loss"] for r in fixed_rows]
    latency = benchmark_deployment(model, normalizer, test, device=device,
                                   warmup=latency_warmup, repeats=latency_repeats)
    costs = {"offline_objective_calls": len(scored),
             "offline_propagation_calls": sum(r["propagation_calls"] for r in scored),
             "offline_integrator_steps": sum(r["total_integrator_steps"] for r in scored),
             "offline_scoring_seconds": sum(r["offline_scoring_seconds"] for r in scored),
             "observable_construction_seconds": context_seconds,
             "online_simulator_calls": 0}
    return _json_safe({
        "schema_version": 1, "evaluation": evaluation,
        "proposal_diagnostics": {"rows": rows, "summary": aggregate_proposal_decomposition(
            rows, bootstrap_resamples=n_resamples, seed=seed)},
        "frozen_proposal_diagnostics": {
            "rows": fixed_rows, "paired_new_critic_vs_old_critic": _paired(new_fixed, old_fixed, parents, seed, n_resamples),
            "checkpoint": str(frozen_checkpoint),
            "checkpoint_sha256": hashlib.sha256(Path(frozen_checkpoint).read_bytes()).hexdigest(),
            "provenance": frozen_provenance,
            "scope": "Same frozen waveforms, different critics; not a separately trained critic-only ablation."},
        "deployment_latency": latency, "costs": costs,
        "evaluation_wall_seconds": perf_counter() - started,
        "scope": "Untouched held-out parents; all proposal labels are offline diagnostics after selection; no QPU or quantum-advantage claim."})
