"""Reproducible held-out scoring and explicitly charged control search.

No simulator outcome participates in neural policy selection. Exact-family
search is a separately labeled, opt-in test-time adaptation experiment.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np

from .evaluation import paired_parent_bootstrap
from .generation import CompiledInstance, Embedding, IsingProblem, output_observables
from .physics import AnnealPath, HamiltonianTerms, propagate
from .schedules import Schedule
from .search import CONTROL_FAMILIES, optimize_control_family


def _scalar(record, key, default=None):
    value = record.get(key, default)
    return value.item() if isinstance(value, np.ndarray) and value.ndim == 0 else value


def _json_safe(value):
    """JSON RFC 8259: unresolved/nonfinite quantities are null, never NaN/Inf."""
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _identity(record):
    return {"record_id": str(_scalar(record, "record_id", "unknown")),
            "parent_id": str(_scalar(record, "parent_id")),
            "family": str(_scalar(record, "family", "unknown")),
            "logical_n": len(record["logical_h"]), "physical_n": len(record["physical_h"]),
            "runtime": float(_scalar(record, "runtime")), "split": str(_scalar(record, "split"))}


def _numerical_settings(tolerance, initial_steps, max_steps, norm_tolerance, backend):
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    if not np.isfinite(norm_tolerance) or not 0 < norm_tolerance <= 1e-6:
        raise ValueError("norm_tolerance must be finite and in (0,1e-6]")
    if any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) or x < 1
           for x in (initial_steps, max_steps)) or max_steps < 2 * initial_steps:
        raise ValueError("positive integer steps with max_steps >= 2*initial_steps required")
    if backend not in {"numpy", "cupy"}:
        raise ValueError("backend must be numpy or cupy")


def record_physics(record):
    """Reconstruct the exact stored problem and path; never drop XX or scale."""
    catalyst = float(_scalar(record, "catalyst_strength", 0.))
    xx_edges, xx_weights = record.get("xx_edges"), record.get("xx_weights")
    if catalyst != 0 and (xx_edges is None or xx_weights is None):
        raise ValueError("nonzero catalyst requires stored xx_edges and xx_weights")
    terms = HamiltonianTerms(len(record["physical_h"]), record["physical_h"],
                             record["physical_edges"], record["physical_J"], xx_edges, xx_weights)
    path = AnnealPath(catalyst_strength=catalyst, energy_scale=float(_scalar(record, "energy_scale", 1.)))
    logical = IsingProblem(record["logical_h"], record["logical_edges"], record["logical_J"])
    physical = IsingProblem(record["physical_h"], record["physical_edges"], record["physical_J"])
    embedding = Embedding(record["membership"], record["physical_edges"])
    compiled = CompiledInstance(logical, embedding, physical, record["problem_J"], record["chain_J"],
                                float(_scalar(record, "programmed_scale")),
                                float(np.asarray(record["chain_J"]).sum()),
                                float(_scalar(record, "chain_strength")))
    return terms, path, output_observables(compiled)


def _segmented_evolution(terms, path, schedule, runtime, counts, backend):
    # Align every true window/pause switch exactly. Local callables intentionally
    # traverse subintervals of s, with the previous state supplied explicitly.
    state = None
    for left, right, steps in zip(schedule.tau_knots[:-1], schedule.tau_knots[1:], counts):
        def segment(tau, left=left, right=right):
            return schedule(left + (right - left) * np.asarray(tau))
        result = propagate(terms, segment, runtime * (right - left), steps=int(steps),
                           path=path, backend=backend, initial_state=state)
        state = result.state
    return state


def score_schedule(record, schedule: Schedule, *, backend="numpy", tolerance=5e-4,
                   initial_steps=128, max_steps=8192, norm_tolerance=1e-9,
                   max_ds_dtau=4., physics_context=None):
    """Score ONE fixed control, with switch-aligned step-doubling diagnostics.

    ``max_steps`` caps the accepted fine trajectory's total steps. All rejected
    attempts and both coarse/fine trajectories are included in cost counts.
    Error-derived ambiguity radii are diagnostics, NOT certified true bounds.
    """
    _numerical_settings(tolerance, initial_steps, max_steps, norm_tolerance, backend)
    if not np.isfinite(max_ds_dtau) or max_ds_dtau < 1:
        raise ValueError("max_ds_dtau must be finite and >=1")
    runtime = float(_scalar(record, "runtime"))
    if not np.isfinite(runtime) or runtime <= 0:
        raise ValueError("positive finite runtime required")
    schedule.validate_slope(runtime=runtime, max_slope=max_ds_dtau / runtime)
    context_started = perf_counter()
    terms, path, observables = record_physics(record) if physics_context is None else physics_context
    context_seconds = perf_counter() - context_started if physics_context is None else 0.
    counts = np.maximum(1, np.ceil(initial_steps * np.diff(schedule.tau_knots)).astype(int))
    start, attempts, total_steps = perf_counter(), 0, 0
    while True:
        if int(2 * counts.sum()) > max_steps:
            raise ArithmeticError("schedule scoring failed convergence within max_steps")
        coarse = _segmented_evolution(terms, path, schedule, runtime, counts, backend)
        fine = _segmented_evolution(terms, path, schedule, runtime, 2 * counts, backend)
        attempts += 1
        total_steps += int(3 * counts.sum())
        error = float(np.linalg.norm(fine - coarse) / 3.)
        norm_error = float(abs(np.vdot(fine, fine).real - 1.))
        if not np.isfinite(error) or not np.isfinite(norm_error) or norm_error > norm_tolerance:
            raise ArithmeticError("schedule scoring failed norm/finite-state gate")
        if error <= tolerance:
            break
        counts *= 2
    probability = np.abs(fine) ** 2
    metrics = {key: float(probability @ value) for key, value in observables.items()}
    if not -norm_tolerance <= metrics["success"] <= 1 + norm_tolerance:
        raise ArithmeticError("success probability outside numerical tolerance")
    # Only roundoff clipping after the norm gate; no pseudo-count smoothing.
    metrics["success"] = float(np.clip(metrics["success"], 0., 1.))
    metrics.update(loss=1 - metrics["success"], state_error_diagnostic=error,
                   loss_ambiguity_indicator=2 * error + error**2,
                   norm_error=norm_error, accepted_steps=int(2 * counts.sum()),
                   total_integrator_steps=total_steps, convergence_attempts=attempts,
                   propagation_calls=2 * attempts * len(counts),
                   offline_scoring_seconds=perf_counter() - start, backend=backend,
                   observable_construction_seconds=context_seconds,
                   switching_knots_aligned=True, waveform=schedule.to_dict(),
                   uncertainty_is_certificate=False)
    return _json_safe(metrics)


def probability_tts(probability: float, runtime: float, target_success=.99):
    """Ideal independent-read TTS for a simulated probability, not shot-count CI."""
    if not np.isfinite(probability) or not 0 <= probability <= 1 or not np.isfinite(runtime) or runtime <= 0:
        raise ValueError("probability in [0,1] and positive runtime required")
    if not np.isfinite(target_success) or not 0 < target_success < 1:
        raise ValueError("target_success must lie in (0,1)")
    if probability == 0:
        return {"nominal_reads": None, "nominal_time": None, "censored": True,
                "status": "zero_probability_unbounded_tts", "target_success": target_success}
    reads = 1 if probability == 1 else int(np.ceil(np.log1p(-target_success) / np.log1p(-probability)))
    return {"nominal_reads": reads, "nominal_time": reads * runtime, "censored": False,
            "status": "ideal_iid_simulated_probability_not_shot_estimate", "target_success": target_success}


def validate_candidate_banks(records: Sequence[Mapping[str, Any]]) -> dict:
    """Require the SAME indexed normalized waveforms, not merely equal sizes."""
    if not records:
        raise ValueError("candidate bank validation requires nonempty records")
    reference = np.asarray(records[0]["candidate_schedules"], float)
    tau = np.asarray(records[0].get("candidate_tau", np.linspace(0., 1., reference.shape[-1])), float)
    ids = np.asarray(records[0].get("candidate_ids", np.arange(len(reference))), str)
    if reference.ndim != 2 or len(reference) < 1 or ids.shape != (len(reference),) or len(set(ids)) != len(ids):
        raise ValueError("invalid common candidate bank")
    for record in records:
        waves = np.asarray(record["candidate_schedules"], float)
        current_tau = np.asarray(record.get("candidate_tau", tau), float)
        current_ids = np.asarray(record.get("candidate_ids", np.arange(len(waves))), str)
        if (waves.shape != reference.shape or current_tau.shape != tau.shape or
                not np.allclose(waves, reference, rtol=0., atol=1e-12) or
                not np.allclose(current_tau, tau, rtol=0., atol=1e-12) or
                not np.array_equal(current_ids, ids)):
            raise ValueError("candidate banks differ: global validation index is not transferable")
        losses = np.asarray(record["candidate_losses"], float)
        if losses.shape != (len(reference),) or not np.isfinite(losses).all():
            raise ValueError("invalid candidate losses")
        for wave in waves:
            Schedule(current_tau, wave)
    linear = np.flatnonzero(np.all(np.isclose(reference, tau[None, :], rtol=0., atol=1e-12), axis=1))
    if not len(linear):
        raise ValueError("common candidate bank lacks an actual linear waveform")
    digest = hashlib.sha256(reference.astype('<f8').tobytes() + tau.astype('<f8').tobytes()).hexdigest()
    return {"candidate_count": len(reference), "linear_index": int(linear[0]),
            "waveform_sha256": digest, "candidate_ids": ids.tolist()}


def _parent_mean(values, parents):
    unique = sorted(set(parents))
    return float(np.mean([np.mean([value for value, parent in zip(values, parents) if parent == item])
                          for item in unique]))


def _paired(method, baseline, parents, seed, n_resamples):
    if len(set(parents)) < 2:
        return {"mean_difference": _parent_mean(np.asarray(method) - np.asarray(baseline), parents),
                "ci_low": None, "ci_high": None, "n_parents": len(set(parents)),
                "n_observations": len(parents), "weighting": "equal_parent_mean",
                "status": "insufficient_independent_parents_for_ci"}
    return asdict(paired_parent_bootstrap(method, baseline, parents, seed=seed, n_resamples=n_resamples))


def _bank_metrics(record, index):
    loss = float(record["candidate_losses"][index])
    result = {"loss": loss, "success": float(np.clip(1 - loss, 0, 1))}
    for source, target in (("candidate_decoded_energy", "decoded_energy"),
                           ("candidate_any_chain_break", "any_chain_break"),
                           ("candidate_chain_break_fraction", "chain_break_fraction"),
                           ("candidate_state_error", "state_error_diagnostic"),
                           ("candidate_norm_error", "norm_error"),
                           ("candidate_seconds", "offline_scoring_seconds")):
        result[target] = float(record[source][index]) if source in record else None
    delta = result["state_error_diagnostic"]
    result["loss_ambiguity_indicator"] = 2 * delta + delta**2 if delta is not None else None
    result["tts"] = probability_tts(result["success"], float(_scalar(record, "runtime")))
    return result


def _checkpoint_provenance(payload, training, validation, test):
    train_ids = set(payload.get("train_parent_ids", []))
    validation_ids = set(payload.get("validation_parent_ids", []))
    if not train_ids or not validation_ids or train_ids & validation_ids:
        raise ValueError("checkpoint lacks valid disjoint train/validation parent provenance")
    expected_train = {str(_scalar(r, "parent_id")) for r in training}
    expected_validation = {str(_scalar(r, "parent_id")) for r in validation}
    test_ids = {str(_scalar(r, "parent_id")) for r in test}
    if (train_ids | validation_ids) & test_ids:
        raise ValueError("test parent appears in checkpoint training/validation provenance")
    if train_ids != expected_train or validation_ids != expected_validation:
        raise ValueError("checkpoint training/validation parents do not match this dataset")
    strict = "data_fingerprints" in payload
    if strict:
        actual = {str(_scalar(r, "fingerprint")) for r in training + validation}
        if set(payload["data_fingerprints"]) != actual:
            raise ValueError("checkpoint data fingerprint mismatch")
    for key, records in (("train_record_ids", training), ("validation_record_ids", validation)):
        if key in payload and payload[key] is not None:
            if set(payload[key]) != {str(_scalar(r, "record_id")) for r in records}:
                raise ValueError(f"checkpoint {key} mismatch")
    provenance = payload.get("data_provenance", {})
    content_verified = False
    if "train_content_sha256" in provenance or "validation_content_sha256" in provenance:
        from .learning import records_content_digest
        for key, records in (("train_content_sha256", training), ("validation_content_sha256", validation)):
            if provenance.get(key) != records_content_digest(records):
                raise ValueError(f"checkpoint {key} content mismatch")
        content_verified = True
    if payload.get("checkpoint_role") in {"latest", "resume"}:
        raise ValueError("evaluate the validation-selected best checkpoint, not the latest resume checkpoint")
    return {"parent_splits_verified": True, "record_fingerprints_verified": strict,
            "training_validation_content_verified": content_verified,
            "legacy_checkpoint_warning": None if strict else "v0.1 checkpoint has parent-only data provenance"}


def validate_training_augmentation(original, training_records):
    """Verify append-only training labels without changing the held-out problem.

    Candidate rows may be appended per instance. Every original physics field,
    label, ID and candidate waveform remains exact; candidate_tau is a grid, not
    a candidate-aligned column. The checkpoint content digest is checked later
    against these explicit records, never against the smaller original bank.
    """
    original, training_records = list(original), list(training_records)
    def indexed(records):
        ids = [str(_scalar(r, "record_id")) for r in records]
        if len(ids) != len(set(ids)):
            raise ValueError("training augmentation contains duplicate record IDs")
        return dict(zip(ids, records))
    before, after = indexed(original), indexed(training_records)
    if set(before) != set(after):
        raise ValueError("training augmentation must retain exactly the original record IDs")
    for record_id, old in before.items():
        new = after[record_id]
        if str(_scalar(old, "split")) != "train" or str(_scalar(new, "split")) != "train":
            raise ValueError("augmentation accepts training records only")
        count = len(old["candidate_losses"])
        new_count = len(new["candidate_losses"])
        if new_count < count:
            raise ValueError("training augmentation removed original candidate rows")
        for key, value in old.items():
            if key not in new:
                raise ValueError(f"training augmentation removed {key}")
            if key == "payload_fingerprint":
                from .pipeline import _payload_fingerprint
                if str(_scalar(new, key)) != _payload_fingerprint(dict(new)):
                    raise ValueError("training augmentation payload fingerprint mismatch")
                continue
            if key == "metadata_json":
                old_metadata = json.loads(str(_scalar(old, key)))
                new_metadata = json.loads(str(_scalar(new, key)))
                for field, entry in old_metadata.items():
                    if field == "acquisitions":
                        if new_metadata.get(field, [])[:len(entry)] != entry:
                            raise ValueError("training augmentation changed acquisition history")
                    elif new_metadata.get(field) != entry:
                        raise ValueError(f"training augmentation changed original metadata {field}")
                continue
            candidate_column = key.startswith("candidate_") and key != "candidate_tau"
            expected, actual = np.asarray(value), np.asarray(new[key])
            if candidate_column:
                if expected.ndim < 1 or len(expected) != count or actual.ndim < 1 or len(actual) != new_count:
                    raise ValueError(f"training augmentation has misaligned {key}")
                actual = actual[:count]
            # array_equal without equal_nan also supports string-valued IDs.
            equal = np.array_equal(expected, actual)
            if not equal and expected.dtype.kind in "fc" and actual.dtype.kind in "fc":
                equal = np.array_equal(expected, actual, equal_nan=True)
            if not equal:
                raise ValueError(f"training augmentation changed original {key}")
        for key, value in new.items():
            if key.startswith("candidate_") and key != "candidate_tau":
                column = np.asarray(value)
                if column.ndim < 1 or len(column) != new_count:
                    raise ValueError(f"training augmentation has misaligned {key}")
        waves, losses = np.asarray(new["candidate_schedules"]), np.asarray(new["candidate_losses"])
        if losses.shape != (new_count,) or not np.isfinite(losses).all() or not np.isfinite(waves).all():
            raise ValueError("training augmentation contains invalid candidate labels/waveforms")
        if "candidate_ids" in new and len(set(np.asarray(new["candidate_ids"], str))) != new_count:
            raise ValueError("training augmentation contains duplicate candidate IDs")
    return training_records


def evaluate_checkpoint(data_dir, checkpoint_path, *, device="cpu", direct=True, seed=0,
                        backend="numpy", tolerance=5e-4, initial_steps=128, max_steps=8192,
                        norm_tolerance=1e-9, n_resamples=2000, max_ds_dtau=4., training_records=None):
    """Load, validate, select without outcomes, then score held-out records.

    Explicit ``training_records`` support append-only acquisition experiments.
    They are validated against the original data and checkpoint content hash;
    the common deployment bank and validation choice always use original data.
    """
    import torch
    from .learning import load_checkpoint
    from .models import graph_from_record
    from .pipeline import environment, load_records

    _numerical_settings(tolerance, initial_steps, max_steps, norm_tolerance, backend)
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, int) or n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer")
    started = perf_counter()
    training, validation, test = (load_records(data_dir, split) for split in ("train", "validation", "test"))
    if not training or not validation or not test:
        raise ValueError("evaluation requires nonempty explicit train, validation and test splits")
    actual_training = (training if training_records is None else
                       validate_training_augmentation(training, training_records))
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    provenance = _checkpoint_provenance(payload, actual_training, validation, test)
    if training_records is not None and not provenance["training_validation_content_verified"]:
        raise ValueError("explicit training records require checkpoint content provenance")
    provenance["explicit_training_records_verified"] = training_records is not None
    bank = validate_candidate_banks(training + validation + test)
    # Parent-equal validation average; no test outcome enters this choice.
    vp = [str(_scalar(r, "parent_id")) for r in validation]
    val_scores = [_parent_mean([float(r["candidate_losses"][i]) for r in validation], vp)
                  for i in range(bank["candidate_count"])]
    global_index, linear_index = int(np.argmin(val_scores)), bank["linear_index"]
    model, normalizer = load_checkpoint(checkpoint_path, device=device)
    model.eval()
    if not np.isfinite(max_ds_dtau) or max_ds_dtau < 1:
        raise ValueError("max_ds_dtau must be finite and >=1")
    if direct and not np.isclose(float(model.config.get("max_ds_dtau", 4.)), max_ds_dtau):
        raise ValueError("policy slew differs from evaluation max_ds_dtau; use a matched control contract")
    rows, direct_rows = [], []
    bank_seconds = 0.
    for record in test:
        began = perf_counter()
        graph = normalizer.transform(graph_from_record(record, device=device))
        graph_seconds = perf_counter() - began
        with torch.no_grad():
            schedules = torch.as_tensor(record["candidate_schedules"], dtype=torch.float32, device=device)
            prediction = model.predict_losses(graph, schedules)
            if not torch.isfinite(prediction).all():
                raise ArithmeticError("nonfinite predicted candidate losses")
            selected = int(prediction.argmin())
        inference_seconds = perf_counter() - began
        bank_seconds += inference_seconds
        # Outcomes are inspected only AFTER the model's selection is frozen.
        losses = np.asarray(record["candidate_losses"], float)
        best_index = int(losses.argmin())
        metrics = {name: _bank_metrics(record, index) for name, index in
                   (("bank_policy", selected), ("linear", linear_index),
                    ("global_validation", global_index), ("best_bank_reference", best_index))}
        row = {**_identity(record), "selected_index": selected, "candidate_count": len(losses),
               "selected_loss": float(losses[selected]), "linear_loss": float(losses[linear_index]),
               "global_loss": float(losses[global_index]), "bank_best_loss": float(losses[best_index]),
               "best_bank_loss": float(losses[best_index]), "bank_regret": float(losses[selected] - losses[best_index]),
               "inference_seconds": inference_seconds, "metrics": metrics}
        radius = metrics["bank_policy"]["loss_ambiguity_indicator"]
        baseline_radius = metrics["linear"]["loss_ambiguity_indicator"]
        row["vs_linear_numerically_ambiguous"] = (None if radius is None or baseline_radius is None else
                                                   abs(row["selected_loss"] - row["linear_loss"]) <= radius + baseline_radius)
        rows.append(row)
        if direct:
            began = perf_counter()
            with torch.no_grad():
                proposals = model(graph)["proposal_schedules"]
                predicted = model.predict_losses(graph, proposals)
                if not torch.isfinite(predicted).all() or not torch.isfinite(proposals).all():
                    raise ArithmeticError("nonfinite direct policy outputs")
                index = int(predicted.argmin())
                wave = proposals[index].detach().cpu().double().numpy().copy()
            wave[0], wave[-1] = 0., 1.
            chosen = Schedule(np.linspace(0., 1., len(wave)), wave)
            direct_seconds = perf_counter() - began + graph_seconds
            # Small float32 construction tolerance; do not repair interior knots.
            score = score_schedule(record, chosen, backend=backend, tolerance=tolerance,
                                   initial_steps=initial_steps, max_steps=max_steps,
                                   norm_tolerance=norm_tolerance, max_ds_dtau=max_ds_dtau * (1 + 1e-6))
            score["tts"] = probability_tts(score["success"], row["runtime"])
            direct_rows.append({**_identity(record), **score, "selected_proposal": index,
                                "inference_seconds": direct_seconds,
                                "reference_bank_regret": score["loss"] - row["bank_best_loss"],
                                "vs_linear_numerically_ambiguous": (None if baseline_radius is None else
                                    abs(score["loss"] - row["linear_loss"]) <= score["loss_ambiguity_indicator"] + baseline_radius)})
    parents = [r["parent_id"] for r in rows]
    learned, linear, global_loss, best = ([r[key] for r in rows] for key in
                                         ("selected_loss", "linear_loss", "global_loss", "bank_best_loss"))
    result = {"schema_version": 2, "selection_mode": "finite_bank_critic", "n_records": len(rows),
              "records": rows, "mean_loss": float(np.mean(learned)),
              "mean_bank_regret": float(np.mean(np.asarray(learned) - best)),
              "mean_linear_loss": float(np.mean(linear)), "mean_global_candidate_loss": float(np.mean(global_loss)),
              "parent_mean_loss": _parent_mean(learned, parents),
              "parent_mean_linear_loss": _parent_mean(linear, parents),
              "parent_mean_global_loss": _parent_mean(global_loss, parents),
              "n_test_parents": len(set(parents)), "bank_inference_seconds": bank_seconds,
              "global_validation_candidate_index": global_index,
              "global_validation_candidate_parent_mean_losses": val_scores,
              "paired_loss_difference_vs_linear": _paired(learned, linear, parents, seed, n_resamples),
              "paired_loss_difference_vs_global": _paired(learned, global_loss, parents, seed, n_resamples),
              "training_seed": payload.get("seed"), "evaluation_seed": seed,
              "checkpoint": str(Path(checkpoint_path)), "checkpoint_sha256": hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest(),
              "provenance": provenance, "common_bank": bank, "environment": environment(),
              "read_budget_at_deployment": 0,
              "costs": {"bank_inference_seconds": bank_seconds,
                        "stored_test_bank_label_seconds": float(sum(np.sum(r.get("candidate_seconds", 0.)) for r in test)),
                        "stored_validation_bank_label_seconds": float(sum(np.sum(r.get("candidate_seconds", 0.)) for r in validation)),
                        "online_simulator_search_calls": 0},
              "uncertainty_note": "Parent bootstrap excludes training-seed uncertainty; numerical ambiguity indicators are not certificates.",
              "timing_note": "Bank/direct standalone inference both include preprocessing; shared work must not be summed as actual wall time.",
              "scope": "held-out closed-system simulation; no hardware, quantum advantage or conference-readiness claim"}
    if direct:
        direct_losses = [row["loss"] for row in direct_rows]
        result["direct_policy"] = {"selection": "critic_selected_among_policy_proposals_then_true_simulator_scored",
                                   "records": direct_rows, "mean_loss": float(np.mean(direct_losses)),
                                   "parent_mean_loss": _parent_mean(direct_losses, parents),
                                   "paired_vs_linear": _paired(direct_losses, linear, parents, seed, n_resamples),
                                   "paired_vs_global": _paired(direct_losses, global_loss, parents, seed, n_resamples),
                                   "paired_vs_bank": _paired(direct_losses, learned, parents, seed, n_resamples)}
        result["costs"].update(direct_inference_seconds=sum(r["inference_seconds"] for r in direct_rows),
                               direct_offline_scoring_seconds=sum(r["offline_scoring_seconds"] for r in direct_rows),
                               direct_offline_scoring_calls=len(direct_rows),
                               direct_integrator_steps=sum(r["total_integrator_steps"] for r in direct_rows))
    result["evaluation_wall_seconds"] = perf_counter() - started
    return _json_safe(result)


def benchmark_record_controls(record, *, budget_per_family=32, families=CONTROL_FAMILIES,
                              seed=0, allow_test_adaptation=False, backend="numpy", tolerance=5e-4,
                              initial_steps=128, max_steps=8192, norm_tolerance=1e-9,
                              teacher_methods=(), max_ds_dtau=4., teacher_grid_points=33,
                              teacher_gap_epsilon=0., strategy="sobol_local"):
    """Equal objective-call family comparison; every trial's true waveform saved.

    Teacher baselines are privileged and separately charged, never folded into
    the equal-budget claim. Linear takes one call because it has no parameters.
    """
    _numerical_settings(tolerance, initial_steps, max_steps, norm_tolerance, backend)
    split = str(_scalar(record, "split"))
    if split == "test" and not allow_test_adaptation:
        raise ValueError("test control benchmark requires explicit allow_test_adaptation=True")
    if not families or len(set(families)) != len(families) or any(f not in CONTROL_FAMILIES for f in families):
        raise ValueError("families must be a nonempty unique subset of CONTROL_FAMILIES")
    if isinstance(budget_per_family, bool) or not isinstance(budget_per_family, int) or budget_per_family < 1:
        raise ValueError("budget_per_family must be a positive integer")
    if isinstance(teacher_grid_points, bool) or not isinstance(teacher_grid_points, int) or teacher_grid_points < 3:
        raise ValueError("teacher_grid_points must be an integer >=3")
    if (len(set(teacher_methods)) != len(teacher_methods) or
            any(method not in {"gap_inverse_square", "d2"} for method in teacher_methods)):
        raise ValueError("teacher_methods must be unique gap_inverse_square/d2 names")
    if not np.isfinite(teacher_gap_epsilon) or teacher_gap_epsilon < 0:
        raise ValueError("teacher_gap_epsilon must be finite and nonnegative")
    if not np.isfinite(max_ds_dtau) or max_ds_dtau < 1:
        raise ValueError("max_ds_dtau must be finite and >=1")
    began = perf_counter()
    context = record_physics(record)
    runtime = float(_scalar(record, "runtime"))
    settings = dict(backend=backend, tolerance=tolerance, initial_steps=initial_steps,
                    max_steps=max_steps, norm_tolerance=norm_tolerance,
                    max_ds_dtau=max_ds_dtau, physics_context=context)
    family_results = {}
    for family in families:
        outcomes = []
        def objective(schedule):
            score = score_schedule(record, schedule, **settings)
            outcomes.append(score)
            return score["loss"]
        search = optimize_control_family(objective, family, budget=budget_per_family,
                                         runtime=runtime, max_slope=max_ds_dtau / runtime,
                                         seed=seed + CONTROL_FAMILIES.index(family) * 1009, split=split,
                                         allow_test_adaptation=allow_test_adaptation,
                                         strategy=strategy)
        incumbent = float("inf")
        records = []
        for trial, outcome in zip(search.records, outcomes):
            incumbent = min(incumbent, trial.loss)
            records.append({**outcome, "candidate_id": trial.candidate.candidate_id,
                            "parameters": trial.candidate.parameters, "evaluation_index": trial.evaluation_index,
                            "incumbent_loss": incumbent, "elapsed_seconds": trial.elapsed_seconds})
        best_index = search.best.evaluation_index
        family_results[family] = {"best_loss": search.best.loss, "best_index": best_index,
                                  "best": records[best_index], "records": records,
                                  "n_evaluations": search.n_evaluations,
                                  "objective_budget": 1 if family == "linear" else budget_per_family,
                                  "parameter_free": family == "linear",
                                  "online_adaptation": search.online_adaptation,
                                  "search_seconds": search.elapsed_seconds,
                                  "total_integrator_steps": sum(r["total_integrator_steps"] for r in records),
                                  "reference_status": search.reference_status,
                                  "exact_switching_waveforms": True}
    teacher_results = {}
    for method in teacher_methods:
        from .physics_baselines import exact_teacher_baseline
        teacher = exact_teacher_baseline(context[0], method, runtime=runtime, max_slope=max_ds_dtau / runtime,
                                         path=context[1], s_grid=np.linspace(0, 1, teacher_grid_points),
                                         gap_epsilon=teacher_gap_epsilon, audit_seed=seed)
        row = {"method": teacher.method, "status": teacher.status,
               "teacher_seconds": teacher.teacher_seconds, "teacher_evaluations": teacher.teacher_evaluations,
               "privileged_per_instance": True, "excluded_from_equal_budget_claim": True,
               "diagnostics": teacher.diagnostics, "outcome_evaluations": 0}
        if teacher.schedule is not None:
            row["outcome"] = score_schedule(record, teacher.schedule, **settings)
            row["outcome_evaluations"] = 1
        teacher_results[method] = row
    return _json_safe({"schema_version": 2, **_identity(record), "seed": seed,
                       "families": family_results, "privileged_teachers": teacher_results,
                       "budget_per_tunable_family": budget_per_family,
                       "search_strategy": strategy,
                       "test_adaptation_explicitly_allowed": allow_test_adaptation,
                       "total_objective_calls": sum(x["n_evaluations"] for x in family_results.values()),
                       "wall_seconds": perf_counter() - began,
                       "scope": "best found by Sobol plus incumbent-local search, not global family optima",
                       "numerical_ambiguity_is_certificate": False})
