"""Train-only, budget-matched acquisition of simulator labels.

Policy proposals, a continuation of the original bank, and random logits passed
through the policy decoder are three different acquisition rules. Their effect
must be measured; using the decoder or adding labels is not itself evidence of
improved control. This is dataset aggregation, not a claim of canonical DAgger
imitation-learning guarantees. Validation and test banks stay untouched.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .telemetry import _safe


class AcquisitionError(RuntimeError):
    """A failed matched acquisition; ``audit`` preserves the partial cost ledger."""

    def __init__(self, message: str, audit: Mapping[str, Any]):
        super().__init__(message)
        self.audit = _safe(audit)


def _train_only(split: str) -> None:
    if split != "train":
        raise ValueError("acquisition and augmentation are train-only; using validation/test labels is leakage")


def _positive_count(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be an integer of at least one")


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(_safe(value), sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _record_digest(record: Mapping[str, Any]) -> str:
    from .pipeline import _payload_fingerprint

    return _payload_fingerprint(dict(record))


def _training_records(data_dir, split):
    from .pipeline import load_records

    _train_only(split)
    records = load_records(data_dir, split)
    if not records:
        raise ValueError(f"split {split!r} of {data_dir} contains no records")
    _validate_records(records)
    return records


def _validate_records(records):
    seen = set()
    if not records:
        raise ValueError("nonempty training records required")
    for record in records:
        _train_only(str(np.asarray(record.get("split", "missing")).item()))
        identifier = str(np.asarray(record["record_id"]).item())
        if identifier in seen:
            raise ValueError(f"duplicate training record {identifier!r}")
        seen.add(identifier)
        stored = np.asarray(record["candidate_schedules"], dtype=float)
        if stored.ndim != 2 or stored.shape[1] < 2 or not np.isfinite(stored).all():
            raise ValueError("finite candidate_schedules[B,K] required")
        expected_tau = np.linspace(0., 1., stored.shape[1])
        if "candidate_tau" in record and not np.array_equal(record["candidate_tau"], expected_tau):
            raise ValueError("acquisition requires the model's common uniform knot grid")


def _collect(records, waveforms: Callable, *, arm: str, n_extra: int, round_id: str,
             provenance: Mapping[str, Any], backend, tolerance, initial_steps, max_steps,
             max_ds_dtau) -> dict:
    from .benchmarking import _numerical_settings, score_schedule
    from .schedules import Schedule

    _positive_count(n_extra, "n_extra")
    _validate_records(records)
    _numerical_settings(tolerance, initial_steps, max_steps, 1e-9, backend)
    if not np.isfinite(max_ds_dtau) or max_ds_dtau < 1:
        raise ValueError("max_ds_dtau must be finite and >= 1")
    if not isinstance(round_id, str) or not round_id.strip():
        raise ValueError("round_id must be a nonempty string")
    settings = dict(backend=backend, tolerance=tolerance, initial_steps=initial_steps,
                    max_steps=max_steps, max_ds_dtau=max_ds_dtau,
                    slope_roundoff_relative_tolerance=1e-6, norm_tolerance=1e-9)
    identity = dict(arm=arm, round_id=round_id, n_extra=int(n_extra), settings=settings, provenance=provenance,
                    record_digests={str(np.asarray(r["record_id"]).item()): _record_digest(r)
                                    for r in records})
    acquisition_id = _digest(identity)
    began = perf_counter()
    result = dict(schema_version=2, status="collecting", arm=arm, round_id=round_id,
                  acquisition_id=acquisition_id, split="train", records={}, n_records=len(records),
                  proposals_per_record=int(n_extra), settings=settings, provenance=dict(provenance),
                  requested_objective_calls=len(records) * n_extra, objective_calls=0,
                  successful_objective_calls=0, propagation_calls=0, total_integrator_steps=0,
                  convergence_attempts=0, offline_scoring_seconds=0.,
                  cost_counts_complete=True, failures=[], uncertainty_is_certificate=False,
                  scope="training labels only; fixed original validation and test banks")
    # Validate the entire requested design before spending any scoring budget.
    pending = []
    for record in records:
        identifier = str(np.asarray(record["record_id"]).item())
        waves = np.asarray(waveforms(record), dtype=float)
        points = np.asarray(record["candidate_schedules"]).shape[1]
        if waves.shape != (n_extra, points) or not np.isfinite(waves).all():
            raise ValueError(f"record {identifier!r} requires exactly {n_extra} finite waveforms with {points} knots")
        schedules = [Schedule(np.linspace(0., 1., points), row) for row in waves]
        runtime = float(np.asarray(record["runtime"]).item())
        for schedule in schedules:
            schedule.validate_slope(runtime=runtime, max_slope=max_ds_dtau * (1 + 1e-6) / runtime)
        pending.append((record, identifier, waves, schedules))
    for record, identifier, waves, schedules in pending:
        outcomes = []
        for index, schedule in enumerate(schedules):
            result["objective_calls"] += 1
            try:
                outcome = score_schedule(record, schedule, backend=backend, tolerance=tolerance,
                                         initial_steps=initial_steps, max_steps=max_steps,
                                         max_ds_dtau=max_ds_dtau * (1 + 1e-6))
            except ArithmeticError as error:
                # No rejection sampling: a difficult trajectory fails the arm.
                # The scorer does not expose its failed internal work, so these
                # totals are lower bounds and must never be labelled complete.
                result["cost_counts_complete"] = False
                result["status"] = "failed"
                result["wall_seconds"] = perf_counter() - began
                result["failures"].append(dict(record_id=identifier, proposal_index=index,
                                               error_type=type(error).__name__, reason=str(error)))
                raise AcquisitionError(f"matched acquisition failed on {identifier!r}, candidate {index}: {error}",
                                       result) from error
            outcomes.append(outcome)
            result["successful_objective_calls"] += 1
            for key in ("propagation_calls", "total_integrator_steps", "convergence_attempts",
                        "offline_scoring_seconds"):
                result[key] += outcome[key]
        prefix = "dagger" if arm == "policy" else arm
        result["records"][identifier] = dict(
            waveforms=waves, losses=np.asarray([o["loss"] for o in outcomes]), outcomes=outcomes,
            record_digest=identity["record_digests"][identifier],
            candidate_ids=[f"{prefix}_{acquisition_id[:16]}_{i:04d}" for i in range(n_extra)])
    result.update(status="complete", wall_seconds=perf_counter() - began,
                  # Historical name meant objective evaluations, not low-level
                  # propagation calls. Keep it as an explicitly documented alias.
                  propagations=result["successful_objective_calls"],
                  propagations_alias="successful_objective_calls", infeasible_proposals=0)
    result["artifact_digest"] = _digest(result)
    return result


def collect_labelled_proposals(data_dir, checkpoint, *, split: str = "train", device: str = "cpu",
                               backend: str = "numpy", tolerance: float = 5e-4,
                               initial_steps: int = 128, max_steps: int = 8192,
                               max_ds_dtau: float = 4.0, round_id: str = "round1") -> dict:
    """Score every frozen-policy proposal on training records, or fail explicitly."""
    _train_only(split)
    import torch

    from .learning import load_checkpoint
    from .models import graph_from_record

    records = _training_records(data_dir, split)
    model, normalizer = load_checkpoint(checkpoint, device=device)
    model.eval()
    if not np.isclose(model.max_ds_dtau, max_ds_dtau, rtol=0., atol=1e-12):
        raise ValueError("checkpoint decoder slope envelope differs from acquisition settings")

    def propose(record):
        graph = normalizer.transform(graph_from_record(record, device=device))
        with torch.no_grad():
            proposals = model(graph)["proposal_schedules"].detach().cpu().double().numpy()
        proposals[:, 0], proposals[:, -1] = 0., 1.
        return proposals

    result = _collect(records, propose, arm="policy", n_extra=model.proposals, round_id=round_id,
                      provenance=dict(checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                                      checkpoint=str(checkpoint), device=device),
                      backend=backend, tolerance=tolerance, initial_steps=initial_steps,
                      max_steps=max_steps, max_ds_dtau=max_ds_dtau)
    return result


def collect_bank_extension(data_dir, *, split: str = "train", n_extra: int = 3,
                           bank_seed: int, bank_size: int, n_segments: int = 8,
                           backend: str = "numpy", tolerance: float = 5e-4,
                           initial_steps: int = 128, max_steps: int = 8192,
                           max_ds_dtau: float = 4.0, round_id: str = "round1") -> dict:
    """Continue the original Sobol bank after checking its entire stored prefix."""
    _train_only(split)
    _positive_count(n_extra, "n_extra")
    _positive_count(bank_size, "bank_size")
    _positive_count(n_segments, "n_segments")
    from .search import shared_candidate_bank

    records = _training_records(data_dir, split)

    def propose(record):
        runtime = float(np.asarray(record["runtime"]).item())
        stored = np.asarray(record["candidate_schedules"], dtype=float)
        if stored.shape[0] != bank_size:
            raise ValueError("bank_size must equal the complete original bank; augmented banks are not a matched prefix")
        tau = np.linspace(0., 1., stored.shape[1])
        bank = shared_candidate_bank(n=bank_size + n_extra, n_segments=n_segments, seed=bank_seed,
                                     runtime=runtime, max_slope=max_ds_dtau / runtime)
        waves = np.stack([candidate.schedule(tau) for candidate in bank])
        if len(waves) != bank_size + n_extra or not np.allclose(waves[:bank_size], stored, rtol=0., atol=1e-9):
            raise ValueError("regenerated bank prefix differs from the complete stored bank; this control arm is not matched")
        return waves[bank_size:]

    return _collect(records, propose, arm="bankext", n_extra=n_extra, round_id=round_id,
                    provenance=dict(bank_seed=bank_seed, bank_size=bank_size, n_segments=n_segments),
                    backend=backend, tolerance=tolerance, initial_steps=initial_steps,
                    max_steps=max_steps, max_ds_dtau=max_ds_dtau)


def collect_decoder_random(data_dir, *, split: str = "train", n_extra: int = 3, seed: int,
                           n_segments: int = 8, logit_std: float = 1.0,
                           backend: str = "numpy", tolerance: float = 5e-4,
                           initial_steps: int = 128, max_steps: int = 8192,
                           max_ds_dtau: float = 4.0, round_id: str = "round1") -> dict:
    """Shared random decoder controls, with a preregisterable IID normal prior.

    The same logits are used for every training record, like the shared original
    bank. They are sampled before looking at any labels, rounded to float32, and
    passed through the actual CPU Torch policy decoder. This matches its feasible
    waveform family, not the learned proposal distribution or marginal variance.
    ``logit_std`` must be fixed before held-out evaluation.
    """
    _train_only(split)
    _positive_count(n_extra, "n_extra")
    _positive_count(n_segments, "n_segments")
    if not np.isfinite(logit_std) or logit_std <= 0:
        raise ValueError("logit_std must be finite and positive")
    import torch

    from .models import monotone_samples

    records = _training_records(data_dir, split)
    logits = (np.random.default_rng(seed).standard_normal((n_extra, n_segments)) * logit_std).astype(np.float32)
    with torch.no_grad():
        waves = monotone_samples(torch.from_numpy(logits), max_ds_dtau=max_ds_dtau).double().numpy()
    return _collect(records, lambda record: waves.copy(), arm="decoder_random", n_extra=n_extra,
                    round_id=round_id,
                    provenance=dict(seed=seed, n_segments=n_segments, logit_std=logit_std,
                                    logit_distribution="iid_normal_zero_mean", shared_across_records=True,
                                    decoder="models.monotone_samples", decoder_dtype="float32", logits=logits),
                    backend=backend, tolerance=tolerance, initial_steps=initial_steps,
                    max_steps=max_steps, max_ds_dtau=max_ds_dtau)


def augment_records(records: Sequence[Mapping[str, Any]], collected: Mapping[str, Any]) -> list[dict]:
    """Append all real labels to exactly the original training records.

    Both acquisition payload integrity and base-record content are checked. No
    diagnostic is imputed from old means; loss ambiguity remains an explicitly
    non-certified numerical indicator. JSON-round-tripped collections work too.
    """
    from .pipeline import _payload_fingerprint

    _train_only(collected.get("split", "missing"))
    _validate_records(records)
    if collected.get("schema_version") != 2 or collected.get("status") != "complete":
        raise ValueError("a complete schema_version=2 acquisition with real outcomes is required")
    count = collected["proposals_per_record"]
    _positive_count(count, "proposals_per_record")
    identifiers = {str(np.asarray(r["record_id"]).item()) for r in records}
    if identifiers != set(collected["records"]):
        raise ValueError("no collected proposals for one or more records, or extra records; use exactly the original training split")
    payload = {key: value for key, value in collected.items() if key != "artifact_digest"}
    if collected.get("artifact_digest") != _digest(payload):
        raise ValueError("acquisition artifact digest mismatch")
    columns = {"candidate_losses": "loss", "candidate_success": "success",
               "candidate_decoded_energy": "decoded_energy", "candidate_any_chain_break": "any_chain_break",
               "candidate_chain_break_fraction": "chain_break_fraction", "candidate_norm_error": "norm_error",
               "candidate_state_error": "state_error_diagnostic", "candidate_steps": "accepted_steps",
               "candidate_seconds": "offline_scoring_seconds", "candidate_loss_uncertainty": "loss_ambiguity_indicator"}
    augmented = []
    for record in records:
        identifier = str(np.asarray(record["record_id"]).item())
        entry = collected["records"][identifier]
        if entry["record_digest"] != _record_digest(record):
            raise ValueError(f"base record digest mismatch for {identifier!r}")
        stored = np.asarray(record["candidate_schedules"], dtype=float)
        waves = np.asarray(entry["waveforms"], dtype=float)
        outcomes = entry["outcomes"]
        if waves.shape != (count, stored.shape[1]) or len(outcomes) != count:
            raise ValueError("acquisition does not contain its exact per-record budget")
        if not np.allclose(entry["losses"], [o["loss"] for o in outcomes], rtol=0., atol=0.):
            raise ValueError("acquired losses disagree with their simulator outcomes")
        old_ids = [str(x) for x in np.asarray(record["candidate_ids"])]
        new_ids = list(entry["candidate_ids"])
        if len(new_ids) != count or len(set(old_ids + new_ids)) != len(old_ids) + count:
            raise ValueError("candidate IDs collide; the same acquisition cannot be appended twice")
        new = dict(record)
        new["candidate_schedules"] = np.vstack((stored, waves))
        new["candidate_ids"] = np.asarray(old_ids + new_ids)
        for key, outcome_key in columns.items():
            if key == "candidate_loss_uncertainty" and key not in record:
                if "candidate_state_error" not in record:
                    raise ValueError("base labels need state-error diagnostics or explicit loss uncertainty")
                delta = np.asarray(record["candidate_state_error"], dtype=float)
                original = 2 * delta + delta**2
            elif key not in record:
                continue
            else:
                original = np.asarray(record[key], dtype=float)
            if original.shape != (len(stored),):
                raise ValueError(f"{key} is not aligned with the original candidate bank")
            extra = np.asarray([outcome[outcome_key] for outcome in outcomes], dtype=float)
            if not np.isfinite(extra).all():
                raise ValueError(f"nonfinite simulator outcome {outcome_key}")
            new[key] = np.concatenate((original, extra))
        metadata = json.loads(str(np.asarray(record.get("metadata_json", "{}")).item()))
        metadata.setdefault("acquisitions", []).append(dict(acquisition_id=collected["acquisition_id"],
             artifact_digest=collected["artifact_digest"], arm=collected["arm"], round_id=collected["round_id"],
             n_extra=count, original_record_digest=entry["record_digest"],
             numerical_uncertainty_is_certificate=False))
        new["metadata_json"] = np.array(json.dumps(metadata, sort_keys=True))
        new["payload_fingerprint"] = np.array(_payload_fingerprint(new))
        augmented.append(new)
    return augmented
