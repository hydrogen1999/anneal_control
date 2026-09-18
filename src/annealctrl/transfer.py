"""Frozen bank selection on logically unseen parents, without target adaptation.

Disjointness is based on exact labelled logical Ising coefficients. It is invariant
under embedding, runtime, generator metadata and source revision; it does not
certify graph/gauge-isomorphism disjointness. Dataset-local names and compiled
record fingerprints alone are never proof of a new logical problem.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence
import re

import numpy as np

from .headroom import _bootstrap, _parent_means
from .telemetry import _safe


def _scalar(record, key):
    if key not in record or np.asarray(record[key]).size != 1:
        raise ValueError(f"record requires scalar {key!r}")
    return np.asarray(record[key]).item()


def record_logical_fingerprint(record: Mapping[str, Any]) -> str:
    """Recompute from coefficients; reject a stale or forged stored fingerprint."""
    from .generation import IsingProblem, logical_fingerprint
    if not {"logical_h", "logical_edges", "logical_J"} <= record.keys():
        raise ValueError("logical coefficient provenance is required; IDs are not proof")
    fingerprint = logical_fingerprint(IsingProblem(record["logical_h"], record["logical_edges"], record["logical_J"]))
    if "logical_fingerprint" in record and str(_scalar(record, "logical_fingerprint")) != fingerprint:
        raise ValueError("logical coefficient fingerprint mismatch")
    return fingerprint


def _fingerprint_set(values, label):
    if not isinstance(values, (list, tuple)) or not values or any(
            not isinstance(v, str) or re.fullmatch(r"[0-9a-f]{64}", v) is None for v in values):
        raise ValueError(f"incomplete {label} logical fingerprint provenance")
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate {label} logical fingerprints")
    return set(values)


def _seen_identifiers(payload: Mapping[str, Any]) -> tuple[set[str], str]:
    """Read complete modern provenance, failing closed for legacy checkpoints."""
    provenance = payload.get("data_provenance", {})
    if provenance.get("logical_provenance_version") != 1:
        raise ValueError("checkpoint lacks complete logical provenance; supply independently verified source records for migration")
    parts = [_fingerprint_set(provenance.get(f"{split}_logical_fingerprints"), split)
             for split in ("train", "validation")]
    if parts[0] & parts[1]:
        raise ValueError("checkpoint train/validation logical provenance overlaps")
    for split, values in zip(("train", "validation"), parts):
        if provenance.get(f"{split}_logical_parent_count") != len(values):
            raise ValueError("incomplete logical provenance count")
        mapping = provenance.get(f"{split}_parent_logical_fingerprints")
        ids = provenance.get(f"{split}_parent_ids")
        if not isinstance(mapping, dict) or not ids or set(mapping) != set(ids) or set(mapping.values()) != values:
            raise ValueError("incomplete parent-to-logical provenance mapping")
    return parts[0] | parts[1], "logical_fingerprint"


def _migrate_logical_provenance(payload, source_records):
    """Verify old checkpoint/source association BEFORE recovering logical hashes.

    Full content digests are preferred. Older checkpoints can be associated via
    exact complete compiled-record fingerprints plus split membership. Names alone
    never suffice. Source records must come from ``pipeline.load_records`` so the
    source manifest, file hashes and logical coefficients have been audited.
    """
    from .learning import records_content_digest
    provenance = payload.get("data_provenance", {})
    selected, all_fingerprints = {}, set()
    for split in ("train", "validation"):
        ids = provenance.get(f"{split}_parent_ids", payload.get(f"{split}_parent_ids"))
        if not ids:
            raise ValueError("legacy checkpoint lacks complete train/validation provenance")
        subset = [r for r in source_records if str(_scalar(r, "split")) == split
                  and str(_scalar(r, "parent_id")) in set(ids)]
        if {str(_scalar(r, "parent_id")) for r in subset} != set(ids):
            raise ValueError("source records do not cover saved parent provenance")
        record_ids = provenance.get(f"{split}_record_ids", payload.get(f"{split}_record_ids"))
        if record_ids is not None and set(record_ids) != {str(_scalar(r, "record_id")) for r in subset}:
            raise ValueError("source records do not match saved record provenance")
        selected[split] = subset
        for record in subset:
            all_fingerprints.add(str(_scalar(record, "fingerprint")))
    has_digests = all(provenance.get(f"{s}_content_sha256") for s in selected)
    if has_digests:
        for split, records in selected.items():
            if records_content_digest(records) != provenance[f"{split}_content_sha256"]:
                raise ValueError("source content digest does not match legacy checkpoint; augmented training records must be supplied exactly")
        basis = "source_content_digests"
    else:
        saved = payload.get("data_fingerprints", provenance.get("data_fingerprints"))
        if not saved or any(re.fullmatch(r"[0-9a-f]{64}", f) is None for f in saved) or set(saved) != all_fingerprints:
            raise ValueError("legacy migration requires complete matching source content fingerprints, never IDs alone")
        basis = "complete_source_record_fingerprints"
    fingerprints = {split: {record_logical_fingerprint(r) for r in records}
                    for split, records in selected.items()}
    if fingerprints["train"] & fingerprints["validation"]:
        raise ValueError("source train/validation logical provenance overlaps")
    return fingerprints["train"] | fingerprints["validation"], basis



def validate_source_reference(payload: Mapping[str, Any], source_records: Sequence[Mapping[str, Any]]) -> dict:
    """Associate a checkpoint with original source physics and validation labels.

    Modern logical provenance can describe training whose candidate labels were
    augmented. We verify the complete record/parent/logical/compiled identity and
    EXACT original validation content, while explicitly distinguishing unverified
    augmented training labels from reconstructed full training content. Legacy
    checkpoints still require the stronger migration association below.
    """
    from .learning import records_content_digest
    provenance = payload.get("data_provenance", {})
    version = provenance.get("logical_provenance_version")
    if version not in (None, 1):
        raise ValueError("unsupported logical provenance version")
    if version != 1:
        seen, basis = _migrate_logical_provenance(payload, source_records)
        return {"basis": basis, "seen_logical_parent_count": len(seen),
                "training_content_verified": basis == "source_content_digests",
                "validation_content_verified": basis == "source_content_digests",
                "training_labels_reconstructed": basis == "source_content_digests"}
    seen, _ = _seen_identifiers(payload)
    subsets, fingerprints = {}, set()
    for split in ("train", "validation"):
        records = [r for r in source_records if str(_scalar(r, "split")) == split]
        mapping = {}
        record_ids = []
        for record in records:
            parent = str(_scalar(record, "parent_id"))
            fingerprint = record_logical_fingerprint(record)
            if parent in mapping and mapping[parent] != fingerprint:
                raise ValueError("source has inconsistent logical parent identity")
            mapping[parent] = fingerprint
            record_ids.append(str(_scalar(record, "record_id")))
            fingerprints.add(str(_scalar(record, "fingerprint")))
        if mapping != provenance[f"{split}_parent_logical_fingerprints"]:
            raise ValueError("source logical mappings do not match checkpoint")
        if len(record_ids) != len(set(record_ids)) or sorted(record_ids) != provenance.get(f"{split}_record_ids"):
            raise ValueError("complete source record IDs do not match checkpoint")
        subsets[split] = records
    expected = provenance.get("data_fingerprints", payload.get("data_fingerprints"))
    if not expected or set(expected) != fingerprints:
        raise ValueError("complete source compiled-record fingerprints do not match checkpoint")
    if records_content_digest(subsets["validation"]) != provenance.get("validation_content_sha256"):
        raise ValueError("source validation content must exactly match frozen checkpoint selection data")
    exact_train = records_content_digest(subsets["train"]) == provenance.get("train_content_sha256")
    return {"basis": "source_content_digests" if exact_train else "logical_and_compiled_source_identity_with_exact_validation",
            "seen_logical_parent_count": len(seen), "training_content_verified": exact_train,
            "validation_content_verified": True, "training_labels_reconstructed": exact_train,
            "scope": ("source logical/compiled identities and validation content verified; training label content "
                      + ("exactly verified" if exact_train else "differs and is not reconstructed; e.g. acquisition augmentation"))}


def validate_transfer_provenance(payload: Mapping[str, Any], records: Sequence[Mapping[str, Any]], *,
                                 source_records: Sequence[Mapping[str, Any]] | None = None) -> dict:
    """Require unseen logical parents relative to BOTH fitting and selection data."""
    if not records:
        raise ValueError("transfer requires nonempty target records")
    provenance = payload.get("data_provenance", {})
    if provenance.get("logical_provenance_version") not in (None, 1):
        raise ValueError("unsupported logical provenance version")
    if provenance.get("logical_provenance_version") == 1:
        seen, _ = _seen_identifiers(payload)
        migration_basis = None
    else:
        if source_records is None:
            raise ValueError("checkpoint lacks complete logical provenance; independently verified source records required for legacy migration")
        seen, migration_basis = _migrate_logical_provenance(payload, source_records)
    target = {record_logical_fingerprint(r) for r in records}
    if seen & target:
        raise ValueError(f"target logical parents must be disjoint from train AND validation; {len(seen & target)} overlap")
    return {"basis": "logical_fingerprint", "seen_logical_fingerprints": sorted(seen),
            "target_logical_fingerprints": sorted(target), "migration_basis": migration_basis,
            "scope": "exact labelled logical coefficients; not graph/gauge-isomorphism disjointness"}


def _validated_bank(record):
    waves = np.asarray(record["candidate_schedules"], dtype=float)
    losses = np.asarray(record["candidate_losses"], dtype=float)
    if waves.ndim != 2 or waves.shape[0] < 1 or waves.shape[1] != 9 or losses.shape != (len(waves),):
        raise ValueError("invalid candidate waveform/loss dimensions; expected nonempty K-by-9 bank")
    if not np.isfinite(waves).all() or not np.isfinite(losses).all() or np.any(losses < 0) or np.any(losses > 1):
        raise ValueError("candidate waveforms/losses must be finite; success losses must be in [0,1]")
    tau = np.asarray(record.get("candidate_tau", np.linspace(0, 1, 9)), dtype=float)
    if tau.shape != (9,) or not np.allclose(tau, np.linspace(0, 1, 9), atol=1e-12, rtol=0):
        raise ValueError("selector requires the shared uniformly sampled control grid")
    if np.any(np.abs(waves[:, 0]) > 1e-9) or np.any(np.abs(waves[:, -1] - 1) > 1e-9) or np.any(np.diff(waves) < -1e-9):
        raise ValueError("bank requires monotone schedules with exact endpoint constraints")
    linear = np.flatnonzero(np.max(np.abs(waves - tau), axis=1) <= 1e-9)
    if not len(linear):
        raise ValueError("bank must contain a genuine linear schedule; nearest waveform is not linear")
    return waves, losses, int(linear[0])


def source_global_schedule(source_records):
    """Choose a fixed waveform on SOURCE validation only, with equal parent weight."""
    validation = [r for r in source_records if str(_scalar(r, "split")) == "validation"]
    if not validation:
        raise ValueError("source global baseline requires source validation records")
    bank = _validated_bank(validation[0])[0]
    parent_values = {}
    for record in validation:
        waves, losses, _ = _validated_bank(record)
        if waves.shape != bank.shape or not np.allclose(waves, bank, atol=1e-12, rtol=0):
            raise ValueError("source validation banks must share identical ordered waveforms")
        parent_values.setdefault(record_logical_fingerprint(record), []).append(losses)
    scores = np.mean([np.mean(v, axis=0) for v in parent_values.values()], axis=0)
    index = int(np.argmin(scores))
    return bank[index].copy(), {"source_validation_parent_count": len(parent_values),
                               "selected_index": index, "mean_validation_loss": float(scores[index]),
                               "selection": "source_validation_equal_parent_mean_only"}


def _load_transfer_model(payload, device):
    """Support known inference schemas explicitly; never drop normalization."""
    from .learning import FeatureNormalizer, NORMALIZED_FIELDS
    from .models import AnnealController
    version = payload.get("checkpoint_version", 1)
    if version not in (1, 2):
        raise ValueError(f"unsupported checkpoint_version {version!r}")
    config = payload.get("model_config", payload.get("config"))
    if not isinstance(config, dict) or "model_state" not in payload:
        raise ValueError("checkpoint requires model_config and model_state")
    statistics = payload.get("normalizer")
    if not isinstance(statistics, dict) or set(statistics) != set(NORMALIZED_FIELDS):
        raise ValueError("checkpoint requires complete fitted source normalizer; target refitting is forbidden")
    import torch
    for mean, scale in statistics.values():
        if not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("invalid checkpoint normalizer statistics")
    model = AnnealController(**config).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, FeatureNormalizer(statistics)


def evaluate_transfer(checkpoint_path, records: Sequence[Mapping[str, Any]], *, device: str = "cpu",
                      source_records: Sequence[Mapping[str, Any]] | None = None,
                      source_global_schedule: np.ndarray | None = None) -> list[dict]:
    """Select before reading outcome labels; reuse frozen source normalization."""
    import torch
    from .models import graph_from_record
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    provenance = validate_transfer_provenance(payload, records, source_records=source_records)
    validated = [_validated_bank(record) for record in records]
    identities = [str(_scalar(record, "record_id")) for record in records]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate target record IDs")
    model, normalizer = _load_transfer_model(payload, device)
    rows = []
    for record, (waves, stored, linear_index) in zip(records, validated):
        graph = normalizer.transform(graph_from_record(record, device=device))
        with torch.no_grad():
            predicted = model.predict_losses(graph, torch.as_tensor(waves, dtype=torch.float32, device=device))
            if predicted.shape != (len(waves),) or not torch.isfinite(predicted).all():
                raise ValueError("model returned invalid candidate loss predictions")
            index = int(predicted.argmin())
        global_index = None
        if source_global_schedule is not None:
            global_wave = np.asarray(source_global_schedule, dtype=float)
            if global_wave.shape != (9,) or not np.isfinite(global_wave).all():
                raise ValueError("source global schedule must be a finite 9-knot waveform")
            matches = np.flatnonzero(np.max(np.abs(waves - global_wave), axis=1) <= 1e-9)
            if not len(matches):
                raise ValueError("source-selected global waveform absent from target bank; evaluate it separately, never substitute a target-selected waveform")
            global_index = int(matches[0])
        rows.append({"disjointness_basis": provenance["basis"], "provenance_migration_basis": provenance["migration_basis"],
                     "record_id": str(_scalar(record, "record_id")), "parent_id": str(_scalar(record, "parent_id")),
                     "logical_fingerprint": record_logical_fingerprint(record),
                     "selected_index": index, "selected_loss": float(stored[index]),
                     "linear_index": linear_index, "linear_loss": float(stored[linear_index]),
                     "source_global_index": global_index,
                     "source_global_loss": None if global_index is None else float(stored[global_index]),
                     "bank_best_loss": float(stored.min()), "physical_n": int(np.asarray(record["physical_h"]).size),
                     "runtime": float(_scalar(record, "runtime"))})
    return rows


def transfer_report(rows: Sequence[Mapping[str, Any]], *, bootstrap_resamples: int = 20000, seed: int = 0) -> dict:
    """Equal-parent means and paired parent CIs, conditional on one checkpoint."""
    rows = [dict(r, parent_id=r.get("logical_fingerprint", r["parent_id"])) for r in rows]
    if not rows:
        raise ValueError("transfer_report requires a nonempty row set")
    if type(bootstrap_resamples) is not int or bootstrap_resamples < 1:
        raise ValueError("bootstrap_resamples must be a positive integer")
    for row in rows:
        for key in ("selected_loss", "linear_loss", "bank_best_loss"):
            if not np.isfinite(row[key]) or not 0 <= row[key] <= 1:
                raise ValueError("report requires finite success losses in [0,1]")
    metrics = {key: _parent_means(rows, lambda r, k=key: r[k])[0]
               for key in ("selected_loss", "linear_loss", "bank_best_loss")}
    differences = metrics["selected_loss"] - metrics["linear_loss"]
    interval = _bootstrap(differences, n_resamples=bootstrap_resamples, seed=seed)
    enough = len(differences) >= 2
    better = enough and interval["high"] is not None and interval["high"] < 0
    worse = enough and interval["low"] is not None and interval["low"] > 0
    result = {"schema_version": 2, "n_records": len(rows), "n_parents": len(differences),
              **{"mean_" + key: float(values.mean()) for key, values in metrics.items()},
              "mean_bank_regret": float((metrics["selected_loss"] - metrics["bank_best_loss"]).mean()),
              "beats_linear_fraction": float((differences < 0).mean()),
              "vs_linear": {"mean_difference": float(differences.mean()), "parent_bootstrap_ci": interval,
                            "separated": bool(better or worse), "sign_convention": "negative favors selector"},
              "verdict": "beats_linear" if better else "worse_than_linear" if worse else "inconclusive",
              "unit_of_independence": "logical_parent", "weighting": "equal_logical_parent",
              "inference_scope": "descriptive unadjusted interval conditional on this checkpoint; non-rejection is not equivalence",
              "scope": "bank selection; source normalizer frozen; no target adaptation or selection by target outcomes"}
    has_global = [r.get("source_global_loss") is not None for r in rows]
    if any(has_global):
        if not all(has_global) or any(not np.isfinite(r["source_global_loss"]) or not 0 <= r["source_global_loss"] <= 1 for r in rows):
            raise ValueError("source global comparison requires finite outcomes on every target record")
        global_values = _parent_means(rows, lambda r: r["source_global_loss"])[0]
        delta = metrics["selected_loss"] - global_values
        result["mean_source_global_loss"] = float(global_values.mean())
        result["vs_source_global"] = {"mean_difference": float(delta.mean()),
                                      "parent_bootstrap_ci": _bootstrap(delta, n_resamples=bootstrap_resamples, seed=seed)}
    return _safe(result)
