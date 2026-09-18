"""Does a selector trained on one distribution work on another?

Every learned number elsewhere in this project is in-distribution: the model is
trained and evaluated on splits of the same generator. That is the weakest kind
of ML evidence, and until this module the project had no cross-distribution
measurement at all.

The provenance guard is inverted here, deliberately. ``benchmarking`` refuses a
checkpoint whose training parents do not *match* the dataset, which is right when
asking "how well did this model learn its own distribution". For transfer the
requirement is the opposite: the target parents must be **disjoint** from
everything the checkpoint saw. A transfer number measured on parents the model
trained on is leakage wearing a different name, so the disjointness is asserted
rather than assumed, and a checkpoint that cannot prove what it trained on is
refused outright.

What is measured is bank selection only. The selector consults no outcome: it
scores the record's stored candidates and picks one, and the reported loss is
that candidate's true stored loss. The normalizer travels with the checkpoint and
is **not** refitted on the target, because refitting is itself an adaptation and
would answer a different question.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .headroom import _bootstrap, _parent_means
from .telemetry import _safe


def _seen_identifiers(payload: Mapping[str, Any]) -> tuple[set[str], str]:
    """What the checkpoint saw, preferring content fingerprints over names.

    Parent ids are **dataset-local**. Two independently generated datasets both
    number their parents from `parent_0000`, so an id-based disjointness check
    reports overlap between problems that share nothing but a counter. That is
    not a hypothetical: the first run of this evaluation refused every checkpoint
    on seven such collisions while the content fingerprints overlapped by zero.

    Content fingerprints are global, so they are used when the checkpoint records
    them. Names are the fallback, and the caller is told which was used, because
    a disjointness claim resting on names is weaker than one resting on content.
    """
    fingerprints = {str(x) for x in (payload.get("data_fingerprints") or [])}
    if fingerprints:
        return fingerprints, "fingerprint"
    train = set(payload.get("train_parent_ids") or [])
    validation = set(payload.get("validation_parent_ids") or [])
    if not train or not validation:
        raise ValueError("checkpoint lacks train/validation parent provenance; a transfer claim "
                         "cannot be made about a model that cannot say what it trained on")
    return {str(x) for x in train | validation}, "parent_id"


def evaluate_transfer(checkpoint_path, records: Sequence[Mapping[str, Any]], *,
                      device: str = "cpu") -> list[dict]:
    """Bank-select on records the checkpoint never trained on.

    Returns one row per record. Nothing here consults a true outcome before the
    selection: the model ranks the stored candidates, and the loss reported is
    the stored loss of whichever it picked.
    """
    import torch

    from .learning import load_checkpoint
    from .models import graph_from_record

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    seen, basis = _seen_identifiers(payload)

    def identify(record):
        key = "fingerprint" if basis == "fingerprint" else "parent_id"
        if key not in record:
            raise ValueError(f"records carry no {key!r}; the checkpoint's provenance is recorded "
                             f"by {basis} and the two must be comparable")
        value = record[key]
        return value if isinstance(value, str) else str(np.asarray(value).item())

    target = {identify(r) for r in records}
    overlap = seen & target
    if overlap:
        raise ValueError(f"target records must be disjoint from what the checkpoint trained on, "
                         f"compared by {basis}; {len(overlap)} overlap, e.g. "
                         f"{sorted(overlap)[:5]}. A transfer number measured on trained-on "
                         "records is leakage.")

    model, normalizer = load_checkpoint(checkpoint_path, device=device)
    model.eval()
    rows = []
    for record in records:
        graph = graph_from_record(record, device=device)
        if normalizer is not None:
            graph = normalizer.transform(graph)
        stored = np.asarray(record["candidate_losses"], dtype=float)
        with torch.no_grad():
            bank = torch.as_tensor(record["candidate_schedules"], dtype=torch.float32, device=device)
            predicted = model.predict_losses(graph, bank)
            index = int(predicted.argmin())
        linear_index = int(np.argmin([
            float(np.abs(np.asarray(w) - np.linspace(0.0, 1.0, len(w))).max())
            for w in np.asarray(record["candidate_schedules"])]))
        rows.append({
            "disjointness_basis": basis,
            "record_id": str(np.asarray(record["record_id"]).item()),
            "parent_id": str(np.asarray(record["parent_id"]).item()),
            "selected_index": index,
            "selected_loss": float(stored[index]),
            "linear_loss": float(stored[linear_index]),
            "bank_best_loss": float(stored.min()),
            "physical_n": int(np.asarray(record["physical_h"]).size),
            "runtime": float(np.asarray(record["runtime"]).item()),
        })
    return rows


def transfer_report(rows: Sequence[Mapping[str, Any]], *, bootstrap_resamples: int = 20000,
                    seed: int = 0) -> dict:
    """Parent-level summary of a transfer evaluation.

    ``verdict`` is deliberately coarse. The question a transfer test answers first
    is not "by how much" but "at all": a selector that cannot beat a linear ramp
    off its training distribution has not transferred, whatever its margin.
    """
    rows = list(rows)
    if not rows:
        raise ValueError("transfer_report requires a nonempty row set")

    selected, parents = _parent_means(rows, lambda r: r.get("selected_loss"))
    differences, _ = _parent_means(
        [{"parent_id": r["parent_id"], "d": float(r["selected_loss"]) - float(r["linear_loss"])}
         for r in rows], lambda r: r["d"])
    regret, _ = _parent_means(
        [{"parent_id": r["parent_id"], "d": float(r["selected_loss"]) - float(r["bank_best_loss"])}
         for r in rows], lambda r: r["d"])

    beats = sum(1 for r in rows if float(r["selected_loss"]) < float(r["linear_loss"])) / len(rows)
    interval = _bootstrap(differences, n_resamples=bootstrap_resamples, seed=seed)
    separated = interval["low"] is not None and (interval["low"] > 0 or interval["high"] < 0)
    mean_difference = float(differences.mean())
    return _safe({
        "schema_version": 1,
        "n_records": len(rows),
        "n_parents": len(parents),
        "mean_selected_loss": float(selected.mean()),
        "mean_linear_loss": float(np.mean([float(r["linear_loss"]) for r in rows])),
        "mean_bank_best_loss": float(np.mean([float(r["bank_best_loss"]) for r in rows])),
        "mean_bank_regret": float(regret.mean()),
        "beats_linear_fraction": beats,
        "vs_linear": {"mean_difference": mean_difference, "parent_bootstrap_ci": interval,
                      "separated": bool(separated),
                      "sign_convention": "negative means the transferred selector is better"},
        "verdict": ("beats_linear" if mean_difference < 0 and beats > 0.5
                    else "worse_than_linear" if mean_difference > 0 and beats < 0.5
                    else "indistinguishable_from_linear"),
        "unit_of_independence": "logical_parent",
        "scope": ("bank selection only, no outcome consulted before the choice; the checkpoint's "
                  "own normalizer is used unrefitted, because refitting on the target is an "
                  "adaptation and answers a different question"),
    })
