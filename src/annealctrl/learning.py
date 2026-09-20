"""Auditable small-instance training and finite-candidate-bank evaluation.

Graphs are evaluated individually inside memory-bounded gradient-accumulation
batches. There is no padding, cross-graph message passing, or label interpolation.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from pathlib import Path
import math
import random
import hashlib
import json
import os
import tempfile
from typing import Any, Mapping, Sequence

import torch
import numpy as np
from torch import Tensor
from torch.nn import functional as F

from .models import AnnealController, GraphInput, graph_from_record


NORMALIZED_FIELDS = ("node_features", "edge_features", "logical_node_features",
                     "logical_edge_features", "context")


@dataclass
class FeatureNormalizer:
    """Affine, invertible feature scaling fitted on TRAIN graphs only."""
    statistics: dict[str, tuple[Tensor, Tensor]]

    @classmethod
    def fit(cls, graphs: Sequence[GraphInput]) -> "FeatureNormalizer":
        if not graphs:
            raise ValueError("Cannot fit normalization without training graphs")
        statistics = {}
        for name in NORMALIZED_FIELDS:
            values = [getattr(g, name).detach().cpu() for g in graphs]
            if name == "context":
                values = [v.unsqueeze(0) for v in values]
            merged = torch.cat(values, dim=0)
            if len(merged):
                mean = merged.mean(0)
                std = merged.std(0, unbiased=False)
                std = torch.where(std < 1e-6, torch.ones_like(std), std)
            else:
                mean = torch.zeros(values[0].shape[-1])
                std = torch.ones_like(mean)
            statistics[name] = (mean, std)
        return cls(statistics)

    def transform(self, graph: GraphInput) -> GraphInput:
        transformed = {}
        for name, (mean, std) in self.statistics.items():
            value = getattr(graph, name)
            transformed[name] = (value - mean.to(value)) / std.to(value)
        return replace(graph, **transformed)


def soft_targets(losses: Tensor, temperature: float = 0.05) -> Tensor:
    """Lower loss gets higher probability; temperature is not physical time."""
    if not math.isfinite(temperature) or temperature <= 0 or losses.ndim != 1 or not len(losses) or not torch.isfinite(losses).all():
        raise ValueError("Expected a nonempty finite loss vector and positive label temperature")
    return torch.softmax(-(losses - losses.min()) / temperature, dim=0)


def proposal_bank_logits(proposals: Tensor, mixture_logits: Tensor, bank: Tensor,
                         bandwidth: float = 0.1) -> Tensor:
    """A mixture kernel evaluated on a discrete bank for soft-target distillation.

    This is a waveform-space surrogate, NOT a differentiable physical outcome
    loss or a density over the continuous feasible-control manifold.
    """
    if not math.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("bandwidth must be finite and positive")
    distance = (bank[:, None, :] - proposals[None, :, :]).square().mean(-1)
    return torch.logsumexp(F.log_softmax(mixture_logits, dim=0)[None, :]
                           - distance / (2 * bandwidth**2), dim=-1)


# The design document's spectral-target ladder asks which spectral summary is
# worth supervising a network with, not only which predicts best. The datasets
# carry two: three moments per s-point, and eight frequency bins. The loop used
# to hard-code the moments, which left the third rung of that ladder untested.
RESPONSE_TARGETS = {"moments": "response_moments", "bins": "response_bins"}


def masked_response_loss(prediction: Tensor, labels: Tensor, mask: Tensor | None = None) -> Tensor:
    if labels.shape != prediction.shape:
        raise ValueError("Response label and prediction shapes must match")
    valid = torch.isfinite(labels) & (labels >= 0)
    if mask is not None:
        if mask.ndim == 1:
            mask = mask.unsqueeze(-1)
        valid = valid & torch.broadcast_to(mask.bool(), valid.shape)
    if not valid.any():
        return prediction.sum() * 0
    return F.smooth_l1_loss(torch.log1p(prediction[valid]), torch.log1p(labels[valid]))


def candidate_loss(output: Mapping[str, Tensor], schedules: Tensor, losses: Tensor, *,
                   label_temperature: float = 0.05, policy_weight: float = 0.2,
                   ranking_weight: float = 0.1, response_weight: float = 0.05,
                   response_labels: Tensor | None = None, response_mask: Tensor | None = None,
                   ranking_tolerance: float = 1e-5, bandwidth: float = 0.1,
                   loss_uncertainty: Tensor | None = None) -> dict[str, Tensor]:
    """Huber outcomes + uncertainty-aware pair ranking + multimodal distillation."""
    if any(not math.isfinite(v) or v < 0 for v in (policy_weight, ranking_weight, response_weight, ranking_tolerance)):
        raise ValueError("Loss weights and ranking_tolerance must be finite and nonnegative")
    if not math.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("bandwidth must be finite and positive")
    if loss_uncertainty is not None:
        if loss_uncertainty.shape != losses.shape or not torch.isfinite(loss_uncertainty).all() or (loss_uncertainty < 0).any():
            raise ValueError("Loss uncertainties must be finite, nonnegative and candidate-aligned")
        label_temperature = max(label_temperature, float(loss_uncertainty.max().detach()))
    targets = soft_targets(losses, label_temperature)
    predicted = output["predicted_losses"]
    outcome = F.smooth_l1_loss(predicted, losses)
    logits = proposal_bank_logits(output["proposal_schedules"], output["proposal_logits"], schedules, bandwidth)
    policy = -(targets * F.log_softmax(logits, dim=0)).sum()
    i, j = torch.triu_indices(len(losses), len(losses), offset=1, device=losses.device)
    difference = losses[i] - losses[j]
    threshold = ranking_tolerance
    if loss_uncertainty is not None:
        threshold = threshold + loss_uncertainty[i] + loss_uncertainty[j]
    valid = difference.abs() > threshold
    if valid.any():
        # When i is worse (larger loss), predicted_loss_i-predicted_loss_j > 0.
        ranking = F.binary_cross_entropy_with_logits((predicted[i][valid] - predicted[j][valid]) / label_temperature,
                                                     (difference[valid] > 0).to(predicted.dtype))
    else:
        ranking = predicted.sum() * 0
    response = output["response"].sum() * 0
    if response_labels is not None:
        response = masked_response_loss(output["response"], response_labels, response_mask)
    total = outcome + policy_weight * policy + ranking_weight * ranking + response_weight * response
    return {"total": total, "outcome": outcome, "policy": policy, "ranking": ranking, "response": response}


def _identity(record: Mapping[str, Any], key: str) -> str:
    if key not in record:
        raise ValueError(f"Record missing required {key}; do not invent split membership")
    return str(record[key])


def validate_splits(train_records: Sequence[Mapping[str, Any]],
                    validation_records: Sequence[Mapping[str, Any]]) -> None:
    if not train_records or not validation_records:
        raise ValueError("Both explicit train and validation records are required")
    for records, expected in ((train_records, "train"), (validation_records, "validation")):
        if any(_identity(r, "split") != expected for r in records):
            raise ValueError(f"Only records explicitly marked {expected!r} belong in this argument")
    train_ids = {_identity(r, "parent_id") for r in train_records}
    validation_ids = {_identity(r, "parent_id") for r in validation_records}
    if train_ids & validation_ids:
        raise ValueError("Logical-parent leakage between train and validation")


def _tensor(record: Mapping[str, Any], key: str, device: torch.device) -> Tensor:
    return torch.as_tensor(record[key], dtype=torch.float32, device=device)


@torch.no_grad()
def evaluate_records(model: AnnealController, records: Sequence[Mapping[str, Any]],
                     normalizer: FeatureNormalizer | None = None) -> dict[str, Any]:
    """Rank fixed pre-evaluated banks; never interpolate proposal outcome labels.

    Report the selection mode as ``finite_bank_critic`` in paper tables. This
    does not evaluate direct proposals or certify continuous-control regret.
    """
    model.eval()
    device = next(model.parameters()).device
    results = []
    for record in records:
        graph = graph_from_record(record, device=device)
        if normalizer is not None:
            graph = normalizer.transform(graph)
        schedules, losses = _tensor(record, "candidate_schedules", device), _tensor(record, "candidate_losses", device)
        if len(losses) != len(schedules) or not len(losses) or not torch.isfinite(losses).all():
            raise ValueError("A finite, nonempty matching candidate loss bank is required")
        predicted = model.predict_losses(graph, schedules)
        if not torch.isfinite(predicted).all():
            raise FloatingPointError("Model produced nonfinite candidate losses during evaluation")
        selected = int(predicted.argmin())
        selected_loss, best_loss = float(losses[selected]), float(losses.min())
        results.append({"parent_id": _identity(record, "parent_id"),
                        "selected_index": selected, "candidate_count": len(losses),
                        "selected_loss": selected_loss, "best_bank_loss": best_loss,
                        "bank_regret": selected_loss - best_loss})
    if not results:
        raise ValueError("Evaluation requires at least one record")
    return {"selection_mode": "finite_bank_critic", "n_records": len(results),
            "mean_loss": sum(r["selected_loss"] for r in results) / len(results),
            "mean_bank_regret": sum(r["bank_regret"] for r in results) / len(results),
            "records": results}


@dataclass
class FitResult:
    model: AnnealController
    normalizer: FeatureNormalizer
    history: list[dict[str, float | int]]
    best_epoch: int
    last_epoch: int = -1
    stopped_early: bool = False


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and Torch before constructing a model.

    Deterministic CUDA algorithms may raise on unsupported operations; there is
    deliberately no silent CPU fallback. Cross-device bitwise identity is not
    promised. Set CUBLAS_WORKSPACE_CONFIG before first CUDA work when needed.
    """
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = deterministic


def records_content_digest(records: Sequence[Mapping[str, Any]]) -> str:
    """Hash the actual records, not just IDs or mutable file paths."""
    digest = hashlib.sha256()
    def update(value: Any) -> None:
        if isinstance(value, Mapping):
            digest.update(b"mapping\0")
            for key in sorted(value):
                update(str(key)); update(value[key])
        elif isinstance(value, (list, tuple)):
            digest.update(f"sequence:{len(value)}\0".encode())
            for item in value:
                update(item)
        elif isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                update(value.tolist())
            else:
                digest.update(f"array:{value.dtype}:{value.shape}\0".encode())
                digest.update(np.ascontiguousarray(value).tobytes())
        elif isinstance(value, Tensor):
            update(value.detach().cpu().numpy())
        elif isinstance(value, np.generic):
            update(value.item())
        else:
            digest.update(json.dumps(value, sort_keys=True, allow_nan=True, separators=(",", ":")).encode())
            digest.update(b"\0")
    update(records)
    return digest.hexdigest()


def _data_provenance(train_records: Sequence[Mapping[str, Any]], validation_records: Sequence[Mapping[str, Any]],
                     dataset_fingerprint: str | None) -> dict[str, Any]:
    result: dict[str, Any] = {"train_content_sha256": records_content_digest(train_records),
                              "validation_content_sha256": records_content_digest(validation_records),
                              "dataset_fingerprint": dataset_fingerprint}
    for name, records in (("train", train_records), ("validation", validation_records)):
        result[f"{name}_parent_ids"] = sorted({_identity(r, "parent_id") for r in records})
        if any("record_id" in r for r in records) and not all("record_id" in r for r in records):
            raise ValueError(f"Inconsistent {name} record ID provenance")
        if all("record_id" in r for r in records):
            ids = [_identity(r, "record_id") for r in records]
            if len(set(ids)) != len(ids):
                raise ValueError(f"Duplicate {name} record IDs")
            result[f"{name}_record_ids"] = sorted(ids)
    # Logical identity excludes runtime, embedding, generator metadata and source revision.
    # Legacy/minimal synthetic fixtures without coefficients remain trainable, but
    # cannot make a transfer claim until independently verified source migration.
    coefficient_keys = {"logical_h", "logical_edges", "logical_J"}
    if all(coefficient_keys <= r.keys() for r in (*train_records, *validation_records)):
        from .transfer import record_logical_fingerprint
        logical_parts = {}
        for name, records in (("train", train_records), ("validation", validation_records)):
            mapping = {}
            for record in records:
                parent, fingerprint = _identity(record, "parent_id"), record_logical_fingerprint(record)
                if parent in mapping and mapping[parent] != fingerprint:
                    raise ValueError("one parent ID carries different logical coefficients")
                mapping[parent] = fingerprint
            values = sorted(set(mapping.values()))
            result[f"{name}_logical_fingerprints"] = values
            result[f"{name}_logical_parent_count"] = len(values)
            result[f"{name}_parent_logical_fingerprints"] = mapping
            logical_parts[name] = set(values)
        if logical_parts["train"] & logical_parts["validation"]:
            raise ValueError("Logical coefficient leakage between train and validation")
        result["logical_provenance_version"] = 1
    result["data_fingerprints"] = sorted({str(r["fingerprint"]) for r in (*train_records, *validation_records)
                                           if "fingerprint" in r})
    return result


def _atomic_checkpoint(payload: Mapping[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            torch.save(dict(payload), stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _cpu_copy(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def _rng_state(order_rng: random.Random) -> dict[str, Any]:
    np_state = np.random.get_state()
    return {"order": order_rng.getstate(), "python": random.getstate(),
            "numpy": (np_state[0], np_state[1].tolist(), np_state[2], np_state[3], np_state[4]),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": [s.cpu() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []}


def _restore_rng(state: Mapping[str, Any], order_rng: random.Random) -> None:
    order_rng.setstate(state["order"])
    random.setstate(state["python"])
    name, keys, pos, gaussian, cached = state["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), pos, gaussian, cached))
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"] and torch.cuda.is_available():
        if len(state["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("Exact resume requires the same visible CUDA device count")
        torch.cuda.set_rng_state_all([s.cpu() for s in state["torch_cuda"]])


def configure_trainable_scope(model: AnnealController, scope: str) -> list[str]:
    """Select disjoint control heads while keeping every shared module fixed.

    Attention is shared by policy and critic and must therefore stay frozen in
    all head-only comparisons. The schedule encoder belongs only to the critic;
    the learned policy query belongs only to proposal generation.
    """
    if scope not in {"all", "critic", "policy", "heads"}:
        raise ValueError("trainable_scope must be all, critic, policy, or heads")
    names = []
    for name, parameter in model.named_parameters():
        critic = name.startswith(("schedule_encoder.", "critic_head."))
        policy = name == "policy_query" or name.startswith("policy_head.")
        enabled = (scope == "all" or (scope in {"critic", "heads"} and critic)
                   or (scope in {"policy", "heads"} and policy))
        parameter.requires_grad_(enabled)
        if enabled:
            names.append(name)
    return names


def fit_records(train_records: Sequence[Mapping[str, Any]],
                validation_records: Sequence[Mapping[str, Any]], *, model: AnnealController | None = None,
                epochs: int = 50, learning_rate: float = 1e-3, patience: int = 10,
                seed: int = 0, device: str = "cpu", checkpoint: str | Path | None = None,
                label_temperature: float = 0.05, policy_weight: float = 0.2,
                ranking_weight: float = 0.1, response_weight: float = 0.05,
                response_target: str = "moments",
                model_config: Mapping[str, Any] | None = None,
                resume_from: str | Path | None = None, latest_checkpoint: str | Path | None = None,
                batch_size: int = 1, accumulation_steps: int = 1, weight_decay: float = 1e-4,
                max_grad_norm: float = 1.0, deterministic: bool = True, bandwidth: float = 0.1,
                ranking_tolerance: float = 1e-5, dataset_fingerprint: str | None = None,
                initialize_from: str | Path | None = None, trainable_scope: str = "all",
                selection_mode: str = "finite_bank_critic",
                gauge_augment: bool = False) -> FitResult:
    """AdamW with exact epoch-boundary resume and validation-only selection.

    ``epochs`` is the total target, not additional epochs. Every optimizer update
    averages at most ``batch_size * accumulation_steps`` independent graphs; the
    last short group uses its actual size. Computation is graphwise, so batch_size
    does not claim fused GPU throughput. Data and graphs stay on CPU until used.

    Resume requires unchanged data contents/order and training settings, except
    total epochs may increase and device may change (cross-device bitwise equality
    is not guaranteed). Both files contain the complete latest resume state;
    ``checkpoint`` exposes the best validation model for inference, whereas
    ``latest_checkpoint`` exposes the current epoch model. Legacy checkpoints
    remain loadable for inference but are not silently treated as resumable.

    ``initialize_from`` starts a fresh optimizer from a checkpoint and preserves
    its normalizer. Head-only mechanism fits require the same training parents
    and exactly the same validation records. ``fixed_epochs`` selects the final
    predeclared epoch and disables early stopping: a frozen critic would make
    bank-regret selection insensitive to all policy updates.
    """
    validate_splits(train_records, validation_records)
    if any(not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in (epochs, patience, batch_size, accumulation_steps)):
        raise ValueError("Positive integer epochs, patience, batch_size and accumulation_steps required")
    if any(not math.isfinite(v) or v <= 0 for v in (learning_rate, label_temperature, max_grad_norm, bandwidth)):
        raise ValueError("Finite positive learning rate, temperature, gradient norm and bandwidth required")
    if response_target not in RESPONSE_TARGETS:
        raise ValueError(f"response_target must be one of {sorted(RESPONSE_TARGETS)}; "
                         f"got {response_target!r}")
    if any(not math.isfinite(v) or v < 0 for v in (weight_decay, policy_weight, ranking_weight, response_weight, ranking_tolerance)):
        raise ValueError("Weights and ranking tolerance must be finite and nonnegative")
    if model is not None and model_config is not None:
        raise ValueError("Provide model or model_config, not both")
    if initialize_from is not None and model is not None:
        raise ValueError("Provide initialize_from or model, not both")
    if selection_mode not in {"finite_bank_critic", "fixed_epochs"}:
        raise ValueError("Unknown checkpoint selection_mode")
    if trainable_scope != "all" and initialize_from is None and resume_from is None:
        raise ValueError("Head-only training requires a frozen initialization checkpoint")
    if trainable_scope == "policy" and (selection_mode != "fixed_epochs" or policy_weight <= 0):
        raise ValueError("Policy-only training requires fixed_epochs and positive policy_weight")
    if checkpoint is not None and latest_checkpoint is not None and Path(checkpoint).resolve() == Path(latest_checkpoint).resolve():
        raise ValueError("Best and latest checkpoints must have distinct paths")
    seed_everything(seed, deterministic=deterministic)
    rng = random.Random(seed)
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training was requested but CUDA is unavailable; choose device='cpu' explicitly")
    payload = torch.load(resume_from, map_location="cpu", weights_only=True) if resume_from is not None else None
    initial = torch.load(initialize_from, map_location="cpu", weights_only=True) if initialize_from is not None else None
    initialization_sha = (hashlib.sha256(Path(initialize_from).read_bytes()).hexdigest()
                          if initialize_from is not None else
                          (payload.get("training_config", {}).get("initialization_sha256") if payload else None))
    if payload is not None and (payload.get("checkpoint_version", 0) < 2 or "resume_state" not in payload):
        raise ValueError("Legacy checkpoint lacks exact resume state; use load_checkpoint for explicit fine-tuning")
    if model is None:
        config = (dict(model_config) if model_config is not None else
                  ((payload or initial)["model_config"] if (payload or initial) is not None else {}))
        model = AnnealController(**config)
    if initial is not None:
        # Constructor fills defaults; compare the complete architecture.
        if model.config != initial["model_config"]:
            raise ValueError("Initialization model configuration mismatch")
        model.load_state_dict(initial["model_state"])
    model = model.to(target_device)
    # The response head is sized by model_config, the target by training config,
    # and nothing connected the two: a three-wide head asked to predict eight
    # bins failed deep in the loss with "shapes must match", naming neither the
    # head nor the target. Check it once, up front, in the caller's vocabulary.
    _target_field = RESPONSE_TARGETS[response_target]
    for _record in list(train_records) + list(validation_records):
        if _target_field in _record:
            _width = int(np.asarray(_record[_target_field]).shape[-1])
            if _width != model.config["response_dim"]:
                raise ValueError(
                    f"response_target={response_target!r} supplies {_width} values per "
                    f"s-point but the model's response_dim is {model.config['response_dim']}; "
                    f"set response_dim={_width} in the model config")
            break
    trainable_names = configure_trainable_scope(model, trainable_scope)
    training_config = {"epochs": epochs, "patience": patience, "learning_rate": learning_rate,
                       "weight_decay": weight_decay, "label_temperature": label_temperature,
                       "policy_weight": policy_weight, "ranking_weight": ranking_weight,
                       "response_weight": response_weight, "response_target": response_target,
                       "batch_size": batch_size,
                       "accumulation_steps": accumulation_steps, "max_grad_norm": max_grad_norm,
                       "deterministic": deterministic, "bandwidth": bandwidth,
                       "ranking_tolerance": ranking_tolerance, "seed": seed,
                       "gauge_augment": bool(gauge_augment),
                       "initialization_sha256": initialization_sha, "trainable_scope": trainable_scope,
                       "selection_mode": selection_mode}
    provenance = _data_provenance(train_records, validation_records, dataset_fingerprint)
    if initial is not None and trainable_scope != "all":
        initial_provenance = initial.get("data_provenance", {})
        for key in ("train_parent_ids", "validation_content_sha256"):
            if initial_provenance.get(key) != provenance[key]:
                raise ValueError(f"Mechanism initialization requires unchanged {key}")
    if payload is not None:
        old_config = {k: v for k, v in payload["training_config"].items() if k != "epochs"}
        for key, default in (("initialization_sha256", None), ("trainable_scope", "all"),
                             ("selection_mode", "finite_bank_critic")):
            old_config.setdefault(key, default)
        if old_config != {k: v for k, v in training_config.items() if k != "epochs"}:
            raise ValueError("Resume training configuration mismatch; begin a new fine-tuning run instead")
        if epochs < payload["training_config"]["epochs"]:
            raise ValueError("Resume epochs must not decrease the previous total target")
        if model.config != payload["model_config"]:
            raise ValueError("Resume model configuration mismatch")
        if provenance != payload["data_provenance"]:
            raise ValueError("Resume dataset provenance/content mismatch")
    # Memory grows with the source records, not accelerator dataset residency.
    # Gauge augmentation, when requested, re-draws a spin reversal for every
    # training record at every epoch. The transformation is exact -- the
    # spectrum is unchanged, so every stored candidate loss stays correct --
    # and it preserves frustration, which is the condition the design document
    # attaches to this arm. Validation is never augmented: the model is scored
    # on the gauge the dataset actually stores.
    gauge_augment = bool(training_config["gauge_augment"])
    gauge_rng = np.random.default_rng(seed + 917) if gauge_augment else None
    graphs = [graph_from_record(r) for r in train_records]
    origin = payload or initial
    normalizer = FeatureNormalizer(origin["normalizer"]) if origin is not None else FeatureNormalizer.fit(graphs)
    graphs = [normalizer.transform(g) for g in graphs]
    for record in (*train_records, *validation_records):
        schedules = np.asarray(record["candidate_schedules"])
        losses = np.asarray(record["candidate_losses"])
        if schedules.ndim != 2 or schedules.shape[1] != model.schedule_points:
            raise ValueError("Model schedule_points must match dataset waveform knots; no label interpolation is allowed")
        if losses.shape != (len(schedules),) or not len(losses) or not np.isfinite(losses).all():
            raise ValueError("A finite nonempty candidate bank with matching losses is required")
        if not np.isfinite(schedules).all():
            raise ValueError("Candidate schedule values must be finite")
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                 lr=learning_rate, weight_decay=weight_decay)
    history, best_epoch, stale = [], -1, 0
    best_regret = float("inf")
    best_state, best_optimizer = None, None
    start_epoch, last_epoch, stopped_early = 0, -1, False
    if payload is not None:
        state = payload["resume_state"]
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        history = copy.deepcopy(payload["history"])
        best_epoch, best_regret, stale = int(payload["best_epoch"]), float(payload["best_regret"]), int(state["stale"])
        best_state, best_optimizer = _cpu_copy(payload["best_model_state"]), _cpu_copy(payload["best_optimizer_state"])
        last_epoch = int(state["epoch"])
        start_epoch = last_epoch + 1
        stopped_early = bool(state["stopped_early"])
        _restore_rng(state["rng_state"], rng)
    resume_state = payload["resume_state"] if payload is not None else None
    frozen_names = sorted(set(model.state_dict()) - set(trainable_names))
    frozen_digest = (records_content_digest([{name: model.state_dict()[name] for name in frozen_names}])
                     if trainable_scope != "all" else None)
    effective_batch = batch_size * accumulation_steps
    for epoch in range(start_epoch, epochs) if not stopped_early else ():
        model.train()
        order = list(range(len(train_records)))
        rng.shuffle(order)
        epoch_parts = {name: 0.0 for name in ("total", "outcome", "policy", "ranking", "response")}
        updates = 0
        for start in range(0, len(order), effective_batch):
            group = order[start:start + effective_batch]
            optimizer.zero_grad(set_to_none=True)
            for index in group:
                record = train_records[index]
                if gauge_augment:
                    # A fresh spin reversal every epoch. Exact, so the stored
                    # candidate losses below remain the correct labels, and
                    # frustration-preserving, so this is not the sign-deleting
                    # invariance the design document refuses.
                    from .gauge import gauge_record, gauge_signs
                    signs = gauge_signs(int(np.asarray(record["logical_h"]).size), gauge_rng)
                    graph = normalizer.transform(
                        graph_from_record(gauge_record(record, signs))).to(target_device)
                else:
                    graph = graphs[index].to(target_device)
                schedules, losses = _tensor(record, "candidate_schedules", target_device), _tensor(record, "candidate_losses", target_device)
                output = model(graph, schedules)
                field = RESPONSE_TARGETS[response_target]
                labels = _tensor(record, field, target_device) if field in record else None
                if response_target == "moments":
                    mask_key = "response_mask" if "response_mask" in record else "response_moments_mask"
                    mask = _tensor(record, mask_key, target_device) if mask_key in record else None
                else:
                    # response_mask is (n_s, n_moments) and its columns are identical:
                    # it is a per-s-point validity flag wearing the moments' width.
                    # Bins are a different width, so reduce it to one dimension and
                    # let masked_response_loss broadcast it rather than reshaping the
                    # flag to a size it never meant.
                    mask = None
                    if "response_mask" in record:
                        flag = _tensor(record, "response_mask", target_device)
                        mask = flag[:, 0] if flag.ndim == 2 else flag
                uncertainty = None
                if "candidate_loss_uncertainty" in record:
                    uncertainty = _tensor(record, "candidate_loss_uncertainty", target_device)
                elif "candidate_state_error" in record:
                    # Numerical ambiguity indicator, NOT a certified true bound.
                    delta = _tensor(record, "candidate_state_error", target_device)
                    uncertainty = 2 * delta + delta.square()
                losses_by_task = candidate_loss(output, schedules, losses, label_temperature=label_temperature,
                                                 policy_weight=policy_weight, ranking_weight=ranking_weight,
                                                 response_weight=response_weight, response_labels=labels, response_mask=mask,
                                                 loss_uncertainty=uncertainty, bandwidth=bandwidth,
                                                 ranking_tolerance=ranking_tolerance)
                # These objectives depend only on the designated head(s).
                # Label targets are the fixed, simulator-evaluated bank losses;
                # no updated critic supplies pseudo-labels to the policy.
                if trainable_scope == "critic":
                    losses_by_task["total"] = losses_by_task["outcome"] + ranking_weight * losses_by_task["ranking"]
                elif trainable_scope == "policy":
                    losses_by_task["total"] = policy_weight * losses_by_task["policy"]
                elif trainable_scope == "heads":
                    losses_by_task["total"] = (losses_by_task["outcome"] + ranking_weight * losses_by_task["ranking"]
                                                + policy_weight * losses_by_task["policy"])
                if not torch.isfinite(losses_by_task["total"]):
                    raise FloatingPointError(f"Nonfinite training loss at epoch {epoch}, graph {index}")
                (losses_by_task["total"] / len(group)).backward()
                for name, value in losses_by_task.items():
                    epoch_parts[name] += float(value.detach())
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            updates += 1
        validation = evaluate_records(model, validation_records, normalizer)
        if frozen_digest is not None:
            current_digest = records_content_digest([{name: model.state_dict()[name] for name in frozen_names}])
            if current_digest != frozen_digest:
                raise RuntimeError("Frozen mechanism parameters changed during training")
        regret = validation["mean_bank_regret"]
        if not math.isfinite(regret):
            raise FloatingPointError("Nonfinite validation bank regret")
        history.append({"epoch": epoch, "train_loss": epoch_parts["total"] / len(train_records),
                        **{f"train_{k}_loss": v / len(train_records) for k, v in epoch_parts.items() if k != "total"},
                        "validation_bank_regret": regret, "validation_mean_loss": validation["mean_loss"],
                        "optimizer_updates": updates})
        if selection_mode == "fixed_epochs" or regret < best_regret:
            best_regret, best_epoch, stale = regret, epoch, 0
            best_state = _cpu_copy(model.state_dict())
            best_optimizer = _cpu_copy(optimizer.state_dict())
        else:
            stale += 1
        last_epoch, stopped_early = epoch, selection_mode != "fixed_epochs" and stale >= patience
        resume_state = {"epoch": epoch, "model_state": _cpu_copy(model.state_dict()),
                        "optimizer_state": _cpu_copy(optimizer.state_dict()),
                        "stale": stale, "stopped_early": stopped_early, "rng_state": _rng_state(rng)}
        saved = {"checkpoint_version": 2, "model_config": model.config,
                 "normalizer": _cpu_copy(normalizer.statistics), "best_epoch": best_epoch,
                 "best_regret": best_regret, "history": history, "seed": seed,
                 "selection_mode": selection_mode, "training_config": training_config,
                 "trainable_parameter_names": trainable_names,
                 "frozen_parameter_sha256": frozen_digest,
                 "training_device": str(target_device), "data_provenance": provenance,
                 "parameter_count": sum(p.numel() for p in model.parameters()),
                 "software_versions": {"torch": str(torch.__version__), "numpy": str(np.__version__)},
                 **{key: value for key, value in provenance.items() if key.endswith("_ids") or key == "data_fingerprints"},
                 "best_model_state": best_state, "best_optimizer_state": best_optimizer,
                 "resume_state": resume_state}
        if latest_checkpoint is not None:
            _atomic_checkpoint({**saved, "checkpoint_role": "latest", "model_state": resume_state["model_state"],
                                "optimizer_state": resume_state["optimizer_state"]}, latest_checkpoint)
        if checkpoint is not None:
            _atomic_checkpoint({**saved, "checkpoint_role": "best", "model_state": best_state,
                                "optimizer_state": best_optimizer}, checkpoint)
        if stopped_early:
            break
    assert best_state is not None
    if payload is not None and last_epoch < start_epoch:
        # A completed/early-stopped resume remains a valid, materialized result
        # even when the caller requests new destination paths.
        saved = {**payload, "training_config": training_config, "training_device": str(target_device)}
        if latest_checkpoint is not None:
            _atomic_checkpoint({**saved, "checkpoint_role": "latest", "model_state": resume_state["model_state"],
                                "optimizer_state": resume_state["optimizer_state"]}, latest_checkpoint)
        if checkpoint is not None:
            _atomic_checkpoint({**saved, "checkpoint_role": "best", "model_state": best_state,
                                "optimizer_state": best_optimizer}, checkpoint)
    model.load_state_dict(best_state)
    model.eval()
    return FitResult(model, normalizer, history, best_epoch, last_epoch, stopped_early)


def load_checkpoint(path: str | Path, *, device: str = "cpu") -> tuple[AnnealController, FeatureNormalizer]:
    """Load tensor/basic-container checkpoints only (not arbitrary pickled objects)."""
    payload = torch.load(path, map_location=device, weights_only=True)
    model = AnnealController(**payload["model_config"]).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, FeatureNormalizer(payload["normalizer"])
