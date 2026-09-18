"""Predeclared multi-seed transfer evaluation with immutable inputs and raw rows.

This runner evaluates existing checkpoints and banks. It never trains, adapts
normalization, selects target checkpoints or spends online simulator queries.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
import uuid

import numpy as np

from .experiments import content_hash, output_lock, source_hash
from .pipeline import load_records, write_json
from .transfer import (evaluate_transfer, transfer_report, source_global_schedule,
                       validate_transfer_provenance, validate_source_reference,
                       record_logical_fingerprint)

AXES = {"topology", "family", "physical_size", "runtime", "heldout_parents"}


def load_transfer_study(path):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text())
    cfg["source_data"] = str((path.parent / cfg["source_data"]).resolve())
    for item in cfg["checkpoints"]:
        item["path"] = str((path.parent / item["path"]).resolve())
    for item in cfg["targets"]:
        item["data"] = str((path.parent / item["data"]).resolve())
    validate_transfer_study(cfg)
    return cfg


def validate_transfer_study(cfg):
    allowed = {"schema_version", "source_data", "checkpoints", "targets", "device", "threads",
               "bootstrap_resamples", "bootstrap_seed", "offline_costs"}
    if {k for k in cfg if not k.startswith("_")} - allowed:
        raise ValueError("unknown transfer-study configuration keys")
    if cfg.get("schema_version", 1) != 1 or not isinstance(cfg.get("source_data"), str):
        raise ValueError("transfer study requires schema_version=1 and source_data")
    checkpoints = cfg.get("checkpoints", [])
    if not checkpoints:
        raise ValueError("at least one frozen checkpoint is required")
    seen, methods = set(), {}
    for item in checkpoints:
        if set(item) != {"method", "seed", "path"} or not item["method"] or not isinstance(item["path"], str):
            raise ValueError("each checkpoint requires method, seed and path")
        if type(item["seed"]) is not int or item["seed"] < 0:
            raise ValueError("checkpoint seed must be a nonnegative integer")
        key = (item["method"], item["seed"])
        if key in seen:
            raise ValueError("duplicate method/seed checkpoint")
        seen.add(key)
        methods.setdefault(item["method"], set()).add(item["seed"])
    if any(seeds != next(iter(methods.values())) for seeds in methods.values()):
        raise ValueError("methods must use the same predeclared training seeds")
    names = set()
    for target in cfg.get("targets", []):
        if set(target) - {"name", "data", "axes", "split", "physical_sizes", "families", "runtimes"}:
            raise ValueError("unknown transfer target keys")
        if not target.get("name") or target["name"] in names or not isinstance(target.get("data"), str):
            raise ValueError("targets require unique names and data paths")
        names.add(target["name"])
        axes = target.get("axes")
        if not isinstance(axes, list) or not axes or len(axes) != len(set(axes)) or set(axes) - AXES:
            raise ValueError("declare valid distinct transfer axes")
        if "heldout_parents" in axes and len(axes) > 1:
            raise ValueError("heldout_parents describes an in-distribution reference, not an OOD axis")
        if target.get("split", "test") != "test":
            raise ValueError("the frozen transfer study evaluates target test only")
        for field in ("physical_sizes", "families", "runtimes"):
            if field in target and (not isinstance(target[field], list) or not target[field]):
                raise ValueError(f"target {field} filter must be a nonempty list")
    if not names:
        raise ValueError("at least one target is required")
    for key, default in (("threads", 1), ("bootstrap_resamples", 10000)):
        if type(cfg.get(key, default)) is not int or cfg.get(key, default) < 1:
            raise ValueError(f"{key} must be a positive integer")
    if type(cfg.get("bootstrap_seed", 0)) is not int or cfg.get("bootstrap_seed", 0) < 0:
        raise ValueError("bootstrap_seed must be a nonnegative integer")
    costs = cfg.get("offline_costs", {})
    if set(costs) - {"source_generation_seconds", "training_seconds_by_method_seed", "evidence_note"}:
        raise ValueError("unknown offline_costs field")
    values = ([costs["source_generation_seconds"]] if "source_generation_seconds" in costs else [])
    values += list(costs.get("training_seconds_by_method_seed", {}).values())
    if any(not isinstance(v, (int, float)) or not np.isfinite(v) or v < 0 for v in values):
        raise ValueError("declared offline costs must be finite and nonnegative")
    if values and not costs.get("evidence_note"):
        raise ValueError("declared offline costs require an evidence_note; otherwise leave unknown")


def plan_transfer_study(cfg):
    validate_transfer_study(cfg)
    return {"schema_version": 1, "checkpoint_evaluations": len(cfg["checkpoints"]) * len(cfg["targets"]),
            "targets": cfg["targets"], "checkpoints": cfg["checkpoints"],
            "training": "none; all checkpoints frozen before targets are evaluated",
            "baselines": ["linear", "source_validation_selected_global", "best_target_bank_diagnostic_only"],
            "normalization": "source_checkpoint_statistics_only",
            "cost_scope": "zero online simulator calls; source training and target evaluation labels are separate costs",
            "scope": "plan only, not empirical transfer evidence"}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _inputs(cfg):
    paths = [Path(cfg["source_data"]) / "manifest.json"]
    paths += [Path(x["data"]) / "manifest.json" for x in cfg["targets"]]
    paths += [Path(x["path"]) for x in cfg["checkpoints"]]
    return {str(p.resolve()): _sha(p) for p in paths}


def _assert_inputs(cfg, manifest):
    if _inputs(cfg) != manifest["input_sha256"]:
        raise RuntimeError("input_drift: checkpoint or dataset manifest changed during transfer study")
    if source_hash() != manifest["source_hash"]:
        raise RuntimeError("source_drift: code changed during transfer study")


def _describe(records, manifest):
    config = manifest["config"]
    hw = config.get("hardware")
    return {"topology": hw.get("topology", "explicit_hardware_graph") if hw else "synthetic",
            "family": sorted({str(np.asarray(r["family"]).item()) for r in records}),
            "physical_size": sorted({len(r["physical_h"]) for r in records}),
            "runtime": sorted({float(np.asarray(r["runtime"]).item()) for r in records}),
            "logical_parent_count": len({record_logical_fingerprint(r) for r in records})}


def validate_axes(source, target, axes):
    for axis in axes:
        if axis == "heldout_parents":
            if source["topology"] != target["topology"] or any(
                    not set(target[field]) <= set(source[field]) for field in ("family", "physical_size", "runtime")):
                raise ValueError("heldout_parents reference has observed distribution shift; declare the actual transfer axis")
            continue
        if axis == "topology":
            if source[axis] == target[axis]:
                raise ValueError("declared topology transfer requires different observed topologies")
        elif set(source[axis]) & set(target[axis]):
            raise ValueError(f"declared unseen {axis} requires disjoint observed values; filter target explicitly")
    return {"axes": list(axes), "source": source, "target": target,
            "in_distribution_reference": axes == ["heldout_parents"],
            "in_distribution_scope": "observed topology/support-compatible reference only; not proof of identical generating distributions",
            "scope": "declared axes verified on loaded data; exact-parent disjointness checked separately"}


def _filter(records, target):
    selected = [r for r in records
                if ("physical_sizes" not in target or len(r["physical_h"]) in target["physical_sizes"])
                and ("families" not in target or str(np.asarray(r["family"]).item()) in target["families"])
                and ("runtimes" not in target or float(np.asarray(r["runtime"]).item()) in target["runtimes"])]
    if not selected:
        raise ValueError("declared target filters leave no records")
    return selected


def _receipt(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())



def _verify_receipts(root, manifest):
    receipts = manifest.get("receipts", {})
    paths = {p.name: p for p in (root / "attempts").glob("*.json")}
    if set(paths) != set(receipts) or any(_sha(paths[name]) != digest for name, digest in receipts.items()):
        raise ValueError("attempt receipt missing, unregistered or changed; preserve evidence and start a new audited run")


def _bound_receipt(root, manifest, name, value):
    path = root / "attempts" / name
    _receipt(path, value)
    manifest.setdefault("receipts", {})[name] = _sha(path)
    write_json(root / "manifest.json", manifest)


def summarize_transfer_study(rows, *, bootstrap_resamples=10000, seed=0):
    from .contrasts import paired_parent_seed_contrast
    output = []
    for target in sorted({r["target"] for r in rows}):
        for method in sorted({r["method"] for r in rows if r["target"] == target}):
            subset = [r for r in rows if r["target"] == target and r["method"] == method]
            seeds = sorted({r["seed"] for r in subset})
            per_seed = {str(s): transfer_report([r for r in subset if r["seed"] == s],
                                               bootstrap_resamples=bootstrap_resamples, seed=seed) for s in seeds}
            paired = []
            for r in subset:
                for name, key in (("selector", "selected_loss"), ("linear", "linear_loss"), ("source_global", "source_global_loss")):
                    paired.append({"method": name, "mode": "bank", "loss": r[key], "seed": r["seed"],
                                   "record_id": r["record_id"], "parent_id": r["logical_fingerprint"]})
            contrasts = {name: paired_parent_seed_contrast(paired, name, "selector", mode="bank",
                         bootstrap_resamples=bootstrap_resamples, seed=seed) for name in ("linear", "source_global")}
            output.append({"target": target, "method": method, "n_seeds": len(seeds), "per_seed": per_seed,
                           "mean_selected_loss": float(np.mean([v["mean_selected_loss"] for v in per_seed.values()])),
                           "crossed_parent_seed_contrasts": contrasts,
                           "verdict": "descriptive_only; assess predeclared contrasts and multiplicity in paper"})
    return output


def run_transfer_study(cfg, output, *, resume=False):
    """Run a frozen study; resumed stages must match inputs, source and row hashes."""
    import torch
    if isinstance(cfg, (str, Path)):
        cfg = load_transfer_study(cfg)
    validate_transfer_study(cfg)
    cfg = copy.deepcopy(cfg)
    root = Path(output).resolve()
    torch.set_num_threads(cfg.get("threads", 1))
    device = cfg.get("device", "cpu")
    with output_lock(root):
        path = root / "manifest.json"
        if path.exists():
            manifest = json.loads(path.read_text())
            if not resume or manifest["config_hash"] != content_hash(cfg):
                raise ValueError("output already exists or transfer configuration changed")
            _assert_inputs(cfg, manifest)
            _verify_receipts(root, manifest)
        else:
            manifest = {"schema_version": 1, "status": "preflight", "config": cfg,
                        "config_hash": content_hash(cfg), "source_hash": source_hash(),
                        "input_sha256": _inputs(cfg), "stages": {}, "receipts": {}}
            write_json(path, manifest)
        # Only source fitting/selection records are opened; source test labels are unnecessary.
        source_records = load_records(cfg["source_data"], "train") + load_records(cfg["source_data"], "validation")
        source_manifest = json.loads((Path(cfg["source_data"]) / "manifest.json").read_text())
        source_description = _describe(source_records, source_manifest)
        global_wave, global_details = source_global_schedule(source_records)
        targets, descriptions, payloads, checkpoint_provenance = {}, {}, {}, {}
        for target in cfg["targets"]:
            records = _filter(load_records(target["data"], "test"), target)
            target_manifest = json.loads((Path(target["data"]) / "manifest.json").read_text())
            targets[target["name"]] = records
            descriptions[target["name"]] = validate_axes(source_description, _describe(records, target_manifest), target["axes"])
        # Validate ALL checkpoint provenance before starting any inference stage.
        for item in cfg["checkpoints"]:
            key = f'{item["method"]}:{item["seed"]}'
            payload = torch.load(item["path"], map_location="cpu", weights_only=True)
            if payload.get("seed") != item["seed"]:
                raise ValueError("declared checkpoint seed differs from saved training seed")
            checkpoint_provenance[key] = validate_source_reference(payload, source_records)
            payloads[key] = payload
            for records in targets.values():
                validate_transfer_provenance(payload, records, source_records=source_records)
        manifest.update(status="running", transfer_axes=descriptions, global_baseline=global_details,
                        checkpoint_source_association=checkpoint_provenance)
        write_json(path, manifest)
        rows = []
        for target in cfg["targets"]:
            for item in cfg["checkpoints"]:
                key = content_hash({"target": target["name"], "method": item["method"], "seed": item["seed"]})[:20]
                raw_path = root / "rows" / f"{key}.json"
                if key in manifest["stages"]:
                    stage = manifest["stages"][key]
                    if not raw_path.exists() or _sha(raw_path) != stage["raw_sha256"]:
                        raise ValueError("completed transfer raw artifact changed or missing")
                    rows.extend(json.loads(raw_path.read_text()))
                    continue
                _assert_inputs(cfg, manifest)
                attempt = uuid.uuid4().hex
                request = {"attempt": attempt, "stage": key, "target": target["name"],
                           "method": item["method"], "seed": item["seed"], "record_count": len(targets[target["name"]]),
                           "checkpoint_sha256": _sha(item["path"]), "online_simulator_calls": 0}
                _bound_receipt(root, manifest, f"{attempt}.request.json", request)
                started = perf_counter()
                try:
                    if str(device).startswith("cuda"):
                        torch.cuda.synchronize(device)
                    result = evaluate_transfer(item["path"], targets[target["name"]], device=device,
                                               source_records=source_records, source_global_schedule=global_wave)
                    if str(device).startswith("cuda"):
                        torch.cuda.synchronize(device)
                    elapsed = perf_counter() - started
                    _assert_inputs(cfg, manifest)
                    result = [dict(r, target=target["name"], method=item["method"], seed=item["seed"]) for r in result]
                    write_json(raw_path, result)
                    stage = {**request, "elapsed_seconds": elapsed, "raw_sha256": _sha(raw_path),
                             "status": "complete", "timing_scope": "checkpoint load, provenance, graph construction and inference; no latency-only claim"}
                    _bound_receipt(root, manifest, f"{attempt}.result.json", stage)
                    manifest["stages"][key] = stage
                    write_json(path, manifest)
                    rows.extend(result)
                except Exception as error:
                    if not (root / "attempts" / f"{attempt}.result.json").exists():
                        _bound_receipt(root, manifest, f"{attempt}.result.json",
                             {**request, "status": "failed", "elapsed_seconds": perf_counter() - started,
                              "error": f"{type(error).__name__}: {error}"})
                    manifest["status"] = "failed"
                    write_json(path, manifest)
                    raise
        _assert_inputs(cfg, manifest)
        _verify_receipts(root, manifest)
        attempts = [json.loads(p.read_text()) for p in (root / "attempts").glob("*.result.json")]
        requests = list((root / "attempts").glob("*.request.json"))
        costs = {"online_simulator_calls": 0, "incremental_training_calls": 0,
                 "attempt_count": len(requests), "completed_or_failed_receipts": len(attempts),
                 "interrupted_attempts_without_receipt": len(requests) - len(attempts),
                 "known_inference_seconds_including_retries": sum(a["elapsed_seconds"] for a in attempts),
                 "timing_is_lower_bound": len(requests) != len(attempts),
                 "source_train_validation_bank_label_count": sum(len(r["candidate_losses"]) for r in source_records),
                 "target_evaluation_bank_label_count": sum(len(r["candidate_losses"]) for recs in targets.values() for r in recs),
                 "offline_costs_user_declared_not_measured_here": cfg.get("offline_costs", {}),
                 "scope": "pre-existing label counts are not original generation propagation counts; unknown training cost is not zero"}
        report = {"schema_version": 1, "config_hash": manifest["config_hash"], "source_hash": manifest["source_hash"],
                  "input_sha256": manifest["input_sha256"], "transfer_axes": descriptions,
                  "global_baseline": global_details, "results": summarize_transfer_study(rows,
                  bootstrap_resamples=cfg.get("bootstrap_resamples", 10000), seed=cfg.get("bootstrap_seed", 0)), "costs": costs}
        write_json(root / "summary.json", report)
        manifest.update(status="complete", summary_sha256=_sha(root / "summary.json"))
        write_json(path, manifest)
        return report
