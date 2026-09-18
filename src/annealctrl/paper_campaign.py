"""Frozen local campaign orchestration for the paper's decisive experiments.

An executable plan is not evidence of a successful scientific hypothesis. Each
step keeps its own scientific receipts; this module records order, dependencies,
source identity and immutable result artifacts. It never submits hardware jobs.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from time import perf_counter

from .experiments import content_hash, output_lock, source_hash
from .pipeline import environment, write_json

KINDS = {"generate", "acquisition", "transfer", "budget", "literature", "noise"}


def _resolve(value, root):
    if isinstance(value, dict):
        return {key: _resolve(item, root) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve(item, root) for item in value]
    if isinstance(value, str) and "${RUN}" in value:
        return value.replace("${RUN}", str(root))
    return value


def _input_paths(config, kind, base):
    """Match standalone study loaders when a step references another JSON file."""
    config = deepcopy(config)
    fields = {"budget": ("source_data", "target_data", "checkpoint"),
              "noise": ("source_data", "data", "checkpoint"),
              "transfer": ("source_data",), "literature": ("data",)}.get(kind, ())
    for key in fields:
        if key in config:
            config[key] = str((base / config[key]).resolve())
    if kind == "transfer":
        for item in config.get("checkpoints", []):
            item["path"] = str((base / item["path"]).resolve())
        for item in config.get("targets", []):
            item["data"] = str((base / item["data"]).resolve())
    if kind == "budget":
        cost = config.get("offline_cost", {})
        if "acquisition_study" in cost:
            cost["acquisition_study"] = str((base / cost["acquisition_study"]).resolve())
        for component in cost.get("components", {}).values():
            component["receipt"] = str((base / component["receipt"]).resolve())
    return config


def load_campaign(path, output):
    path, root = Path(path).resolve(), Path(output).resolve()
    config = json.loads(path.read_text())
    unknown = {k for k in config if not k.startswith("_")} - {"schema_version", "name", "scope", "steps"}
    if unknown or config.get("schema_version") != 1:
        raise ValueError(f"Invalid campaign schema/keys: {sorted(unknown)}")
    if not isinstance(config.get("steps"), list) or not config["steps"]:
        raise ValueError("campaign requires nonempty steps")
    seen, outputs = set(), set()
    steps = []
    for spec in config["steps"]:
        if not isinstance(spec, dict) or set(spec) - {"id", "kind", "config", "output", "depends", "backend", "workers"}:
            raise ValueError("Invalid campaign step keys")
        identifier = spec.get("id", "")
        if not re.fullmatch(r"[a-z][a-z0-9_]*", identifier) or identifier in seen:
            raise ValueError("step ids must be unique simple identifiers")
        if spec.get("kind") not in KINDS:
            raise ValueError(f"Unknown step kind: {spec.get('kind')}")
        dependencies = spec.get("depends", [])
        if not isinstance(dependencies, list) or any(d not in seen for d in dependencies):
            raise ValueError("dependencies must name earlier steps; cycles and forward references are refused")
        destination = (root / spec.get("output", identifier)).resolve()
        if destination == root or root not in destination.parents:
            raise ValueError("step output must be below the campaign directory")
        if any(destination == prior or prior in destination.parents or destination in prior.parents for prior in outputs):
            raise ValueError("step outputs cannot overlap")
        config_value = spec.get("config")
        if isinstance(config_value, str):
            source = (path.parent / config_value).resolve()
            if spec["kind"] == "acquisition":
                from .acquisition_study import load_study
                step_config = load_study(source)
            else:
                step_config = json.loads(source.read_text())
        elif isinstance(config_value, dict):
            step_config = deepcopy(config_value)
        else:
            raise ValueError("each step requires a JSON config object or filename")
        step_config = _resolve(step_config, root)
        if isinstance(config_value, str):
            step_config = _input_paths(step_config, spec["kind"], source.parent)
        steps.append({**spec, "config": step_config, "output": str(destination)})
        seen.add(identifier)
        outputs.add(destination)
    return {**config, "steps": steps, "output": str(root)}


def plan_campaign(path, output):
    cfg = load_campaign(path, output)
    return {"name": cfg.get("name"), "scope": cfg.get("scope"),
            "steps": [{"id": s["id"], "kind": s["kind"], "output": s["output"],
                       "depends": s.get("depends", []), "config_hash": content_hash(s["config"])}
                      for s in cfg["steps"]],
            "scientific_verdict": "unmeasured", "executes_training": False,
            "note": "Dry run resolves configuration and order; it does not establish GPU capacity or performance."}


def _artifacts(folder):
    result = {}
    for path in sorted(Path(folder).rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Artifact symlink cannot certify frozen content: {path}")
        if path.is_file() and not path.name.endswith(".lock"):
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            result[str(path.relative_to(folder))] = digest.hexdigest()
    if not result:
        raise ValueError("completed step produced no artifacts")
    return result


def _dispatch(step, *, resume):
    cfg, output = step["config"], step["output"]
    if step["kind"] == "generate":
        from .pipeline import generate_dataset
        return generate_dataset(cfg, output, resume=resume,
                                backend=step.get("backend", "numpy"), workers=step.get("workers", 1))
    if step["kind"] == "acquisition":
        from .acquisition_study import run_study
        return run_study(cfg, output, resume=resume)
    if step["kind"] == "transfer":
        from .transfer_study import run_transfer_study
        return run_transfer_study(cfg, output, resume=resume)
    if step["kind"] == "budget":
        from .budget_study import run_budget_study
        return run_budget_study(cfg, output, resume=resume)
    if step["kind"] == "noise":
        from .noise_study import run_noise_study
        return run_noise_study(cfg, output, resume=resume)
    from .literature_baselines import run_literature_study
    return run_literature_study(cfg, output, resume=resume)


def run_campaign(path, output, *, resume=False, through=None):
    cfg = load_campaign(path, output)
    root = Path(cfg["output"])
    if through is not None and through not in {s["id"] for s in cfg["steps"]}:
        raise ValueError("--through must name a declared step")
    frozen_hash = content_hash(cfg)
    with output_lock(root):
        manifest_path = root / "campaign.json"
        if manifest_path.exists():
            if not resume:
                raise FileExistsError("campaign exists; use --resume to verify and continue")
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("config_hash") != frozen_hash or manifest.get("source_hash") != source_hash():
                raise ValueError("campaign config/source changed; use a new output directory")
        else:
            manifest = {"schema_version": 1, "config": cfg, "config_hash": frozen_hash,
                        "source_hash": source_hash(), "environment": environment(),
                        "status": "planned", "steps": {}, "scientific_verdict": "not_assigned"}
            write_json(manifest_path, manifest)
        for step in cfg["steps"]:
            if source_hash() != manifest["source_hash"]:
                raise RuntimeError("source_drift: campaign source changed during execution")
            state = manifest["steps"].setdefault(step["id"], {"status": "pending", "attempts": []})
            if state["status"] == "complete":
                if state["artifacts"] != _artifacts(step["output"]):
                    raise ValueError(f"completed artifacts changed: {step['id']}")
            else:
                if any(manifest["steps"].get(dep, {}).get("status") != "complete"
                       for dep in step.get("depends", [])):
                    raise ValueError("dependency is incomplete")
                # Each sub-study owns its query ledger. An interrupted outer
                # attempt cannot certify complete wall time or scientific work.
                for old in state["attempts"]:
                    if old["status"] == "running":
                        old.update(status="interrupted", wall_seconds_complete=False)
                attempt = {"status": "running", "number": len(state["attempts"]) + 1}
                state["attempts"].append(attempt)
                state["status"] = manifest["status"] = "running"
                write_json(manifest_path, manifest)
                started = perf_counter()
                try:
                    _dispatch(step, resume=resume and attempt["number"] > 1)
                    if source_hash() != manifest["source_hash"]:
                        raise RuntimeError("source_drift: source changed during step")
                    state["artifacts"] = _artifacts(step["output"])
                    state["status"] = attempt["status"] = "complete"
                    attempt["wall_seconds_complete"] = True
                except BaseException as error:
                    state["status"] = attempt["status"] = manifest["status"] = "failed"
                    attempt["error"] = {"type": type(error).__name__, "message": str(error)}
                    raise
                finally:
                    attempt["observed_wall_seconds"] = perf_counter() - started
                    write_json(manifest_path, manifest)
            if step["id"] == through:
                break
        manifest["status"] = ("complete" if all(manifest["steps"].get(s["id"], {}).get("status") == "complete"
                                                  for s in cfg["steps"]) else "partial")
        manifest["observed_wall_seconds"] = sum(a.get("observed_wall_seconds", 0.)
                                                for s in manifest["steps"].values() for a in s["attempts"])
        manifest["wall_seconds_complete"] = all(a.get("wall_seconds_complete", False)
                                                 for s in manifest["steps"].values() for a in s["attempts"])
        write_json(manifest_path, manifest)
        return manifest
