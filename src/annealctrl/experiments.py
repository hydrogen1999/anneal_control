"""Config-driven, resumable experiments and validation-only grid tuning.

All commands are local. The manifest freezes configuration and source before any
test outcomes are read. A stage is complete only after its artifacts are saved.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import shutil
from time import perf_counter

from .pipeline import environment, generate_dataset, load_records, write_json
from .telemetry import unknown_config_keys


def content_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def source_hash() -> str:
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _assert_frozen(manifest):
    if source_hash() != manifest["source_hash"]:
        raise RuntimeError("source_drift: source changed during this run; do not attribute mixed artifacts to the frozen revision")


@contextmanager
def output_lock(root: Path):
    """Fail on concurrent/stale locks; never steal a lock from a live job."""
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".experiment.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise RuntimeError(f"Run is locked: {lock}. Check for an active job before manually removing a stale lock.") from error
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump({"pid": os.getpid()}, stream)
        yield
    finally:
        lock.unlink(missing_ok=True)


def load_experiment(path: str | Path) -> dict:
    """Resolve dataset JSON relative to the experiment file, then freeze its bytes."""
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    config = deepcopy(config)
    if isinstance(config.get("dataset"), str):
        config["dataset"] = json.loads((path.parent / config["dataset"]).read_text(encoding="utf-8"))
    validate_experiment(config)
    return config


def validate_experiment(cfg: dict) -> None:
    allowed = {"schema_version", "dataset", "seeds", "methods", "training", "execution", "evaluation", "report"}
    if not isinstance(cfg, dict):
        raise ValueError("experiment configuration must be a mapping")
    # Keys beginning with "_" are free-form annotations, never read. The dataset
    # configs already use that convention; the allowlist stays strict for
    # everything else so a typo like "excution" is still an error.
    unknown = set(unknown_config_keys(cfg, allowed))
    if unknown:
        raise ValueError(f"Unknown experiment keys: {sorted(unknown)}")
    if cfg.get("schema_version", 1) != 1 or not isinstance(cfg.get("dataset"), dict):
        raise ValueError("schema_version=1 and dataset object/path required")
    from .pipeline import _validate_config
    _validate_config(cfg["dataset"])
    seeds = cfg.get("seeds", [0])
    if not seeds or any(type(s) is not int or s < 0 for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique nonnegative integers")
    methods = cfg.get("methods", [])
    names = []
    if not methods:
        raise ValueError("at least one explicit method required")
    for method in methods:
        if set(method) - {"name", "model", "training"}:
            raise ValueError("method accepts name, model and training only")
        name = method.get("name", "")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name):
            raise ValueError("method name must be a short filesystem-safe identifier")
        if not isinstance(method.get("model", {}), dict) or not isinstance(method.get("training", {}), dict):
            raise ValueError("model/training must be objects")
        names.append(name)
    if len(set(names)) != len(names):
        raise ValueError("duplicate method names")
    training_allowed = {"epochs", "learning_rate", "patience", "label_temperature", "policy_weight",
                        "ranking_weight", "response_weight", "batch_size", "accumulation_steps",
                        "weight_decay", "max_grad_norm", "deterministic", "bandwidth", "ranking_tolerance"}
    for train_cfg in [cfg.get("training", {})] + [m.get("training", {}) for m in methods]:
        if set(train_cfg) - training_allowed:
            raise ValueError(f"Unknown training keys: {sorted(set(train_cfg) - training_allowed)}")
    execution = cfg.get("execution", {})
    if set(execution) - {"device", "backend", "threads", "workers"}:
        raise ValueError("Unknown execution key")
    if execution.get("backend", "numpy") not in {"numpy", "cupy"}:
        raise ValueError("backend must be numpy or cupy")
    for key in ("threads", "workers"):
        if type(execution.get(key, 1)) is not int or execution.get(key, 1) < 1:
            raise ValueError(f"{key} must be a positive integer")
    if set(cfg.get("evaluation", {})) - {"direct", "tolerance", "initial_steps", "max_steps", "norm_tolerance", "n_resamples"}:
        raise ValueError("Unknown evaluation key")
    if set(cfg.get("report", {})) - {"plots", "bootstrap_resamples"}:
        raise ValueError("Unknown report key")


def plan_experiment(cfg: dict) -> dict:
    validate_experiment(cfg)
    data = cfg["dataset"]
    paths = data["parents"] * len(data["variants"]) * len(data["chain_strengths"])
    tasks = paths * len(data["runtimes"])
    return {"config_hash": content_hash(cfg), "source_hash": source_hash(),
            "parents": data["parents"], "physical_paths": paths, "tasks": tasks,
            "candidate_outcomes": tasks * data["candidates"],
            "training_runs": len(cfg["methods"]) * len(cfg.get("seeds", [0])),
            "methods": [m["name"] for m in cfg["methods"]], "seeds": cfg.get("seeds", [0]),
            "warning": "Counts are requested workload, not measured performance or acceptance. No jobs launched."}


def _device(execution: dict) -> str:
    import torch
    value = execution.get("device", "cpu")
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if str(value).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA training requested but unavailable; select cpu explicitly or configure CUDA PyTorch")
    torch.set_num_threads(execution.get("threads", 1))
    return value


def _check_manifest(root: Path, cfg: dict, resume: bool) -> dict:
    path = root / "experiment.json"
    if path.exists():
        if not resume:
            raise FileExistsError("Experiment exists; use --resume with identical config/source or a new output")
        manifest = json.loads(path.read_text())
        if manifest["config_hash"] != content_hash(cfg) or manifest["source_hash"] != source_hash():
            raise ValueError("Experiment config/source changed; create a new output to avoid stale results")
        return manifest
    extras = [p for p in root.iterdir() if p.name != ".experiment.lock"]
    if extras:
        raise FileExistsError("Nonempty output without experiment manifest; refusing to overwrite")
    manifest = {"schema_version": 1, "status": "in_progress", "config": cfg,
                "config_hash": content_hash(cfg), "source_hash": source_hash(),
                "environment": environment(), "stages": {}, "runs": {}}
    write_json(path, manifest)
    write_json(root / "resolved_config.json", cfg)
    return manifest


def run_experiment(cfg: dict, output: str | Path, *, stage: str = "all", resume: bool = False) -> dict:
    """Run frozen methods/seeds. Training never reads test records; tuning is separate."""
    validate_experiment(cfg)
    if stage not in {"all", "generate", "train", "evaluate", "report"}:
        raise ValueError("unknown experiment stage")
    root = Path(output).resolve()
    with output_lock(root):
        manifest = _check_manifest(root, cfg, resume)
        manifest_path = root / "experiment.json"
        execution = cfg.get("execution", {})
        data_dir = root / "data"
        selected = {"generate", "train", "evaluate", "report"} if stage == "all" else {stage}
        try:
            if "generate" in selected and not manifest["stages"].get("generate"):
                print("[generate] building/reusing verified dataset", flush=True)
                began = perf_counter()
                data_manifest = generate_dataset(cfg["dataset"], data_dir, resume=resume and (data_dir / "manifest.json").exists(),
                    backend=execution.get("backend", "numpy"), workers=execution.get("workers", 1))
                _assert_frozen(manifest)
                from .reporting import audit_dataset
                write_json(root / "data_audit_train.json", audit_dataset(data_dir, split="train"))
                manifest["stages"]["generate"] = {"seconds": perf_counter() - began,
                    "record_count": data_manifest["record_count"]}
                write_json(manifest_path, manifest)
            if {"train", "evaluate"} & selected:
                if not (data_dir / "manifest.json").exists():
                    raise FileNotFoundError("Run generate stage first")
                device = _device(execution)
                data_manifest = json.loads((data_dir / "manifest.json").read_text())
                data_fingerprint = content_hash({"config_hash": data_manifest["config_hash"],
                    "records": [(r["record_id"], r["fingerprint"]) for r in data_manifest["records"]]})
                for method in cfg["methods"]:
                    for seed in cfg.get("seeds", [0]):
                        run_id = f"{method['name']}/seed_{seed}"
                        folder = root / "models" / method["name"] / f"seed_{seed}"
                        folder.mkdir(parents=True, exist_ok=True)
                        state = manifest["runs"].setdefault(run_id, {"method": method["name"], "seed": seed})
                        checkpoint, latest = folder / "best.pt", folder / "latest.pt"
                        if "train" in selected and not state.get("trained"):
                            _assert_frozen(manifest)
                            from .learning import fit_records
                            training = {**cfg.get("training", {}), **method.get("training", {})}
                            print(f"[train] {run_id}", flush=True)
                            began = perf_counter()
                            restarting = resume and latest.exists()
                            state["training_cost_complete"] = not restarting
                            try:
                                fitted = fit_records(load_records(data_dir, "train"), load_records(data_dir, "validation"),
                                    model_config=method.get("model", {}), seed=seed, device=device,
                                    checkpoint=checkpoint, latest_checkpoint=latest,
                                    resume_from=latest if restarting else None,
                                    dataset_fingerprint=data_fingerprint, **training)
                            finally:
                                state["observed_training_seconds"] = state.get("observed_training_seconds", 0.) + perf_counter() - began
                                write_json(manifest_path, manifest)
                            _assert_frozen(manifest)
                            state.update(trained=True, best_epoch=fitted.best_epoch,
                                         training_seconds=state["observed_training_seconds"],
                                         checkpoint=str(checkpoint.relative_to(root)))
                            write_json(folder / "history.json", {"history": fitted.history, **state})
                            write_json(manifest_path, manifest)
                        if "evaluate" in selected and not state.get("evaluated"):
                            if not state.get("trained") or not checkpoint.exists():
                                raise FileNotFoundError(f"Train {run_id} before evaluating")
                            from .benchmarking import evaluate_checkpoint
                            print(f"[evaluate] {run_id}", flush=True)
                            began = perf_counter()
                            result = evaluate_checkpoint(data_dir, checkpoint, device=device, seed=seed,
                                backend=execution.get("backend", "numpy"), **cfg.get("evaluation", {}))
                            _assert_frozen(manifest)
                            result.update(method=method["name"], training_seed=seed,
                                experiment_config_hash=manifest["config_hash"], training_seconds=state["training_seconds"],
                                training_cost_complete=state["training_cost_complete"])
                            evaluation_path = root / "evaluations" / f"{method['name']}__seed_{seed}.json"
                            write_json(evaluation_path, result)
                            state.update(evaluated=True, evaluation_seconds=perf_counter() - began,
                                         evaluation=str(evaluation_path.relative_to(root)))
                            write_json(manifest_path, manifest)
            if "report" in selected:
                _assert_frozen(manifest)
                from .reporting import build_report
                missing = [f"{m['name']}/seed_{s}" for m in cfg["methods"] for s in cfg.get("seeds", [0])
                           if not manifest["runs"].get(f"{m['name']}/seed_{s}", {}).get("evaluated")]
                if missing:
                    raise ValueError(f"Report requires all predeclared evaluations; missing {missing}")
                build_report(root, **cfg.get("report", {}))
                manifest["stages"]["report"] = True
            manifest["status"] = "complete" if manifest["stages"].get("report") else "in_progress"
            _assert_frozen(manifest)
            manifest.pop("error", None)
            write_json(manifest_path, manifest)
        except Exception as error:
            manifest["status"] = "failed"
            manifest["error"] = {"type": type(error).__name__, "message": str(error)}
            write_json(manifest_path, manifest)
            raise
    return manifest


def expand_tuning(config: dict) -> list[dict]:
    """Finite Cartesian grid over training/model keys, never test scores."""
    if set(config) - {"experiment", "search_space", "max_trials"}:
        raise ValueError("Tuning accepts experiment, search_space, max_trials only")
    base, space = config["experiment"], config["search_space"]
    validate_experiment(base)
    if len(base["methods"]) != 1 or not space:
        raise ValueError("Tuning requires one base method and a nonempty search space")
    for key, values in space.items():
        if not re.fullmatch(r"(model|training)\.[A-Za-z_]+", key) or not isinstance(values, list) or not values:
            raise ValueError("search keys must be model.key or training.key with nonempty lists")
    keys = sorted(space)
    count = 1
    for key in keys:
        count *= len(space[key])
    maximum = config.get("max_trials", 32)
    if type(maximum) is not int or maximum < 1 or count > maximum:
        raise ValueError(f"Requested {count} tuning trials exceeds valid max_trials")
    trials = []
    for values in itertools.product(*(space[k] for k in keys)):
        trial = deepcopy(base)
        for key, value in zip(keys, values):
            group, name = key.split(".")
            trial["methods"][0].setdefault(group, {})[name] = value
        validate_experiment(trial)
        trials.append(trial)
    return trials


def tune_validation(config: dict, output: str | Path, *, resume: bool = False) -> dict:
    """Training-only grid; test generation is allowed, test loading/evaluation is not."""
    trials = expand_tuning(config)
    root = Path(output).resolve()
    with output_lock(root):
        saved = root / "tuning.json"
        fingerprint = content_hash(config)
        if saved.exists():
            if not resume or json.loads(saved.read_text())["config_hash"] != fingerprint:
                raise ValueError("Tuning exists/config mismatch; use identical --resume or new output")
        elif any(p.name != ".experiment.lock" for p in root.iterdir()):
            raise FileExistsError("Tuning output nonempty")
        summary = {"config_hash": fingerprint, "selection": "validation_only_equal_seed_mean_bank_regret", "trials": []}
        write_json(saved, summary)
        # Generate once. Trial datasets are private copies, not mutable symlinks.
        shared = root / "shared_generation"
        shared_manifest = run_experiment(trials[0], shared, stage="generate",
            resume=resume and (shared / "experiment.json").exists())
        for index, trial in enumerate(trials):
            folder = root / f"trial_{index:03d}"
            with output_lock(folder):
                trial_manifest = _check_manifest(folder, trial, resume=resume)
                if not trial_manifest["stages"].get("generate"):
                    # Interrupted copies can be reused only after full loader integrity checks.
                    shutil.copytree(shared / "data", folder / "data", dirs_exist_ok=True)
                    load_records(folder / "data", "train")
                    trial_manifest["stages"]["generate"] = {**shared_manifest["stages"]["generate"],
                        "reused_from": "shared_generation/data", "additional_generation_seconds": 0.0}
                    write_json(folder / "experiment.json", trial_manifest)
            run_experiment(trial, folder, stage="train", resume=True)
            regrets = []
            for seed in trial.get("seeds", [0]):
                history = json.loads((folder / "models" / trial["methods"][0]["name"] / f"seed_{seed}" / "history.json").read_text())
                regrets.append(min(h["validation_bank_regret"] for h in history["history"]))
            summary["trials"].append({"trial": index, "mean_validation_regret": sum(regrets) / len(regrets), "seed_regrets": regrets})
            write_json(saved, summary)
        best = min(summary["trials"], key=lambda x: (x["mean_validation_regret"], x["trial"]))["trial"]
        summary["selected_trial"] = best
        summary["test_evaluated"] = False
        write_json(root / "selected_experiment.json", trials[best])
        write_json(saved, summary)
        return summary
