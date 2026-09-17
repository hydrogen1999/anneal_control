"""Frozen, matched one-round acquisition study with untouched evaluation banks.

The policy arm is simulator outcome relabelling, not oracle-action imitation;
DAgger's no-regret theorem is not a theorem for this procedure.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .experiments import _assert_frozen, _device, content_hash, output_lock, source_hash, validate_experiment
from .pipeline import environment, generate_dataset, load_records, write_json

ARMS = ("control", "bankext", "decoder_random", "policy")


def load_study(path):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text())
    if isinstance(cfg.get("dataset"), str):
        cfg["dataset"] = json.loads((path.parent / cfg["dataset"]).read_text())
    validate_study(cfg)
    return cfg


def validate_study(cfg):
    allowed = {"schema_version", "dataset", "model", "training", "execution", "evaluation",
               "seeds", "acquisition", "latency", "bootstrap_resamples"}
    unknown = {k for k in cfg if not k.startswith("_")} - allowed
    if unknown:
        raise ValueError(f"Unknown acquisition-study keys: {sorted(unknown)}")
    validate_experiment({"schema_version": cfg.get("schema_version", 1), "dataset": cfg.get("dataset"),
                         "seeds": cfg.get("seeds", [0, 1, 2]),
                         "methods": [{"name": "study", "model": cfg.get("model", {})}],
                         "training": cfg.get("training", {}), "execution": cfg.get("execution", {}),
                         "evaluation": cfg.get("evaluation", {})})
    acquisition = cfg.get("acquisition", {})
    if set(acquisition) - {"n_extra", "random_seed", "logit_std"}:
        raise ValueError("Unknown acquisition key")
    count = acquisition.get("n_extra", cfg.get("model", {}).get("proposals", 3))
    if type(count) is not int or count < 1 or count != cfg.get("model", {}).get("proposals", 3):
        raise ValueError("n_extra must equal the positive policy proposal count for matched label budgets")
    if "n_segments" in cfg.get("model", {}) or cfg.get("model", {}).get("schedule_points", 9) != 9:
        raise ValueError("shared dataset bank currently requires model.schedule_points=9")
    if type(acquisition.get("random_seed", 1701)) is not int or acquisition.get("random_seed", 1701) < 0:
        raise ValueError("random_seed must be a nonnegative integer")
    if not np.isfinite(acquisition.get("logit_std", 1.)) or acquisition.get("logit_std", 1.) <= 0:
        raise ValueError("logit_std must be finite and positive")
    if cfg.get("evaluation", {}).get("direct", True) is not True:
        raise ValueError("acquisition study requires direct evaluation")
    if cfg.get("evaluation", {}).get("norm_tolerance", 1e-9) != 1e-9:
        raise ValueError("matched acquisition currently fixes norm_tolerance=1e-9")
    latency = cfg.get("latency", {})
    if set(latency) - {"warmup", "repeats"}:
        raise ValueError("Unknown latency key")
    for name, value in (("warmup", latency.get("warmup", 2)), ("repeats", latency.get("repeats", 5)),
                        ("bootstrap_resamples", cfg.get("bootstrap_resamples", 2000))):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")


def plan_study(cfg):
    validate_study(cfg)
    seeds = cfg.get("seeds", [0, 1, 2])
    return {"arms": list(ARMS), "seeds": seeds, "training_runs": 5 * len(seeds),
            "added_labels_per_training_record_per_acquisition_arm": cfg.get("acquisition", {}).get(
                "n_extra", cfg.get("model", {}).get("proposals", 3)),
            "retraining": "from scratch, identical initialization seed and recipe for all arms",
            "selection": "original validation bank regret",
            "primary_contrasts": ["policy minus bankext", "policy minus decoder_random"],
            "scope": "one matched round; plan is not experimental evidence"}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _verify_artifact(root, state, key):
    path = root / state[key]
    if not path.exists() or _sha(path) != state[key + "_sha256"]:
        raise ValueError(f"missing or changed frozen artifact: {path}")
    return path


def _fit(root, state, records, validation, cfg, seed, device, resume):
    from .learning import fit_records
    folder = root / state["folder"]
    folder.mkdir(parents=True, exist_ok=True)
    best, latest = folder / "best.pt", folder / "latest.pt"
    restarting = resume and latest.exists()
    began = perf_counter()
    fitted = fit_records(records, validation, model_config=cfg.get("model", {}), seed=seed,
                         device=device, checkpoint=best, latest_checkpoint=latest,
                         resume_from=latest if restarting else None, **cfg.get("training", {}))
    state.update(trained=True, checkpoint=str(best.relative_to(root)), checkpoint_sha256=_sha(best),
                 training_seconds=perf_counter() - began, training_cost_complete=not restarting,
                 best_epoch=fitted.best_epoch, last_epoch=fitted.last_epoch)
    write_json(folder / "history.json", {"history": fitted.history, **state})


def _acquire(data, baseline, arm, cfg, data_config, device):
    from .dagger import collect_bank_extension, collect_decoder_random, collect_labelled_proposals
    acquisition = cfg.get("acquisition", {})
    count = acquisition.get("n_extra", cfg.get("model", {}).get("proposals", 3))
    scoring = {k: v for k, v in cfg.get("evaluation", {}).items()
               if k in {"tolerance", "initial_steps", "max_steps"}}
    common = dict(split="train", backend=cfg.get("execution", {}).get("backend", "numpy"),
                  max_ds_dtau=cfg.get("model", {}).get("max_ds_dtau", 4.), **scoring)
    if arm == "policy":
        return collect_labelled_proposals(data, baseline, device=device, **common)
    if arm == "bankext":
        return collect_bank_extension(data, n_extra=count, bank_seed=data_config["seed"],
                                      bank_size=data_config["candidates"], **common)
    if arm == "decoder_random":
        return collect_decoder_random(data, n_extra=count, seed=acquisition.get("random_seed", 1701),
                                      logit_std=acquisition.get("logit_std", 1.), **common)
    raise ValueError("control has no acquisition")


def _rows(evaluation, arm, seed):
    result = []
    for row in evaluation["records"]:
        for mode, field in (("bank", "selected_loss"), ("linear", "linear_loss"), ("global", "global_loss")):
            result.append({"method": arm, "seed": seed, "mode": mode, "parent_id": row["parent_id"],
                           "record_id": row["record_id"], "loss": row[field]})
    for row in evaluation["direct_policy"]["records"]:
        result.append({"method": arm, "seed": seed, "mode": "direct", "parent_id": row["parent_id"],
                       "record_id": row["record_id"], "loss": row["loss"]})
    return result


def summarize_study(root, manifest):
    from .contrasts import paired_parent_seed_contrast
    root = Path(root)
    rows, costs, diagnostics = [], {}, {}
    for seed in manifest["config"].get("seeds", [0, 1, 2]):
        for arm in ARMS:
            name = f"{arm}/seed_{seed}"
            state = manifest["runs"][name]
            result = json.loads(_verify_artifact(root, state, "evaluation").read_text())
            rows.extend(_rows(result["evaluation"], arm, seed))
            base = manifest["runs"][f"baseline/seed_{seed}"]
            costs[name] = {"training_seconds": state["training_seconds"],
                           "shared_frozen_baseline_training_seconds": base["training_seconds"],
                           "shared_frozen_baseline_training_cost_complete": base["training_cost_complete"],
                           "cost_note": "baseline training is shared within a seed; charge it once, not once per arm",
                           "training_cost_complete": state["training_cost_complete"],
                           "acquisition": state.get("acquisition_cost", {"objective_calls": 0}),
                           "offline_evaluation_diagnostics": result["costs"],
                           "deployment_latency": result["deployment_latency"]}
            diagnostics[name] = {k: result[k] for k in ("proposal_diagnostics", "frozen_proposal_diagnostics")}
    count = manifest["config"].get("bootstrap_resamples", 2000)
    contrasts = {}
    for comparator in ("control", "bankext", "decoder_random"):
        contrasts[f"policy_minus_{comparator}"] = paired_parent_seed_contrast(
            rows, comparator, "policy", mode="direct", bootstrap_resamples=count)
    deployment = {}
    for arm in ARMS:
        selected = [dict(row, method=row["mode"], mode="deployment") for row in rows
                    if row["method"] == arm and row["mode"] in {"bank", "direct", "global"}]
        deployment[arm] = {f"direct_minus_{ref}": paired_parent_seed_contrast(
            selected, ref, "direct", mode="deployment", bootstrap_resamples=count) for ref in ("bank", "global")}
    means = {}
    for arm in ARMS:
        means[arm] = {}
        for mode in ("bank", "direct", "global", "linear"):
            cells = {}
            for row in rows:
                if row["method"] == arm and row["mode"] == mode:
                    cells.setdefault((row["parent_id"], row["seed"]), []).append(row["loss"])
            means[arm][mode] = float(np.mean([np.mean(v) for v in cells.values()]))
    result = {"schema_version": 1, "config_hash": manifest["config_hash"], "source_hash": manifest["source_hash"],
              "means": means, "acquisition_contrasts": contrasts, "deployment_contrasts": deployment,
              "costs": costs, "diagnostics": diagnostics, "records": rows,
              "inference": "crossed parent/seed percentile intervals are descriptive and unadjusted; no automatic discovery claim",
              "scope": "closed-system one-round experiment; shared-bank random extension and decoder-random are distinct controls"}
    write_json(root / "summary.json", result)
    lines = ["# Controlled acquisition study", "", "Lower loss is better. Parent/seed cells receive equal weight.", "",
             "| Arm | Bank | Direct | Global | Linear |", "|---|---:|---:|---:|---:|"]
    for arm, values in means.items():
        lines.append("| " + arm + " | " + " | ".join(f"{values[k]:.6f}" for k in ("bank", "direct", "global", "linear")) + " |")
    lines += ["", "Full paired contrasts, raw rows, diagnostics and costs: `summary.json`.", "",
              "These results do not establish architectural superiority, hardware performance, or conference readiness.",
              "Primary attribution requires policy acquisition to improve over BOTH bank extension and decoder-random across independent parents and seeds.",
              "All checkpoints use the original validation bank; test labels enter only the frozen evaluation stage.",
              "Both bank and direct deployment require zero online simulator calls. Offline diagnostic scoring is separate."]
    (root / "RESULTS.md").write_text("\n".join(lines) + "\n")
    return result


def run_study(cfg, output, *, data_dir=None, stage="all", resume=False):
    """Finish all acquisition/training arms before opening any test outcome."""
    validate_study(cfg)
    if stage not in {"all", "train", "evaluate", "report"}:
        raise ValueError("unknown study stage")
    root = Path(output).resolve()
    data = Path(data_dir).resolve() if data_dir is not None else root / "data"
    with output_lock(root):
        path = root / "study.json"
        if path.exists():
            if not resume:
                raise FileExistsError("study exists; use --resume with frozen config/source/data")
            manifest = json.loads(path.read_text())
            if (manifest["config_hash"] != content_hash(cfg) or manifest["source_hash"] != source_hash()
                    or manifest["data_dir"] != str(data)):
                raise ValueError("study config/source/data path changed; choose a new output")
            for state in manifest["runs"].values():
                for key in ("checkpoint", "acquisition", "evaluation"):
                    if key in state:
                        _verify_artifact(root, state, key)
        else:
            if any(p.name != ".experiment.lock" for p in root.iterdir()):
                raise FileExistsError("nonempty output without study manifest")
            manifest = {"schema_version": 1, "config": cfg, "config_hash": content_hash(cfg),
                        "source_hash": source_hash(), "environment": environment(), "data_dir": str(data),
                        "status": "in_progress", "runs": {}, "plan": plan_study(cfg)}
            write_json(path, manifest)
        try:
            execution = cfg.get("execution", {})
            if data_dir is None and not manifest.get("data_manifest_sha256"):
                generate_dataset(cfg["dataset"], data, resume=resume and (data / "manifest.json").exists(),
                                 backend=execution.get("backend", "numpy"), workers=execution.get("workers", 1))
            data_hash = _sha(data / "manifest.json")
            if manifest.get("data_manifest_sha256", data_hash) != data_hash:
                raise ValueError("frozen dataset manifest changed")
            manifest["data_manifest_sha256"] = data_hash
            write_json(path, manifest)
            data_config = json.loads((data / "manifest.json").read_text())["config"]
            scientific_config = lambda value: {k: v for k, v in value.items()
                                                if k != "backend" and not k.startswith("_")}
            if scientific_config(data_config) != scientific_config(cfg["dataset"]):
                raise ValueError("existing dataset config differs from frozen study dataset config")
            device = _device(execution)
            training, validation = load_records(data, "train"), load_records(data, "validation")
            from .learning import records_content_digest
            for key, records in (("training_digest", training), ("validation_digest", validation)):
                digest = records_content_digest(records)
                if manifest.get(key, digest) != digest:
                    raise ValueError(f"frozen {key} changed")
                manifest[key] = digest
            write_json(path, manifest)
            from .dagger import augment_records
            if stage in {"all", "train"}:
                for seed in cfg.get("seeds", [0, 1, 2]):
                    base = manifest["runs"].setdefault(f"baseline/seed_{seed}", {"folder": f"models/baseline/seed_{seed}"})
                    if not base.get("trained"):
                        _assert_frozen(manifest)
                        print(f"[baseline] seed={seed}", flush=True)
                        _fit(root, base, training, validation, cfg, seed, device, resume)
                        write_json(path, manifest)
                    baseline = _verify_artifact(root, base, "checkpoint")
                    for arm in ARMS:
                        name = f"{arm}/seed_{seed}"
                        state = manifest["runs"].setdefault(name, {"folder": f"models/{name}"})
                        if state.get("trained"):
                            continue
                        _assert_frozen(manifest)
                        records = training
                        if arm != "control":
                            if "acquisition" not in state:
                                print(f"[acquire] {name}", flush=True)
                                collected = _acquire(data, baseline, arm, cfg, data_config, device)
                                artifact = root / "acquisitions" / f"{arm}_seed_{seed}.json"
                                write_json(artifact, collected)
                                state.update(acquisition=str(artifact.relative_to(root)), acquisition_sha256=_sha(artifact),
                                             acquisition_cost={k: v for k, v in collected.items() if k != "records"})
                                write_json(path, manifest)
                            collected = json.loads(_verify_artifact(root, state, "acquisition").read_text())
                            records = augment_records(training, collected)
                        print(f"[retrain] {name}", flush=True)
                        _fit(root, state, records, validation, cfg, seed, device, resume)
                        write_json(path, manifest)
                manifest["training_complete"] = True
                _assert_frozen(manifest)
                write_json(path, manifest)
            if stage in {"all", "evaluate", "report"} and not manifest.get("training_complete"):
                raise ValueError("complete all frozen training arms before evaluation/report")
            if stage in {"all", "evaluate"}:
                from .acquisition_evaluation import evaluate_acquisition_arm
                for seed in cfg.get("seeds", [0, 1, 2]):
                    baseline = _verify_artifact(root, manifest["runs"][f"baseline/seed_{seed}"], "checkpoint")
                    for arm in ARMS:
                        state = manifest["runs"][f"{arm}/seed_{seed}"]
                        if "evaluation" in state:
                            continue
                        _assert_frozen(manifest)
                        records = training if arm == "control" else augment_records(training,
                            json.loads(_verify_artifact(root, state, "acquisition").read_text()))
                        evaluation = {k: v for k, v in cfg.get("evaluation", {}).items()
                                      if k not in {"direct", "norm_tolerance"}}
                        latency = cfg.get("latency", {})
                        print(f"[evaluate] {arm}/seed_{seed}", flush=True)
                        result = evaluate_acquisition_arm(data, _verify_artifact(root, state, "checkpoint"),
                            training_records=records, frozen_checkpoint=baseline, seed=seed, device=device,
                            backend=execution.get("backend", "numpy"), **evaluation,
                            max_ds_dtau=cfg.get("model", {}).get("max_ds_dtau", 4.),
                            latency_warmup=latency.get("warmup", 2), latency_repeats=latency.get("repeats", 5))
                        _assert_frozen(manifest)
                        artifact = root / "evaluations" / f"{arm}_seed_{seed}.json"
                        write_json(artifact, result)
                        state.update(evaluation=str(artifact.relative_to(root)), evaluation_sha256=_sha(artifact))
                        write_json(path, manifest)
            if stage in {"all", "report"}:
                summarize_study(root, manifest)
                manifest["status"] = "complete"
            elif manifest["status"] != "complete":
                manifest["status"] = "in_progress"
            manifest.pop("error", None)
            _assert_frozen(manifest)
            write_json(path, manifest)
        except Exception as error:
            manifest.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
            if getattr(error, "audit", None) is not None:
                write_json(root / "failed_acquisition.json", error.audit)
                manifest["error"]["acquisition_audit"] = "failed_acquisition.json"
            write_json(path, manifest)
            raise
    return manifest
