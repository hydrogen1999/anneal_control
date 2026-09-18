"""Frozen, matched one-round acquisition study with untouched evaluation banks.

The policy arm is simulator outcome relabelling, not oracle-action imitation;
DAgger's no-regret theorem is not a theorem for this procedure.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from .experiments import _assert_frozen, _device, content_hash, output_lock, source_hash, validate_experiment
from .pipeline import environment, generate_dataset, jsonable, load_records, write_json

ARMS = ("control", "bankext", "decoder_random", "policy")
MECHANISM_SCOPES = ("critic", "policy", "heads")


def mechanism_arms(cfg):
    mechanism = cfg.get("mechanism", {})
    if not mechanism.get("enabled", False):
        return ()
    return tuple(f"mechanism_{scope}_{labels}" for scope in mechanism.get("scopes", MECHANISM_SCOPES)
                 for labels in ("original", "acquired"))


def study_arms(cfg):
    return ARMS + mechanism_arms(cfg)


def _has_acquisition(arm):
    return arm in ARMS[1:] or arm.endswith("_acquired")


def load_study(path):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text())
    if isinstance(cfg.get("dataset"), str):
        cfg["dataset"] = json.loads((path.parent / cfg["dataset"]).read_text())
    validate_study(cfg)
    return cfg


def validate_study(cfg):
    allowed = {"schema_version", "dataset", "model", "training", "execution", "evaluation",
               "seeds", "acquisition", "latency", "bootstrap_resamples", "mechanism"}
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
    mechanism = cfg.get("mechanism", {})
    if set(mechanism) - {"enabled", "epochs", "scopes"}:
        raise ValueError("Unknown mechanism key")
    if type(mechanism.get("enabled", False)) is not bool:
        raise ValueError("mechanism.enabled must be boolean")
    if type(mechanism.get("epochs", 20)) is not int or mechanism.get("epochs", 20) < 1:
        raise ValueError("mechanism.epochs must be a positive predeclared integer")
    scopes = mechanism.get("scopes", list(MECHANISM_SCOPES))
    if (not isinstance(scopes, (list, tuple)) or not scopes or
            any(s not in MECHANISM_SCOPES for s in scopes) or len(set(scopes)) != len(scopes)):
        raise ValueError("mechanism.scopes must contain unique critic, policy, and/or heads scopes")
    if mechanism.get("enabled") and any(s in {"policy", "heads"} for s in scopes):
        if cfg.get("training", {}).get("policy_weight", 0.2) <= 0:
            raise ValueError("Policy mechanism requires positive training.policy_weight")
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
    arms = study_arms(cfg)
    return {"arms": list(arms), "primary_arms": list(ARMS), "mechanism_arms": list(mechanism_arms(cfg)),
            "seeds": seeds, "training_runs": (1 + len(arms)) * len(seeds),
            "added_labels_per_training_record_per_acquisition_arm": cfg.get("acquisition", {}).get(
                "n_extra", cfg.get("model", {}).get("proposals", 3)),
            "retraining": "primary arms: from scratch, identical initialization seed and recipe",
            "selection": "primary arms: original validation bank regret; mechanism arms: fixed final epoch",
            "primary_contrasts": ["policy minus bankext", "policy minus decoder_random"],
            "mechanism_design": {"initialization": "same frozen baseline weights and normalizer",
                                 "fixed_epochs": cfg.get("mechanism", {}).get("epochs", 20),
                                 "paired_control": "same head scope trained on original labels",
                                 "label_source": "one shared frozen-baseline policy acquisition",
                                 "shared_modules": "encoder, attention, response head all frozen",
                                 "targets": "soft targets from true fixed simulator outcomes, never critic pseudo-labels"},
            "scope": "one matched round; plan is not experimental evidence"}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _verify_artifact(root, state, key):
    path = root / state[key]
    if not path.exists() or _sha(path) != state[key + "_sha256"]:
        raise ValueError(f"missing or changed frozen artifact: {path}")
    return path


def _write_once_json(path, value):
    """Create an immutable attempt receipt; never replace an earlier attempt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, default=jsonable, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _attempt_costs(root, state):
    """Sum known work, including work discarded by failed attempts.

    A failed scorer does not expose its internal step count. Those sums are
    lower bounds, even after a later retry succeeds. An interrupted attempt can
    also have unknown objective calls; its known contribution is zero.
    """
    keys = ("objective_calls", "successful_objective_calls", "propagation_calls",
            "total_integrator_steps", "convergence_attempts", "offline_scoring_seconds", "wall_seconds")
    totals = {key: 0 for key in keys}
    totals.update(attempt_count=0, failed_attempts=0, failed_attempt_objective_calls=0,
                  failed_objective_calls=0, cost_counts_complete=True, objective_calls_complete=True,
                  scope="cumulative known work over all acquisition attempts, including discarded failed attempts")
    for attempt in state.get("acquisition_attempts", []):
        _verify_artifact(root, attempt, "request")
        audit = json.loads(_verify_artifact(root, attempt, "audit").read_text())
        totals["attempt_count"] += 1
        for key in keys:
            totals[key] += audit.get(key, 0)
        if attempt["status"] != "complete":
            totals["failed_attempts"] += 1
            totals["failed_attempt_objective_calls"] += audit.get("objective_calls", 0)
            totals["failed_objective_calls"] += max(0, audit.get("objective_calls", 0)
                                                       - audit.get("successful_objective_calls", 0))
        totals["cost_counts_complete"] &= bool(audit.get("cost_counts_complete", False))
        totals["objective_calls_complete"] &= bool(audit.get("objective_calls_complete",
                                                              "objective_calls" in audit))
    totals["counts_are_lower_bounds"] = not totals["cost_counts_complete"]
    return totals


def _finish_attempt(root, attempt, audit):
    path = root / attempt["request"].replace(".request.json", ".audit.json")
    _write_once_json(path, audit)
    attempt.update(status=audit["status"], audit=str(path.relative_to(root)), audit_sha256=_sha(path))


def _acquire_recorded(root, manifest, state, *, data, baseline, arm, seed, cfg, data_config, device):
    attempts = state.setdefault("acquisition_attempts", [])
    index = len(attempts) + 1
    request = root / "acquisition_attempts" / arm / f"seed_{seed}" / f"attempt_{index:04d}.request.json"
    _write_once_json(request, {"arm": arm, "seed": seed, "attempt": index,
                               "config_hash": manifest["config_hash"], "source_hash": manifest["source_hash"],
                               "baseline_sha256": _sha(baseline)})
    attempt = {"index": index, "status": "started", "request": str(request.relative_to(root)),
               "request_sha256": _sha(request)}
    attempts.append(attempt)
    write_json(root / "study.json", manifest)
    try:
        collected = _acquire(data, baseline, arm, cfg, data_config, device)
    except Exception as error:
        audit = getattr(error, "audit", None)
        if audit is None:
            audit = {"status": "failed", "cost_counts_complete": False, "objective_calls_complete": False,
                     "error": {"type": type(error).__name__, "message": str(error)},
                     "scope": "No scorer ledger was returned; work before the exception is unknown."}
        _finish_attempt(root, attempt, audit)
        state["acquisition_cost"] = _attempt_costs(root, state)
        error.acquisition_audit = attempt["audit"]
        write_json(root / "study.json", manifest)
        raise
    _finish_attempt(root, attempt, collected)
    state.update(acquisition=attempt["audit"], acquisition_sha256=attempt["audit_sha256"],
                 acquisition_cost=_attempt_costs(root, state))
    write_json(root / "study.json", manifest)
    return collected


def _fit(root, state, records, validation, cfg, seed, device, resume, *, initialize_from=None, scope="all"):
    from .learning import fit_records
    folder = root / state["folder"]
    folder.mkdir(parents=True, exist_ok=True)
    best, latest = folder / "best.pt", folder / "latest.pt"
    restarting = resume and latest.exists()
    began = perf_counter()
    settings = dict(cfg.get("training", {}))
    if scope != "all":
        settings.update(epochs=cfg.get("mechanism", {}).get("epochs", 20), response_weight=0.,
                        initialize_from=initialize_from, trainable_scope=scope, selection_mode="fixed_epochs")
    fitted = fit_records(records, validation, model_config=cfg.get("model", {}), seed=seed,
                         device=device, checkpoint=best, latest_checkpoint=latest,
                         resume_from=latest if restarting else None, **settings)
    state.update(trained=True, checkpoint=str(best.relative_to(root)), checkpoint_sha256=_sha(best),
                 training_seconds=perf_counter() - began, training_cost_complete=not restarting,
                 best_epoch=fitted.best_epoch, last_epoch=fitted.last_epoch)
    if scope != "all":
        state.update(initialization_checkpoint_sha256=_sha(initialize_from), trainable_scope=scope,
                     selection_mode="fixed_epochs", target_source="fixed_simulator_outcomes")
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
        return collect_bank_extension(data, n_extra=count,
                                      bank_seed=data_config.get("candidate_seed", data_config["seed"]),
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
        for arm in study_arms(manifest["config"]):
            name = f"{arm}/seed_{seed}"
            state = manifest["runs"][name]
            result = json.loads(_verify_artifact(root, state, "evaluation").read_text())
            rows.extend(_rows(result["evaluation"], arm, seed))
            base = manifest["runs"][f"baseline/seed_{seed}"]
            costs[name] = {"training_seconds": state["training_seconds"],
                           "shared_frozen_baseline_training_seconds": base["training_seconds"],
                           "shared_frozen_baseline_training_cost_complete": base["training_cost_complete"],
                           "cost_note": "baseline training is shared within a seed; charge it once, not once per arm",
                           "shared_acquisition_from": state.get("shared_acquisition_from"),
                           "training_cost_complete": state["training_cost_complete"],
                           "acquisition": (_attempt_costs(root, state) if state.get("acquisition_attempts")
                                           else state.get("acquisition_cost", {"objective_calls": 0})),
                           "offline_evaluation_diagnostics": result["costs"],
                           "deployment_latency": result["deployment_latency"]}
            diagnostics[name] = {k: result[k] for k in ("proposal_diagnostics", "frozen_proposal_diagnostics")}
    count = manifest["config"].get("bootstrap_resamples", 2000)
    contrasts = {}
    for comparator in ("control", "bankext", "decoder_random"):
        contrasts[f"policy_minus_{comparator}"] = paired_parent_seed_contrast(
            rows, comparator, "policy", mode="direct", bootstrap_resamples=count)
    deployment = {}
    mechanisms = {}
    for scope in manifest["config"].get("mechanism", {}).get("scopes", MECHANISM_SCOPES):
        original, acquired = f"mechanism_{scope}_original", f"mechanism_{scope}_acquired"
        if original in study_arms(manifest["config"]):
            mechanisms[scope] = {mode: paired_parent_seed_contrast(
                rows, original, acquired, mode=mode, bootstrap_resamples=count) for mode in ("bank", "direct")}
    for arm in study_arms(manifest["config"]):
        selected = [dict(row, method=row["mode"], mode="deployment") for row in rows
                    if row["method"] == arm and row["mode"] in {"bank", "direct", "global"}]
        deployment[arm] = {f"direct_minus_{ref}": paired_parent_seed_contrast(
            selected, ref, "direct", mode="deployment", bootstrap_resamples=count) for ref in ("bank", "global")}
    means = {}
    for arm in study_arms(manifest["config"]):
        means[arm] = {}
        for mode in ("bank", "direct", "global", "linear"):
            cells = {}
            for row in rows:
                if row["method"] == arm and row["mode"] == mode:
                    cells.setdefault((row["parent_id"], row["seed"]), []).append(row["loss"])
            means[arm][mode] = float(np.mean([np.mean(v) for v in cells.values()]))
    result = {"schema_version": 1, "config_hash": manifest["config_hash"], "source_hash": manifest["source_hash"],
              "means": means, "acquisition_contrasts": contrasts, "deployment_contrasts": deployment,
              "mechanism_contrasts": mechanisms,
              "mechanism_scope": "acquired minus original labels within the same frozen-backbone head scope; fixed-budget continuation, not primary from-scratch arms",
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
              "Primary checkpoints use the original validation bank; mechanism checkpoints use the predeclared final epoch. Test labels enter only frozen evaluation.",
              "Both bank and direct deployment require zero online simulator calls. Offline diagnostic scoring is separate."]
    if mechanisms:
        lines += ["", "Mechanism arms share the frozen baseline, normalizer, policy-acquired labels, and fixed epoch budget.",
                  "Only designated heads change; encoder/attention/response remain fixed. Compare acquired vs original within each scope.",
                  "Policy targets use true simulator labels; updated critics never create pseudo-labels. No mechanism result by itself proves distribution-shift causality."]
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
                for attempt in state.get("acquisition_attempts", []):
                    _verify_artifact(root, attempt, "request")
                    if attempt["status"] == "started":
                        _finish_attempt(root, attempt, {
                            "status": "interrupted", "cost_counts_complete": False,
                            "objective_calls_complete": False,
                            "scope": "Persisted request has no terminal receipt; unreported work is unknown."})
                    else:
                        _verify_artifact(root, attempt, "audit")
                if state.get("acquisition_attempts"):
                    state["acquisition_cost"] = _attempt_costs(root, state)
            write_json(path, manifest)
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
                    for arm in study_arms(cfg):
                        name = f"{arm}/seed_{seed}"
                        state = manifest["runs"].setdefault(name, {"folder": f"models/{name}"})
                        if state.get("trained"):
                            continue
                        _assert_frozen(manifest)
                        records = training
                        if _has_acquisition(arm):
                            if arm.startswith("mechanism_"):
                                source_name = f"policy/seed_{seed}"
                                source = manifest["runs"][source_name]
                                _verify_artifact(root, source, "acquisition")
                                state.update(acquisition=source["acquisition"], acquisition_sha256=source["acquisition_sha256"],
                                             shared_acquisition_from=source_name,
                                             acquisition_cost={"objective_calls": 0, "cost_counts_complete": True,
                                                 "objective_calls_complete": True,
                                                 "scope": "shared policy acquisition charged once to the primary policy arm"})
                            if "acquisition" not in state:
                                print(f"[acquire] {name}", flush=True)
                                _acquire_recorded(root, manifest, state, data=data, baseline=baseline,
                                                  arm=arm, seed=seed, cfg=cfg,
                                                  data_config=data_config, device=device)
                            collected = json.loads(_verify_artifact(root, state, "acquisition").read_text())
                            records = augment_records(training, collected)
                        print(f"[retrain] {name}", flush=True)
                        if arm.startswith("mechanism_"):
                            _fit(root, state, records, validation, cfg, seed, device, resume,
                                 initialize_from=baseline, scope=arm.split("_")[1])
                        else:
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
                    for arm in study_arms(cfg):
                        state = manifest["runs"][f"{arm}/seed_{seed}"]
                        if "evaluation" in state:
                            continue
                        _assert_frozen(manifest)
                        records = training if not _has_acquisition(arm) else augment_records(training,
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
            if getattr(error, "acquisition_audit", None) is not None:
                manifest["error"]["acquisition_audit"] = error.acquisition_audit
            write_json(path, manifest)
            raise
    return manifest
