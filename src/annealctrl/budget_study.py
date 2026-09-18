"""Frozen learned warm starts versus search, with charged anytime trajectories.

All optimizers use the same eight-bin duration family. The original learned
waveforms remain zero-online-query methods; projected hints are different
controls and always incur an outcome query. These are finite-budget best-found
references, never globally optimal controls or QPU performance claims.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from time import perf_counter
from typing import Mapping

import numpy as np

from .pipeline import jsonable, source_fingerprint, write_json
from .schedules import Schedule
from .search import decode_eight_bin, eight_bin_hint, optimize_control_family


def _digest(value):
    return hashlib.sha256(json.dumps(value, default=jsonable, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _text(record, key):
    return str(np.asarray(record[key]).item())


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def validate_budget_config(cfg):
    allowed = {"schema_version", "source_data", "target_data", "checkpoint", "split", "budgets",
               "seeds", "strategies", "warm_modes", "hint_mode", "hint_tolerance", "execution",
               "numerics", "bootstrap_resamples", "offline_cost", "quality_thresholds",
               "allow_test_adaptation", "max_records", "family", "literature", "latency"}
    unknown = [k for k in cfg if k not in allowed and not k.startswith("_")]
    if unknown:
        raise ValueError(f"unknown budget-study config keys: {unknown}")
    for key in ("source_data", "checkpoint", "budgets", "seeds"):
        if key not in cfg:
            raise ValueError(f"missing {key}")
    if cfg.get("schema_version", 1) != 1:
        raise ValueError("unsupported budget-study schema_version")
    budgets = cfg["budgets"]
    if not isinstance(budgets, list) or not budgets or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in budgets):
        raise ValueError("budgets must be nonnegative integer list")
    if budgets != sorted(set(budgets)) or budgets[0] != 0 or max(budgets) < 2:
        raise ValueError("sorted unique budgets must contain 0 and a horizon >= 2")
    seeds = cfg["seeds"]
    if not seeds or len(set(seeds)) != len(seeds) or any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in seeds):
        raise ValueError("seeds must be distinct nonnegative integers")
    if cfg.get("split", "test") not in {"validation", "test"}:
        raise ValueError("budget-study target split must be validation or test")
    if cfg.get("split", "test") == "test" and cfg.get("allow_test_adaptation") is not True:
        raise ValueError("test search requires allow_test_adaptation=true")
    if cfg.get("family", "eight_bin") != "eight_bin":
        raise ValueError("matched warm-start study only supports eight_bin")
    strategies = cfg.get("strategies", ["sobol_local", "bayesian", "finzgar_gp_ucb"])
    if not strategies or len(set(strategies)) != len(strategies) or set(strategies) - {"sobol_local", "bayesian", "finzgar_gp_ucb"}:
        raise ValueError("invalid or repeated search strategies")
    modes = cfg.get("warm_modes", ["bank", "direct", "source_global"])
    if not modes or len(set(modes)) != len(modes) or set(modes) - {"bank", "direct", "source_global"}:
        raise ValueError("invalid or repeated warm modes")
    if cfg.get("hint_mode", "project") not in {"reject", "project"}:
        raise ValueError("hint_mode must be reject or project")
    if "max_records" in cfg:
        _positive(cfg["max_records"], "max_records")
    _positive(cfg.get("bootstrap_resamples", 2000), "bootstrap_resamples")
    for value in cfg.get("quality_thresholds", []):
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("preregistered quality thresholds must be losses in [0,1]")
    for group, valid in [("execution", {"device", "backend", "threads"}),
                         ("numerics", {"tolerance", "initial_steps", "max_steps", "norm_tolerance", "max_ds_dtau"}),
                         ("literature", {"n_initial", "acquisition_candidates", "acquisition_restarts", "gp_restarts", "noise"}),
                         ("latency", {"warmup", "repeats"})]:
        if set(cfg.get(group, {})) - valid:
            raise ValueError(f"unknown settings in {group}")
    literature = cfg.get("literature", {})
    for key in ("n_initial", "acquisition_candidates", "acquisition_restarts", "gp_restarts"):
        if key in literature:
            _positive(literature[key], key)
    if literature.get("n_initial", 10) < 2:
        raise ValueError("warm Finzgar initial design needs at least two trials")
    if "noise" in literature and (not np.isfinite(literature["noise"]) or literature["noise"] <= 0):
        raise ValueError("literature noise must be positive and finite")
    return cfg


class ObjectiveLedger:
    """Write-ahead objective receipts and deterministic replay without re-querying.

    Every requested call is durable before simulator entry. An interrupted request
    is retried under a new attempt and remains an explicitly incomplete cost.
    Completed failures are replayed as failures, not retried until they succeed.
    Timing of resumed trajectories is marked incomplete because optimizer replay
    is administrative work; query/loss traces remain exactly reproducible.
    """
    def __init__(self, path, objective, *, resume=False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.objective = objective
        self.events = []
        self.cursor = 0
        self.started = perf_counter()
        self.resumed = self.path.exists()
        if self.resumed and not resume:
            raise FileExistsError(self.path)
        if self.resumed:
            for line in self.path.read_text().splitlines():
                self.events.append(json.loads(line))
        self.replayed_calls = 0

    def _append(self, row):
        with self.path.open("a") as stream:
            stream.write(json.dumps(row, default=jsonable, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.events.append(row)

    def __call__(self, schedule):
        index = self.cursor
        self.cursor += 1
        waveform = schedule.to_dict()
        digest = _digest(waveform)
        requested = [r for r in self.events if r["index"] == index and r["event"] == "requested"]
        if any(r["waveform_sha256"] != digest for r in requested):
            raise ValueError("resumed search proposed a different waveform; ledger/source mismatch")
        finished = [r for r in self.events if r["index"] == index and r["event"] == "finished"]
        if finished:
            self.replayed_calls += 1
            row = finished[-1]
            if row["status"] != "ok":
                if row.get("numerical_failure") is True:
                    raise ArithmeticError(row["error"])
                raise ValueError("replayed nonnumerical objective error: " + row["error"])
            return row["metrics"]
        attempt = len(requested)
        self._append({"event": "requested", "index": index, "attempt": attempt,
                      "waveform_sha256": digest, "waveform": waveform})
        began = perf_counter()
        row = {"event": "finished", "index": index, "attempt": attempt,
               "waveform_sha256": digest, "waveform": waveform}
        try:
            outcome = self.objective(schedule)
            metrics = dict(outcome) if isinstance(outcome, Mapping) else {"loss": float(outcome)}
            if not np.isfinite(float(metrics["loss"])):
                raise ArithmeticError("nonfinite objective loss")
            row.update(status="ok", metrics=metrics, loss=float(metrics["loss"]))
        except Exception as exc:
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}", error_type=type(exc).__name__,
                       numerical_failure=isinstance(exc, ArithmeticError), loss=None, metrics=None)
            row.update(objective_seconds=perf_counter() - began,
                       search_elapsed_seconds=None if self.resumed else perf_counter() - self.started)
            self._append(row)
            raise
        row.update(objective_seconds=perf_counter() - began,
                   search_elapsed_seconds=None if self.resumed else perf_counter() - self.started)
        self._append(row)
        return metrics

    def outcome_rows(self):
        rows = {}
        for row in self.events:
            if row["event"] == "finished":
                rows[row["index"]] = row
        return [{**rows[i], "objective_attempts": sum(r["event"] == "requested" and r["index"] == i for r in self.events)}
                for i in sorted(rows)]

    def costs(self):
        requested = {(r["index"], r["attempt"]) for r in self.events if r["event"] == "requested"}
        finished = {(r["index"], r["attempt"]) for r in self.events if r["event"] == "finished"}
        failures = [r for r in self.events if r["event"] == "finished" and r["status"] != "ok"]
        successful = [r for r in self.events if r["event"] == "finished" and r["status"] == "ok"]
        return {"attempted_objective_calls": len(requested), "completed_objective_calls": len(finished),
                "failed_calls": len(failures), "interrupted_calls": len(requested - finished),
                "replayed_without_query": self.replayed_calls,
                "known_objective_seconds": sum(r["objective_seconds"] for r in successful + failures),
                "known_propagation_calls": sum(r["metrics"].get("propagation_calls", 0) for r in successful),
                "known_integrator_steps": sum(r["metrics"].get("total_integrator_steps", 0) for r in successful),
                "inner_cost_is_lower_bound": bool(failures or requested - finished),
                "walltime_valid": not self.resumed}


def run_search_trajectory(objective, *, strategy, horizon, seed, runtime=1., max_ds_dtau=4.,
                          hint=None, literature=None):
    """Charge the initial linear and optional hint inside the total horizon."""
    if strategy == "finzgar_gp_ucb":
        from .literature_baselines import optimize_finzgar_schedule
        settings = dict(literature or {})
        result = optimize_finzgar_schedule(
            objective, budget=horizon, seed=seed, n_segments=8, max_ds_dtau=max_ds_dtau,
            initial_point=hint, split="test", allow_test_adaptation=True,
            _schedule_decoder=lambda p: decode_eight_bin(p, runtime=runtime, max_slope=max_ds_dtau / runtime),
            **settings)
        return {"status": "complete", "optimizer": result["method"], "settings": result["settings"],
                "reference": result["reference"], "reproduction": False,
                "representation": "eight_bin_duration_common_to_all_budget_study_arms",
                "horizon": horizon}
    try:
        result = optimize_control_family(lambda s: float(objective(s)["loss"]), "eight_bin",
                                         budget=horizon, seed=seed, runtime=runtime,
                                         max_slope=max_ds_dtau / runtime, strategy=strategy,
                                         warm_start=hint, split="test", allow_test_adaptation=True)
        return {"status": "complete", "optimizer": strategy, "horizon": horizon,
                "n_evaluations": result.n_evaluations}
    except ArithmeticError as exc:
        # No fabricated penalty labels: censor the remaining trajectory.
        return {"status": "numerically_censored", "optimizer": strategy, "horizon": horizon,
                "error": str(exc)}


def trajectory_points(rows, budgets, *, deployment_seconds=0., setup_seconds=0., costs=None):
    result = []
    for budget in budgets:
        if budget == 0:
            continue
        prefix = [r for r in rows if r["index"] < budget]
        valid = [r for r in prefix if r["status"] == "ok"]
        complete = len(prefix) == budget
        best = min(valid, key=lambda r: r["loss"]) if valid else None
        result.append({"budget": budget, "online_queries": sum(r.get("objective_attempts", 1) for r in prefix),
                       "selected_loss": None if best is None else best["loss"],
                       "selected_waveform": None if best is None else best["waveform"],
                       "status": "ok" if complete and len(valid) == budget and sum(r.get("objective_attempts", 1) for r in prefix) == budget else "censored",
                       "failed_queries": len(prefix) - len(valid),
                       "objective_seconds": sum(r["objective_seconds"] for r in prefix),
                       "deployment_seconds": deployment_seconds,
                       "online_wall_seconds": (deployment_seconds + setup_seconds + prefix[-1]["search_elapsed_seconds"]
                                               if complete and prefix[-1]["search_elapsed_seconds"] is not None else None),
                       "trajectory_cost": costs})
    return result


def _parent_mean(rows, field):
    parents = sorted({r["logical_fingerprint"] for r in rows})
    return float(np.mean([np.mean([r[field] for r in rows if r["logical_fingerprint"] == p]) for p in parents]))


def _bootstrap(values, n_resamples, seed):
    values = np.asarray(values, float)
    if len(values) < 2:
        return {"low": None, "high": None, "n_parents": len(values)}
    rng = np.random.default_rng(seed)
    means = np.mean(values[rng.integers(0, len(values), (n_resamples, len(values)))], axis=1)
    return {"low": float(np.quantile(means, .025)), "high": float(np.quantile(means, .975)), "n_parents": len(values)}


def equal_quality_break_even(*, offline_seconds, learned_seconds, baseline_seconds,
                             learned_loss, baseline_loss, quality_threshold):
    """Arithmetic for a preregistered common quality target, not a quality proof.

    Callers must report uncertainty/coverage separately. Unknown offline cost,
    either method missing the fixed target, or no positive per-instance savings
    produces no break-even claim.
    """
    values = [offline_seconds, learned_seconds, baseline_seconds, learned_loss, baseline_loss, quality_threshold]
    if any(x is None for x in values):
        return {"status": "unknown_cost_or_quality", "instances": None}
    if not np.isfinite(values).all() or min(offline_seconds, learned_seconds, baseline_seconds) < 0:
        raise ValueError("cost and quality inputs must be finite; costs nonnegative")
    if learned_loss > quality_threshold or baseline_loss > quality_threshold:
        return {"status": "common_quality_target_not_met", "instances": None}
    saving = baseline_seconds - learned_seconds
    if saving <= 0:
        return {"status": "no_per_instance_time_saving", "instances": None}
    return {"status": "conditional_on_measured_mean_cost_and_common_quality_target",
            "instances": int(math.ceil(offline_seconds / saving)), "seconds_saved_per_instance": saving,
            "quality_threshold": quality_threshold, "uncertainty": "not a confidence bound"}


def budget_study_report(rows, *, bootstrap_resamples=2000, seed=0, offline_cost=None, quality_thresholds=()):
    """Equal-parent summaries and paired curves on a common complete population.

    Seeds are averaged within records before parent bootstrap. This CI describes
    logical-parent variability conditional on these checkpoints/search seeds.
    It does not measure checkpoint-training uncertainty.
    """
    rows = list(rows)
    groups = {}
    for row in rows:
        groups.setdefault((row["method"], row["budget"]), []).append(row)
    summaries = []
    for (method, budget), group in sorted(groups.items()):
        complete = [r for r in group if r["status"] == "ok"]
        entry = {"method": method, "budget": budget, "requested_rows": len(group),
                 "completed_rows": len(complete), "censored_rows": len(group) - len(complete),
                 "n_parents": len({r["logical_fingerprint"] for r in complete}),
                 "parent_mean_loss": _parent_mean(complete, "selected_loss") if complete else None,
                 "mean_online_queries": float(np.mean([r["online_queries"] for r in group]))}
        timed = [r for r in complete if r["online_wall_seconds"] is not None]
        entry["parent_mean_online_seconds"] = _parent_mean(timed, "online_wall_seconds") if len(timed) == len(complete) and timed else None
        summaries.append(entry)
    contrasts = []
    for (method, budget), group in sorted(groups.items()):
        if "/" not in method or method.endswith("/cold"):
            continue
        baseline = method.split("/")[0] + "/cold"
        cold = groups.get((baseline, budget), [])
        keyed = {(r["record_id"], r["seed"]): r for r in cold if r["status"] == "ok"}
        paired = [{"logical_fingerprint": r["logical_fingerprint"],
                   "difference": r["selected_loss"] - keyed[(r["record_id"], r["seed"])]["selected_loss"]}
                  for r in group if r["status"] == "ok" and (r["record_id"], r["seed"]) in keyed]
        parents = sorted({r["logical_fingerprint"] for r in paired})
        values = [np.mean([r["difference"] for r in paired if r["logical_fingerprint"] == p]) for p in parents]
        contrasts.append({"method": method, "baseline": baseline, "budget": budget,
                          "matched_rows": len(paired), "requested_rows": len(group),
                          "complete_population": len(paired) == len(group) == len(cold),
                          "population_scope": "matched successful record-seed subset; missing cells may change within-parent weighting",
                          "mean_difference": float(np.mean(values)) if values else None,
                          "parent_bootstrap_ci": _bootstrap(values, bootstrap_resamples, seed),
                          "inference": "descriptive_unadjusted; negative favors warm; conditional on seeds"})
    # Pareto membership only compares fully observed points from the identical
    # record/seed population. Censored subsets never enter a leaderboard.
    population = {(r["record_id"], r["seed"]) for r in rows}
    eligible = [s for s in summaries if s["completed_rows"] == s["requested_rows"] and
                {(r["record_id"], r["seed"]) for r in groups[(s["method"], s["budget"]) ]} == population]
    for entry in summaries:
        entry["quality_queries_pareto"] = None
        entry["quality_walltime_pareto"] = None
        if entry not in eligible:
            continue
        for metric, target in [("mean_online_queries", "quality_queries_pareto"),
                               ("parent_mean_online_seconds", "quality_walltime_pareto")]:
            if entry[metric] is None:
                continue
            entry[target] = not any(other[metric] is not None and other[metric] <= entry[metric]
                                     and other["parent_mean_loss"] <= entry["parent_mean_loss"]
                                     and (other[metric] < entry[metric] or other["parent_mean_loss"] < entry["parent_mean_loss"])
                                     for other in eligible)
    costs = dict(offline_cost or {"status": "unknown", "total_seconds": None})
    break_even = []
    total = costs.get("total_seconds") if costs.get("status") == "complete" else None
    for threshold in quality_thresholds:
        qualified = [s for s in eligible if s["parent_mean_loss"] <= threshold and s["parent_mean_online_seconds"] is not None]
        cold = [s for s in qualified if s["method"].endswith("/cold")]
        learned = [s for s in qualified if s["method"] in {"bank", "direct"} or s["method"].endswith(("/bank", "/direct"))]
        if not cold or not learned:
            break_even.append({"quality_threshold": threshold, "status": "no_fully_observed_common_quality_target", "instances": None})
            continue
        base = min(cold, key=lambda s: s["parent_mean_online_seconds"])
        best = min(learned, key=lambda s: s["parent_mean_online_seconds"])
        result = equal_quality_break_even(offline_seconds=total, learned_seconds=best["parent_mean_online_seconds"],
                    baseline_seconds=base["parent_mean_online_seconds"], learned_loss=best["parent_mean_loss"],
                    baseline_loss=base["parent_mean_loss"], quality_threshold=threshold)
        result.update(learned_method=best["method"], learned_budget=best["budget"], baseline=base["method"], baseline_budget=base["budget"],
                      selection_scope="descriptive_best_measured_grid_points; not an independently validated deployment policy")
        break_even.append(result)
    return {"schema_version": 1, "curves": summaries, "paired_warm_minus_cold": contrasts,
            "offline_cost": costs, "break_even": break_even,
            "inference": "parent-bootstrap descriptive CIs, seeds averaged, no multiple-testing significance claims",
            "reference_status": "best_found_within_evaluated_candidates_not_global_optimum"}


def _acquisition_offline_cost(spec):
    """Read measured preparation work; never infer missing/restarted costs."""
    from .acquisition_study import _attempt_costs, _verify_artifact, plan_study
    root = Path(spec["acquisition_study"])
    path = root / "study.json"
    manifest = json.loads(path.read_text())
    data_path = Path(manifest["data_dir"]) / "manifest.json"
    data = json.loads(data_path.read_text())
    receipts = {str(path): _sha(path), str(data_path): _sha(data_path)}
    reasons = []
    if receipts[str(data_path)] != manifest.get("data_manifest_sha256"):
        raise ValueError("acquisition-study data manifest changed")
    if not manifest.get("training_complete"):
        reasons.append("training stage incomplete")
    if data.get("status") != "complete" or data.get("generation_cost_complete") is not True:
        reasons.append("data-generation cost missing or incomplete after resume")
    if spec.get("fixed_recipe") is not True:
        reasons.append("no explicit declaration of fixed recipe without external tuning")
    runs = manifest.get("runs", {})
    if len(runs) != plan_study(manifest["config"])["training_runs"]:
        reasons.append("missing training-run receipts")
    training, acquisition, calls = 0., 0., 0
    def finite_seconds(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value) and value >= 0
    for name, state in runs.items():
        if not state.get("trained") or state.get("training_cost_complete") is not True or not finite_seconds(state.get("training_seconds")):
            reasons.append(f"incomplete training cost: {name}")
        if finite_seconds(state.get("training_seconds")):
            training += state["training_seconds"]
        if state.get("checkpoint"):
            artifact = _verify_artifact(root, state, "checkpoint")
            receipts[str(artifact)] = _sha(artifact)
        if state.get("acquisition_attempts"):
            try:
                cost = _attempt_costs(root, state)
            except (KeyError, FileNotFoundError):
                reasons.append(f"interrupted acquisition cost: {name}")
                continue
            if not cost.get("cost_counts_complete") or not cost.get("objective_calls_complete"):
                reasons.append(f"incomplete acquisition cost: {name}")
            acquisition += cost["wall_seconds"]
            calls += cost["objective_calls"]
            for attempt in state["acquisition_attempts"]:
                for key in ("request", "audit"):
                    artifact = _verify_artifact(root, attempt, key)
                    receipts[str(artifact)] = _sha(artifact)
        elif state.get("acquisition_cost", {}).get("objective_calls", 0):
            reasons.append(f"legacy acquisition lacks cumulative attempt receipts: {name}")
    generation = data.get("elapsed_seconds")
    if not finite_seconds(generation):
        reasons.append("missing data-generation seconds")
        generation = None
    complete = not reasons
    return {"status": "complete" if complete else "unknown", "reason": reasons,
            "total_seconds": generation + training + acquisition if complete else None,
            "known_components": {"data_generation_seconds": generation, "training_seconds": training,
                                 "acquisition_seconds": acquisition, "acquisition_objective_calls": calls,
                                 "tuning_seconds": 0. if spec.get("fixed_recipe") is True else None},
            "receipt_sha256": receipts, "source": "acquisition_study_automatically_measured_receipts",
            "scope": "whole frozen study preparation: source data, every baseline/control/acquisition/mechanism fit, unique acquisition attempts; shared fits counted once",
            "exclusions": "held-out scientific evaluation, external project development and unreported exploratory tuning; not a minimal single-method preparation cost",
            "fixed_recipe_declaration": spec.get("fixed_recipe") is True}


def _offline_provenance(spec):
    if spec and "acquisition_study" in spec:
        return _acquisition_offline_cost(spec)
    if not spec or spec.get("status") != "complete":
        return {"status": "unknown", "total_seconds": None,
                "declared": spec, "scope": "missing offline cost cannot be treated as zero"}
    components = spec.get("components", {})
    required = {"data_generation", "training", "acquisition", "tuning"}
    if set(components) != required:
        raise ValueError("complete offline cost needs data_generation/training/acquisition/tuning components")
    receipts, total = {}, 0.
    for name, component in components.items():
        seconds = component.get("seconds")
        if not isinstance(seconds, (int, float)) or isinstance(seconds, bool) or not np.isfinite(seconds) or seconds < 0:
            raise ValueError(f"invalid offline seconds for {name}")
        receipt = Path(component["receipt"])
        receipts[name] = {**component, "receipt_sha256": _sha(receipt)}
        total += seconds
    return {"status": "complete", "components": receipts, "total_seconds": total,
            "scope": "operator-declared stage walltimes backed by hashed receipts; not accelerator rental cost"}


def _select_controls(model, normalizer, record, global_schedule, *, device, warmup=1, repeats=3):
    import torch
    from .acquisition_evaluation import _synchronize
    from .models import graph_from_record
    _positive(repeats, "latency repeats")
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("latency warmup must be nonnegative")

    def deploy(mode):
        graph = graph_from_record(record, device=device)
        if normalizer is not None:
            graph = normalizer.transform(graph)
        waveforms = (torch.as_tensor(record["candidate_schedules"], dtype=torch.float32, device=device)
                     if mode == "bank" else model(graph)["proposal_schedules"])
        prediction = model.predict_losses(graph, waveforms)
        if not torch.isfinite(prediction).all() or not torch.isfinite(waveforms).all():
            raise ArithmeticError("nonfinite deployed model output")
        index = int(prediction.argmin())
        waveform = waveforms[index].detach().cpu().double().numpy().copy()
        waveform[0], waveform[-1] = 0., 1.
        tau = (np.asarray(record.get("candidate_tau", np.linspace(0, 1, len(waveform))), float)
               if mode == "bank" else np.linspace(0, 1, len(waveform)))
        return Schedule(tau, waveform), index

    result = {}
    samples = {"bank": [], "direct": []}
    with torch.inference_mode():
        for _ in range(warmup):
            for mode in samples:
                deploy(mode)
                _synchronize(device)
        for repeat in range(repeats):
            for mode in (["bank", "direct"] if repeat % 2 == 0 else ["direct", "bank"]):
                _synchronize(device)
                start = perf_counter()
                schedule, index = deploy(mode)
                _synchronize(device)
                samples[mode].append(perf_counter() - start)
                result[mode] = {"waveform": schedule.to_dict(), "selected_index": index}
    for mode in samples:
        result[mode].update(deployment_seconds=float(np.median(samples[mode])), latency_samples=samples[mode])
    result["source_global"] = {"waveform": global_schedule.to_dict(), "deployment_seconds": 0.,
                               "latency_scope": "stored fixed waveform; no model inference"}
    result["linear"] = {"waveform": Schedule.linear().to_dict(), "deployment_seconds": 0.}
    return result


def run_budget_study(config, output_dir, *, resume=False):
    """Run a checkpoint-conditioned, matched-record anytime budget campaign.

    Input paths resolve relative to cwd. Source validation chooses one fixed
    waveform before target scoring. Target logical Hamiltonians must be disjoint
    from every train/validation logical Hamiltonian seen by the checkpoint.
    A source held-out test split and a genuinely external target both work.
    """
    import torch
    from .benchmarking import record_physics, score_schedule, validate_candidate_banks, _numerical_settings, _parent_mean as parent_mean
    from .pipeline import _payload_fingerprint, dataset_lock, load_records
    from .transfer import validate_transfer_provenance, validate_source_reference, _load_transfer_model
    cfg = json.loads(Path(config).read_text()) if isinstance(config, (str, Path)) else dict(config)
    validate_budget_config(cfg)
    source = Path(cfg["source_data"])
    target = Path(cfg.get("target_data", source))
    checkpoint = Path(cfg["checkpoint"])
    execution = cfg.get("execution", {})
    device, backend = execution.get("device", "cpu"), execution.get("backend", "numpy")
    if "threads" in execution:
        _positive(execution["threads"], "threads")
        torch.set_num_threads(execution["threads"])
    source_records = load_records(source)
    records = load_records(target, cfg.get("split", "test"))
    records = sorted(records, key=lambda r: (_text(r, "parent_id"), _text(r, "record_id")))
    if cfg.get("max_records"):
        records = records[:cfg["max_records"]]
    if not records:
        raise ValueError("budget-study requires target records")
    ids = [_text(r, "record_id") for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate target record ids")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    source_association = validate_source_reference(payload, source_records)
    provenance = validate_transfer_provenance(payload, records, source_records=source_records)
    provenance["source_association"] = source_association
    validation = [r for r in source_records if _text(r, "split") == "validation"]
    bank = validate_candidate_banks(validation)
    val_parents = [_text(r, "logical_fingerprint") for r in validation]
    losses = [parent_mean([float(r["candidate_losses"][i]) for r in validation], val_parents)
              for i in range(bank["candidate_count"])]
    global_index = int(np.argmin(losses))
    wave = np.asarray(validation[0]["candidate_schedules"])[global_index]
    global_schedule = Schedule(validation[0].get("candidate_tau", np.linspace(0., 1., len(wave))), wave)
    numerics = {"tolerance": 5e-4, "initial_steps": 128, "max_steps": 8192, "max_ds_dtau": 4.,
                **cfg.get("numerics", {})}
    _numerical_settings(numerics["tolerance"], numerics["initial_steps"], numerics["max_steps"],
                        numerics.get("norm_tolerance", 1e-9), backend)
    model, normalizer = _load_transfer_model(payload, device)
    model.eval()
    if not np.isclose(model.config.get("max_ds_dtau", 4.), numerics["max_ds_dtau"]):
        raise ValueError("checkpoint and budget-study slope envelopes differ")
    offline_cost = _offline_provenance(cfg.get("offline_cost"))
    frozen = {"config": cfg, "source_fingerprint": source_fingerprint(),
              "checkpoint_sha256": _sha(checkpoint), "source_manifest_sha256": _sha(source / "manifest.json"),
              "target_manifest_sha256": _sha(target / "manifest.json"),
              "record_payload_hashes": {rid: _payload_fingerprint(dict(record)) for rid, record in zip(ids, records)},
              "provenance": provenance, "offline_cost": offline_cost,
              "source_global": {"candidate_index": global_index, "waveform": global_schedule.to_dict(),
                                "selection_split": "source_validation", "parent_mean_losses": losses},
              "search_horizon": max(cfg["budgets"]),
              "curve_protocol": "prefixes of a single fixed-horizon trajectory, not independent runs per budget",
              "claim_scope": "finite-budget classical simulation, no global-optimality or QPU claim"}
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    with dataset_lock(root):
        manifest_path = root / "manifest.json"
        if manifest_path.exists():
            if not resume:
                raise FileExistsError("budget study exists; use resume with the identical frozen configuration")
            manifest = json.loads(manifest_path.read_text())
            if manifest["frozen_sha256"] != _digest(frozen):
                raise ValueError("frozen config/source/checkpoint/data changed; cannot resume")
            for name, digest in manifest["artifacts"].items():
                if _sha(root / name) != digest:
                    raise ValueError(f"budget-study artifact changed: {name}")
        else:
            manifest = {"schema_version": 1, "frozen": frozen, "frozen_sha256": _digest(frozen), "artifacts": {}}
            write_json(manifest_path, manifest)

        def save(path, value):
            if source_fingerprint() != frozen["source_fingerprint"]:
                raise RuntimeError("source changed during study; query receipts retained, result not publishable")
            write_json(path, value)
            manifest["artifacts"][str(path.relative_to(root))] = _sha(path)
            write_json(manifest_path, manifest)

        all_rows, all_costs = [], []
        for record in records:
            record_id, logical = _text(record, "record_id"), _text(record, "logical_fingerprint")
            directory = root / "records" / _digest(record_id)[:20]
            selection_path = directory / "selection.json"
            if selection_path.exists():
                selection = json.loads(selection_path.read_text())
            else:
                selection = _select_controls(model, normalizer, record, global_schedule, device=device,
                                             **cfg.get("latency", {}))
                save(selection_path, selection)
            runtime = float(np.asarray(record["runtime"]).item())
            setup_started = perf_counter()
            physics = record_physics(record)
            setup_seconds = perf_counter() - setup_started

            def objective(schedule):
                # Float32 network output can exceed the exact slope by rounding.
                # This gate only covers that declared construction tolerance.
                settings = {**numerics, "max_ds_dtau": numerics["max_ds_dtau"] * (1 + 1e-6)}
                return score_schedule(record, schedule, backend=backend, physics_context=physics, **settings)

            for mode in ("bank", "direct", "source_global", "linear"):
                path = directory / f"zero_{mode}.json"
                ledger_path = directory / f"zero_{mode}.jsonl"
                if path.exists():
                    result = json.loads(path.read_text())
                else:
                    ledger = ObjectiveLedger(ledger_path, objective, resume=resume)
                    chosen = Schedule(**selection[mode]["waveform"])
                    try:
                        outcome = ledger(chosen)
                        result = {"selected_loss": outcome["loss"], "status": "ok", "metrics": outcome}
                    except ArithmeticError as exc:
                        result = {"selected_loss": None, "status": "censored", "error": str(exc)}
                    result.update(cost=ledger.costs(), method=mode, budget=0, online_queries=0,
                                  diagnostic_queries=1, selected_waveform=chosen.to_dict(),
                                  online_wall_seconds=selection[mode]["deployment_seconds"])
                    save(path, result)
                    manifest["artifacts"][str(ledger_path.relative_to(root))] = _sha(ledger_path)
                    write_json(manifest_path, manifest)
                all_costs.append({"record_id": record_id, "role": "offline_diagnostic", "method": mode, **result["cost"]})
                for seed in cfg["seeds"]:
                    all_rows.append({**result, "record_id": record_id, "logical_fingerprint": logical,
                                     "seed": seed, "repeated_zero_query_row": "one deterministic selection, repeated only to pair search seeds"})

            for seed in cfg["seeds"]:
                # All variants of a logical parent share this deterministic stream;
                # cold/warm and the methods receive the same seed.
                search_seed = int(_digest([seed, logical])[:8], 16)
                for strategy in cfg.get("strategies", ["sobol_local", "bayesian", "finzgar_gp_ucb"]):
                    for mode in ["cold", *cfg.get("warm_modes", ["bank", "direct", "source_global"])]:
                        method = f"{strategy}/{mode}"
                        path = directory / f"{strategy}_{mode}_{seed}.json"
                        ledger_path = path.with_suffix(".jsonl")
                        if path.exists():
                            result = json.loads(path.read_text())
                        else:
                            hint = None
                            projection_started = perf_counter()
                            if mode != "cold":
                                original = Schedule(**selection[mode]["waveform"])
                                # Tiny float32 slope excess is allowed only here;
                                # inverse projection itself uses the exact bound.
                                hint = eight_bin_hint(original, runtime=runtime,
                                            max_slope=numerics["max_ds_dtau"] / runtime,
                                            mode=cfg.get("hint_mode", "project"),
                                            tolerance=cfg.get("hint_tolerance", 1e-6))
                            projection_seconds = 0. if mode == "cold" else perf_counter() - projection_started
                            ledger = ObjectiveLedger(ledger_path, objective, resume=resume)
                            meta = run_search_trajectory(ledger, strategy=strategy, horizon=max(cfg["budgets"]),
                                seed=search_seed, runtime=runtime, max_ds_dtau=numerics["max_ds_dtau"],
                                hint=None if hint is None else hint["unit_parameters"], literature=cfg.get("literature"))
                            cost = ledger.costs()
                            rows = trajectory_points(ledger.outcome_rows(), cfg["budgets"],
                                    deployment_seconds=0. if mode == "cold" else selection[mode]["deployment_seconds"] + projection_seconds,
                                    setup_seconds=setup_seconds, costs=cost)
                            result = {"method": method, "seed": seed, "search_seed": search_seed,
                                      "metadata": meta, "hint": hint, "rows": rows, "cost": cost}
                            save(path, result)
                            manifest["artifacts"][str(ledger_path.relative_to(root))] = _sha(ledger_path)
                            write_json(manifest_path, manifest)
                        all_costs.append({"record_id": record_id, "role": "online_search", "method": method, "seed": seed, **result["cost"]})
                        all_rows.extend({**row, "record_id": record_id, "logical_fingerprint": logical,
                                         "seed": seed, "method": method} for row in result["rows"])
        report = budget_study_report(all_rows, bootstrap_resamples=cfg.get("bootstrap_resamples", 2000),
                                    offline_cost=offline_cost, quality_thresholds=cfg.get("quality_thresholds", []))
        report.update(frozen_sha256=manifest["frozen_sha256"], cost_ledger=all_costs,
                      model_training_seeds="one checkpoint per campaign; search seeds do not count as training replications",
                      n_records=len(records), n_logical_parents=len({r["logical_fingerprint"] for r in all_rows}))
        save(root / "rows.json", all_rows)
        save(root / "report.json", report)
        manifest["status"] = "complete"
        write_json(manifest_path, manifest)
        return report
