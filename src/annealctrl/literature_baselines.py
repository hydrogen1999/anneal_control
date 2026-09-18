"""Task-constrained adaptation of Finzgar et al. (PR Research 6, 023063).

This implements their GP confidence-bound search strategy, not a reproduction
of their p-spin or Rydberg experiments. In particular, our forward monotone
controls exclude the nonmonotone schedules central to some of their results.
See docs/literature_baseline.md for the correspondence and changed choices.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Callable, Mapping

import numpy as np
from scipy.linalg import cho_solve, solve_triangular
from scipy.optimize import minimize

from .schedules import Schedule
from .search import capped_simplex_samples

REFERENCE = "https://doi.org/10.1103/PhysRevResearch.6.023063"


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def exploration_weight(iteration: int, n_iterations: int) -> float:
    """First half at 2; geometrically decay the second half to .01.

    Iterations are zero based. With 50 model steps this gives 25 steps at 2,
    followed by 25 decay steps. Other budgets use the same fractional schedule.
    """
    _positive_integer(n_iterations, "n_iterations")
    if not 0 <= iteration < n_iterations:
        raise ValueError("iteration outside the adaptive search budget")
    hold = (n_iterations + 1) // 2
    if iteration < hold:
        return 2.0
    fraction = (iteration - hold + 1) / (n_iterations - hold)
    return float(2.0 * (0.01 / 2.0) ** fraction)


def _kernel(a, b, length_scale):
    distance = np.sqrt(np.maximum(((a[:, None] - b[None, :]) ** 2).sum(-1), 0.))
    r = np.sqrt(5.) * distance / length_scale
    return (1. + r + r * r / 3.) * np.exp(-r)


def _fit_surrogate(x, y, *, rng, noise, restarts):
    """Isotropic Matern5/2, standardized responses, bounded continuous MLE."""
    centre = float(np.mean(y))
    spread = float(np.std(y))
    scale = spread if spread > 1e-12 else 1.0
    target = (y - centre) / scale

    def factor(log_length):
        length = float(np.exp(np.asarray(log_length).item()))
        cov = _kernel(x, x, length) + (noise + 1e-10) * np.eye(len(x))
        chol = np.linalg.cholesky(cov)
        alpha = cho_solve((chol, True), target, check_finite=False)
        nll = .5 * float(target @ alpha) + float(np.log(np.diag(chol)).sum())
        return nll, length, chol, alpha

    def objective(theta):
        try:
            return factor(theta)[0]
        except np.linalg.LinAlgError:
            return 1e30

    starts = [0., *rng.uniform(-4., 4., max(0, restarts - 1))]
    fits = [minimize(objective, [start], method="L-BFGS-B", bounds=[(-11.5, 11.5)])
            for start in starts]
    best = min(fits, key=lambda fit: float(fit.fun))
    _, length, chol, alpha = factor(best.x)

    def predict(query):
        query = np.atleast_2d(query)
        cross = _kernel(query, x, length)
        mean = cross @ alpha
        variance = np.maximum(1. - (solve_triangular(
            chol, cross.T, lower=True, check_finite=False) ** 2).sum(axis=0), 0.)
        return mean * scale + centre, np.sqrt(variance) * scale

    return predict, length


def decode_parameters(parameters, *, parameterization="policy_decoder", max_ds_dtau=4.,
                      real_zeta=.5, logit_bound=6.) -> Schedule:
    """Decode a finite unit-box point into this task's feasible control set.

    ``real`` has K-1 equispaced interior values j/K + zeta*(2*x_j-1)/K.
    Restricting zeta <= .5 guarantees monotonicity (the paper usually uses 2).
    ``policy_decoder`` uses K capped-simplex logits in [-logit_bound,logit_bound].
    """
    x = np.asarray(parameters, dtype=float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all() or np.any((x < 0) | (x > 1)):
        raise ValueError("parameters must be a finite vector in the unit box")
    if not np.isfinite(max_ds_dtau) or max_ds_dtau < 1:
        raise ValueError("max_ds_dtau must be finite and >= 1")
    if parameterization == "real":
        if not np.isfinite(real_zeta) or not 0 < real_zeta <= .5:
            raise ValueError("task-constrained real_zeta must lie in (0,.5]")
        if max_ds_dtau < 1 + 2 * real_zeta:
            raise ValueError("max_ds_dtau must cover the entire real parameter box")
        segments = len(x) + 1
        interior = (np.arange(1, segments) + real_zeta * (2 * x - 1)) / segments
        schedule = Schedule(np.linspace(0., 1., segments + 1), np.r_[0., interior, 1.])
    elif parameterization == "policy_decoder":
        if not np.isfinite(logit_bound) or logit_bound <= 0:
            raise ValueError("logit_bound must be finite and positive")
        waveform = capped_simplex_samples(logit_bound * (2 * x - 1),
                                          max_ds_dtau=max_ds_dtau)[0]
        schedule = Schedule(np.linspace(0., 1., len(waveform)), waveform)
    else:
        raise ValueError("parameterization must be real or policy_decoder")
    schedule.validate_slope(runtime=1., max_slope=max_ds_dtau)
    return schedule


def optimize_finzgar_schedule(objective: Callable, *, budget=60, n_segments=8,
                             parameterization="policy_decoder", seed=0,
                             max_ds_dtau=4., real_zeta=.5, logit_bound=6.,
                             n_initial=10, acquisition_candidates=1024,
                             acquisition_restarts=3, gp_restarts=3, noise=1e-6,
                             split="validation", allow_test_adaptation=False,
                             initial_point=None, search_method="gp_ucb",
                             event_callback=None, _schedule_decoder=None):
    """Run exactly ``budget`` attempted outcome queries, including failures.

    Objective accepts a Schedule and returns either loss or a score_schedule
    metrics mapping. ArithmeticError/nonfinite outcomes are recorded as failed
    queries and never converted into training labels. ValueError propagates:
    configuration/programming errors must not look like numerical censoring.
    """
    for value, name in [(budget, "budget"), (n_segments, "n_segments"),
                        (n_initial, "n_initial"), (acquisition_candidates, "acquisition_candidates"),
                        (acquisition_restarts, "acquisition_restarts"), (gp_restarts, "gp_restarts")]:
        _positive_integer(value, name)
    if n_segments < 2:
        raise ValueError("n_segments must be at least two")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if search_method not in {"gp_ucb", "uniform_random"}:
        raise ValueError("search_method must be gp_ucb or uniform_random")
    if parameterization not in {"real", "policy_decoder"}:
        raise ValueError("parameterization must be real or policy_decoder")
    if event_callback is not None and not callable(event_callback):
        raise ValueError("event_callback must be callable")
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    if split == "test" and not allow_test_adaptation:
        raise ValueError("test search requires allow_test_adaptation=True")
    if not np.isfinite(noise) or noise <= 0:
        raise ValueError("noise must be finite and positive")
    dimension = n_segments if parameterization == "policy_decoder" else n_segments - 1
    # Private hook for the separate original-system reference harness; the
    # embedded-task entry point always uses the constrained decoder above.
    decode = _schedule_decoder if _schedule_decoder is not None else (
        lambda p: decode_parameters(p, parameterization=parameterization,
                                    max_ds_dtau=max_ds_dtau, real_zeta=real_zeta,
                                    logit_bound=logit_bound))
    decode(np.full(dimension, .5))  # Validate all decoder settings before queries.
    rng = np.random.default_rng(seed)
    initial_count = min(budget, n_initial)
    design = [np.full(dimension, .5), *rng.random((initial_count - 1, dimension))]
    hint = None
    if initial_point is not None:
        hint = np.asarray(initial_point, dtype=float)
        if (initial_count < 2 or hint.shape != (dimension,) or
                not np.isfinite(hint).all() or np.any((hint < 0) | (hint > 1))):
            raise ValueError("initial_point requires two initialization calls and a finite unit-box vector of the correct dimension")
        # Keep every other initial draw identical to the cold-start design.
        design[1] = hint.copy()
    observed_x, observed_y, history = [], [], []
    best_index = None
    started = perf_counter()
    for index in range(budget):
        length_scale, kappa = None, None
        if index < initial_count:
            point, source = design[index], "linear" if index == 0 else "random_initial"
            if index == 1 and hint is not None:
                source = "warm_start"
        elif search_method == "uniform_random":
            point, source = rng.random(dimension), "uniform_random"
        elif len(observed_x) < 2:
            point, source = rng.random(dimension), "random_insufficient_valid_labels"
        else:
            kappa = exploration_weight(index - initial_count, budget - initial_count)
            predict, length_scale = _fit_surrogate(np.asarray(observed_x), np.asarray(observed_y),
                                                   rng=rng, noise=noise, restarts=gp_restarts)
            # UCB on success is equivalent to LCB on failure loss.
            def acquisition(points):
                mean, sd = predict(points)
                return mean - kappa * sd
            pool = rng.random((acquisition_candidates, dimension))
            rank = np.argsort(acquisition(pool))[:acquisition_restarts]
            local = [minimize(lambda z: float(acquisition(z)[0]), pool[j], method="L-BFGS-B",
                              bounds=[(0., 1.)] * dimension) for j in rank]
            pool = np.vstack([pool, *[fit.x for fit in local if np.isfinite(fit.fun)]])
            # Avoid exact repeats, which only consume calls for deterministic labels.
            unseen = np.min(np.max(np.abs(pool[:, None] - np.asarray(observed_x)[None]),
                                   axis=-1), axis=-1) > 1e-9
            pool = pool[unseen]
            if not len(pool):
                pool = rng.random((1, dimension))
            point, source = pool[int(np.argmin(acquisition(pool)))], "lower_confidence_bound"
        schedule = decode(point)
        step_start = perf_counter()
        row = {"evaluation": index + 1, "source": source, "parameters": point.tolist(),
               "waveform": schedule.to_dict(), "kappa": kappa,
               "gp_length_scale": length_scale, "loss": None, "metrics": None}
        if event_callback is not None:
            event_callback({"event": "query_started", **row})
        try:
            outcome = objective(schedule)
            metrics = dict(outcome) if isinstance(outcome, Mapping) else None
            value = float(metrics["loss"] if metrics is not None else outcome)
            if not np.isfinite(value):
                raise ArithmeticError("objective returned a nonfinite loss")
            row.update(status="ok", loss=value, metrics=metrics)
            observed_x.append(point)
            observed_y.append(value)
            if best_index is None or value < history[best_index]["loss"]:
                best_index = index
        except ArithmeticError as exc:
            row.update(status="numerical_failure", error=str(exc))
        except BaseException as exc:
            # Record the attempted expensive call, then preserve the original
            # exception. Programmer errors must not become censored labels.
            if event_callback is not None:
                event_callback({"event": "query_aborted", **row,
                                "error_type": type(exc).__name__, "error": str(exc),
                                "objective_seconds": perf_counter() - step_start})
            raise
        row["objective_seconds"] = perf_counter() - step_start
        history.append(row)
        row["best_so_far"] = None if best_index is None else history[best_index]["loss"]
        if event_callback is not None:
            event_callback({"event": "query_finished", **row})
    best = None if best_index is None else history[best_index]
    return {"method": ("finzgar_gp_ucb_task_adaptation" if search_method == "gp_ucb"
                       else "matched_uniform_random"), "reference": REFERENCE,
            "reproduction": False, "parameterization": parameterization,
            "decoder_scope": "custom_explicit_decoder" if _schedule_decoder is not None else "embedded_task_forward_feasible",
            "settings": {"budget": int(budget), "n_segments": int(n_segments), "seed": int(seed),
                         "max_ds_dtau": max_ds_dtau, "real_zeta": real_zeta,
                         "logit_bound": logit_bound, "n_initial": int(n_initial),
                         "acquisition_candidates": int(acquisition_candidates),
                         "acquisition_restarts": int(acquisition_restarts),
                         "gp_restarts": int(gp_restarts), "noise": noise,
                         "search_method": search_method,
                         "initial_point": None if hint is None else hint.tolist()},
            "split": split, "online_adaptation": split == "test",
            "n_objective_calls": len(history), "n_initial_calls": initial_count,
            "n_acquisition_queries": sum(r["source"] == "lower_confidence_bound" for r in history),
            "n_failed_calls": sum(r["status"] != "ok" for r in history),
            "best_loss": None if best is None else best["loss"],
            "best_waveform": None if best is None else best["waveform"],
            "wall_seconds": perf_counter() - started, "history": history,
            "reference_status": "best_found_within_charged_queries_not_global_optimum"}


def benchmark_finzgar_record(record, *, backend="numpy", tolerance=5e-4,
                            initial_steps=128, max_steps=8192, **search_settings):
    """Use the same converged physical objective as other project baselines."""
    from .benchmarking import _identity, _numerical_settings, record_physics, score_schedule
    from .pipeline import _payload_fingerprint
    _numerical_settings(tolerance, initial_steps, max_steps, 1e-9, backend)
    identity = _identity(record)
    supplied_split = search_settings.pop("split", identity["split"])
    if supplied_split != identity["split"]:
        raise ValueError("search split cannot override the record split")
    max_slope = search_settings.get("max_ds_dtau", 4.)
    context_start = perf_counter()
    context = record_physics(record)
    context_seconds = perf_counter() - context_start
    def objective(schedule):
        return score_schedule(record, schedule, backend=backend, tolerance=tolerance,
                              initial_steps=initial_steps, max_steps=max_steps,
                              max_ds_dtau=max_slope, physics_context=context)
    result = optimize_finzgar_schedule(objective, split=supplied_split, **search_settings)
    result.update(identity=identity, simulator={"backend": backend, "tolerance": tolerance,
                                              "initial_steps": initial_steps, "max_steps": max_steps},
                  record_payload_fingerprint=_payload_fingerprint(dict(record)),
                  observable_construction_seconds=context_seconds)
    metrics = [row["metrics"] for row in result["history"] if row["metrics"] is not None]
    result["cost"] = {"objective_calls": result["n_objective_calls"],
                      "known_propagation_calls": sum(m["propagation_calls"] for m in metrics),
                      "known_integrator_steps": sum(m["total_integrator_steps"] for m in metrics),
                      "inner_cost_is_lower_bound": result["n_failed_calls"] > 0,
                      "failed_inner_cost_unavailable": result["n_failed_calls"]}
    return result


def validate_campaign(config):
    """Resolve a frozen paired search protocol before any outcome query."""
    allowed = {"schema_version", "split", "allow_test_adaptation", "seeds", "budgets",
               "search", "simulator", "max_parents", "bootstrap_resamples"}
    unknown = {k for k in config if not k.startswith("_")} - allowed
    if unknown or config.get("schema_version", 1) != 1:
        raise ValueError(f"unknown literature-campaign keys or schema: {sorted(unknown)}")
    cfg = {"schema_version": 1, "split": "validation", "allow_test_adaptation": False,
           "seeds": [0, 1, 2, 3, 4], "budgets": [10, 20, 40, 60, 100],
           "bootstrap_resamples": 2000, "search": {}, "simulator": {}, **config}
    for key in ("seeds", "budgets"):
        values = cfg[key]
        if (not isinstance(values, (list, tuple)) or not values or
                any(type(x) is not int or x < (0 if key == "seeds" else 1) for x in values) or
                len(set(values)) != len(values)):
            raise ValueError(f"{key} must be a nonempty unique integer sequence")
        cfg[key] = list(values)
    if cfg["split"] not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    if type(cfg["allow_test_adaptation"]) is not bool:
        raise ValueError("allow_test_adaptation must be boolean")
    if cfg["split"] == "test" and not cfg["allow_test_adaptation"]:
        raise ValueError("test campaign requires allow_test_adaptation=True")
    if cfg.get("max_parents") is not None:
        _positive_integer(cfg["max_parents"], "max_parents")
    _positive_integer(cfg["bootstrap_resamples"], "bootstrap_resamples")
    search_allowed = {"n_segments", "parameterization", "max_ds_dtau", "real_zeta",
                      "logit_bound", "n_initial", "acquisition_candidates",
                      "acquisition_restarts", "gp_restarts", "noise"}
    if set(cfg["search"]) - search_allowed:
        raise ValueError("unknown literature search setting")
    sim_allowed = {"backend", "tolerance", "initial_steps", "max_steps"}
    if set(cfg["simulator"]) - sim_allowed:
        raise ValueError("unknown literature simulator setting")
    cfg["simulator"] = {"backend": "numpy", "tolerance": 5e-4,
                        "initial_steps": 128, "max_steps": 8192, **cfg["simulator"]}
    from .benchmarking import _numerical_settings
    sim = cfg["simulator"]
    _numerical_settings(sim["tolerance"], sim["initial_steps"], sim["max_steps"], 1e-9, sim["backend"])
    # A constant cheap objective validates all decoder/search arguments without
    # reading any scientific label or consuming a simulator call.
    optimize_finzgar_schedule(lambda schedule: 0., budget=1, **cfg["search"])
    return cfg


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize_campaign(results, *, bootstrap_resamples=2000):
    """Parent/optimizer-seed pairing with no silent censoring of failed runs."""
    from .contrasts import paired_parent_seed_contrast
    summaries = []
    for budget in sorted({r["settings"]["budget"] for r in results}):
        selected = [r for r in results if r["settings"]["budget"] == budget]
        runs = {(r["identity"]["record_id"], r["settings"]["seed"], r["settings"]["search_method"]): r
                for r in selected}
        if len(runs) != len(selected):
            raise ValueError("duplicate record/seed/method literature result")
        paired = {(rid, seed) for rid, seed, method in runs}
        if any((rid, seed, m) not in runs for rid, seed in paired for m in ("gp_ucb", "uniform_random")):
            raise ValueError("missing matched literature arm")
        for rid, seed in paired:
            bo, random = (runs[rid, seed, m] for m in ("gp_ucb", "uniform_random"))
            count = min(bo["n_initial_calls"], random["n_initial_calls"])
            if ([r["parameters"] for r in bo["history"][:count]] !=
                    [r["parameters"] for r in random["history"][:count]]):
                raise ValueError("paired methods do not share their initial design")
        missing = sum(r["best_loss"] is None for r in selected)
        report = {"budget": budget, "n_runs": len(selected), "n_matched_record_seed_pairs": len(paired),
                  "n_failed_queries": sum(r["n_failed_calls"] for r in selected),
                  "n_runs_without_incumbent": missing, "gp_ucb_minus_uniform_random": None,
                  "scope": "conditional on frozen dataset and simulator; independent runs at each budget"}
        if not missing:
            rows = [{**r["identity"], "method": r["settings"]["search_method"], "mode": "search",
                     "seed": r["settings"]["seed"], "loss": r["best_loss"]} for r in selected]
            contrast = paired_parent_seed_contrast(
                rows, "uniform_random", "gp_ucb", mode="search",
                bootstrap_resamples=bootstrap_resamples)
            contrast["unit_of_resampling"] = "crossed_logical_parent_and_optimizer_seed"
            contrast["scope"] = ("descriptive crossed-factor percentile interval; optimizer seeds and logical parents "
                                 "resampled independently; not multiplicity-adjusted; few seeds poorly resolve uncertainty")
            report["gp_ucb_minus_uniform_random"] = contrast
        else:
            report["status"] = "unresolved_runs_no_complete_population_contrast"
        summaries.append(report)
    return {"schema_version": 1, "reference": REFERENCE, "reproduction": False,
            "budgets": summaries, "note": "All budget contrasts are descriptive; no best-budget or best-seed selection."}


def _campaign_cost(root, state):
    started, finished, aborted, failures, propagation_calls, steps = 0, 0, 0, 0, 0, 0
    for unit in state["runs"].values():
        for attempt in unit["attempts"]:
            path = root / attempt["ledger"]
            if not path.exists():
                # Attempt recorded before file creation: no callback ran.
                continue
            if attempt.get("ledger_sha256") and _file_hash(path) != attempt["ledger_sha256"]:
                raise ValueError("literature query ledger changed")
            for line in path.read_text().splitlines():
                event = json.loads(line)
                started += event["event"] == "query_started"
                aborted += event["event"] == "query_aborted"
                if event["event"] == "query_finished":
                    finished += 1
                    failures += event["status"] != "ok"
                    metrics = event.get("metrics") or {}
                    propagation_calls += metrics.get("propagation_calls", 0)
                    steps += metrics.get("total_integrator_steps", 0)
    return {"reserved_objective_calls_all_attempts": started,
            "completed_objective_calls_all_attempts": finished,
            "aborted_objective_calls_all_attempts": aborted,
            "unresolved_call_reservations": started - finished,
            "numerical_failures_all_attempts": failures,
            "known_propagation_calls": propagation_calls, "known_integrator_steps": steps,
            "inner_cost_is_lower_bound": started != finished or failures > 0,
            "note": "Reservations precede the objective; an interrupted reservation may or may not have executed. Aborted queries consumed an attempted call but expose no solver work."}


def run_campaign(config, *, data, out, resume=False):
    """Frozen paired campaign with durable per-query receipts and cumulative cost.

    Every budget is an independently optimized run: its exploration decay is
    not a prefix of another budget's schedule. Interrupted runs restart as new
    attempts; previously spent calls remain charged in the cumulative ledger.
    """
    from .experiments import output_lock
    from .pipeline import _payload_fingerprint, load_records, source_fingerprint, write_json
    cfg = validate_campaign(config)
    data, root = Path(data).resolve(), Path(out).resolve()
    manifest_path = data / "manifest.json"
    dataset_hash = _file_hash(manifest_path)
    records = load_records(data, cfg["split"])
    if cfg.get("max_parents") is not None:
        parents = sorted({str(np.asarray(r["parent_id"]).item()) for r in records})[:cfg["max_parents"]]
        records = [r for r in records if str(np.asarray(r["parent_id"]).item()) in parents]
    if not records:
        raise ValueError("no literature campaign records selected")
    identities = [str(np.asarray(r["record_id"]).item()) for r in records]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate selected record IDs")
    fingerprint = source_fingerprint()
    frozen = {"config": cfg, "dataset_manifest_sha256": dataset_hash,
              "source_fingerprint": fingerprint,
              "selected_records": [{"record_id": rid, "payload_sha256": _payload_fingerprint(r)}
                                   for rid, r in zip(identities, records)]}
    def check_frozen():
        if source_fingerprint() != fingerprint or _file_hash(manifest_path) != dataset_hash:
            raise RuntimeError("source or dataset manifest changed during literature campaign")
    root.mkdir(parents=True, exist_ok=True)
    with output_lock(root):
        state_path = root / "campaign.json"
        if state_path.exists():
            if not resume:
                raise FileExistsError("campaign exists; explicit resume required")
            state = json.loads(state_path.read_text())
            if state["frozen"] != frozen:
                raise ValueError("cannot resume changed source, configuration, or dataset")
            for saved in state["runs"].values():
                for attempt in saved["attempts"]:
                    if attempt.get("ledger_sha256") and _file_hash(root / attempt["ledger"]) != attempt["ledger_sha256"]:
                        raise ValueError("literature query ledger changed")
        else:
            state = {"schema_version": 1, "status": "running", "frozen": frozen, "runs": {}}
            write_json(state_path, state)
        results = []
        for record_index, record in enumerate(records):
            for budget in cfg["budgets"]:
                for seed in cfg["seeds"]:
                    # Alternate arm order across pairs to limit systematic load bias.
                    methods = ("gp_ucb", "uniform_random") if (record_index + seed) % 2 == 0 else ("uniform_random", "gp_ucb")
                    for method in methods:
                        key = f"record_{record_index:06d}_budget_{budget}_seed_{seed}_{method}"
                        unit = state["runs"].setdefault(key, {"attempts": []})
                        if unit.get("status") == "complete":
                            result_path = root / unit["result"]
                            if _file_hash(result_path) != unit["result_sha256"]:
                                raise ValueError("completed literature artifact changed")
                            for attempt in unit["attempts"]:
                                if attempt.get("ledger_sha256") and _file_hash(root / attempt["ledger"]) != attempt["ledger_sha256"]:
                                    raise ValueError("literature query ledger changed")
                            results.append(json.loads(result_path.read_text()))
                            continue
                        check_frozen()
                        folder = root / "runs" / key
                        folder.mkdir(parents=True, exist_ok=True)
                        attempt_index = len(unit["attempts"]) + 1
                        ledger = folder / f"attempt_{attempt_index:03d}.jsonl"
                        attempt = {"status": "running", "ledger": str(ledger.relative_to(root))}
                        unit["attempts"].append(attempt)
                        write_json(state_path, state)
                        with ledger.open("x", encoding="utf-8") as stream:
                            def receipt(event):
                                if event["event"] == "query_started":
                                    check_frozen()
                                stream.write(json.dumps(event, allow_nan=False) + "\n")
                                stream.flush()
                                os.fsync(stream.fileno())
                            try:
                                result = benchmark_finzgar_record(
                                    record, budget=budget, seed=seed, search_method=method,
                                    event_callback=receipt, allow_test_adaptation=cfg["allow_test_adaptation"],
                                    **cfg["search"], **cfg["simulator"])
                                check_frozen()
                            except BaseException as error:
                                attempt.update(status="interrupted_or_failed", error_type=type(error).__name__,
                                               error=str(error), ledger_sha256=_file_hash(ledger))
                                state["status"] = "interrupted_or_failed"
                                state["cost"] = _campaign_cost(root, state)
                                write_json(state_path, state)
                                raise
                        attempt.update(status="complete", ledger_sha256=_file_hash(ledger))
                        result_path = folder / f"attempt_{attempt_index:03d}.result.json"
                        write_json(result_path, result)
                        unit.update(status="complete", result=str(result_path.relative_to(root)),
                                    result_sha256=_file_hash(result_path))
                        results.append(result)
                        write_json(state_path, state)
        check_frozen()
        summary = summarize_campaign(results, bootstrap_resamples=cfg["bootstrap_resamples"])
        summary["cost"] = _campaign_cost(root, state)
        write_json(root / "summary.json", summary)
        state.update(status="complete", summary="summary.json", summary_sha256=_file_hash(root / "summary.json"),
                     cost=summary["cost"])
        write_json(state_path, state)
        return summary


def run_literature_study(config, output, *, resume=False):
    """CLI/campaign entry point; relative dataset paths follow the config file."""
    base = Path.cwd()
    if isinstance(config, (str, Path)):
        path = Path(config).resolve()
        base, config = path.parent, json.loads(path.read_text())
    config = dict(config)
    data = config.pop("data", None)
    if not isinstance(data, str) or not data:
        raise ValueError("literature study requires a data directory")
    return run_campaign(config, data=base / data, out=output, resume=resume)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="validation")
    parser.add_argument("--allow-test-adaptation", action="store_true")
    parser.add_argument("--parameterization", choices=["policy_decoder", "real"], default="policy_decoder")
    parser.add_argument("--budget", type=int, default=60)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--backend", choices=["numpy", "cupy"], default="numpy")
    parser.add_argument("--tolerance", type=float, default=5e-4)
    parser.add_argument("--initial-steps", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=8192)
    args = parser.parse_args(argv)
    if args.max_records is not None:
        _positive_integer(args.max_records, "max_records")
    from .pipeline import load_records, source_fingerprint
    records = load_records(args.data, args.split)
    if args.max_records is not None:
        records = records[:args.max_records]
    if not records:
        raise ValueError("no records selected")
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest_bytes = (Path(args.data) / "manifest.json").read_bytes()
    provenance = {"manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                  "source_fingerprint_at_start": source_fingerprint(),
                  "dataset_fingerprint": json.loads(manifest_bytes).get("dataset_fingerprint")}
    # Exclusive creation prevents accidental overwriting of an archived experiment.
    with destination.open("x") as stream:
        for record in records:
            for seed in args.seeds:
                result = benchmark_finzgar_record(
                    record, budget=args.budget, seed=seed, parameterization=args.parameterization,
                    allow_test_adaptation=args.allow_test_adaptation, backend=args.backend,
                    tolerance=args.tolerance, initial_steps=args.initial_steps, max_steps=args.max_steps)
                result["provenance"] = provenance
                result["source_fingerprint_at_end"] = source_fingerprint()
                result["source_frozen"] = result["source_fingerprint_at_end"] == provenance["source_fingerprint_at_start"]
                stream.write(json.dumps(result, allow_nan=False) + "\n")
                stream.flush()
                if not result["source_frozen"]:
                    raise RuntimeError("source changed during baseline run; retained row is invalid for the frozen revision")


if __name__ == "__main__":
    main()
