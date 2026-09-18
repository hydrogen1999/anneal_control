"""Scientific correctness gates for equal-budget warm-start studies."""
import json

import numpy as np
import pytest

from annealctrl.budget_study import (
    ObjectiveLedger, budget_study_report, equal_quality_break_even,
    run_search_trajectory, trajectory_points, validate_budget_config,
)
from annealctrl.schedules import Schedule, decode_durations, pause_schedule
from annealctrl.search import decode_eight_bin, eight_bin_hint, optimize_control_family


def test_inverse_recovers_eligible_duration_schedule():
    schedule = decode_durations(np.arange(8) * .23, runtime=3., max_slope=4 / 3)
    mapped = eight_bin_hint(schedule, runtime=3., max_slope=4 / 3)
    reconstructed = decode_eight_bin(mapped["unit_parameters"], runtime=3., max_slope=4 / 3)
    union = np.union1d(schedule.tau_knots, reconstructed.tau_knots)
    assert np.max(np.abs(schedule(union) - reconstructed(union))) < 1e-12
    assert mapped["exact_within_tolerance"]


def test_policy_knots_and_pause_are_not_silently_misrepresented():
    schedule = Schedule([0, .3, .75, 1], [0, .17, .87, 1])
    with pytest.raises(ValueError, match="not exactly representable"):
        eight_bin_hint(schedule)
    mapped = eight_bin_hint(schedule, mode="project")
    assert mapped["sup_waveform_error"] > 1e-3
    assert mapped["evaluation_required"]
    for wave in (schedule, pause_schedule(.45, .2, max_slope=4)):
        projected = eight_bin_hint(wave, mode="project")
        assert projected["sup_waveform_error"] > 0
        Schedule(**projected["waveform"]).validate_slope(1, 4)


@pytest.mark.parametrize("bad", [np.full(8, np.nan), np.full(8, np.inf), np.full(8, -.1), np.full(8, 1.1), np.zeros(7)])
def test_bad_warm_hints_fail_before_objective(bad):
    calls = []
    with pytest.raises(ValueError, match="warm_start"):
        optimize_control_family(lambda s: calls.append(s) or .5, "eight_bin", budget=8, warm_start=bad)
    assert not calls


@pytest.mark.parametrize("strategy", ["sobol_local", "bayesian", "finzgar_gp_ucb"])
def test_every_optimizer_charges_linear_and_warm_start_inside_budget(tmp_path, strategy):
    expected = decode_eight_bin(np.linspace(.2, .7, 8))
    seen = []
    def objective(schedule):
        seen.append(schedule)
        return {"loss": float((schedule(.3) - .1) ** 2), "propagation_calls": 2}
    ledger = ObjectiveLedger(tmp_path / f"{strategy}.jsonl", objective)
    result = run_search_trajectory(ledger, strategy=strategy, horizon=7, seed=11,
                                  hint=np.linspace(.2, .7, 8),
                                  literature={"n_initial": 3, "acquisition_candidates": 12,
                                              "acquisition_restarts": 1, "gp_restarts": 1})
    assert result["status"] == "complete"
    assert len(seen) == 7
    assert seen[0](.3) == pytest.approx(.3)
    assert seen[1](.3) == pytest.approx(expected(.3))
    assert ledger.costs()["attempted_objective_calls"] == 7
    assert ledger.costs()["known_propagation_calls"] == 14
    points = trajectory_points(ledger.outcome_rows(), [0, 1, 2, 7])
    assert [p["online_queries"] for p in points] == [1, 2, 7]
    assert points[-1]["selected_loss"] <= points[0]["selected_loss"]


def test_completed_queries_replay_exactly_without_rescoring(tmp_path):
    calls = []
    def objective(schedule):
        calls.append(schedule)
        return {"loss": float(schedule(.25))}
    path = tmp_path / "queries.jsonl"
    first = ObjectiveLedger(path, objective)
    run_search_trajectory(first, strategy="bayesian", horizon=5, seed=3)
    replay = ObjectiveLedger(path, objective, resume=True)
    run_search_trajectory(replay, strategy="bayesian", horizon=5, seed=3)
    assert len(calls) == 5
    assert replay.costs()["replayed_without_query"] == 5
    assert [r["loss"] for r in replay.outcome_rows()] == [r["loss"] for r in first.outcome_rows()]
    bad = ObjectiveLedger(path, objective, resume=True)
    with pytest.raises(ValueError, match="different waveform"):
        bad(decode_eight_bin(np.linspace(.2, .7, 8)))


def test_interrupted_queries_are_retained_and_retry_is_not_equal_budget(tmp_path):
    path = tmp_path / "queries.jsonl"
    interrupted = ObjectiveLedger(path, lambda s: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        interrupted(Schedule.linear())
    retry = ObjectiveLedger(path, lambda s: {"loss": .5}, resume=True)
    retry(Schedule.linear())
    costs = retry.costs()
    assert costs["attempted_objective_calls"] == 2
    assert costs["interrupted_calls"] == 1
    assert costs["inner_cost_is_lower_bound"]
    point = trajectory_points(retry.outcome_rows(), [1])[0]
    assert point["online_queries"] == 2
    assert point["status"] == "censored"
    assert point["online_wall_seconds"] is None


def test_numerical_failure_is_charged_and_not_retried(tmp_path):
    calls = []
    def fail(schedule):
        calls.append(schedule)
        raise ArithmeticError("failed convergence")
    ledger = ObjectiveLedger(tmp_path / "fail.jsonl", fail)
    result = run_search_trajectory(ledger, strategy="sobol_local", horizon=5, seed=0)
    assert result["status"] == "numerically_censored"
    replay = ObjectiveLedger(tmp_path / "fail.jsonl", fail, resume=True)
    run_search_trajectory(replay, strategy="sobol_local", horizon=5, seed=0)
    assert len(calls) == 1
    assert replay.costs()["failed_calls"] == 1
    assert replay.costs()["inner_cost_is_lower_bound"]


def test_break_even_requires_known_cost_and_common_quality():
    args = dict(offline_seconds=100., learned_seconds=1., baseline_seconds=3., learned_loss=.4,
                baseline_loss=.41, quality_threshold=.42)
    assert equal_quality_break_even(**args)["instances"] == 50
    assert equal_quality_break_even(**{**args, "offline_seconds": None})["instances"] is None
    assert equal_quality_break_even(**{**args, "learned_loss": .5})["instances"] is None
    assert equal_quality_break_even(**{**args, "baseline_seconds": .5})["instances"] is None


def test_paired_parent_report_excludes_censored_rows_and_never_claims_equivalence():
    rows = []
    for p in range(3):
        for seed in range(2):
            for mode, loss in [("cold", .5), ("direct", .4)]:
                rows.append({"record_id": f"r{p}", "logical_fingerprint": f"p{p}", "seed": seed,
                             "method": f"sobol_local/{mode}", "budget": 3, "online_queries": 3,
                             "selected_loss": loss, "status": "ok", "online_wall_seconds": 2.})
    rows[-1]["status"] = "censored"
    report = budget_study_report(rows, bootstrap_resamples=50)
    contrast = report["paired_warm_minus_cold"][0]
    assert contrast["matched_rows"] == 5
    assert contrast["mean_difference"] == pytest.approx(-.1)
    direct = [s for s in report["curves"] if s["method"].endswith("direct")][0]
    assert direct["quality_queries_pareto"] is None
    assert report["offline_cost"]["total_seconds"] is None


def test_config_refuses_silent_unfair_budget_or_unknown_settings():
    base = dict(source_data="data", checkpoint="model.pt", budgets=[0, 1, 3], seeds=[0], allow_test_adaptation=True)
    validate_budget_config(base)
    for patch in ({"budgets": [0, 1]}, {"budgets": [0, 3, 1]}, {"budgets": [0, True, 3]},
                  {"allow_test_adaptation": False}, {"family": "pause"}, {"mystery": True},
                  {"numerics": {"toleranse": .1}}):
        with pytest.raises(ValueError):
            validate_budget_config({**base, **patch})


def test_runner_scores_fixed_selections_separately_and_resumes_without_queries(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from annealctrl import budget_study, benchmarking
    from annealctrl.learning import fit_records
    from annealctrl.pipeline import generate_dataset, load_records
    torch.set_num_threads(1)
    data, output, checkpoint = tmp_path / "data", tmp_path / "study", tmp_path / "best.pt"
    generate_dataset({"seed": 73011, "parents": 10, "families": ["spin_glass"], "logical_qubits": 3,
                    "chain_lengths": [1, 1, 1], "variants": [{"shape": "path", "ports": 1}],
                    "chain_strengths": [1.5], "runtimes": [2.], "candidates": 4,
                    "spectral_points": 3, "teacher": {"mode": "none"}, "steps": 16,
                    "max_steps": 1024, "label_state_tolerance": .005, "max_physical_qubits": 3}, data)
    fit_records(load_records(data, "train"), load_records(data, "validation"), epochs=1,
                model_config={"encoder_variant": "summary", "width": 8, "proposals": 3},
                checkpoint=checkpoint, response_weight=0.)
    cfg = {"source_data": str(data), "checkpoint": str(checkpoint), "budgets": [0, 1, 2, 3],
           "seeds": [0, 1], "allow_test_adaptation": True, "strategies": ["sobol_local"],
           "warm_modes": ["direct", "bank", "source_global"], "hint_mode": "project", "max_records": 1,
           "execution": {"device": "cpu", "backend": "numpy", "threads": 1},
           "numerics": {"tolerance": .005, "initial_steps": 16, "max_steps": 1024},
           "bootstrap_resamples": 20, "latency": {"warmup": 0, "repeats": 1}}
    report = budget_study.run_budget_study(cfg, output)
    assert report["n_records"] == 1
    assert len(report["curves"]) == 4 + 4 * 3
    costs = report["cost_ledger"]
    assert sum(c["attempted_objective_calls"] for c in costs if c["role"] == "online_search") == 24
    assert sum(c["attempted_objective_calls"] for c in costs if c["role"] == "offline_diagnostic") == 4
    rows = json.loads((output / "rows.json").read_text())
    assert all(r["online_queries"] == 0 for r in rows if r["budget"] == 0)
    assert all(r["online_queries"] == r["budget"] for r in rows if r["budget"] > 0)
    monkeypatch.setattr(benchmarking, "score_schedule", lambda *a, **k: pytest.fail("resume reran simulator"))
    assert budget_study.run_budget_study(cfg, output, resume=True) == report
    with pytest.raises(ValueError, match="changed"):
        budget_study.run_budget_study({**cfg, "seeds": [7]}, output, resume=True)
    artifact = next((output / "records").glob("*/zero_direct.json"))
    artifact.write_text("{}")
    with pytest.raises(ValueError, match="artifact changed"):
        budget_study.run_budget_study(cfg, output, resume=True)


def test_replayed_programming_failure_never_becomes_numerical_censoring(tmp_path):
    def wrong(schedule):
        raise ValueError("invalid configuration")
    path = tmp_path / "wrong.jsonl"
    ledger = ObjectiveLedger(path, wrong)
    with pytest.raises(ValueError, match="invalid configuration"):
        ledger(Schedule.linear())
    replay = ObjectiveLedger(path, wrong, resume=True)
    with pytest.raises(ValueError, match="nonnumerical"):
        run_search_trajectory(replay, strategy="sobol_local", horizon=3, seed=0)


def test_automatic_offline_cost_refuses_missing_timing_or_unproven_fixed_recipe(tmp_path, monkeypatch):
    from annealctrl import acquisition_study
    from annealctrl.budget_study import _acquisition_offline_cost, _sha
    data = tmp_path / "data"
    data.mkdir()
    (data / "manifest.json").write_text(json.dumps({"status": "complete", "elapsed_seconds": 10.,
                                                  "generation_cost_complete": True}))
    state = {"trained": True, "training_seconds": 3., "training_cost_complete": True}
    study = {"data_dir": str(data), "data_manifest_sha256": _sha(data / "manifest.json"),
             "training_complete": True, "config": {}, "runs": {"baseline": state}}
    (tmp_path / "study.json").write_text(json.dumps(study))
    monkeypatch.setattr(acquisition_study, "plan_study", lambda cfg: {"training_runs": 1})
    args = {"acquisition_study": str(tmp_path), "fixed_recipe": True}
    complete = _acquisition_offline_cost(args)
    assert complete["total_seconds"] == 13.
    assert complete["status"] == "complete"
    assert _acquisition_offline_cost({**args, "fixed_recipe": False})["total_seconds"] is None
    state["training_cost_complete"] = False
    (tmp_path / "study.json").write_text(json.dumps(study))
    assert _acquisition_offline_cost(args)["status"] == "unknown"
