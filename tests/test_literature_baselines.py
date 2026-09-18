"""External-method correspondence, query accounting, and physical adapter gates."""
import json

import numpy as np
import pytest

from annealctrl.literature_baselines import (
    benchmark_finzgar_record, decode_parameters, exploration_weight,
    optimize_finzgar_schedule, run_campaign, validate_campaign,
)


def _fast(**kwargs):
    return dict(n_segments=3, n_initial=4, acquisition_candidates=24,
                acquisition_restarts=1, gp_restarts=1, **kwargs)


def test_ucb_decay_matches_50_adaptive_step_protocol():
    values = [exploration_weight(i, 50) for i in range(50)]
    assert values[:25] == [2.] * 25
    assert values[-1] == pytest.approx(.01)
    assert np.all(np.diff(values[25:]) < 0)
    assert np.allclose(np.asarray(values[26:]) / values[25:-1], (.01 / 2.) ** (1 / 25))


@pytest.mark.parametrize("parameterization", ["real", "policy_decoder"])
def test_decoder_whole_box_is_forward_feasible(parameterization):
    rng = np.random.default_rng(12)
    points = np.vstack([np.zeros(7), np.ones(7), rng.random((40, 7))])
    for point in points:
        schedule = decode_parameters(point, parameterization=parameterization)
        schedule.validate_slope(runtime=1., max_slope=4.)
        assert np.all(np.diff(schedule.s_knots) >= 0)
    linear = decode_parameters(np.full(7, .5), parameterization=parameterization)
    assert np.allclose(linear.s_knots, linear.tau_knots)


def test_original_nonmonotonic_box_is_not_silently_projected():
    with pytest.raises(ValueError, match="real_zeta"):
        decode_parameters([.5, .5], parameterization="real", real_zeta=2.)


def test_budget_initialization_and_reproducible_bo():
    calls = []
    def objective(schedule):
        calls.append(schedule)
        return float((schedule(.5) - .2) ** 2)
    result = optimize_finzgar_schedule(objective, budget=8, seed=3, **_fast())
    assert len(calls) == result["n_objective_calls"] == 8
    assert result["n_initial_calls"] == 4
    assert [r["source"] for r in result["history"][:4]] == ["linear"] + ["random_initial"] * 3
    assert all(r["source"] == "lower_confidence_bound" for r in result["history"][4:])
    assert np.all(np.diff([r["best_so_far"] for r in result["history"]]) <= 0)
    repeat = optimize_finzgar_schedule(objective, budget=8, seed=3, **_fast())
    assert [r["loss"] for r in result["history"]] == [r["loss"] for r in repeat["history"]]
    assert result["best_loss"] < result["history"][0]["loss"]
    json.dumps(result, allow_nan=False)


def test_failed_queries_are_charged_and_not_fabricated_labels():
    calls = 0
    def objective(schedule):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ArithmeticError("convergence gate")
        if calls == 2:
            return float("nan")
        return .4
    result = optimize_finzgar_schedule(objective, budget=6, **_fast())
    assert calls == 6
    assert result["n_failed_calls"] == 2
    assert result["history"][0]["best_so_far"] is None
    assert result["history"][1]["loss"] is None
    assert result["best_loss"] == .4
    json.dumps(result, allow_nan=False)


def test_all_failed_queries_return_no_incumbent():
    def objective(schedule):
        raise ArithmeticError("failed")
    result = optimize_finzgar_schedule(objective, budget=5, **_fast())
    assert result["n_failed_calls"] == 5
    assert result["best_waveform"] is result["best_loss"] is None
    assert result["history"][-1]["source"] == "random_insufficient_valid_labels"


def test_invalid_configuration_is_not_hidden_as_a_failed_query():
    def objective(schedule):
        raise ValueError("invalid physical record")
    with pytest.raises(ValueError, match="invalid physical record"):
        optimize_finzgar_schedule(objective, budget=3, **_fast())


def test_test_adaptation_is_explicit():
    with pytest.raises(ValueError, match="allow_test_adaptation"):
        optimize_finzgar_schedule(lambda s: 0., split="test")
    result = optimize_finzgar_schedule(lambda s: 0., split="test", allow_test_adaptation=True, budget=1)
    assert result["online_adaptation"]


def _record():
    from annealctrl.generation import Embedding, IsingProblem, compile_embedding
    logical = IsingProblem(np.array([.3, -.2]), np.array([[0, 1]]), np.array([-.6]))
    embedding = Embedding(np.array([0, 1]), np.array([[0, 1]]))
    compiled = compile_embedding(logical, embedding, 1., np.random.default_rng(0))
    return dict(record_id="r0", parent_id="p0", split="validation", family="test",
                logical_h=logical.h, logical_edges=logical.edges, logical_J=logical.J,
                physical_h=compiled.physical.h, physical_edges=compiled.physical.edges,
                physical_J=compiled.physical.J, problem_J=compiled.problem_J,
                chain_J=compiled.chain_J, membership=embedding.membership, runtime=2.,
                programmed_scale=compiled.programmed_scale, chain_strength=1.)


def test_adapter_preserves_scores_and_costs():
    result = benchmark_finzgar_record(_record(), budget=3, initial_steps=16, max_steps=2048, **_fast())
    assert result["cost"]["objective_calls"] == 3
    assert result["cost"]["known_propagation_calls"] > 0
    assert not result["cost"]["inner_cost_is_lower_bound"]
    assert result["history"][0]["metrics"]["state_error_diagnostic"] <= 5e-4
    assert result["identity"]["physical_n"] == 2
    json.dumps(result, allow_nan=False)


def test_record_split_cannot_be_overridden_to_bypass_test_gate():
    record = _record()
    record["split"] = "test"
    with pytest.raises(ValueError, match="override"):
        benchmark_finzgar_record(record, split="train", budget=1)


def test_warm_hint_is_charged_and_preserves_other_initial_draws():
    cold = optimize_finzgar_schedule(lambda s: .5, budget=4, seed=3, **_fast())
    warm = optimize_finzgar_schedule(lambda s: .5, budget=4, seed=3,
                                    initial_point=[.1, .2, .3], **_fast())
    assert warm["n_objective_calls"] == cold["n_objective_calls"] == 4
    assert warm["history"][0]["source"] == "linear"
    assert warm["history"][1]["source"] == "warm_start"
    assert warm["history"][1]["parameters"] == [.1, .2, .3]
    assert [r["parameters"] for r in warm["history"][2:]] == [r["parameters"] for r in cold["history"][2:]]
    with pytest.raises(ValueError, match="initial_point"):
        optimize_finzgar_schedule(lambda s: 0., budget=1, initial_point=[.1, .2, .3], **_fast())


def test_matched_uniform_initial_design_and_budget():
    bo = optimize_finzgar_schedule(lambda s: float(s(.5)), budget=7, seed=12, **_fast())
    random = optimize_finzgar_schedule(lambda s: float(s(.5)), budget=7, seed=12,
                                      search_method="uniform_random", **_fast())
    assert bo["n_objective_calls"] == random["n_objective_calls"] == 7
    assert [r["parameters"] for r in bo["history"][:4]] == [r["parameters"] for r in random["history"][:4]]
    assert random["n_acquisition_queries"] == 0
    assert random["history"][4]["source"] == "uniform_random"


def test_query_receipts_preserve_programming_error_attempt():
    events = []
    def broken(schedule):
        raise ValueError("bad physical data")
    with pytest.raises(ValueError, match="bad physical"):
        optimize_finzgar_schedule(broken, budget=3, event_callback=events.append, **_fast())
    assert [event["event"] for event in events] == ["query_started", "query_aborted"]
    assert events[-1]["evaluation"] == 1


@pytest.mark.parametrize("config", [
    {"seeds": [0, 0]}, {"budgets": [0]}, {"seeds": [True]},
    {"split": "test"}, {"search": {"unknown": 3}}, {"max_parents": 0},
])
def test_campaign_rejects_invalid_protocol_before_queries(config):
    with pytest.raises(ValueError):
        validate_campaign(config)


def _campaign_fixture(tmp_path, monkeypatch):
    import annealctrl.pipeline as pipeline
    data = tmp_path / "data"
    data.mkdir()
    (data / "manifest.json").write_text('{"status":"complete"}')
    monkeypatch.setattr(pipeline, "load_records", lambda path, split: [_record()])
    monkeypatch.setattr(pipeline, "source_fingerprint", lambda: "frozen-source")
    config = {"seeds": [0], "budgets": [3], "search": _fast(),
              "simulator": {"initial_steps": 16, "max_steps": 2048}, "bootstrap_resamples": 20}
    return data, config


def test_campaign_resume_does_not_query_completed_results(tmp_path, monkeypatch):
    data, config = _campaign_fixture(tmp_path, monkeypatch)
    out = tmp_path / "campaign"
    result = run_campaign(config, data=data, out=out)
    assert result["cost"]["completed_objective_calls_all_attempts"] == 6
    assert result["cost"]["unresolved_call_reservations"] == 0
    import annealctrl.benchmarking as benchmarking
    monkeypatch.setattr(benchmarking, "score_schedule", lambda *a, **k: pytest.fail("resume rescored a finished result"))
    assert run_campaign(config, data=data, out=out, resume=True) == result
    with pytest.raises(ValueError, match="changed source, configuration"):
        run_campaign({**config, "budgets": [4]}, data=data, out=out, resume=True)


def test_campaign_retry_preserves_failed_attempt_cost(tmp_path, monkeypatch):
    data, config = _campaign_fixture(tmp_path, monkeypatch)
    out = tmp_path / "campaign"
    import annealctrl.benchmarking as benchmarking
    original = benchmarking.score_schedule
    calls = 0
    def broken_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("injected interruption")
        return original(*args, **kwargs)
    monkeypatch.setattr(benchmarking, "score_schedule", broken_second)
    with pytest.raises(ValueError, match="injected"):
        run_campaign(config, data=data, out=out)
    monkeypatch.setattr(benchmarking, "score_schedule", original)
    result = run_campaign(config, data=data, out=out, resume=True)
    assert result["cost"]["reserved_objective_calls_all_attempts"] == 8
    assert result["cost"]["completed_objective_calls_all_attempts"] == 7
    assert result["cost"]["unresolved_call_reservations"] == 1
    assert result["cost"]["inner_cost_is_lower_bound"]
