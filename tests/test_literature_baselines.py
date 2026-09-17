"""External-method correspondence, query accounting, and physical adapter gates."""
import json

import numpy as np
import pytest

from annealctrl.literature_baselines import (
    benchmark_finzgar_record, decode_parameters, exploration_weight,
    optimize_finzgar_schedule,
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
