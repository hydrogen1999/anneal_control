import numpy as np
import pytest

from annealctrl.search import equal_budget_refinement, evaluate_candidates, finite_difference_interventions, shared_candidate_bank


def loss_fn(schedule):
    return float((schedule(0.5) - 0.3) ** 2)


def test_bank_reproducible_and_diverse():
    a = shared_candidate_bank(n=64, n_segments=8, seed=7)
    b = shared_candidate_bank(n=64, n_segments=8, seed=7)
    assert len(a) == 64
    assert {x.family for x in a} == {"linear", "one_window", "two_window", "pause", "duration_logits"}
    for left, right in zip(a, b):
        assert left.candidate_id == right.candidate_id
        np.testing.assert_array_equal(left.schedule.tau_knots, right.schedule.tau_knots)
        left.schedule.validate_slope(1, 10)
    assert len(shared_candidate_bank(n=1)) == 1
    assert "pause" not in {x.family for x in shared_candidate_bank(n=16, runtime=0.1, max_slope=10)}


def test_evaluation_counts_every_call():
    calls = []
    def counted(schedule):
        calls.append(schedule)
        return loss_fn(schedule)
    result = evaluate_candidates(counted, shared_candidate_bank(16), budget=9, split="test")
    assert result.n_evaluations == len(calls) == 9
    assert result.best.loss == min(record.loss for record in result.records)
    assert result.reference_status == "best_found_within_evaluated_candidates"
    assert not result.online_adaptation
    with pytest.raises(ValueError):
        evaluate_candidates(counted, shared_candidate_bank(2), budget=3)


def test_refinement_budget_and_test_guard():
    calls = []
    def counted(schedule):
        calls.append(schedule)
        return loss_fn(schedule)
    result = equal_budget_refinement(counted, [0, 0, 0, 0], budget=11, seed=4)
    assert result.n_evaluations == len(calls) == 11
    assert result.best.loss <= result.records[0].loss
    with pytest.raises(ValueError, match="explicitly"):
        equal_budget_refinement(counted, [0, 0], budget=3, split="test")
    result = equal_budget_refinement(counted, [0, 0], budget=3, split="test", allow_test_adaptation=True)
    assert result.online_adaptation


def test_interventions_are_feasible_counted_and_schedule_conditioned():
    calls = []
    def objective(schedule):
        schedule.validate_slope(2, 1)
        calls.append(schedule)
        # First segment duration has a known derivative through softmax.
        return schedule.tau_knots[1]
    logits = np.array([0.3, -0.2, 0.1])
    labels = finite_difference_interventions(objective, logits, runtime=2, max_slope=1, epsilon=1e-5)
    p = np.exp(logits) / np.exp(logits).sum()
    expected = 0.5 * p[0] * (np.eye(3)[0] - p)
    np.testing.assert_allclose(labels.derivatives, expected, atol=1e-10)
    assert labels.n_evaluations == len(calls) == 6
    np.testing.assert_array_equal(labels.baseline_logits, logits)
    assert labels.runtime == 2
