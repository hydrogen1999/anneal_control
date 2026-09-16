import math

import numpy as np
import pytest

from annealctrl.evaluation import amortized_cost, audit_parent_splits, paired_parent_bootstrap, teacher_relative_regret, time_to_solution


def test_negative_teacher_regret_is_preserved():
    np.testing.assert_allclose(teacher_relative_regret([0.1, 0.3], [0.2, 0.2]), [-0.1, 0.1])


def test_parent_split_leakage():
    assert audit_parent_splits(["a", "a", "b", "c"], ["train", "train", "validation", "test"]) == {"train": 1, "validation": 1, "test": 1}
    with pytest.raises(ValueError, match="leakage"):
        audit_parent_splits(["a", "a"], ["train", "test"])


def test_bootstrap_uses_equal_parent_weight_not_variant_count():
    # Parent a has three correlated variants, b only one. Equal-parent mean=0.
    result = paired_parent_bootstrap([0, 0, 0, 2], [1, 1, 1, 1], ["a", "a", "a", "b"], n_resamples=1000, seed=5)
    assert result.mean_difference == 0
    assert result.n_parents == 2
    assert result.ci_low <= 0 <= result.ci_high
    other = paired_parent_bootstrap([0, 0, 0, 2], [1, 1, 1, 1], ["a", "a", "a", "b"], n_resamples=1000, seed=5)
    assert result == other
    with pytest.raises(ValueError, match="two independent"):
        paired_parent_bootstrap([0, 1], [1, 1], ["a", "a"])


def test_tts_zero_success_is_censored():
    result = time_to_solution(0, 100, 2e-5)
    assert result.censored
    assert math.isinf(result.nominal_seconds)
    assert math.isinf(result.seconds_ci[1])
    assert result.seconds_ci[0] > 0
    assert result.probability_ci[0] == 0


def test_tts_regular_and_all_success():
    result = time_to_solution(50, 100, 1, target_success=0.99)
    assert result.nominal_reads == 7
    assert result.seconds_ci[0] <= 7 <= result.seconds_ci[1]
    assert not result.censored
    certain = time_to_solution(100, 100, 1)
    assert certain.nominal_reads == 1
    assert certain.probability_ci[0] < 1
    assert certain.probability_ci[1] == 1


def test_amortization_keeps_all_costs():
    result = amortized_cost(data_seconds=80, training_seconds=20, deployment_instances=10, inference_seconds=1, online_search_seconds=2, execution_seconds=3, embedding_seconds=4, programming_seconds=5)
    assert result["online_seconds_per_instance"] == 15
    assert result["total_for_one_instance_seconds"] == 115
    assert result["amortized_seconds_per_instance"] == 25
    assert result["deployment_total_seconds"] == 250
