"""An actual Bayesian optimiser, because the protocol names its absence.

docs/paper_protocol.md section 6 asks for 'stronger equal-budget classical
optimization, including Bayesian optimization where appropriate' and states
plainly that the shipped random local search is not one. A learned policy
compared only against random search is compared against a weak opponent.
"""
import numpy as np
import pytest

from annealctrl.bayesopt import expected_improvement, fit_gp, minimise


def quadratic(x):
    """Smooth, single optimum at 0.3 in every coordinate."""
    return float(np.sum((np.asarray(x) - 0.3) ** 2))


def rough(x):
    x = np.asarray(x)
    return float(np.sum((x - 0.7) ** 2) + 0.1 * np.sin(20 * x).sum())


# --- the surrogate -----------------------------------------------------------

def test_gp_interpolates_its_training_points():
    rng = np.random.default_rng(0)
    X = rng.random((8, 2))
    y = np.array([quadratic(x) for x in X])
    gp = fit_gp(X, y)
    mean, sd = gp(X)
    np.testing.assert_allclose(mean, y, atol=1e-4)
    assert np.all(sd < 1e-2), "uncertainty must vanish at observed points"


def test_gp_is_uncertain_away_from_its_data():
    rng = np.random.default_rng(1)
    X = rng.random((6, 2)) * 0.2
    y = np.array([quadratic(x) for x in X])
    gp = fit_gp(X, y)
    _, near = gp(X[:1])
    _, far = gp(np.array([[0.95, 0.95]]))
    assert far[0] > near[0]


def test_gp_refuses_mismatched_shapes():
    with pytest.raises(ValueError):
        fit_gp(np.zeros((4, 2)), np.zeros(3))


def test_expected_improvement_is_nonnegative_and_rewards_promise():
    ei = expected_improvement(mean=np.array([0.5, 0.1]), sd=np.array([0.1, 0.1]), best=0.3)
    assert np.all(ei >= 0)
    assert ei[1] > ei[0], "a lower predicted mean must look more promising"


def test_expected_improvement_is_zero_without_uncertainty_or_gain():
    ei = expected_improvement(mean=np.array([0.9]), sd=np.array([0.0]), best=0.3)
    assert ei[0] == pytest.approx(0.0)


# --- the optimiser -----------------------------------------------------------

def test_minimise_spends_exactly_its_budget():
    calls = []

    def counted(x):
        calls.append(x)
        return quadratic(x)

    result = minimise(counted, dimension=3, budget=20, seed=0)
    assert len(calls) == 20
    assert result["n_evaluations"] == 20
    assert len(result["history"]) == 20


def test_minimise_is_deterministic_for_a_seed():
    a = minimise(quadratic, dimension=3, budget=18, seed=5)
    b = minimise(quadratic, dimension=3, budget=18, seed=5)
    assert a["best_value"] == pytest.approx(b["best_value"])
    np.testing.assert_allclose(a["best_parameters"], b["best_parameters"])


def test_minimise_stays_inside_the_unit_box():
    result = minimise(quadratic, dimension=4, budget=24, seed=1)
    for point in result["history"]:
        assert np.all(np.asarray(point["parameters"]) >= 0.0)
        assert np.all(np.asarray(point["parameters"]) <= 1.0)


def test_minimise_beats_random_search_on_a_smooth_objective():
    """The whole reason to add it: it has to actually be stronger."""
    budget, wins = 24, 0
    for seed in range(8):
        bo = minimise(quadratic, dimension=3, budget=budget, seed=seed)
        rng = np.random.default_rng(1000 + seed)
        random_best = min(quadratic(x) for x in rng.random((budget, 3)))
        wins += bo["best_value"] < random_best
    assert wins >= 7, f"Bayesian optimisation won only {wins}/8 against random search"


def test_minimise_also_helps_on_a_rougher_objective():
    wins = 0
    for seed in range(8):
        bo = minimise(rough, dimension=2, budget=24, seed=seed)
        rng = np.random.default_rng(2000 + seed)
        random_best = min(rough(x) for x in rng.random((24, 2)))
        wins += bo["best_value"] < random_best
    assert wins >= 6


def test_minimise_accepts_a_supplied_incumbent_and_charges_it():
    calls = []
    result = minimise(lambda x: calls.append(x) or quadratic(x), dimension=3, budget=10,
                      seed=0, initial=[np.full(3, 0.5)])
    assert len(calls) == 10
    np.testing.assert_allclose(result["history"][0]["parameters"], np.full(3, 0.5))


def test_minimise_refuses_a_budget_below_the_initial_design():
    with pytest.raises(ValueError, match="budget"):
        minimise(quadratic, dimension=3, budget=0, seed=0)


def test_minimise_reports_when_the_budget_was_all_initial_design():
    result = minimise(quadratic, dimension=5, budget=3, seed=0)
    assert result["n_model_steps"] == 0
    assert result["degenerate_to_random_design"] is True
