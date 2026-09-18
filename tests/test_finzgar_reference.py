import numpy as np
import pytest
from scipy.integrate import solve_ivp

from annealctrl.finzgar_reference import (
    ReferenceSchedule, reference_real_schedule, run_reference, score_reference, symmetric_pspin,
)


def test_original_real_bounds_allow_nonmonotonicity_and_preserve_endpoints():
    schedule = reference_real_schedule([1., 0., 1., 0.])
    assert schedule(0.) == 0 and schedule(1.) == 1
    assert np.any(np.diff(schedule.s_knots) < 0)
    lower = reference_real_schedule(np.zeros(4)).s_knots[1:-1]
    upper = reference_real_schedule(np.ones(4)).s_knots[1:-1]
    assert np.allclose(lower, (np.arange(1, 5) - 2.) / 5.)
    assert np.allclose(upper, (np.arange(1, 5) + 2.) / 5.)
    with pytest.raises(ValueError):
        ReferenceSchedule([0., 1.], [1., 0.])


def test_symmetric_sector_matches_independent_full_hilbert_evolution():
    n, gamma, runtime = 3, 2., .7
    schedule = reference_real_schedule([1., 0., .7])
    result = score_reference(schedule, n=n, gamma=gamma, runtime=runtime,
                             rtol=1e-10, atol=1e-12, return_state=True)
    dimension = 2 ** n
    full_driver = np.zeros((dimension, dimension))
    for state in range(dimension):
        for bit in range(n):
            full_driver[state, state ^ (1 << bit)] = -gamma
    magnetization = np.array([n - 2 * state.bit_count() for state in range(dimension)])
    diagonal = -n * (magnetization / n) ** 3
    state = np.ones(dimension, complex) / np.sqrt(dimension)
    for left, right in zip(schedule.tau_knots[:-1], schedule.tau_knots[1:]):
        def rhs(tau, vector):
            u = schedule(tau)
            return -1j * runtime * ((1 - u) * full_driver @ vector + u * diagonal * vector)
        state = solve_ivp(rhs, (left, right), state, method="DOP853", rtol=1e-11,
                          atol=1e-13).y[:, -1]
    assert result["fidelity"] == pytest.approx(abs(state[0]) ** 2, abs=1e-9)
    assert result["norm_error"] < 1e-9
    for k in range(n + 1):
        sector = [j for j in range(dimension) if j.bit_count() == k]
        reduced_amplitude = state[sector].sum() / np.sqrt(len(sector))
        assert result["state"][k] == pytest.approx(reduced_amplitude, abs=1e-9)


def test_pauli_sum_normalization_initial_state_and_target():
    driver, target, state = symmetric_pspin(n=5, gamma=5.)
    assert np.allclose(driver @ state, -25 * state)
    assert target[0, 0] == -5 and target[-1, -1] == 5
    assert abs(np.linalg.norm(state) - 1) < 1e-14


def test_linear_fidelity_is_stable_when_integration_tolerance_tightens():
    schedule = reference_real_schedule(np.full(4, .5))
    result = score_reference(schedule)
    tighter = score_reference(schedule, rtol=1e-10, atol=1e-12)
    assert result["fidelity"] == pytest.approx(tighter["fidelity"], abs=2e-9)


def test_paired_initial_design_and_budget():
    result = run_reference(seeds=(7,), budget=3, n=3, runtime=.2)
    bo, random = result["runs"]
    assert result["total_objective_calls"] == 6
    assert result["total_failed_queries"] == 0
    assert [r["parameters"] for r in bo["history"]] == [r["parameters"] for r in random["history"]]
    assert bo["best_loss"] == random["best_loss"]
    assert result["paired_summary"]["mean_difference"] == 0.
    assert result["cost"]["known_trajectory_calls"] == 12
    assert result["setting"] == "custom_original_model_setting"


@pytest.mark.parametrize("settings", [{"seeds": [0, 0]}, {"seeds": []},
                                      {"seeds": [-1]}, {"runtime": 0.}, {"zeta": 0.}])
def test_reference_rejects_invalid_campaign_before_queries(settings):
    events = []
    with pytest.raises(ValueError):
        run_reference(budget=1, event_callback=events.append, **settings)
    assert not events
