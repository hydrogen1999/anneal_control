import copy

import numpy as np
import pytest

from annealctrl.adapters import CalibratedPath, simulate_lindblad
from annealctrl.physics import AnnealPath, HamiltonianTerms, dense_reference_propagate
from annealctrl.schedules import Schedule


def calibration(**updates):
    data = {"s": [0., .5, 1.], "a": [1., .5, 0.], "b": [0., .5, 1.],
            "c": [0., 0., 0.], "coefficient_unit": "dimensionless",
            "time_unit": "dimensionless", "component_factors": {"a": 1., "b": 1., "c": 1.},
            "calibration_id": "fixture-not-device", "source": "analytic-test"}
    data.update(updates)
    return data


def test_calibration_matches_canonical_path_and_derivative():
    path = CalibratedPath.from_mapping(calibration())
    for s in [0., .2, .5, .8, 1.]:
        np.testing.assert_allclose(path.coefficients(s), AnnealPath().coefficients(s))
        np.testing.assert_allclose(path.derivative(s), [-1., 1., 0.])


def test_ghz_hbar_and_half_factor_are_explicit():
    path = CalibratedPath.from_mapping(calibration(coefficient_unit="GHz", time_unit="us",
                                                  component_factors={"a": .5, "b": .5, "c": 1.}))
    np.testing.assert_allclose(path.coefficients(.5), [.5 * np.pi * 1000, .5 * np.pi * 1000, 0.])
    angular = CalibratedPath.from_mapping(calibration(coefficient_unit="rad/us", time_unit="us"))
    np.testing.assert_allclose(angular.coefficients(.5), [.5, .5, 0.])


@pytest.mark.parametrize("updates", [
    {"s": [.1, .5, 1.]}, {"s": [0., 0., 1.]}, {"s": [0., .5, .9]},
    {"a": [1., 0.]}, {"a": [1., np.nan, 0.]}, {"source": ""},
    {"time_unit": "us"}, {"coefficient_unit": "GHz"},
    {"component_factors": {"a": 1., "b": 1.}}, {"coefficient_unit": "eV"},
])
def test_calibration_rejects_ambiguous_or_invalid_inputs(updates):
    with pytest.raises(ValueError):
        CalibratedPath.from_mapping(calibration(**updates))


def test_calibration_requires_factors_and_refuses_extrapolation(tmp_path):
    import json
    data = calibration()
    data.pop("component_factors")
    with pytest.raises(ValueError, match="missing"):
        CalibratedPath.from_mapping(data)
    target = tmp_path / "path.json"
    target.write_text(json.dumps(calibration()))
    path = CalibratedPath.from_json(target)
    for value in [-.1, 1.1, np.nan]:
        with pytest.raises(ValueError, match="domain"):
            path.coefficients(value)


@pytest.mark.parametrize("catalyst", [0., .3])
def test_lindblad_zero_noise_agrees_with_independent_state_solver(catalyst):
    terms = HamiltonianTerms(2, [.3, -.1], [[0, 1]], [-.7], [[0, 1]], [.2])
    path = AnnealPath(catalyst)
    schedule = Schedule([0., .3, .7, 1.], [0., .4, .4, 1.])
    density = simulate_lindblad(terms, schedule, 1.4, path=path)
    state = dense_reference_propagate(terms, schedule, 1.4, path=path).state
    np.testing.assert_allclose(density.density, np.outer(state, state.conj()), atol=5e-8)
    assert density.diagnostics["physicality_passed"]
    assert density.diagnostics["state_repaired"] is False


def test_pure_dephasing_analytic_coherence_decay():
    terms = HamiltonianTerms(1, [0.], [], [])
    psi = np.ones(2) / np.sqrt(2)
    result = simulate_lindblad(terms, lambda _: 1., 2., initial_state=psi, dephasing_rates=.4)
    np.testing.assert_allclose(result.density, [[.5, .5 * np.exp(-1.6)], [.5 * np.exp(-1.6), .5]], atol=1e-10)


def test_relaxation_analytic_and_lsb_order():
    terms = HamiltonianTerms(2, [0., 0.], [], [])
    # |q1 q0> = |01>: relaxation of q0 reaches |00>; q1 is untouched.
    psi = np.array([0., 1., 0., 0.])
    result = simulate_lindblad(terms, lambda _: 1., 2., initial_state=psi, relaxation_rates=[.4, 0.])
    np.testing.assert_allclose(result.probabilities, [1 - np.exp(-.8), np.exp(-.8), 0., 0.], atol=1e-10)


def test_density_initialization_zero_time_and_calibrated_units():
    terms = HamiltonianTerms(1, [.2], [], [])
    mixed = np.diag([.3, .7])
    result = simulate_lindblad(terms, Schedule.linear(), 0., initial_density=mixed)
    np.testing.assert_array_equal(result.density, mixed)
    path = CalibratedPath.from_mapping(calibration(coefficient_unit="rad/us", time_unit="us"))
    with pytest.raises(ValueError, match="rates_unit"):
        simulate_lindblad(terms, Schedule.linear(), 1., path=path)
    physical = simulate_lindblad(terms, Schedule.linear(), 1., path=path, rates_unit="1/us")
    assert physical.diagnostics["time_unit"] == "us"


@pytest.mark.parametrize("initial", [np.eye(2), np.diag([1.2, -.2]), [[.5, .2], [0., .5]], [[np.nan, 0.], [0., 1.]]])
def test_invalid_density_is_not_repaired(initial):
    with pytest.raises(ValueError, match="initial_density"):
        simulate_lindblad(HamiltonianTerms(1, [0.], [], []), Schedule.linear(), 1., initial_density=initial)


def test_rate_initialization_and_memory_guards():
    terms = HamiltonianTerms(1, [0.], [], [])
    with pytest.raises(ValueError, match="nonnegative"):
        simulate_lindblad(terms, Schedule.linear(), 1., dephasing_rates=-1.)
    with pytest.raises(ValueError, match="normalized"):
        simulate_lindblad(terms, Schedule.linear(), 1., initial_state=[1., 1.])
    with pytest.raises(ValueError, match="pure-X"):
        simulate_lindblad(terms, lambda _: .5, 1.)
    with pytest.raises(ValueError, match="only"):
        simulate_lindblad(terms, Schedule.linear(), 1., initial_state=[1., 0.], initial_density=np.eye(2) / 2)
    with pytest.raises(MemoryError):
        simulate_lindblad(HamiltonianTerms(9, np.zeros(9), [], []), Schedule.linear(), 1.)


def test_calibrated_noncanonical_initial_hamiltonian_requires_explicit_state():
    path = CalibratedPath.from_mapping(calibration(b=[.1, .5, 1.]))
    with pytest.raises(ValueError, match="pure-X"):
        simulate_lindblad(HamiltonianTerms(1, [1.], [], []), Schedule.linear(), 1., path=path)
