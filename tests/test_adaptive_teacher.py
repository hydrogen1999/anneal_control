import json
from dataclasses import replace

import numpy as np
import pytest

import annealctrl.spectral as spectral
from annealctrl.physics import (
    AnnealPath, HamiltonianOperator, HamiltonianTerms, backend_device_info,
    dense_reference_propagate, estimate_state_workspace, propagate, propagate_batch,
)


def one_qubit(field=1.):
    return HamiltonianTerms(1, [field], [], [])


def two_qubits():
    return HamiltonianTerms(2, [.4, -.3], [[0, 1]], [-.7], [[0, 1]], [.2])


def test_adaptive_labels_match_analytic_one_qubit_full_response():
    field = .8
    result = spectral.adaptive_spectral_profile(one_qubit(field), max_queries=65, audit_points=8)
    for point in result.points + result.audit_points:
        energy = np.sqrt((1 - point.s) ** 2 + (field * point.s) ** 2)
        np.testing.assert_allclose(point.raw_gap, 2 * energy, atol=1e-13)
        np.testing.assert_allclose(point.mu0, field**2 / energy**2, atol=1e-13)
        np.testing.assert_allclose(point.g_ss, field**2 / (4 * energy**4), atol=1e-13)
        np.testing.assert_allclose(point.d2, field / (4 * energy**3), atol=1e-13)
    assert result.diagnostics["sampled_checks_passed"]
    assert not result.diagnostics["uniform_certificate"]
    assert result.diagnostics["query_count"] <= 65


def test_adaptive_allocates_queries_to_curvature_and_reserves_audit_budget():
    result = spectral.adaptive_spectral_profile(one_qubit(.08), initial_grid=[0., 1.],
                                                max_queries=45, audit_points=5,
                                                relative_tolerance=.01)
    locations = np.asarray([point.s for point in result.points])
    assert len(locations) > 3
    assert np.diff(locations).min() < np.diff(locations).max() / 2
    assert result.diagnostics["query_count"] <= 45
    assert len(result.audit_points) == 5
    assert result.diagnostics["profile_query_count"] == len(locations)


def test_insufficient_query_budget_reports_unchecked_intervals_without_lying():
    result = spectral.adaptive_spectral_profile(one_qubit(), max_queries=7, audit_points=2)
    assert len(result.points) == 5
    assert result.diagnostics["query_count"] == 7
    assert result.diagnostics["budget_exhausted"]
    assert not result.diagnostics["sampled_refinement_converged"]
    assert not result.diagnostics["sampled_checks_passed"]
    assert all(item["reasons"] == ["unchecked_budget"] for item in result.diagnostics["intervals"])


def test_rank_change_is_refined_and_remains_explicit():
    result = spectral.adaptive_spectral_profile(one_qubit(0.), max_queries=21, audit_points=2)
    assert result.points[-1].ground_rank == 2
    assert result.points[-2].s > .99
    reasons = [reason for item in result.diagnostics["intervals"] for reason in item["reasons"]]
    assert "ground_rank_change" in reasons or "unchecked_budget" in reasons
    assert not result.diagnostics["sampled_refinement_converged"]
    assert result.diagnostics["budget_exhausted"]


def test_minimum_width_stops_rank_change_without_claiming_convergence():
    result = spectral.adaptive_spectral_profile(one_qubit(0.), max_queries=100,
                                                audit_points=2, min_interval_width=.1)
    assert result.diagnostics["stop_reason"] == "minimum_interval_width"
    assert not result.diagnostics["budget_exhausted"]
    assert not result.diagnostics["sampled_checks_passed"]
    failing = [item for item in result.diagnostics["intervals"] if item["needs_refinement"]]
    assert all(item["right"] - item["left"] <= .1 for item in failing)


def test_audit_seed_does_not_change_adaptive_training_queries():
    first = spectral.adaptive_spectral_profile(one_qubit(), seed=1)
    second = spectral.adaptive_spectral_profile(one_qubit(), seed=2)
    assert [point.s for point in first.points] == [point.s for point in second.points]
    assert [point.s for point in first.audit_points] != [point.s for point in second.audit_points]
    assert set(point.s for point in first.points).isdisjoint(point.s for point in first.audit_points)
    repeated = spectral.adaptive_spectral_profile(one_qubit(), seed=1)
    assert first.to_dict() == repeated.to_dict()


def test_independent_audit_can_fail_after_midpoint_checks_pass(monkeypatch):
    # A deliberately aliased response curve demonstrates why a sampled test
    # cannot be advertised as a uniform certificate. Audit cannot repair it.
    reference = spectral.spectral_teacher(one_qubit(), 0.)

    def aliased(terms, s, **kwargs):
        value = float(np.sin(8 * np.pi * s) ** 2)
        return replace(reference, s=float(s), mu0=value, g_ss=value, d2=value)

    monkeypatch.setattr(spectral, "spectral_teacher", aliased)
    result = spectral.adaptive_spectral_profile(one_qubit(), max_queries=25, audit_points=8)
    assert result.diagnostics["sampled_refinement_converged"]
    assert result.diagnostics["audit_passed"] is False
    assert result.diagnostics["sampled_checks_passed"] is False
    assert len(result.points) == 9


def test_numerical_failure_masks_inverse_moments_and_serializes_strict_json(monkeypatch):
    original = spectral.spectral_teacher

    def bad_numerics(*args, **kwargs):
        return replace(original(*args, **kwargs), max_eigenpair_residual=.1)

    monkeypatch.setattr(spectral, "spectral_teacher", bad_numerics)
    result = spectral.adaptive_spectral_profile(one_qubit(), max_queries=11, audit_points=2)
    assert result.diagnostics["numerical_failures"]
    assert all(not point.moments_resolved for point in result.points)
    assert all(np.isnan(point.g_ss) and np.isnan(point.d2) for point in result.points)
    converted = result.to_dict()
    assert converted["points"][0]["g_ss"] is None
    json.dumps(converted, allow_nan=False)
    json.dumps(result.diagnostics, allow_nan=False)


@pytest.mark.parametrize("kwargs", [
    {"initial_grid": [.1, 1.]}, {"initial_grid": [0., .5, .5, 1.]},
    {"max_queries": 6, "audit_points": 2}, {"max_queries": 6.5},
    {"audit_points": -1}, {"relative_tolerance": 0.},
    {"absolute_tolerance": float("nan")}, {"min_interval_width": 2.},
    {"residual_tolerance": -1.}, {"min_interval_width": 1e-30},
    {"initial_grid": [0., .5, np.nextafter(.5, 1.), 1.]},
])
def test_adaptive_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        spectral.adaptive_spectral_profile(one_qubit(), **kwargs)


def test_adaptivity_does_not_bypass_dense_qubit_cap():
    with pytest.raises(MemoryError, match="max_qubits"):
        spectral.adaptive_spectral_profile(HamiltonianTerms(11, np.zeros(11), [], []))


@pytest.mark.parametrize("catalyst", [0., .6, -.4])
def test_batched_matches_single_and_independent_dense(catalyst):
    terms = two_qubits()
    schedules = [lambda t: t, lambda t: t * t, lambda t: np.sqrt(t)]
    runtimes = [0., 1.2, 2.4]
    path = AnnealPath(catalyst)
    result = propagate_batch(terms, schedules, runtimes, steps=128, path=path, state_tolerance=.002)
    assert result.states.shape == (3, 4)
    assert result.accepted.all()
    for index, (schedule, runtime) in enumerate(zip(schedules, runtimes)):
        single = propagate(terms, schedule, runtime, steps=128, path=path, step_doubling=True)
        reference = dense_reference_propagate(terms, schedule, runtime, path=path)
        np.testing.assert_allclose(result.states[index], single.state, atol=1e-13)
        np.testing.assert_allclose(result.step_doubling_errors[index], single.step_doubling_error, atol=1e-14)
        assert np.linalg.norm(result.states[index] - reference.state) < 2e-4


def test_batch_numerical_gate_is_per_row_and_explicit():
    result = propagate_batch(two_qubits(), [lambda t: t] * 2, [0., 10.], steps=2,
                             state_tolerance=1e-7)
    np.testing.assert_array_equal(result.accepted, [True, False])
    with pytest.raises(RuntimeError, match="rows \\[1\\]"):
        propagate_batch(two_qubits(), [lambda t: t] * 2, [0., 10.], steps=2,
                        state_tolerance=1e-7, require_accepted=True)
    unchecked = propagate_batch(two_qubits(), [lambda t: t], 1., steps=2, step_doubling=False)
    assert unchecked.accepted is None
    assert unchecked.step_doubling_errors is None


def test_batch_custom_initial_states_and_static_schedule():
    initial = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.]], dtype=complex)
    result = propagate_batch(two_qubits(), [lambda t: .4] * 2, 1.2, initial_states=initial, steps=128)
    for index in range(2):
        single = propagate(two_qubits(), lambda t: .4, 1.2, initial_state=initial[index],
                           steps=128, step_doubling=True)
        np.testing.assert_allclose(result.states[index], single.state, atol=1e-13)


def test_batch_memory_and_device_information_are_explicit():
    first = estimate_state_workspace(8, batch_size=1, step_doubling=True)
    second = estimate_state_workspace(8, batch_size=4, step_doubling=True)
    assert second["state_bytes"] == 4 * first["state_bytes"]
    assert second["estimated_peak_bytes"] > first["estimated_peak_bytes"]
    with pytest.raises(MemoryError):
        propagate_batch(two_qubits(), [lambda t: t] * 8, 1., max_state_bytes=100)
    assert backend_device_info()["device"] == "cpu"
    with pytest.raises(ValueError, match="backend"):
        propagate_batch(two_qubits(), [lambda t: t], 1., backend="pretend_gpu")


@pytest.mark.parametrize("kwargs", [
    {"runtimes": [1., 2.]}, {"runtimes": -1.}, {"steps": 0},
    {"initial_states": np.ones(4)}, {"initial_states": np.ones((2, 4))},
    {"norm_tolerance": 0.}, {"state_tolerance": float("nan")},
    {"max_state_bytes": 1.5}, {"step_doubling": False, "require_accepted": True},
])
def test_batch_invalid_arguments(kwargs):
    options = {"runtimes": 1., "steps": 2, **kwargs}
    with pytest.raises(ValueError):
        propagate_batch(two_qubits(), [lambda t: t], **options)


def test_batch_cupy_parity_if_available():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("no CUDA device")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA runtime unavailable")
    schedules = [lambda t: t, lambda t: t * t]
    cpu = propagate_batch(two_qubits(), schedules, [1., 2.], steps=32, path=AnnealPath(.5))
    gpu = propagate_batch(two_qubits(), schedules, [1., 2.], steps=32, path=AnnealPath(.5), backend="cupy")
    np.testing.assert_allclose(cpu.states, gpu.states, atol=1e-12)
    np.testing.assert_allclose(cpu.step_doubling_errors, gpu.step_doubling_errors, atol=1e-12)
    np.testing.assert_array_equal(cpu.accepted, gpu.accepted)


@pytest.mark.parametrize("budget", [0, -1, 1.2, True])
def test_operator_rejects_invalid_memory_budget(budget):
    with pytest.raises(ValueError, match="max_state_bytes"):
        HamiltonianOperator(one_qubit(), max_state_bytes=budget)
