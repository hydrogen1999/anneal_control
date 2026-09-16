import numpy as np
import pytest

from annealctrl.physics import HamiltonianTerms
from annealctrl.physics_baselines import bounded_density_schedule, exact_teacher_baseline


def test_density_residual_allocation_enforces_slope_even_for_sharp_peak():
    grid = np.linspace(0., 1., 9)
    density = np.ones(9)
    density[4] = 1e12
    schedule = bounded_density_schedule(grid, density, runtime=2., max_slope=0.8)
    assert schedule.s_knots[0] == 0 and schedule.s_knots[-1] == 1
    assert schedule.tau_knots[-1] == 1
    assert np.max(schedule.slopes(2.)) <= 0.8 * (1 + 1e-12)
    assert schedule.tau_knots[5] - schedule.tau_knots[3] > 0.5


def test_constant_or_zero_density_gives_linear():
    grid = np.array([0., 0.1, 0.3, 0.9, 1.])
    for density in (np.zeros(5), np.ones(5)):
        schedule = bounded_density_schedule(grid, density, runtime=3., max_slope=2.)
        np.testing.assert_allclose(schedule(grid), grid, atol=1e-15)


def test_first_gap_degeneracy_is_not_silently_clipped():
    terms = HamiltonianTerms(2, [0., 0.], [[0, 1]], [-1.])
    result = exact_teacher_baseline(terms, "gap_inverse_square", runtime=2., max_slope=2.,
                                    s_grid=np.linspace(0., 1., 9), audit_points=0)
    assert result.schedule is None
    assert result.status == "unresolved_spectral_points"
    assert result.unresolved_indices == (8,)
    assert result.teacher_evaluations == 9
    assert result.privileged_per_instance
    assert result.deployment_spectrum_required


def test_dark_gap_and_d2_baselines_are_distinct_and_explicitly_regularized():
    terms = HamiltonianTerms(2, [0., 0.], [[0, 1]], [-1.])
    gap = exact_teacher_baseline(terms, "gap_inverse_square", runtime=2., max_slope=2.,
                                gap_epsilon=0.01, audit_points=0)
    bright = exact_teacher_baseline(terms, "d2", runtime=2., max_slope=2., audit_points=0)
    assert gap.method == "gap_inverse_square_regularized"
    assert bright.method == "d2"
    assert gap.schedule is not None and bright.schedule is not None
    assert bright.diagnostics["rank_changes"]
    tau = np.linspace(0., 1., 101)
    assert np.max(np.abs(gap.schedule(tau) - bright.schedule(tau))) > 0.1
    assert gap.density[-1] == pytest.approx(1 / 0.01**2)
    assert bright.status == "sampled_profile_not_audited"


def test_independent_random_interpolation_audit_is_charged():
    terms = HamiltonianTerms(1, [1.], [], [])
    result = exact_teacher_baseline(terms, "d2", runtime=2., max_slope=1.,
                                    audit_points=7, audit_seed=3, audit_relative_tolerance=0.02)
    assert result.teacher_evaluations == 40
    assert result.status == "sampled_point_audit_passed"
    assert result.diagnostics["audit_max_relative_error"] < 0.02
    assert result.diagnostics["outcome_evaluations"] == 0
    assert not result.diagnostics["uniform_interpolation_certificate"]


def test_coarse_grid_can_fail_independent_audit_without_hiding_it():
    terms = HamiltonianTerms(1, [1.], [], [])
    result = exact_teacher_baseline(terms, "d2", runtime=2., max_slope=1.,
                                    s_grid=[0., 1.], audit_points=8,
                                    audit_relative_tolerance=1e-6)
    assert result.schedule is not None
    assert result.status == "interpolation_audit_failed"


def test_no_runtime_for_feasible_traversal_rejected_before_teacher():
    with pytest.raises(ValueError, match="infeasible"):
        exact_teacher_baseline(HamiltonianTerms(1, [1.], [], []), "d2", runtime=1., max_slope=0.5)
