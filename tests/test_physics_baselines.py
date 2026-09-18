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


def test_gap_schedule_actually_slows_down_at_the_gap_minimum():
    """The construction must realise ds/dt proportional to Delta^2, not merely be named for it.

    This is the single most load-bearing implementation in the project: the
    finding that an equal-budget search beats the privileged spectral schedule is
    only about the adiabatic rule if the waveform really is that rule. A wrong
    construction would lose to search and look exactly as confident -- which is
    what happened once already in this codebase, when a Gaussian score was
    implemented as (z-mu)/sigma instead of (z-mu)/sigma^2 and the resulting arm
    was published as a result about learned search.

    Local adiabaticity means the time density dt/ds goes as 1/Delta^2, so the
    realised density along the waveform must rank-correlate with 1/gap^2, and the
    slowest point must be the gap minimum.
    """
    import numpy as np

    from annealctrl.physics import AnnealPath, HamiltonianTerms
    from annealctrl.spectral import spectral_profile

    rng = np.random.default_rng(11)
    n, runtime = 5, 2.0
    edges = [[i, i + 1] for i in range(n - 1)]
    terms = HamiltonianTerms(n, rng.normal(0, 0.4, n), edges, rng.normal(-1, 0.3, len(edges)))
    path, grid = AnnealPath(), np.linspace(0.0, 1.0, 33)

    result = exact_teacher_baseline(terms, "gap_inverse_square", runtime=runtime,
                                    max_slope=4.0 / runtime, path=path, s_grid=grid,
                                    audit_points=0)
    assert result.schedule is not None, result.status

    points = spectral_profile(terms, grid, path=path, max_qubits=10)
    gaps = np.array([p.raw_gap for p in points])
    tau = np.linspace(0.0, 1.0, 2001)
    tau_of_s = np.interp(grid, result.schedule(tau), tau)
    density = np.gradient(tau_of_s, grid)

    interior = slice(1, -1)
    usable = np.isfinite(gaps[interior]) & (gaps[interior] > 0)
    x, y = density[interior][usable], 1.0 / gaps[interior][usable] ** 2
    rx, ry = np.argsort(np.argsort(x)).astype(float), np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    rho = float(rx @ ry / np.sqrt((rx**2).sum() * (ry**2).sum()))

    assert rho > 0.95, f"time density does not track 1/gap^2: rho={rho:.3f}"
    slowest = grid[interior][np.argmax(density[interior])]
    narrowest = grid[interior][usable][np.argmin(gaps[interior][usable])]
    assert slowest == pytest.approx(narrowest), (slowest, narrowest)
