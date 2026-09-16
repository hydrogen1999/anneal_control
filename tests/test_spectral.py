import numpy as np
import pytest

from annealctrl.physics import AnnealPath, HamiltonianTerms
from annealctrl.spectral import sparse_low_energy, spectral_profile, spectral_teacher


def _dark():
    return HamiltonianTerms(2, [0., 0.], [[0, 1]], [-1.])


@pytest.mark.parametrize("s", [0.1, 0.5, 0.8, 0.95])
def test_dark_gap_requires_higher_bright_state(s):
    point = spectral_teacher(_dark(), s)
    r = np.sqrt(s * s + 4 * (1 - s)**2)
    assert point.raw_gap == pytest.approx(r - s)
    assert point.accessible_gap == pytest.approx(2 * r)
    assert point.d2 == pytest.approx(1 / (2 * r**3))
    assert point.transition_strengths[0] < 1e-24
    assert point.transition_strengths[1] < 1e-24
    assert point.transition_strengths[2] == pytest.approx(4 / r**2)
    assert point.mu0 == pytest.approx(4 / r**2)
    assert point.sum_rule_error < 1e-12


def test_degenerate_endpoint_uses_entire_ground_band():
    point = spectral_teacher(_dark(), 1.)
    assert point.ground_rank == 2
    assert point.raw_gap == 0
    assert point.ground_band_spread == 0
    assert point.moments_resolved
    assert np.isfinite(point.d2)
    assert point.sum_rule_error < 1e-12


def test_single_qubit_analytic_moments():
    s = 0.31
    r = np.sqrt((1 - s)**2 + s**2)
    point = spectral_teacher(HamiltonianTerms(1, [1.], [], []), s)
    assert point.mu0 == pytest.approx(1 / r**2)
    assert point.g_ss == pytest.approx(1 / (4 * r**4))
    assert point.d2 == pytest.approx(1 / (4 * r**3))


def test_scale_identities():
    gamma = 3.1
    base = spectral_teacher(_dark(), 0.63)
    scaled = spectral_teacher(_dark(), 0.63, path=AnnealPath(energy_scale=gamma))
    np.testing.assert_allclose(scaled.gaps, gamma * base.gaps, atol=1e-13)
    assert scaled.mu0 == pytest.approx(gamma**2 * base.mu0)
    assert scaled.g_ss == pytest.approx(base.g_ss)
    assert scaled.d2 == pytest.approx(base.d2 / gamma)


def test_gauge_transformation_preserves_spectrum_and_response():
    fields = np.array([0.2, -0.4, 0.1])
    edges = np.array([[0, 1], [1, 2], [0, 2]])
    weights = np.array([-0.7, 0.3, -0.2])
    gauge = np.array([-1, 1, -1])
    original = HamiltonianTerms(3, fields, edges, weights, [[0, 1]], [0.2])
    transformed = HamiltonianTerms(3, fields * gauge, edges, weights * gauge[edges[:, 0]] * gauge[edges[:, 1]], [[0, 1]], [0.2])
    path = AnnealPath(0.7)
    first = spectral_teacher(original, 0.6, path=path)
    second = spectral_teacher(transformed, 0.6, path=path)
    np.testing.assert_allclose(first.energies, second.energies, atol=2e-14)
    np.testing.assert_allclose([first.mu0, first.g_ss, first.d2], [second.mu0, second.g_ss, second.d2], atol=2e-13)


def test_response_histogram_preserves_mass_including_outside_bins():
    point = spectral_teacher(_dark(), 0.95, frequency_edges=np.geomspace(0.01, 0.1, 9))
    total = point.bin_masses.sum() + point.low_frequency_mass + point.high_frequency_mass + point.unresolved_mass
    assert total == pytest.approx(point.mu0)
    assert point.high_frequency_mass == pytest.approx(point.mu0)
    assert point.omitted_response_mass == 0
    assert not point.truncated


def test_loose_degeneracy_tolerance_does_not_report_false_precise_moments():
    terms = HamiltonianTerms(1, [1e-10], [], [])
    point = spectral_teacher(terms, 1., ground_atol=1e-8)
    assert point.ground_rank == 2
    assert point.near_degenerate_band
    assert not point.moments_resolved
    assert np.isnan(point.d2)


def test_forcing_single_vector_at_degenerate_endpoint_is_unresolved():
    point = spectral_teacher(_dark(), 1., ground_rank=1)
    assert point.unresolved_transition_count >= 1
    assert not point.moments_resolved
    assert np.isnan(point.d2)


def test_dense_teacher_cap_checks_physical_qubit_count():
    with pytest.raises(MemoryError):
        spectral_teacher(HamiltonianTerms(20, np.zeros(20), [], []), 0.4)


def test_profile_and_eigenpair_diagnostics():
    grid = np.linspace(0., 1., 5)
    profile = spectral_profile(_dark(), grid)
    assert len(profile) == 5
    assert profile[-1].ground_rank != profile[-2].ground_rank
    for point in profile:
        assert point.max_eigenpair_residual < 1e-13
        assert point.orthogonality_error < 1e-13
        assert point.bin_masses.shape == (8,)


@pytest.mark.parametrize("edges", [[0., 1.], [1., 0.5], [1., float("nan")]])
def test_bad_frequency_contract_rejected(edges):
    with pytest.raises(ValueError):
        spectral_teacher(_dark(), 0.5, frequency_edges=np.array(edges))


def test_sparse_low_energy_matches_dense_subset_but_marks_incomplete():
    terms = HamiltonianTerms(3, [0.2, -0.4, 0.3], [[0, 1], [1, 2]], [-0.7, 0.6], [[0, 2]], [0.3])
    path = AnnealPath(0.4)
    dense = spectral_teacher(terms, 0.47, path=path)
    sparse = sparse_low_energy(terms, 0.47, 3, path=path)
    np.testing.assert_allclose(sparse.energies, dense.energies[:3], atol=2e-12)
    assert sparse.residuals.max() < 1e-12
    assert sparse.orthogonality_error < 1e-12
    assert sparse.truncated
    assert not sparse.full_response_resolved
    assert not sparse.ground_band_certified


def test_sparse_memory_and_k_validation():
    with pytest.raises(ValueError):
        sparse_low_energy(_dark(), 0.4, 4)
    with pytest.raises(MemoryError):
        sparse_low_energy(HamiltonianTerms(20, np.zeros(20), [], []), 0.4, 10, max_state_bytes=1000)
