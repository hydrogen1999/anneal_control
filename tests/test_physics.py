import numpy as np
import pytest

from annealctrl.physics import (
    AnnealPath, HamiltonianOperator, HamiltonianTerms,
    dense_reference_propagate, plus_state, propagate,
)


def _pauli(n, i, matrix):
    result = np.array([[1.]])
    for site in reversed(range(n)):
        result = np.kron(result, matrix if site == i else np.eye(2))
    return result


def _example(n=3):
    return HamiltonianTerms(n, np.array([0.4, -0.3, 0.2])[:n],
                            np.array([[0, 1], [1, 2]]) if n == 3 else np.array([[0, 1]]),
                            np.array([-0.7, 0.3]) if n == 3 else np.array([-0.7]),
                            np.array([[0, 1]]), np.array([0.2]))


def test_dense_and_matrix_free_agree_with_independent_pauli_products():
    terms = _example()
    x = np.array([[0., 1.], [1., 0.]])
    z = np.diag([1., -1.])
    xs = [_pauli(3, i, x) for i in range(3)]
    zs = [_pauli(3, i, z) for i in range(3)]
    hx = -sum(xs)
    hz = sum(h * zi for h, zi in zip(terms.h, zs))
    hz += sum(w * (zs[i] @ zs[j]) for (i, j), w in zip(terms.zz_edges, terms.zz_weights))
    hxx = 0.2 * xs[0] @ xs[1]
    path = AnnealPath(catalyst_strength=0.8, energy_scale=1.3)
    a, b, c = path.coefficients(0.37)
    expected = a * hx + b * hz + c * hxx
    operator = HamiltonianOperator(terms)
    np.testing.assert_allclose(operator.dense(0.37, path), expected, atol=1e-14)
    psi = np.arange(8) + 1j * np.arange(8)[::-1]
    np.testing.assert_allclose(operator.matvec(psi, 0.37, path), expected @ psi, atol=1e-13)


def test_bit_zero_is_lsb_and_zero_has_positive_spin():
    operator = HamiltonianOperator(HamiltonianTerms(2, [1., 2.], [], []))
    np.testing.assert_array_equal(operator.diagonal, [3., 1., -1., -3.])


def test_analytic_path_derivative():
    operator = HamiltonianOperator(_example())
    path = AnnealPath(0.7, 1.4)
    eps = 1e-6
    finite_difference = (operator.dense(0.41 + eps, path) - operator.dense(0.41 - eps, path)) / (2 * eps)
    exact = operator.dense_parts(path.derivative(0.41))
    np.testing.assert_allclose(exact, finite_difference, atol=2e-10)


@pytest.mark.parametrize("catalyst", [0., 0.6, -0.4])
def test_split_propagation_agrees_with_independent_dense_integrator(catalyst):
    terms = _example()
    path = AnnealPath(catalyst_strength=catalyst)
    schedule = lambda tau: tau * tau
    reference = dense_reference_propagate(terms, schedule, 3.2, path=path)
    result = propagate(terms, schedule, 3.2, steps=256, path=path, step_doubling=True)
    error = np.linalg.norm(result.state - reference.state)
    assert error < 4e-5
    assert result.norm_error < 1e-11
    assert result.step_doubling_error is not None
    assert 0.8 < result.step_doubling_error / error < 1.2
    assert result.steps == 512


def test_second_order_convergence():
    terms = _example(2)
    reference = dense_reference_propagate(terms, lambda t: t, 2.3)
    coarse = propagate(terms, lambda t: t, 2.3, steps=40)
    fine = propagate(terms, lambda t: t, 2.3, steps=80)
    ratio = np.linalg.norm(coarse.state - reference.state) / np.linalg.norm(fine.state - reference.state)
    assert 3.9 < ratio < 4.1


def test_whole_hamiltonian_runtime_covariance():
    terms = _example()
    first = propagate(terms, lambda t: t, 2., steps=100, path=AnnealPath(0.4))
    second = propagate(terms, lambda t: t, 2. / 3., steps=100, path=AnnealPath(0.4, 3.))
    np.testing.assert_allclose(first.state, second.state, atol=2e-14)


def test_pure_commuting_x_xx_rotation_is_exact():
    # Static midpoint schedule and explicit arbitrary initial state; Z part zero.
    terms = HamiltonianTerms(2, [0., 0.], [], [], [[0, 1]], [0.8])
    psi = np.array([1., 0., 0., 0.], dtype=complex)
    path = AnnealPath(0.7)
    reference = dense_reference_propagate(terms, lambda _: 0.4, 2., path=path, initial_state=psi)
    result = propagate(terms, lambda _: 0.4, 2., steps=1, path=path, initial_state=psi)
    np.testing.assert_allclose(result.state, reference.state, atol=2e-10)


def test_zero_runtime_preserves_state():
    terms = _example()
    result = propagate(terms, lambda t: t, 0., steps=4)
    np.testing.assert_allclose(result.state, plus_state(3), atol=1e-16, rtol=0.)


def test_initialization_is_not_silently_repaired():
    with pytest.raises(ValueError, match="normalized"):
        propagate(_example(2), lambda t: t, 1., initial_state=np.ones(4))
    with pytest.raises(ValueError, match="schedule\\(0\\)"):
        propagate(_example(2), lambda _: 0.5, 1.)


@pytest.mark.parametrize("bad_edges,bad_weights", [([[0, 0]], [1.]), ([[0, 2]], [1.]), ([[0, 1], [1, 0]], [1., 2.]), ([[0.0, 1.0]], [1.]), ([[0, 1]], [])])
def test_invalid_edges_rejected(bad_edges, bad_weights):
    with pytest.raises(ValueError):
        HamiltonianTerms(2, [0., 0.], bad_edges, bad_weights)


def test_memory_guards_before_large_allocations():
    terms = HamiltonianTerms(30, np.zeros(30), [], [])
    with pytest.raises(MemoryError):
        HamiltonianOperator(terms)
    with pytest.raises(MemoryError):
        dense_reference_propagate(terms, lambda t: t, 1.)


def test_terms_copy_inputs_and_are_read_only():
    fields = np.ones(2)
    terms = HamiltonianTerms(2, fields, [[0, 1]], [1.])
    fields[0] = 100
    assert terms.h[0] == 1
    with pytest.raises(ValueError):
        terms.h[0] = 5


@pytest.mark.parametrize("runtime,steps", [(-1., 10), (float("nan"), 10), (1., 0), (1., 1.1)])
def test_invalid_integration_arguments(runtime, steps):
    with pytest.raises(ValueError):
        propagate(_example(2), lambda t: t, runtime, steps=steps)


def test_cupy_agreement_if_gpu_available():
    cp = pytest.importorskip("cupy")
    try:
        available = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CuPy installed but CUDA device unavailable")
    if available < 1:
        pytest.skip("no CUDA device")
    terms = _example()
    cpu = propagate(terms, lambda t: t, 1.7, steps=32, path=AnnealPath(0.6))
    gpu = propagate(terms, lambda t: t, 1.7, steps=32, path=AnnealPath(0.6), backend="cupy")
    np.testing.assert_allclose(gpu.state, cpu.state, atol=1e-12)
