"""Cross-module checks: compilation coefficients and decoded objectives."""
import numpy as np
import pytest

from annealctrl.generation import (
    Embedding, IsingProblem, all_spins, compile_embedding,
    output_observables, synthetic_lift,
)
from annealctrl.physics import AnnealPath, HamiltonianOperator, HamiltonianTerms, propagate


def _terms(compiled):
    return HamiltonianTerms(compiled.physical.n, compiled.physical.h,
                            compiled.physical.edges, compiled.physical.J)


def test_physics_diagonal_equals_programmed_physical_energy_for_every_string():
    logical = IsingProblem([0.7, -0.2, 0.1], [[0, 1], [1, 2]], [-0.8, 0.4])
    rng = np.random.default_rng(10)
    embedding = synthetic_lift(logical, [2, 2, 2], rng, ports=2)
    compiled = compile_embedding(logical, embedding, 2.5, rng)
    operator = HamiltonianOperator(_terms(compiled))
    expected = compiled.physical.energy(all_spins(compiled.physical.n))
    np.testing.assert_allclose(operator.diagonal, expected, atol=1e-14)
    # No double application of alpha or double counting chain metadata.
    np.testing.assert_allclose(np.diag(operator.dense(1.)).real, expected, atol=1e-14)


def test_success_is_decoded_optimum_projector_not_physical_ground_projector():
    logical = IsingProblem([-1.], [], [])
    embedding = Embedding([0, 0], [[0, 1]])
    compiled = compile_embedding(logical, embedding, 2., np.random.default_rng(0))
    observables = output_observables(compiled)
    result = propagate(_terms(compiled), lambda tau: tau, 0., steps=1)
    probability = abs(result.state)**2
    physical_ground = np.isclose(observables["physical_energy"], observables["physical_energy"].min())
    assert probability @ observables["success"] == pytest.approx(0.75)
    assert probability @ physical_ground == pytest.approx(0.25)
    assert probability @ observables["any_chain_break"] == pytest.approx(0.5)


def test_hz_autoscaling_is_not_mistaken_for_whole_path_scaling():
    logical = IsingProblem([0.8, -0.1], [[0, 1]], [1.2])
    rng = np.random.default_rng(1)
    embedding = synthetic_lift(logical, [2, 2], rng)
    compiled = compile_embedding(logical, embedding, 2., rng, h_limit=10., j_limit=0.5)
    alpha = compiled.programmed_scale
    assert alpha < 1
    programmed = HamiltonianOperator(_terms(compiled))
    raw = HamiltonianOperator(HamiltonianTerms(compiled.physical.n, compiled.physical.h / alpha,
                                               compiled.physical.edges, compiled.physical.J / alpha))
    # At s=0, both paths use the SAME transverse driver.
    np.testing.assert_allclose(programmed.dense(0.), raw.dense(0.), atol=1e-14)
    # At s=1, only the programmed Ising endpoint is scaled.
    np.testing.assert_allclose(programmed.dense(1.), alpha * raw.dense(1.), atol=1e-14)
    assert not np.allclose(programmed.dense(0.5), raw.dense(0.5, AnnealPath(energy_scale=alpha)))
