"""How much entanglement do these trajectories actually carry?

This decides whether a tensor-network simulation could reach device sizes. An
MPS represents a state exactly with bond dimension chi = exp(S) where S is the
entanglement entropy across the cut, so a trajectory with S around 1-2 nats is
cheap at hundreds of qubits and one approaching the volume law N/2 * ln2 is not
representable at any size worth having.

The measurement is made on the same records and the same controls the rest of
the project uses, so the answer is about this problem family rather than about
entanglement in general.
"""
import numpy as np
import pytest

from annealctrl.entanglement import (bipartition_entropy, required_bond_dimension,
                                     volume_law_entropy)


def test_a_product_state_has_zero_entropy():
    state = np.zeros(2 ** 4, dtype=complex)
    state[0] = 1.0
    assert bipartition_entropy(state, 4, cut=2) == pytest.approx(0.0, abs=1e-12)


def test_a_bell_pair_carries_one_bit():
    state = np.zeros(4, dtype=complex)
    state[0] = state[3] = 1 / np.sqrt(2)
    assert bipartition_entropy(state, 2, cut=1) == pytest.approx(np.log(2))


def test_a_maximally_entangled_cut_saturates_the_volume_law():
    n, cut = 6, 3
    state = np.zeros(2 ** n, dtype=complex)
    dimension = 2 ** cut
    for k in range(dimension):                       # sum_k |k>|k> / sqrt(d)
        state[k * dimension + k] = 1 / np.sqrt(dimension)
    assert bipartition_entropy(state, n, cut=cut) == pytest.approx(cut * np.log(2))
    assert volume_law_entropy(n) == pytest.approx((n // 2) * np.log(2))


def test_bond_dimension_is_the_exponential_of_the_entropy():
    assert required_bond_dimension(0.0) == 1
    assert required_bond_dimension(np.log(2)) == 2
    assert required_bond_dimension(3 * np.log(2)) == 8
    assert required_bond_dimension(np.log(10)) >= 10


def test_entropy_is_bounded_by_the_smaller_side():
    """S <= min(cut, n-cut) * ln 2. Cut 2 and cut 4 are different bipartitions,
    not the same one seen twice, so the symmetry to check is this bound rather
    than equality between them."""
    rng = np.random.default_rng(0)
    n = 6
    state = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n)
    state /= np.linalg.norm(state)
    for cut in range(1, n):
        entropy = bipartition_entropy(state, n, cut=cut)
        assert entropy <= min(cut, n - cut) * np.log(2) + 1e-9, (cut, entropy)


def test_a_random_state_is_near_the_volume_law():
    """Random states are maximally entangled; this is the regime MPS cannot reach."""
    rng = np.random.default_rng(1)
    n = 10
    state = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n)
    state /= np.linalg.norm(state)
    entropy = bipartition_entropy(state, n, cut=n // 2)
    assert entropy > 0.7 * volume_law_entropy(n)


def test_entropy_refuses_a_state_of_the_wrong_length():
    with pytest.raises(ValueError, match="2\\*\\*n"):
        bipartition_entropy(np.ones(7) / np.sqrt(7), 3, cut=1)


def test_the_instrumented_loop_reproduces_the_library_propagator():
    """An instrumented rewrite that drifts from the real propagator measures the wrong thing."""
    from annealctrl.entanglement import entanglement_trajectory
    from annealctrl.physics import AnnealPath, HamiltonianTerms, propagate
    from annealctrl.schedules import Schedule

    rng = np.random.default_rng(3)
    n = 6
    edges = [[i, j] for i in range(n) for j in range(i + 1, n)]
    terms = HamiltonianTerms(n, rng.normal(0, 0.3, n), edges,
                             rng.choice([-1.0, 1.0], size=len(edges)))
    schedule, path, steps = Schedule.linear(), AnnealPath(), 128

    mine = entanglement_trajectory(terms, schedule, 2.0, path=path, steps=steps)["final_state"]
    theirs = propagate(terms, schedule, 2.0, steps=steps, path=path).state
    assert np.abs(mine - np.asarray(theirs)).max() < 1e-10


def test_trajectory_reports_the_bond_dimension_a_tensor_network_would_need():
    from annealctrl.entanglement import entanglement_trajectory
    from annealctrl.physics import AnnealPath, HamiltonianTerms
    from annealctrl.schedules import Schedule

    n = 6
    edges = [[i, i + 1] for i in range(n - 1)]
    terms = HamiltonianTerms(n, np.zeros(n), edges, -np.ones(len(edges)))
    out = entanglement_trajectory(terms, Schedule.linear(), 2.0, path=AnnealPath(), steps=64)
    assert out["n_qubits"] == n and out["cut"] == 3
    assert 0.0 <= out["peak_entropy"] <= out["volume_law_entropy"] + 1e-9
    assert out["required_bond_dimension"] >= 1
    assert len(out["trace"]) >= 2
