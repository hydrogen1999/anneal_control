"""Spin-reversal gauge: exact, label-preserving, and frustration-preserving."""
import numpy as np
import pytest

from annealctrl.gauge import apply_gauge, gauge_signs, loop_product


def frustrated_triangle():
    """Three sites, one negative coupling: loop product < 0 means frustrated."""
    h = np.array([0.3, -0.2, 0.5])
    edges = np.array([[0, 1], [1, 2], [2, 0]])
    J = np.array([1.0, 1.0, -1.0])
    return h, edges, J


# --- the physics the augmentation must preserve ------------------------------

def test_the_spectrum_is_unchanged_by_a_gauge_transformation():
    """If this failed, the stored labels would not transfer and the arm is invalid."""
    from annealctrl.physics import AnnealPath, HamiltonianTerms
    from annealctrl.spectral import spectral_profile

    h, edges, J = frustrated_triangle()
    signs = np.array([1.0, -1.0, 1.0])
    gh, gJ = apply_gauge(h, edges, J, signs)
    grid = np.linspace(0.0, 1.0, 5)
    path = AnnealPath()
    a = spectral_profile(HamiltonianTerms(3, h, edges, J), grid, path=path)
    b = spectral_profile(HamiltonianTerms(3, gh, edges, gJ), grid, path=path)
    for pa, pb in zip(a, b):
        assert pa.raw_gap == pytest.approx(pb.raw_gap, abs=1e-10)


def test_frustration_survives_the_gauge():
    """The document refuses invariance bought by deleting signs. This checks it."""
    h, edges, J = frustrated_triangle()
    before = loop_product(edges, J, np.array([0, 1, 2]))
    assert before < 0
    rng = np.random.default_rng(0)
    for _ in range(20):
        _, gJ = apply_gauge(h, edges, J, gauge_signs(3, rng))
        assert loop_product(edges, gJ, np.array([0, 1, 2])) == pytest.approx(before)


def test_an_unfrustrated_loop_stays_unfrustrated():
    h, edges, _ = frustrated_triangle()
    J = np.array([1.0, 1.0, 1.0])
    rng = np.random.default_rng(1)
    for _ in range(20):
        _, gJ = apply_gauge(h, edges, J, gauge_signs(3, rng))
        assert loop_product(edges, gJ, np.array([0, 1, 2])) > 0


# --- the transformation itself -----------------------------------------------

def test_the_identity_gauge_changes_nothing():
    h, edges, J = frustrated_triangle()
    gh, gJ = apply_gauge(h, edges, J, np.ones(3))
    assert gh == pytest.approx(h) and gJ == pytest.approx(J)


def test_flipping_every_spin_negates_the_fields_but_not_the_couplings():
    h, edges, J = frustrated_triangle()
    gh, gJ = apply_gauge(h, edges, J, -np.ones(3))
    assert gh == pytest.approx(-h)
    assert gJ == pytest.approx(J)


def test_a_gauge_is_an_involution():
    h, edges, J = frustrated_triangle()
    signs = np.array([-1.0, 1.0, -1.0])
    once_h, once_J = apply_gauge(h, edges, J, signs)
    twice_h, twice_J = apply_gauge(once_h, edges, once_J, signs)
    assert twice_h == pytest.approx(h) and twice_J == pytest.approx(J)


# --- refusals ----------------------------------------------------------------

def test_non_unit_signs_are_refused():
    h, edges, J = frustrated_triangle()
    with pytest.raises(ValueError, match="\\+1 or -1"):
        apply_gauge(h, edges, J, np.array([1.0, 0.5, -1.0]))


def test_mismatched_lengths_are_refused():
    h, edges, J = frustrated_triangle()
    with pytest.raises(ValueError, match="one entry per site"):
        apply_gauge(h, edges, J, np.ones(2))
    with pytest.raises(ValueError, match="agree in length"):
        apply_gauge(h, edges, J[:2], np.ones(3))


def test_a_loop_edge_not_in_the_graph_is_refused():
    h, edges, J = frustrated_triangle()
    with pytest.raises(ValueError, match="not in the graph"):
        loop_product(edges, J, np.array([0, 2, 1, 0]))


def embedded_record(chain=2, n_logical=3):
    """A logical chain embedded with `chain` physical qubits per logical spin."""
    logical_edges = np.array([[i, i + 1] for i in range(n_logical - 1)])
    membership = np.repeat(np.arange(n_logical), chain)
    physical_edges, physical_J = [], []
    for i in range(n_logical):
        for k in range(chain - 1):
            physical_edges.append([i * chain + k, i * chain + k + 1])
            physical_J.append(-2.0)                       # intra-chain, ferromagnetic
    for a, b in logical_edges:
        physical_edges.append([a * chain, b * chain])
        physical_J.append(0.7)                            # inter-chain
    return {
        "logical_h": np.array([0.3, -0.2, 0.5][:n_logical]),
        "logical_edges": logical_edges,
        "logical_J": np.full(len(logical_edges), 0.7),
        "physical_h": np.full(n_logical * chain, 0.15),
        "physical_edges": np.asarray(physical_edges),
        "physical_J": np.asarray(physical_J, dtype=float),
        "problem_J": np.full(len(logical_edges), 0.7),
        "membership": membership,
    }, len(physical_J) - len(logical_edges)


def test_the_induced_gauge_leaves_chain_couplings_alone():
    """Chains must stay ferromagnetic, or chain strength stops meaning anything."""
    from annealctrl.gauge import gauge_record

    record, n_chain_edges = embedded_record()
    rng = np.random.default_rng(0)
    for _ in range(20):
        gauged = gauge_record(record, gauge_signs(3, rng))
        assert gauged["physical_J"][:n_chain_edges] == pytest.approx(
            record["physical_J"][:n_chain_edges])


def test_the_induced_gauge_keeps_logical_and_physical_consistent():
    from annealctrl.gauge import gauge_record

    record, n_chain_edges = embedded_record()
    signs = np.array([1.0, -1.0, 1.0])
    gauged = gauge_record(record, signs)
    # Inter-chain physical couplings must match the gauged logical couplings.
    assert gauged["physical_J"][n_chain_edges:] == pytest.approx(
        np.array([0.7, 0.7]) * signs[[0, 1]] * signs[[1, 2]])
    assert gauged["logical_J"] == pytest.approx(
        record["logical_J"] * signs[[0, 1]] * signs[[1, 2]])
    assert gauged["problem_J"] == pytest.approx(gauged["logical_J"])


def test_the_record_gauge_refuses_a_physical_length_sign_vector():
    from annealctrl.gauge import gauge_record

    record, _ = embedded_record()
    with pytest.raises(ValueError, match="one entry per logical spin"):
        gauge_record(record, np.ones(6))


def test_gauging_a_record_does_not_mutate_the_original():
    from annealctrl.gauge import gauge_record

    record, _ = embedded_record()
    before = record["physical_J"].copy()
    gauge_record(record, np.array([-1.0, 1.0, -1.0]))
    assert record["physical_J"] == pytest.approx(before)
