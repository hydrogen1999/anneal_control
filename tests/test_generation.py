"""Independent checks of generation, compilation, splits, and readout semantics."""
from __future__ import annotations

import itertools

import networkx as nx
import numpy as np
import pytest

from annealctrl.generation import (
    Embedding,
    IsingProblem,
    all_spins,
    canonical_edges,
    compile_embedding,
    generate_problem,
    grow_hardware_partition,
    output_observables,
    parent_splits,
    planted_loops,
    sample_chain_lengths,
    synthetic_lift,
    validate_compilation,
)


def triangle_problem() -> IsingProblem:
    return IsingProblem(
        np.array([0.8, -0.3, 0.2]),
        np.array([[0, 1], [1, 2], [0, 2]]),
        np.array([1.1, -0.7, 0.9]),
    )


def test_all_spins_uses_lsb_bit_zero_positive_convention():
    np.testing.assert_array_equal(
        all_spins(2), np.array([[1, 1], [-1, 1], [1, -1], [-1, -1]])
    )
    for n in (0, 21):
        with pytest.raises(ValueError):
            all_spins(n)


@pytest.mark.parametrize("seed", [0, 3, 17])
def test_overlapping_planted_cycles_attain_global_lower_bound(seed):
    n = 5
    edges = np.array(list(itertools.combinations(range(n), 2)))
    problem = planted_loops(n, edges, np.random.default_rng(seed), num_loops=19)
    brute = np.array([
        sum(problem.h[i] * z[i] for i in range(n))
        + sum(J * z[i] * z[j] for (i, j), J in zip(problem.edges, problem.J))
        for z in itertools.product([-1, 1], repeat=n)
    ])
    claimed = problem.metadata["certified_ground_energy"]
    np.testing.assert_allclose(brute.min(), claimed, atol=1e-11, rtol=0)
    np.testing.assert_allclose(
        problem.energy(np.array(problem.metadata["planted_spins"])),
        claimed, atol=1e-11, rtol=0,
    )
    assert len(problem.metadata["cycles"]) == 19
    # The certificate covers endpoint energy, not quantum spectral hardness.
    assert "spectral_hardness" not in problem.metadata


def test_planted_loops_reject_acyclic_support():
    with pytest.raises(ValueError, match="cyclic"):
        planted_loops(3, np.array([[0, 1], [1, 2]]), np.random.default_rng(1))


@pytest.mark.parametrize("field_distribution", ["uniform", "concentrated"])
@pytest.mark.parametrize("coupling_distribution", ["uniform", "random"])
def test_compilation_preserves_aligned_energy_and_common_scale(
    field_distribution, coupling_distribution,
):
    problem = triangle_problem()
    rng = np.random.default_rng(21)
    embedding = synthetic_lift(problem, np.array([2, 3, 1]), rng, ports=3)
    compiled = compile_embedding(
        problem, embedding, chain_strength=3.0, rng=rng,
        field_distribution=field_distribution,
        coupling_distribution=coupling_distribution,
        h_limit=0.15, j_limit=0.4,
    )
    assert 0 < compiled.programmed_scale < 1
    assert np.max(np.abs(compiled.physical.h)) <= 0.15 + 1e-13
    assert np.max(np.abs(compiled.physical.J)) <= 0.4 + 1e-13
    np.testing.assert_allclose(
        compiled.physical.J, compiled.problem_J + compiled.chain_J, atol=1e-13,
    )
    assert compiled.aligned_offset == pytest.approx(compiled.chain_J.sum())
    for z in itertools.product([-1, 1], repeat=problem.n):
        z = np.array(z)
        physical_z = z[embedding.membership]
        expected = compiled.programmed_scale * problem.energy(z) + compiled.aligned_offset
        assert compiled.physical.energy(physical_z) == pytest.approx(expected, abs=1e-12)
    report = validate_compilation(compiled)
    assert report["aligned_energy_max_error"] < 1e-11
    # Every distributed field/coupler reconstructs the scaled logical value.
    owner = embedding.membership
    for v in range(problem.n):
        assert compiled.physical.h[owner == v].sum() == pytest.approx(
            compiled.programmed_scale * problem.h[v]
        )
    for (v, w), coupling in zip(problem.edges, problem.J):
        a, b = owner[embedding.hardware_edges.T]
        cross = ((a == v) & (b == w)) | ((a == w) & (b == v))
        assert compiled.problem_J[cross].sum() == pytest.approx(
            compiled.programmed_scale * coupling
        )


def test_weak_chain_can_fail_ground_state_preservation_without_breaking_identity():
    problem = IsingProblem(np.zeros(3), np.array([[0, 1], [0, 2], [1, 2]]), np.ones(3))
    embedding = Embedding(
        np.array([0, 0, 1, 2]), np.array([[0, 1], [0, 2], [1, 3], [2, 3]])
    )
    result = compile_embedding(problem, embedding, 0.1, np.random.default_rng(0))
    report = validate_compilation(result)
    assert report["aligned_energy_max_error"] < 1e-12
    assert not report["has_aligned_physical_ground_state"]
    assert report["physical_ground_energy"] < report["aligned_ground_energy"]


def test_embeddings_reject_disconnected_chains_and_missing_logical_edge():
    with pytest.raises(ValueError, match="disconnected"):
        Embedding(np.array([0, 0, 1]), np.array([[0, 2], [1, 2]]))
    with pytest.raises(ValueError, match="contiguous"):
        Embedding(np.array([0, 2]), np.empty((0, 2), dtype=int))
    problem = IsingProblem(np.zeros(2), np.array([[0, 1]]), np.ones(1))
    embedding = Embedding(np.array([0, 1]), np.empty((0, 2), dtype=int))
    with pytest.raises(ValueError, match="no physical realization"):
        compile_embedding(problem, embedding, 1.0, np.random.default_rng(1))


@pytest.mark.parametrize("bad_edges", [
    [[0, 1], [1, 0]], [[0, 0]], [[0, 3]], [[-1, 1]], [0, 1],
])
def test_invalid_edge_supports_rejected(bad_edges):
    with pytest.raises(ValueError):
        canonical_edges(bad_edges, 3)


def test_same_logical_parent_produces_distinct_physical_variants_without_mutation():
    problem = triangle_problem()
    saved = (problem.h.copy(), problem.edges.copy(), problem.J.copy())
    variants = []
    for lengths, shape, strength in [([1, 2, 2], "path", 1.0), ([3, 2, 1], "star", 2.0)]:
        rng = np.random.default_rng(5)
        embedding = synthetic_lift(problem, np.array(lengths), rng, shape=shape, ports=2)
        variants.append(compile_embedding(problem, embedding, strength, rng))
    assert all(variant.logical is problem for variant in variants)
    assert variants[0].fingerprint() != variants[1].fingerprint()
    for current, before in zip((problem.h, problem.edges, problem.J), saved):
        np.testing.assert_array_equal(current, before)
    assert all(validate_compilation(v)["aligned_energy_max_error"] < 1e-10 for v in variants)


@pytest.mark.parametrize("distribution", ["uniform", "power_law"])
def test_chain_length_sampling_is_reproducible_and_truncated(distribution):
    a = sample_chain_lengths(4000, 1, 6, np.random.default_rng(42), distribution)
    b = sample_chain_lengths(4000, 1, 6, np.random.default_rng(42), distribution)
    np.testing.assert_array_equal(a, b)
    assert a.min() >= 1 and a.max() <= 6
    assert set(a.tolist()) == set(range(1, 7))
    singleton = sample_chain_lengths(7, 3, 3, np.random.default_rng(2), distribution)
    np.testing.assert_array_equal(singleton, np.full(7, 3))
    if distribution == "power_law":
        assert np.sum(a == 1) > 10 * np.sum(a == 6)


@pytest.mark.parametrize("bad_exponent", [0, -1, np.nan, np.inf])
def test_power_law_rejects_invalid_exponents(bad_exponent):
    with pytest.raises(ValueError):
        sample_chain_lengths(4, 1, 4, np.random.default_rng(1), "power_law", bad_exponent)


@pytest.mark.parametrize("seed", [0, 4, 19])
def test_fixed_hardware_growth_connected_disjoint_quota_metadata(seed):
    original_graph = nx.convert_node_labels_to_integers(nx.grid_2d_graph(4, 4))
    edges = np.array(list(original_graph.edges()))
    target = np.array([3, 4, 3])
    embedding = grow_hardware_partition(16, edges, target, np.random.default_rng(seed))
    repeated = grow_hardware_partition(16, edges, target, np.random.default_rng(seed))
    np.testing.assert_array_equal(embedding.membership, repeated.membership)
    np.testing.assert_array_equal(embedding.hardware_edges, repeated.hardware_edges)
    original_ids = embedding.metadata["original_physical_ids"]
    assert len(original_ids) == len(set(original_ids)) == len(embedding.membership)
    counts = np.bincount(embedding.membership, minlength=len(target))
    assert np.all((counts >= 1) & (counts <= target))
    np.testing.assert_array_equal(counts, embedding.metadata["achieved_lengths"])
    np.testing.assert_array_equal(target, embedding.metadata["target_lengths"])
    assert embedding.metadata["target_met"] == bool(np.array_equal(counts, target))
    assert embedding.metadata["unused_qubits"] == 16 - len(original_ids)
    for v in range(len(target)):
        group = [original_ids[i] for i in np.flatnonzero(embedding.membership == v)]
        assert nx.is_connected(original_graph.subgraph(group))
    for a, b in embedding.hardware_edges:
        assert original_graph.has_edge(original_ids[a], original_ids[b])
    # A quotient-generated logical problem compiles on exactly this grown support.
    problem = generate_problem(len(target), "spin_glass", np.random.default_rng(seed),
                               edges=embedding.quotient_edges)
    compiled = compile_embedding(problem, embedding, 2.0, np.random.default_rng(seed))
    assert validate_compilation(compiled)["aligned_energy_max_error"] < 1e-10


def test_jammed_growth_does_not_claim_exact_target_distribution():
    embedding = grow_hardware_partition(
        6, np.empty((0, 2), dtype=int), np.array([3, 3]), np.random.default_rng(3)
    )
    assert embedding.metadata["target_lengths"] == [3, 3]
    assert embedding.metadata["achieved_lengths"] == [1, 1]
    assert not embedding.metadata["target_met"]
    assert embedding.metadata["unused_qubits"] == 4
    assert embedding.quotient_edges.shape == (0, 2)
    with pytest.raises(ValueError, match="capacity"):
        grow_hardware_partition(2, np.empty((0, 2)), np.array([2, 2]), np.random.default_rng(0))


def test_majority_decoding_accepts_all_decoded_optima_including_broken_chains():
    # Logical +1 is uniquely optimal; two-qubit ties deterministically decode +1.
    problem = IsingProblem(np.array([-1.0]), np.empty((0, 2)), np.empty(0))
    embedding = Embedding(np.array([0, 0]), np.array([[0, 1]]))
    compiled = compile_embedding(problem, embedding, 1.0, np.random.default_rng(0))
    obs = output_observables(compiled)
    np.testing.assert_array_equal(obs["success"], [1, 1, 1, 0])
    np.testing.assert_array_equal(obs["decoded_energy"], [-1, -1, -1, 1])
    np.testing.assert_array_equal(obs["any_chain_break"], [0, 1, 1, 0])
    np.testing.assert_array_equal(obs["chain_break_fraction"], [0, 1, 1, 0])
    # No normalization by only unbroken strings; uniform physical output has p=3/4.
    assert np.ones(4).dot(obs["success"]) / 4 == 0.75
    assert np.array([0, 0, 0, 1]).dot(obs["success"]) == 0.0
    assert np.array([1, 0, 0, 0]).dot(obs["success"]) == 1.0
    np.testing.assert_allclose(obs["physical_energy"], compiled.physical.energy(all_spins(2)))


def test_readout_accepts_degenerate_logical_optimum_entire_subspace():
    problem = IsingProblem(np.zeros(2), np.empty((0, 2)), np.empty(0))
    embedding = Embedding(np.array([0, 0, 1]), np.array([[0, 1]]))
    compiled = compile_embedding(problem, embedding, 1.0, np.random.default_rng(0))
    np.testing.assert_array_equal(output_observables(compiled)["success"], np.ones(8))


def test_parent_split_is_stratified_reproducible_and_has_exact_parent_keys():
    families = [family for family in ("spin_glass", "weighted_maxcut", "planted_loops")
                for _ in range(10)]
    split = parent_splits(families, seed=6)
    assert split == parent_splits(families, seed=6)
    assert len(split) == len(families)
    assert set(split) == {f"parent_{i:04d}" for i in range(len(families))}
    for family in set(families):
        values = [split[f"parent_{i:04d}"] for i, f in enumerate(families) if f == family]
        assert values.count("train") == 6
        assert values.count("validation") == 2
        assert values.count("test") == 2
    with pytest.raises(ValueError, match="at least three"):
        parent_splits(["spin_glass"] * 2)


@pytest.mark.parametrize("cap_name", ["h_limit", "j_limit"])
@pytest.mark.parametrize("bad_cap", [np.nan, np.inf, -np.inf])
def test_compiler_rejects_nonfinite_caps(cap_name, bad_cap):
    problem = triangle_problem()
    rng = np.random.default_rng(0)
    embedding = synthetic_lift(problem, np.ones(problem.n, dtype=int), rng)
    with pytest.raises(ValueError):
        compile_embedding(problem, embedding, 1.0, rng, **{cap_name: bad_cap})


def test_synthetic_lift_rejects_unknown_shape_even_for_singleton_chains():
    problem = triangle_problem()
    with pytest.raises(ValueError, match="shape"):
        synthetic_lift(problem, np.ones(problem.n, dtype=int), np.random.default_rng(0),
                       shape="not_a_shape")
