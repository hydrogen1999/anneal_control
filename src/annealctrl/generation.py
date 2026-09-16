"""Generate logical coefficients and physical embeddings, before any simulation.

Two distinct routes are intentional:
* synthetic_lift: same logical parent, controlled chain geometry/ports;
* grow_hardware_partition: fixed hardware, construct connected chains then quotient.
The latter guarantees a feasible embedding, NOT difficult annealing instances.
Bit i is LSB; bit 0 denotes spin +1 throughout this project.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any

import networkx as nx
import numpy as np


def canonical_edges(edges: Any, n: int) -> np.ndarray:
    given = np.asarray(edges)
    if given.size and (not np.all(np.isfinite(given)) or not np.all(given == np.floor(given))):
        raise ValueError("edge indices must be finite integers")
    raw = np.asarray(given, dtype=int)
    if raw.size == 0:
        return np.empty((0, 2), dtype=int)
    if raw.ndim != 2 or raw.shape[1] != 2:
        raise ValueError("edges must have shape (E, 2)")
    out = np.sort(raw, axis=1)
    if np.any(out < 0) or np.any(out >= n) or np.any(out[:, 0] == out[:, 1]):
        raise ValueError("edge endpoint out of range or self-loop")
    if len(np.unique(out, axis=0)) != len(out):
        raise ValueError("duplicate undirected edges")
    return out


@dataclass
class IsingProblem:
    h: np.ndarray
    edges: np.ndarray
    J: np.ndarray
    family: str = "custom"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.h = np.asarray(self.h, dtype=float)
        if self.h.ndim != 1 or not len(self.h):
            raise ValueError("h must be a nonempty vector")
        self.edges = canonical_edges(self.edges, self.n)
        self.J = np.asarray(self.J, dtype=float)
        if self.J.shape != (len(self.edges),) or not np.all(np.isfinite(self.J)):
            raise ValueError("invalid J")
        if not np.all(np.isfinite(self.h)):
            raise ValueError("invalid h")

    @property
    def n(self) -> int:
        return len(self.h)

    def energy(self, spins: np.ndarray) -> np.ndarray:
        z = np.asarray(spins)
        if z.shape[-1] != self.n:
            raise ValueError("spin dimension mismatch")
        return z @ self.h + np.sum(
            self.J * z[..., self.edges[:, 0]] * z[..., self.edges[:, 1]], axis=-1
        )


def all_spins(n: int, max_qubits: int = 20) -> np.ndarray:
    if not 1 <= n <= max_qubits:
        raise ValueError(f"enumeration limited to 1..{max_qubits} qubits, got {n}")
    bits = (np.arange(1 << n, dtype=np.uint64)[:, None] >> np.arange(n, dtype=np.uint64)) & 1
    return 1 - 2 * bits.astype(np.int8)


def planted_loops(n: int, edges: np.ndarray, rng: np.random.Generator,
                  num_loops: int = 4) -> IsingProblem:
    """Positive weighted sum of singly frustrated simple cycles; certified endpoint.

    Each cycle has one unsatisfied edge at planted z*, attaining its bound
    weight*(2-|cycle|). Summing these bounds certifies z* despite overlapping
    cycles. This certificate says nothing about quantum spectral hardness.
    A cycle basis is used only for generation, never as a permutation feature.
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    edges = canonical_edges(edges, n)
    graph.add_edges_from(edges.tolist())
    cycles = nx.cycle_basis(graph)
    if num_loops < 1 or not cycles:
        raise ValueError("planted loops need a cyclic support and num_loops >= 1")
    planted = rng.choice([-1, 1], size=n)
    weights = {tuple(e): 0.0 for e in edges.tolist()}
    lower_bound = 0.0
    used = []
    for _ in range(num_loops):
        cycle = cycles[int(rng.integers(len(cycles)))]
        amplitude = float(rng.uniform(0.5, 1.5))
        bad = int(rng.integers(len(cycle)))
        for k, u in enumerate(cycle):
            v = cycle[(k + 1) % len(cycle)]
            edge = tuple(sorted((u, v)))
            weights[edge] += amplitude * (1 if k == bad else -1) * planted[u] * planted[v]
        lower_bound += amplitude * (2 - len(cycle))
        used.append({"cycle": cycle, "frustrated_position": bad, "weight": amplitude})
    J = np.array([weights[tuple(e)] for e in edges.tolist()])
    result = IsingProblem(np.zeros(n), edges, J, "planted_loops", {
        "planted_spins": planted.tolist(), "certified_ground_energy": lower_bound,
        "cycles": used, "certificate": "sum_of_attained_cycle_lower_bounds",
    })
    if not np.isclose(result.energy(planted), lower_bound, atol=1e-10):
        raise ArithmeticError("planting certificate failed")
    return result


def generate_problem(n: int, family: str, rng: np.random.Generator,
                     edges: np.ndarray | None = None) -> IsingProblem:
    if n < 2:
        raise ValueError("logical generation requires n >= 2")
    if edges is None:
        edges = np.array([(i, j) for i in range(n) for j in range(i + 1, n)], dtype=int)
    edges = canonical_edges(edges, n)
    if family == "planted_loops":
        return planted_loops(n, edges, rng, num_loops=max(2, n))
    if family == "weighted_maxcut":
        return IsingProblem(np.zeros(n), edges, rng.uniform(0.3, 1.2, len(edges)), family)
    if family == "spin_glass":
        return IsingProblem(rng.uniform(-0.4, 0.4, n), edges, rng.normal(0, 0.7, len(edges)), family)
    if family == "weak_field":
        return IsingProblem(rng.choice([-1., 1.], n) * 1e-3, edges,
                            -rng.uniform(0.3, 1., len(edges)), family)
    raise ValueError(f"unknown family {family!r}")


def sample_chain_lengths(n_logical: int, low: int, high: int, rng: np.random.Generator,
                         distribution: str = "uniform", exponent: float = 2.0) -> np.ndarray:
    """A truncated discrete power law is an experimental option, not a meeting fact."""
    if any(not np.isfinite(x) or int(x) != x for x in (n_logical, low, high)) or n_logical < 1 or low < 1 or high < low:
        raise ValueError("invalid chain length bounds")
    values = np.arange(low, high + 1)
    if distribution == "uniform":
        probs = np.ones(len(values))
    elif distribution in {"power_law", "powerlaw"}:
        if not np.isfinite(exponent) or exponent <= 0:
            raise ValueError("power-law exponent must be positive")
        probs = values.astype(float) ** (-exponent)
    else:
        raise ValueError("distribution must be uniform or power_law")
    return rng.choice(values, size=n_logical, p=probs / probs.sum())


def sample_logical_support(n: int, rng: np.random.Generator, *, kind: str = "complete",
                           edge_probability: float = 0.5) -> np.ndarray:
    """Sample a labelled support without looking at spectral/dynamical labels.

    ``erdos_renyi`` is unconditioned G(n,p), including disconnected supports.
    ``connected_erdos_renyi`` adds a uniformly shuffled path backbone; it is
    deliberately NOT advertised as ER conditioned on connectedness.
    """
    if not isinstance(n, (int, np.integer)) or n < 2:
        raise ValueError("support size must be an integer >= 2")
    if kind == "complete":
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    elif kind in {"cycle", "path"}:
        order = rng.permutation(n)
        pairs = [(int(order[i]), int(order[i + 1])) for i in range(n - 1)]
        if kind == "cycle":
            if n < 3:
                raise ValueError("cycle support needs n >= 3")
            pairs.append((int(order[-1]), int(order[0])))
    elif kind in {"erdos_renyi", "connected_erdos_renyi"}:
        if not np.isfinite(edge_probability) or not 0 <= edge_probability <= 1:
            raise ValueError("edge_probability must lie in [0, 1]")
        chosen = {(i, j) for i in range(n) for j in range(i + 1, n)
                  if rng.random() < edge_probability}
        if kind == "connected_erdos_renyi":
            order = rng.permutation(n)
            chosen.update(tuple(sorted((int(order[i]), int(order[i + 1])))) for i in range(n - 1))
        pairs = sorted(chosen)
    else:
        raise ValueError("unknown logical support kind")
    edges = canonical_edges(np.asarray(pairs, dtype=int).reshape(-1, 2), n)
    return edges[np.lexsort((edges[:, 1], edges[:, 0]))] if len(edges) else edges


def logical_fingerprint(problem: IsingProblem) -> str:
    """Exact labelled coefficient fingerprint; NOT graph/gauge isomorphism.

    Edge order/orientation and signed floating zero are normalized. Family,
    provenance, and parent IDs are excluded so they cannot hide a duplicate.
    """
    order = np.lexsort((problem.edges[:, 1], problem.edges[:, 0]))
    h, J = problem.h.copy(), problem.J[order].copy()
    h[h == 0] = 0.
    J[J == 0] = 0.
    data = {"h": h.tolist(), "edges": problem.edges[order].tolist(), "J": J.tolist()}
    return hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()


@dataclass
class Embedding:
    membership: np.ndarray
    hardware_edges: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        given = np.asarray(self.membership)
        if given.size and (not np.all(np.isfinite(given)) or not np.all(given == np.floor(given))):
            raise ValueError("membership labels must be finite integers")
        self.membership = np.asarray(self.membership, dtype=int)
        if self.membership.ndim != 1 or not len(self.membership):
            raise ValueError("membership must be nonempty")
        if set(self.membership.tolist()) != set(range(int(self.membership.max()) + 1)):
            raise ValueError("logical membership labels must be contiguous and nonnegative")
        self.hardware_edges = canonical_edges(self.hardware_edges, len(self.membership))
        graph = nx.Graph()
        graph.add_nodes_from(range(len(self.membership)))
        graph.add_edges_from(self.hardware_edges.tolist())
        for v in range(self.n_logical):
            nodes = np.flatnonzero(self.membership == v).tolist()
            if not nx.is_connected(graph.subgraph(nodes)):
                raise ValueError(f"chain {v} is disconnected")

    @property
    def n_logical(self) -> int:
        return int(self.membership.max()) + 1

    @property
    def quotient_edges(self) -> np.ndarray:
        pairs = {tuple(sorted((int(self.membership[i]), int(self.membership[j]))))
                 for i, j in self.hardware_edges if self.membership[i] != self.membership[j]}
        return np.array(sorted(pairs), dtype=int).reshape(-1, 2)


def synthetic_lift(problem: IsingProblem, lengths: np.ndarray,
                   rng: np.random.Generator, shape: str = "path", ports: int = 1) -> Embedding:
    """Construct a toy physical graph, NOT an embedding on a commercial topology."""
    original_lengths = np.asarray(lengths)
    if not np.all(np.isfinite(original_lengths)) or not np.all(original_lengths == np.floor(original_lengths)):
        raise ValueError("chain lengths must be finite integers")
    lengths = np.asarray(original_lengths, dtype=int)
    if lengths.shape != (problem.n,) or np.any(lengths < 1) or ports < 1:
        raise ValueError("invalid lengths/ports")
    if shape not in {"path", "star", "random_tree"}:
        raise ValueError("shape must be path/star/random_tree")
    membership = np.repeat(np.arange(problem.n), lengths)
    chains = [np.flatnonzero(membership == i) for i in range(problem.n)]
    hardware = set()
    for chain in chains:
        for k in range(1, len(chain)):
            if shape == "path":
                parent = k - 1
            elif shape == "star":
                parent = 0
            elif shape == "random_tree":
                parent = int(rng.integers(k))
            else:
                raise ValueError("shape must be path/star/random_tree")
            hardware.add(tuple(sorted((int(chain[parent]), int(chain[k])))))
    for v, w in problem.edges:
        possible = [(int(i), int(j)) for i in chains[v] for j in chains[w]]
        chosen = rng.choice(len(possible), size=min(ports, len(possible)), replace=False)
        for idx in chosen:
            hardware.add(tuple(sorted(possible[idx])))
    return Embedding(membership, np.array(sorted(hardware), dtype=int).reshape(-1, 2),
                     {"route": "synthetic_lift", "shape": shape, "ports": ports,
                      "is_commercial_hardware": False, "lengths": lengths.tolist(),
                      "target_lengths": lengths.tolist(), "achieved_lengths": lengths.tolist(),
                      "target_met": True})


def grow_hardware_partition(n_hardware: int, hardware_edges: np.ndarray,
                            target_lengths: np.ndarray, rng: np.random.Generator) -> Embedding:
    """Multi-source connected growth (ink-drop style), with achieved-size reporting.

    Seed disjoint chains, choose a non-full chain with an available frontier,
    attach one free neighbor. Growth may jam; returning smaller chains is
    intentional and MUST NOT be described as sampling exact target lengths.
    Idle hardware sites are excluded from the active operator; original IDs kept.
    """
    hardware_edges = canonical_edges(hardware_edges, n_hardware)
    original = np.asarray(target_lengths)
    if original.size and (not np.all(np.isfinite(original)) or not np.all(original == np.floor(original))):
        raise ValueError("target lengths must be finite integers")
    target = np.asarray(original, dtype=int)
    if target.ndim != 1 or not len(target) or np.any(target < 1) or target.sum() > n_hardware:
        raise ValueError("target lengths invalid or exceed hardware capacity")
    graph = nx.Graph()
    graph.add_nodes_from(range(n_hardware))
    graph.add_edges_from(hardware_edges.tolist())
    owner = np.full(n_hardware, -1, dtype=int)
    seeds = rng.choice(n_hardware, len(target), replace=False)
    owner[seeds] = np.arange(len(target))
    sizes = np.ones(len(target), dtype=int)
    while True:
        frontiers = []
        for v in range(len(target)):
            if sizes[v] >= target[v]:
                continue
            frontier = sorted({u for x in np.flatnonzero(owner == v)
                               for u in graph.neighbors(int(x)) if owner[u] < 0})
            if frontier:
                frontiers.append((v, frontier))
        if not frontiers:
            break
        v, frontier = frontiers[int(rng.integers(len(frontiers)))]
        u = frontier[int(rng.integers(len(frontier)))]
        owner[u] = v
        sizes[v] += 1
    active = np.flatnonzero(owner >= 0)
    remap = {int(old): new for new, old in enumerate(active)}
    edges = [(remap[int(i)], remap[int(j)]) for i, j in hardware_edges if i in remap and j in remap]
    return Embedding(owner[active], np.asarray(edges, dtype=int).reshape(-1, 2), {
        "route": "fixed_hardware_growth", "original_physical_ids": active.tolist(),
        "target_lengths": target.tolist(), "achieved_lengths": sizes.tolist(),
        "target_met": bool(np.array_equal(sizes, target)),
        "unused_qubits": int(n_hardware - len(active)),
        "warning": "quotient support is generated, not guaranteed hard or representative",
    })


@dataclass
class CompiledInstance:
    logical: IsingProblem
    embedding: Embedding
    physical: IsingProblem
    problem_J: np.ndarray
    chain_J: np.ndarray
    programmed_scale: float
    aligned_offset: float
    chain_strength: float

    def fingerprint(self) -> str:
        payload = {"h": self.physical.h.tolist(), "edges": self.physical.edges.tolist(),
                   "J": self.physical.J.tolist(), "membership": self.embedding.membership.tolist(),
                   "logical_h": self.logical.h.tolist(), "logical_edges": self.logical.edges.tolist(),
                   "logical_J": self.logical.J.tolist(), "scale": self.programmed_scale,
                   "decoder": "majority_tie_plus_v1", "schema": 1}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def compile_embedding(problem: IsingProblem, embedding: Embedding, chain_strength: float,
                      rng: np.random.Generator, field_distribution: str = "uniform",
                      coupling_distribution: str = "uniform", h_limit: float = 2.,
                      j_limit: float = 1.) -> CompiledInstance:
    """Compile each logical coefficient once, plus internal chain couplers.

    Limits are explicit symmetric *toy* hardware caps, not D-Wave autoscale.
    Sum raw chain/problem terms before applying the common coefficient scale.
    This scales HZ only: driver and runtime are NOT silently rescaled.
    """
    if field_distribution not in {"uniform", "concentrated"} or coupling_distribution not in {"uniform", "random"}:
        raise ValueError("unknown field/coupling distribution")
    if problem.n != embedding.n_logical:
        raise ValueError("logical graph and embedding disagree")
    if not np.all(np.isfinite([chain_strength, h_limit, j_limit])) or chain_strength <= 0 or min(h_limit, j_limit) <= 0:
        raise ValueError("positive finite chain strength and limits required")
    owner, edges = embedding.membership, embedding.hardware_edges
    h = np.zeros(len(owner))
    pJ, cJ = np.zeros(len(edges)), np.zeros(len(edges))
    for v in range(problem.n):
        members = np.flatnonzero(owner == v)
        if field_distribution == "uniform":
            weights = np.ones(len(members)) / len(members)
        elif field_distribution == "concentrated":
            weights = np.zeros(len(members))
            weights[int(rng.integers(len(members)))] = 1.
        else:
            raise ValueError("unknown field distribution")
        h[members] = problem.h[v] * weights
    cJ[owner[edges[:, 0]] == owner[edges[:, 1]]] = -chain_strength
    for (v, w), coupling in zip(problem.edges, problem.J):
        cross = np.flatnonzero(((owner[edges[:, 0]] == v) & (owner[edges[:, 1]] == w)) |
                               ((owner[edges[:, 0]] == w) & (owner[edges[:, 1]] == v)))
        if not len(cross):
            raise ValueError(f"no physical realization of logical edge {(v, w)}")
        if coupling_distribution == "uniform":
            weights = np.ones(len(cross)) / len(cross)
        elif coupling_distribution == "random":
            weights = rng.dirichlet(np.ones(len(cross)))
        else:
            raise ValueError("unknown coupling distribution")
        pJ[cross] += coupling * weights
    total = pJ + cJ
    divisor = max(1., float(np.max(np.abs(h), initial=0)) / h_limit,
                  float(np.max(np.abs(total), initial=0)) / j_limit)
    alpha = 1 / divisor
    physical = IsingProblem(alpha * h, edges.copy(), alpha * total, "compiled", {
        "coefficient_rule": "symmetric_caps_v1_not_vendor_autoscale",
        "h_limit": h_limit, "j_limit": j_limit,
        "field_distribution": field_distribution, "coupling_distribution": coupling_distribution,
    })
    return CompiledInstance(problem, embedding, physical, alpha * pJ, alpha * cJ,
                            alpha, float(alpha * cJ.sum()), chain_strength)


def spin_chunks(n: int, chunk_size: int = 16384, max_qubits: int = 20):
    """Enumerate exactly in bounded chunks, with the same LSB spin convention."""
    if not 1 <= n <= max_qubits or chunk_size < 1:
        raise ValueError("invalid exact-enumeration size/chunk budget")
    for start in range(0, 1 << n, chunk_size):
        stop = min(1 << n, start + chunk_size)
        bits = (np.arange(start, stop, dtype=np.uint64)[:, None] >> np.arange(n, dtype=np.uint64)) & 1
        yield start, 1 - 2 * bits.astype(np.int8)


def validate_compilation(instance: CompiledInstance, atol: float = 1e-9, *,
                         max_qubits: int = 20, chunk_size: int = 16384) -> dict[str, Any]:
    mismatch, Elogical = 0., float("inf")
    for _, z in spin_chunks(instance.logical.n, chunk_size, max_qubits):
        embedded = z[:, instance.embedding.membership]
        energy = instance.logical.energy(z)
        expected = instance.programmed_scale * energy + instance.aligned_offset
        mismatch = max(mismatch, float(np.max(np.abs(instance.physical.energy(embedded) - expected))))
        Elogical = min(Elogical, float(np.min(energy)))
    if mismatch > atol:
        raise ArithmeticError("unbroken-chain energy identity failed")
    degeneracy = sum(int(np.sum(np.isclose(instance.logical.energy(z), Elogical, atol=atol, rtol=0)))
                     for _, z in spin_chunks(instance.logical.n, chunk_size, max_qubits))
    Ephys = min(float(np.min(instance.physical.energy(z)))
                for _, z in spin_chunks(instance.physical.n, chunk_size, max_qubits))
    predicted = instance.programmed_scale * Elogical + instance.aligned_offset
    ground_ok = bool(np.isclose(Ephys, predicted, atol=atol, rtol=0))
    return {"aligned_energy_max_error": mismatch, "logical_ground_energy": Elogical,
            "physical_ground_energy": Ephys, "aligned_ground_energy": predicted,
            "has_aligned_physical_ground_state": ground_ok,
            "logical_ground_degeneracy": degeneracy}


def output_observables(instance: CompiledInstance, atol: float = 1e-9, *,
                       max_qubits: int = 20, chunk_size: int = 16384,
                       ground_energy: float | None = None) -> dict[str, np.ndarray]:
    """All physical output strings accepted iff deterministic decode is logical-optimal.

    Tie rule +1 is declared (not generally gauge invariant). Chain-preserving
    gauges require consistent decoder un-gauging; never augment h,J alone.
    """
    owner = instance.embedding.membership
    if instance.physical.n > max_qubits:
        raise ValueError("physical endpoint exceeds exact-enumeration cap")
    ground = ground_energy if ground_energy is not None else min(
        float(instance.logical.energy(z).min()) for _, z in spin_chunks(instance.logical.n, chunk_size, max_qubits))
    result = {name: np.empty(1 << instance.physical.n, dtype=float) for name in
              ("success", "decoded_energy", "any_chain_break", "chain_break_fraction", "physical_energy")}
    for start, spins in spin_chunks(instance.physical.n, chunk_size, max_qubits):
        decoded = np.ones((len(spins), instance.logical.n), dtype=np.int8)
        broken = np.zeros((len(spins), instance.logical.n), dtype=bool)
        for v in range(instance.logical.n):
            group = spins[:, owner == v]
            decoded[:, v] = np.where(group.sum(axis=1) >= 0, 1, -1)
            broken[:, v] = np.any(group != group[:, :1], axis=1)
        energy = instance.logical.energy(decoded)
        sl = slice(start, start + len(spins))
        result["success"][sl] = np.isclose(energy, ground, atol=atol, rtol=0)
        result["decoded_energy"][sl] = energy
        result["any_chain_break"][sl] = broken.any(axis=1)
        result["chain_break_fraction"][sl] = broken.mean(axis=1)
        result["physical_energy"][sl] = instance.physical.energy(spins)
    return result


def parent_splits(families: list[str], seed: int = 0) -> dict[str, str]:
    """Stratify parents BEFORE embeddings, runtimes, gauges or coefficient variants."""
    rng = np.random.default_rng(seed)
    mapping = {}
    for family in sorted(set(families)):
        indices = [i for i, f in enumerate(families) if f == family]
        if len(indices) < 3:
            raise ValueError("at least three logical parents per family for train/validation/test")
        rng.shuffle(indices)
        n_holdout = max(1, len(indices) // 5)
        for k, i in enumerate(indices):
            split = "test" if k < n_holdout else "validation" if k < 2 * n_holdout else "train"
            mapping[f"parent_{i:04d}"] = split
    return mapping
