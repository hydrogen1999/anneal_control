"""Spin-reversal gauge: an exact, label-preserving augmentation.

Flipping a subset S of spins conjugates the problem Hamiltonian by
prod_{i in S} X_i, which sends

    h_i  -> -h_i            for i in S
    J_ij -> -J_ij           when exactly one of i, j lies in S

and leaves the transverse driver sum_i X_i untouched. The spectrum of
Hbar(s) is therefore identical at every s, and the decoded logical success is
carried to the gauged good set, so **every stored label remains correct**.
That is what makes this usable as a training augmentation rather than as a
relabelling exercise.

The design document's factorial table asks for exactly this arm:

> Symmetry | Signed baseline; gauge augmentation; specialized covariant model |
> Whether stronger inductive bias improves robustness without deleting
> frustration.

The last clause is the trap. Removing signs to obtain invariance would make a
frustrated loop indistinguishable from an unfrustrated one, and the document
devotes a subsection to refusing it. A gauge *augmentation* keeps every sign
and only changes which gauge representative the encoder sees, so frustration --
the product of couplings around a loop -- is untouched. `loop_product` exists
so a test can assert that rather than a comment claim it.
"""
from __future__ import annotations

import numpy as np


def gauge_signs(n: int, rng) -> np.ndarray:
    """A random +/-1 vector: -1 marks a flipped spin."""
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be a positive integer")
    return np.where(rng.random(int(n)) < 0.5, -1.0, 1.0)


def apply_gauge(h: np.ndarray, edges: np.ndarray, couplings: np.ndarray,
                signs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Transformed fields and couplings under the given spin reversal."""
    h = np.asarray(h, dtype=float)
    edges = np.asarray(edges, dtype=int).reshape(-1, 2)
    couplings = np.asarray(couplings, dtype=float)
    signs = np.asarray(signs, dtype=float)
    if signs.shape != h.shape:
        raise ValueError("signs must have one entry per site")
    if not np.all(np.isin(signs, (-1.0, 1.0))):
        raise ValueError("gauge signs must be +1 or -1")
    if edges.shape[0] != couplings.shape[0]:
        raise ValueError("edges and couplings must agree in length")
    if edges.size and (edges.max() >= h.size or edges.min() < 0):
        raise ValueError("edge endpoints must index the sites")
    return h * signs, couplings * signs[edges[:, 0]] * signs[edges[:, 1]]


def loop_product(edges: np.ndarray, couplings: np.ndarray, loop: np.ndarray) -> float:
    """Sign of the coupling product around a closed loop: the frustration invariant.

    A gauge transformation cannot change it. Anything that does has deleted
    physics rather than removed a redundancy.
    """
    edges = np.asarray(edges, dtype=int).reshape(-1, 2)
    couplings = np.asarray(couplings, dtype=float)
    loop = np.asarray(loop, dtype=int)
    if loop.size < 3:
        raise ValueError("a loop needs at least three sites")
    product = 1.0
    for a, b in zip(loop, np.roll(loop, -1)):
        hit = np.flatnonzero(((edges[:, 0] == a) & (edges[:, 1] == b))
                             | ((edges[:, 0] == b) & (edges[:, 1] == a)))
        if not hit.size:
            raise ValueError(f"loop edge ({a}, {b}) is not in the graph")
        product *= float(couplings[hit[0]])
    return product


def gauge_record(record, signs) -> dict:
    """Apply a LOGICAL gauge to a record, induced on the physical graph.

    The gauge is chosen on logical spins and pushed down through ``membership``,
    so every physical qubit in a chain receives the same sign. Two consequences
    make this the right transformation for an embedded problem rather than an
    arbitrary physical one:

    * intra-chain couplings see ``s_i s_j = sigma^2 = +1`` and are **unchanged**,
      so chains stay ferromagnetic and the chain strength keeps its meaning;
    * inter-chain couplings and both field vectors transform together, so the
      logical and physical descriptions stay consistent with each other.

    Labels are untouched because the spectrum is. The returned dict is a shallow
    copy: only the six transformed arrays are new.
    """
    import numpy as np

    membership = np.asarray(record["membership"], dtype=int)
    signs = np.asarray(signs, dtype=float)
    n_logical = int(np.asarray(record["logical_h"]).size)
    if signs.shape != (n_logical,):
        raise ValueError(f"signs must carry one entry per logical spin ({n_logical})")
    if not np.all(np.isin(signs, (-1.0, 1.0))):
        raise ValueError("gauge signs must be +1 or -1")
    if membership.size and (membership.max() >= n_logical or membership.min() < 0):
        raise ValueError("membership must index the logical spins")

    physical_signs = signs[membership]
    out = dict(record)
    out["logical_h"], out["logical_J"] = apply_gauge(
        record["logical_h"], record["logical_edges"], record["logical_J"], signs)
    out["physical_h"], out["physical_J"] = apply_gauge(
        record["physical_h"], record["physical_edges"], record["physical_J"], physical_signs)
    if "problem_J" in record:
        problem = np.asarray(record["problem_J"], dtype=float)
        edges = np.asarray(record["logical_edges"], dtype=int).reshape(-1, 2)
        if problem.shape[0] == edges.shape[0]:
            out["problem_J"] = problem * signs[edges[:, 0]] * signs[edges[:, 1]]
    return out
