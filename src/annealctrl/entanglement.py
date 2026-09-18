"""How much entanglement these trajectories carry, and therefore whether MPS could scale them.

Exact state-vector simulation stops near 30-40 qubits and the spectral teacher
near 16, while a deployed annealer has thousands. A matrix product state
represents a pure state exactly with bond dimension ``chi = exp(S)``, where ``S``
is the entanglement entropy across the cut, so the feasibility of a tensor
network at hundreds of qubits is a measurable property of the trajectories rather
than a matter of opinion.

Two regimes bound the answer. An area-law trajectory keeps ``S`` roughly constant
in system size and needs a fixed, small ``chi``. A volume-law trajectory has
``S`` growing as ``(N/2) ln 2`` and needs ``chi`` exponential in ``N``, which is
exact simulation wearing a different name.

Measured on the project's own records and controls, so the answer is about this
problem family. It says nothing about entanglement in annealing generally.
"""
from __future__ import annotations

import numpy as np


def bipartition_entropy(state: np.ndarray, n_qubits: int, *, cut: int) -> float:
    """Von Neumann entropy in nats across a contiguous cut of the first ``cut`` qubits."""
    state = np.asarray(state).ravel()
    if state.size != 2 ** int(n_qubits):
        raise ValueError(f"state must have 2**n entries for n={n_qubits}, got {state.size}")
    if not 0 < cut < n_qubits:
        raise ValueError("cut must lie strictly inside the register")
    matrix = state.reshape(2 ** cut, 2 ** (n_qubits - cut))
    singular = np.linalg.svd(matrix, compute_uv=False)
    probabilities = singular ** 2
    total = probabilities.sum()
    if total <= 0:
        return 0.0
    probabilities = probabilities[probabilities > 1e-15] / total
    return float(-(probabilities * np.log(probabilities)).sum())


def volume_law_entropy(n_qubits: int) -> float:
    """The ceiling: a maximally entangled balanced cut carries (N/2) ln 2 nats."""
    return float((int(n_qubits) // 2) * np.log(2.0))


def required_bond_dimension(entropy: float, *, safety: float = 1.0) -> int:
    """Bond dimension an MPS needs to carry this much entropy across one cut.

    ``exp(S)`` is the exact Schmidt rank of a flat spectrum; real spectra decay,
    so this is an upper bound on what a faithful representation costs and a
    lower bound on what a truncated one must keep to stay honest.
    """
    if not np.isfinite(entropy) or entropy < 0:
        raise ValueError("entropy must be finite and nonnegative")
    return int(np.ceil(safety * np.exp(float(entropy))))


def entanglement_trajectory(terms, schedule, runtime: float, *, path=None, steps: int = 256,
                            cut: int | None = None, samples: int = 32) -> dict:
    """Entropy across a balanced cut along the real propagated trajectory.

    The evolution loop mirrors ``physics._split_evolve`` exactly rather than
    approximating it, and a test pins that the final state agrees with
    ``physics.propagate`` to 1e-10. Rewriting a propagator to instrument it is
    how an instrument ends up measuring something other than the thing it
    reports on.
    """
    import numpy as np

    from .physics import AnnealPath, HamiltonianOperator, _path_coordinate

    path = AnnealPath() if path is None else path
    operator = HamiltonianOperator(terms)
    n = terms.n_qubits
    cut = n // 2 if cut is None else int(cut)

    psi = np.zeros(2 ** n, dtype=complex)
    psi[:] = 1.0 / np.sqrt(2 ** n)          # transverse ground state at s = 0
    dt = runtime / steps
    checkpoints = sorted({int(round(k)) for k in np.linspace(0, steps, min(samples, steps + 1))})
    trace = []
    for k in range(steps):
        s = _path_coordinate(float(np.asarray(schedule((k + 0.5) / steps)).item()))
        a, b, c = path.coefficients(s)
        phase = np.exp((-0.5j * dt * b) * operator.diagonal)
        psi = psi * phase
        if a:
            angle = -dt * a
            for i in range(n):
                psi = np.cos(angle) * psi - 1j * np.sin(angle) * psi[operator.indices ^ (1 << i)]
        if c:
            for (i, j), weight in zip(terms.xx_edges, terms.xx_weights):
                angle = dt * c * float(weight)
                mask = (1 << int(i)) | (1 << int(j))
                psi = np.cos(angle) * psi - 1j * np.sin(angle) * psi[operator.indices ^ mask]
        psi = psi * phase
        if (k + 1) in checkpoints:
            entropy = bipartition_entropy(psi, n, cut=cut)
            trace.append({"step": k + 1, "tau": (k + 1) / steps, "s": s, "entropy": entropy})

    peak = max((point["entropy"] for point in trace), default=0.0)
    ceiling = volume_law_entropy(n)
    return {"n_qubits": int(n), "cut": cut, "steps": steps,
            "peak_entropy": float(peak),
            "volume_law_entropy": float(ceiling),
            "fraction_of_volume_law": float(peak / ceiling) if ceiling else None,
            "required_bond_dimension": required_bond_dimension(peak),
            "trace": trace,
            "final_state": psi}
