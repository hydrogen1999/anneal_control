"""Spin-vector Monte Carlo: a semiclassical surrogate that reaches device sizes.

Exact state-vector dynamics costs O(2**N) and stops around 30-40 qubits on any
machine; the spectral teacher stops near 16, because an eigendecomposition is a
2**N x 2**N matrix. A deployed annealer has thousands of qubits. Every claim in
this project is therefore made two to three orders of magnitude below the regime
the claims are about, and no amount of compute changes that.

Spin-vector Monte Carlo (Shin, Smith, Smolin and Vazirani, 2014) replaces each
qubit with a classical O(2) rotor at angle theta, so X_i becomes sin(theta_i) and
Z_i becomes cos(theta_i). The energy is then an ordinary classical function,
Metropolis updates cost O(N) per sweep, and thousands of spins are routine. It is
the standard semiclassical model in the D-Wave literature, which matters: this is
an established reference rather than something invented to make a number larger.

**It is not quantum.** A result produced here is a statement about the surrogate
until the surrogate has been shown to agree with exact dynamics somewhere. The
overlap region is 10-14 physical qubits, where this project already has exact
outcomes for the same records and the same controls. Validate there; only then
run where truth is unreachable; and report the disagreement rather than the
convenient half of it.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .telemetry import _safe


def rotor_energy(theta: np.ndarray, terms, a: float, b: float, c: float) -> float:
    """Classical energy of the rotor configuration under H = a*H_X + b*H_Z + c*H_XX.

    The project's convention is H_X = -sum(X), H_Z = sum(h Z) + sum(J ZZ) and
    H_XX = sum(K XX), so the semiclassical replacement is X -> sin(theta),
    Z -> cos(theta) term by term.
    """
    theta = np.asarray(theta, dtype=float)
    sin, cos = np.sin(theta), np.cos(theta)
    energy = -a * float(sin.sum())
    energy += b * float(np.asarray(terms.h, dtype=float) @ cos)
    if len(terms.zz_edges):
        edges = np.asarray(terms.zz_edges, dtype=int)
        weights = np.asarray(terms.zz_weights, dtype=float)
        energy += b * float((weights * cos[edges[:, 0]] * cos[edges[:, 1]]).sum())
    if c and terms.xx_edges is not None and len(terms.xx_edges):
        edges = np.asarray(terms.xx_edges, dtype=int)
        weights = np.asarray(terms.xx_weights, dtype=float)
        energy += c * float((weights * sin[edges[:, 0]] * sin[edges[:, 1]]).sum())
    return energy


def svmc_cost_estimate(*, n_qubits: int, steps: int, sweeps: int, restarts: int) -> dict:
    """Why this exists at all: the cost is linear in qubits, not exponential."""
    updates = int(n_qubits) * int(steps) * int(sweeps) * int(restarts)
    return {"spin_updates": updates,
            "state_bytes": int(n_qubits) * int(restarts) * 8,
            # Reported in log2 because the point is that it leaves float range:
            # 2**2000 amplitudes is not a number any machine holds.
            "exact_state_log2_bytes": float(int(n_qubits) + 4),
            "note": "O(N * steps * sweeps * restarts); exact dynamics is O(2**N) per step"}


def svmc_success(terms, schedule, path, *, ground_states: np.ndarray, steps: int = 200,
                 sweeps: int = 8, restarts: int = 256, temperature: float = 0.05,
                 seed: int = 0) -> dict:
    """Anneal a population of rotors along ``schedule`` and report ground-state hits.

    ``ground_states`` is the set of accepted spin configurations, supplied by the
    caller so the decoder is the same object the exact pipeline uses rather than
    a second implementation that could drift from it.
    """
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    for name, value in (("steps", steps), ("sweeps", sweeps), ("restarts", restarts)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")

    n = terms.n_qubits
    rng = np.random.default_rng(seed)
    # Start in the transverse ground state: every rotor flat in the x-y plane.
    theta = np.full((restarts, n), np.pi / 2)
    fields = np.asarray(terms.h, dtype=float)
    edges = np.asarray(terms.zz_edges, dtype=int) if len(terms.zz_edges) else np.zeros((0, 2), int)
    weights = np.asarray(terms.zz_weights, dtype=float) if len(terms.zz_edges) else np.zeros(0)

    taus = np.linspace(0.0, 1.0, steps)
    for tau in taus:
        a, b, c = path.coefficients(float(np.asarray(schedule(np.array([tau]))).item()))
        for _ in range(sweeps):
            order = rng.permutation(n)
            for site in order:
                proposal = rng.uniform(0.0, np.pi, size=restarts)
                cos_old, cos_new = np.cos(theta[:, site]), np.cos(proposal)
                sin_old, sin_new = np.sin(theta[:, site]), np.sin(proposal)
                delta = -a * (sin_new - sin_old) + b * fields[site] * (cos_new - cos_old)
                if len(edges):
                    touching = np.where((edges[:, 0] == site) | (edges[:, 1] == site))[0]
                    for index in touching:
                        other = edges[index, 1] if edges[index, 0] == site else edges[index, 0]
                        delta += b * weights[index] * (cos_new - cos_old) * np.cos(theta[:, other])
                accept = (delta <= 0) | (rng.random(restarts) < np.exp(-np.clip(delta, 0, 700) / temperature))
                theta[accept, site] = proposal[accept]

    spins = np.where(np.cos(theta) >= 0, 1.0, -1.0)
    accepted = np.asarray(ground_states, dtype=float)
    hits = np.zeros(restarts, dtype=bool)
    for target in accepted:
        hits |= np.all(spins == target[None, :], axis=1)
    return _safe({
        "success": float(hits.mean()),
        "loss": float(1.0 - hits.mean()),
        "restarts": restarts, "steps": steps, "sweeps": sweeps,
        "temperature": temperature, "n_qubits": int(n),
        "n_accepted_states": int(len(accepted)),
        "scope": ("semiclassical rotor surrogate, not quantum dynamics; a number here is a "
                  "claim about the surrogate until it is checked against exact outcomes"),
    })
