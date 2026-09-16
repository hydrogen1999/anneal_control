"""Auditable closed-system Ising simulation; no QPU calls or hidden units.

Bit ``i`` is the least-significant-bit indexed physical qubit. Computational
bit 0 has Z eigenvalue +1. H_X=-sum X_i; H_XX=sum K_ij X_i X_j.
The state vector still costs O(2**N): matrix-free is not polynomial simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any, Sequence

import numpy as np

Schedule = Callable[[float], float]


def _edges(edges: Any, weights: Any, n: int, name: str) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray([] if edges is None else edges)
    if raw.size == 0:
        raw = np.empty((0, 2), dtype=np.int64)
    if raw.ndim != 2 or raw.shape[1] != 2:
        raise ValueError(f"{name}_edges must have shape (E, 2)")
    if not np.issubdtype(raw.dtype, np.integer):
        raise ValueError(f"{name}_edges must contain integers")
    e = np.array(raw, dtype=np.int64, copy=True)
    w = np.array([] if weights is None else weights, dtype=np.float64, copy=True)
    if w.shape != (len(e),) or not np.isfinite(w).all():
        raise ValueError(f"{name}_weights must be finite and have shape (E,)")
    if np.any(e < 0) or np.any(e >= n) or np.any(e[:, 0] == e[:, 1]):
        raise ValueError(f"{name} edges must have distinct valid endpoints")
    e.sort(axis=1)
    if len(np.unique(e, axis=0)) != len(e):
        raise ValueError(f"duplicate {name} edge: sum programmed contributions first")
    e.flags.writeable = False
    w.flags.writeable = False
    return e, w


@dataclass(frozen=True)
class HamiltonianTerms:
    """Actual programmed coefficients, not separately double-counted metadata."""

    n_qubits: int
    h: np.ndarray
    zz_edges: np.ndarray
    zz_weights: np.ndarray
    xx_edges: np.ndarray | None = None
    xx_weights: np.ndarray | None = None

    def __post_init__(self) -> None:
        if isinstance(self.n_qubits, bool) or not isinstance(self.n_qubits, (int, np.integer)):
            raise ValueError("n_qubits must be an integer")
        if not 1 <= self.n_qubits <= 62:
            raise ValueError("n_qubits must lie in [1, 62]; allocations have separate guards")
        h = np.array(self.h, dtype=np.float64, copy=True)
        if h.shape != (self.n_qubits,) or not np.isfinite(h).all():
            raise ValueError("h must have shape (n_qubits,) and finite real values")
        h.flags.writeable = False
        object.__setattr__(self, "h", h)
        for name in ("zz", "xx"):
            edges, weights = _edges(getattr(self, f"{name}_edges"), getattr(self, f"{name}_weights"), self.n_qubits, name)
            object.__setattr__(self, f"{name}_edges", edges)
            object.__setattr__(self, f"{name}_weights", weights)


@dataclass(frozen=True)
class AnnealPath:
    """Dimensionless path: (a,b,c)=gamma*(1-s,s,4*lambda*s*(1-s)).

    ``energy_scale`` rescales the WHOLE path. It is not Ising-only autoscaling.
    XX is a simulated catalyst, not a claim of native D-Wave programmability.
    """

    catalyst_strength: float = 0.0
    energy_scale: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.catalyst_strength):
            raise ValueError("catalyst_strength must be finite")
        if not np.isfinite(self.energy_scale) or self.energy_scale <= 0:
            raise ValueError("energy_scale must be finite and positive")

    def coefficients(self, s: float) -> tuple[float, float, float]:
        s = _path_coordinate(s)
        g = self.energy_scale
        return g * (1 - s), g * s, g * 4 * self.catalyst_strength * s * (1 - s)

    def derivative(self, s: float) -> tuple[float, float, float]:
        s = _path_coordinate(s)
        g = self.energy_scale
        return -g, g, g * 4 * self.catalyst_strength * (1 - 2 * s)


def _path_coordinate(s: float) -> float:
    value = float(s)
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("schedule/path coordinate must lie in [0, 1]")
    return value


def _array_module(backend: str):
    if backend == "numpy":
        return np
    if backend == "cupy":
        try:
            import cupy as cp
        except ImportError as exc:
            raise ImportError("CuPy backend requested; install the CuPy build matching your CUDA runtime") from exc
        try:
            if cp.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError("CuPy backend requested but no CUDA device is available")
            cp.cuda.runtime.memGetInfo()
        except cp.cuda.runtime.CUDARuntimeError as exc:
            raise RuntimeError("CuPy backend requested but the CUDA device/runtime is not usable; no CPU fallback") from exc
        return cp
    raise ValueError("backend must be 'numpy' or 'cupy'")


def estimate_state_workspace(n_qubits: int, *, batch_size: int = 1,
                             step_doubling: bool = False) -> dict[str, int]:
    """Conservative explicit-array estimate, not measured allocator peak.

    Includes complex128 states, XOR gather/rotation temporaries, diagonal and
    indices, and an extra retained coarse batch during step doubling. Backend
    allocator pools, eigensolver workspaces and Python overhead are excluded.
    """
    if isinstance(n_qubits, bool) or not isinstance(n_qubits, (int, np.integer)) or not 1 <= n_qubits <= 62:
        raise ValueError("n_qubits must be an integer in [1, 62]")
    if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    dimension = 1 << int(n_qubits)
    state_bytes = 16 * dimension * int(batch_size)
    estimated = 32 * dimension + (128 if step_doubling else 112) * dimension * int(batch_size)
    return {"dimension": dimension, "batch_size": int(batch_size),
            "state_bytes": state_bytes, "estimated_peak_bytes": estimated}


def backend_device_info(backend: str = "numpy") -> dict:
    """Read-only backend availability/memory query; never substitutes a device."""
    xp = _array_module(backend)
    if backend == "numpy":
        return {"backend": backend, "device": "cpu", "free_device_bytes": None,
                "total_device_bytes": None, "numpy_version": np.__version__}
    free, total = xp.cuda.runtime.memGetInfo()
    device = int(xp.cuda.runtime.getDevice())
    properties = xp.cuda.runtime.getDeviceProperties(device)
    name = properties.get("name", properties.get(b"name", "cuda"))
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="replace")
    return {"backend": backend, "device": str(name), "device_id": device,
            "free_device_bytes": int(free), "total_device_bytes": int(total),
            "cupy_version": xp.__version__}


def _check_workspace(estimated_bytes: int, max_state_bytes: int, xp, backend: str) -> None:
    if (isinstance(max_state_bytes, bool) or not isinstance(max_state_bytes, (int, np.integer))
            or max_state_bytes < 1):
        raise ValueError("max_state_bytes must be a positive integer")
    if estimated_bytes > max_state_bytes:
        raise MemoryError("state-vector work estimate exceeds max_state_bytes; exponential physical-qubit cost")
    if backend == "cupy":
        free, _ = xp.cuda.runtime.memGetInfo()
        if estimated_bytes > int(0.8 * free):
            raise MemoryError("state-vector work estimate exceeds 80% of currently free CUDA memory; reduce qubits/batch size")


class HamiltonianOperator:
    """O(2**N) storage and O((N+|E|)2**N) matrix-vector action.

    No cache of all bit-flipped indices is retained. The conservative byte
    estimate includes several work vectors; actual backend allocator overhead
    is additional and should be measured for the GPU being used.
    """

    def __init__(self, terms: HamiltonianTerms, backend: str = "numpy", *, max_state_bytes: int = 512 * 2**20):
        self.terms = terms
        self.dimension = 1 << terms.n_qubits
        if (isinstance(max_state_bytes, bool) or not isinstance(max_state_bytes, (int, np.integer))
                or max_state_bytes < 1):
            raise ValueError("max_state_bytes must be a positive integer")
        if 128 * self.dimension > max_state_bytes:
            raise MemoryError("state-vector work estimate exceeds max_state_bytes; exponential physical-qubit cost")
        self.xp = _array_module(backend)
        self.backend = backend
        _check_workspace(128 * self.dimension, max_state_bytes, self.xp, backend)
        xp = self.xp
        self.indices = xp.arange(self.dimension, dtype=xp.int64)
        self.diagonal = xp.zeros(self.dimension, dtype=xp.float64)
        for i, value in enumerate(terms.h):
            self.diagonal += float(value) * (1 - 2 * ((self.indices >> i) & 1))
        for (i, j), value in zip(terms.zz_edges, terms.zz_weights):
            parity = ((self.indices >> int(i)) ^ (self.indices >> int(j))) & 1
            self.diagonal += float(value) * (1 - 2 * parity)

    def apply_parts(self, state: Any, coefficients: tuple[float, float, float]):
        xp = self.xp
        psi = xp.asarray(state, dtype=xp.complex128)
        if psi.shape != (self.dimension,):
            raise ValueError("state has incorrect dimension")
        a, b, c = coefficients
        out = b * self.diagonal * psi
        if a:
            for i in range(self.terms.n_qubits):
                out -= a * psi[self.indices ^ (1 << i)]
        if c:
            for (i, j), weight in zip(self.terms.xx_edges, self.terms.xx_weights):
                out += c * float(weight) * psi[self.indices ^ ((1 << int(i)) | (1 << int(j)))]
        return out

    def matvec(self, state: Any, s: float, path: AnnealPath | None = None):
        return self.apply_parts(state, (path or AnnealPath()).coefficients(s))

    def derivative_matvec(self, state: Any, s: float, path: AnnealPath | None = None):
        return self.apply_parts(state, (path or AnnealPath()).derivative(s))

    def dense_parts(self, coefficients: tuple[float, float, float], *, max_qubits: int = 12) -> np.ndarray:
        """Dense O(4**N) teacher only; returned on host even for a GPU operator."""
        if self.terms.n_qubits > max_qubits:
            raise MemoryError("dense Hamiltonian exceeds max_qubits teacher cap")
        a, b, c = coefficients
        diagonal = self.to_numpy(self.diagonal)
        matrix = np.diag(b * diagonal).astype(np.complex128)
        indices = np.arange(self.dimension)
        for i in range(self.terms.n_qubits):
            matrix[indices ^ (1 << i), indices] -= a
        for (i, j), value in zip(self.terms.xx_edges, self.terms.xx_weights):
            matrix[indices ^ ((1 << int(i)) | (1 << int(j))), indices] += c * float(value)
        return matrix

    def dense(self, s: float, path: AnnealPath | None = None, *, max_qubits: int = 12) -> np.ndarray:
        return self.dense_parts((path or AnnealPath()).coefficients(s), max_qubits=max_qubits)

    def to_numpy(self, value: Any) -> np.ndarray:
        return np.asarray(value) if self.backend == "numpy" else self.xp.asnumpy(value)


@dataclass(frozen=True)
class PropagationResult:
    state: np.ndarray
    norm_error: float
    steps: int
    step_doubling_error: float | None = None
    backend: str = "numpy"
    method: str = "midpoint_strang"


@dataclass(frozen=True)
class BatchPropagationResult:
    states: np.ndarray
    norm_errors: np.ndarray
    steps: int
    step_doubling_errors: np.ndarray | None
    accepted: np.ndarray | None
    backend: str
    estimated_peak_bytes: int
    method: str = "batched_midpoint_strang"


def plus_state(n_qubits: int) -> np.ndarray:
    """Ground state of -sum X at the canonical path's s=0 endpoint."""
    if not isinstance(n_qubits, (int, np.integer)) or not 1 <= n_qubits <= 24:
        raise ValueError("plus_state helper is capped at 24 qubits")
    return np.full(1 << n_qubits, 2.0 ** (-n_qubits / 2), dtype=np.complex128)


def _initial_state(operator: HamiltonianOperator, schedule: Schedule, initial_state: Any):
    xp = operator.xp
    start = _path_coordinate(schedule(0.0))
    _path_coordinate(schedule(1.0))
    if initial_state is None:
        if abs(start) > 1e-14:
            raise ValueError("default |+> initialization requires schedule(0)=0; supply initial_state otherwise")
        psi = xp.full(operator.dimension, 1 / np.sqrt(operator.dimension), dtype=xp.complex128)
    else:
        psi = xp.asarray(initial_state, dtype=xp.complex128).copy()
    if psi.shape != (operator.dimension,) or not bool(xp.isfinite(psi).all()):
        raise ValueError("initial_state must be finite and have dimension 2**N")
    if abs(float(xp.vdot(psi, psi).real) - 1) > 1e-10:
        raise ValueError("initial_state must be normalized; normalization is never silently repaired")
    return psi


def _validate_propagation(runtime: float, steps: int) -> None:
    if not np.isfinite(runtime) or runtime < 0:
        raise ValueError("dimensionless runtime must be finite and nonnegative")
    if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or steps < 1:
        raise ValueError("steps must be a positive integer")


def _split_evolve(operator: HamiltonianOperator, schedule: Schedule, runtime: float, steps: int, path: AnnealPath, initial_state: Any):
    xp = operator.xp
    psi = _initial_state(operator, schedule, initial_state)
    dt = runtime / steps
    for k in range(steps):
        s = _path_coordinate(schedule((k + 0.5) / steps))
        a, b, c = path.coefficients(s)
        # All X_i and X_i X_j commute: their product rotation is exact.
        phase = xp.exp((-0.5j * dt * b) * operator.diagonal)
        psi *= phase
        if a:
            angle = -dt * a  # H_X=-sum X, so exp(-i dt H_X)=exp(+i dt sum X)
            for i in range(operator.terms.n_qubits):
                psi = np.cos(angle) * psi - 1j * np.sin(angle) * psi[operator.indices ^ (1 << i)]
        if c:
            for (i, j), weight in zip(operator.terms.xx_edges, operator.terms.xx_weights):
                angle = dt * c * float(weight)
                mask = (1 << int(i)) | (1 << int(j))
                psi = np.cos(angle) * psi - 1j * np.sin(angle) * psi[operator.indices ^ mask]
        psi *= phase
    return psi


def propagate(terms: HamiltonianTerms, schedule: Schedule, runtime: float, *, steps: int = 512,
              path: AnnealPath | None = None, backend: str = "numpy", initial_state: Any = None,
              step_doubling: bool = False, max_state_bytes: int = 512 * 2**20) -> PropagationResult:
    """Second-order midpoint/Strang propagation in dimensionless time.

    If step_doubling=True the returned state uses 2*steps and the reported
    estimate is ||psi_(2M)-psi_M||/3. This is an asymptotic diagnostic, not a
    rigorous error certificate. Align abrupt schedule switches to the grid;
    smooth-problem second-order convergence need not hold across unresolved
    discontinuities. States are not renormalized. GPU transfer occurs once.
    """
    _validate_propagation(runtime, steps)
    operator = HamiltonianOperator(terms, backend, max_state_bytes=max_state_bytes)
    chosen_path = path or AnnealPath()
    psi = _split_evolve(operator, schedule, runtime, steps, chosen_path, initial_state)
    estimate = None
    if step_doubling:
        fine = _split_evolve(operator, schedule, runtime, 2 * steps, chosen_path, initial_state)
        estimate = float(operator.xp.linalg.norm(fine - psi)) / 3
        psi = fine
        steps *= 2
    norm_error = abs(float(operator.xp.vdot(psi, psi).real) - 1)
    return PropagationResult(operator.to_numpy(psi), norm_error, steps, estimate, backend)


def _split_evolve_batch(operator: HamiltonianOperator, schedules: Sequence[Schedule],
                        runtimes: np.ndarray, steps: int, path: AnnealPath,
                        initial_states: Any):
    xp = operator.xp
    batch_size = len(schedules)
    starts = [_path_coordinate(schedule(0.)) for schedule in schedules]
    for schedule in schedules:
        _path_coordinate(schedule(1.))
    if initial_states is None:
        if any(abs(start) > 1e-14 for start in starts):
            raise ValueError("default |+> initialization requires schedule(0)=0 for every batch row")
        psi = xp.full((batch_size, operator.dimension), 1 / np.sqrt(operator.dimension), dtype=xp.complex128)
    else:
        raw = xp.asarray(initial_states, dtype=xp.complex128)
        if raw.shape == (operator.dimension,):
            psi = xp.broadcast_to(raw, (batch_size, operator.dimension)).copy()
        elif raw.shape == (batch_size, operator.dimension):
            psi = raw.copy()
        else:
            raise ValueError("initial_states must have shape (2**N,) or (batch_size, 2**N)")
    if not bool(xp.isfinite(psi).all()):
        raise ValueError("initial_states must be finite")
    if bool(xp.any(xp.abs(xp.sum(xp.abs(psi) ** 2, axis=1) - 1) > 1e-10)):
        raise ValueError("initial_states must be normalized; normalization is never silently repaired")
    # Evaluate user callables on the host, then transfer the coefficient table
    # once. No scalar host/device synchronization occurs inside the time loop.
    coefficient_table = np.asarray([[path.coefficients(_path_coordinate(schedule((k + .5) / steps)))
                                     for schedule in schedules] for k in range(steps)])
    coefficient_table *= runtimes[None, :, None] / steps
    coefficients = xp.asarray(coefficient_table)
    has_catalyst = bool(path.catalyst_strength and len(operator.terms.xx_edges))
    for k in range(steps):
        a, b, c = (coefficients[k, :, index, None] for index in range(3))
        phase = xp.exp(-.5j * b * operator.diagonal[None, :])
        psi *= phase
        angle = -a
        cosine, sine = xp.cos(angle), xp.sin(angle)
        for i in range(operator.terms.n_qubits):
            psi = cosine * psi - 1j * sine * psi[:, operator.indices ^ (1 << i)]
        if has_catalyst:
            for (i, j), weight in zip(operator.terms.xx_edges, operator.terms.xx_weights):
                angle = c * float(weight)
                mask = (1 << int(i)) | (1 << int(j))
                psi = xp.cos(angle) * psi - 1j * xp.sin(angle) * psi[:, operator.indices ^ mask]
        psi *= phase
    return psi


def propagate_batch(terms: HamiltonianTerms, schedules: Sequence[Schedule],
                    runtimes: float | Sequence[float], *, steps: int = 512,
                    path: AnnealPath | None = None, backend: str = "numpy",
                    initial_states: Any = None, step_doubling: bool = True,
                    state_tolerance: float = 5e-4, norm_tolerance: float = 1e-9,
                    require_accepted: bool = False,
                    max_state_bytes: int = 512 * 2**20) -> BatchPropagationResult:
    """Batched schedules for one Hamiltonian, with per-row numerical gates.

    Rows may have different runtimes and waveforms, but share the path and
    number of integration steps. ``accepted`` is based on norm conservation
    and the same asymptotic ||psi_(2M)-psi_M||/3 diagnostic as ``propagate``.
    Failed rows are NOT silently normalized or accepted: callers may retry
    those rows at a finer step count. ``require_accepted=True`` instead raises.
    Without step doubling, accepted=None (labels have not passed that gate).
    This is an explicit batching API, not a claim that the CLI uses batching.
    """
    schedules = list(schedules)
    if not schedules or not all(callable(schedule) for schedule in schedules):
        raise ValueError("schedules must be a nonempty sequence of callables")
    values = np.asarray(runtimes, dtype=float)
    if values.ndim == 0:
        values = np.full(len(schedules), float(values))
    if values.shape != (len(schedules),):
        raise ValueError("runtimes must be scalar or have one value per schedule")
    for runtime in values:
        _validate_propagation(float(runtime), steps)
    for name, value in (("state_tolerance", state_tolerance), ("norm_tolerance", norm_tolerance)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if require_accepted and not step_doubling:
        raise ValueError("require_accepted requires step_doubling")
    estimate = estimate_state_workspace(terms.n_qubits, batch_size=len(schedules), step_doubling=step_doubling)
    # Account for both host and device coefficient tables in the CPU estimate
    # as well; counting a nonexistent second device on CPU is conservative.
    coefficient_bytes = 2 * (2 if step_doubling else 1) * steps * len(schedules) * 3 * 8
    peak = estimate["estimated_peak_bytes"] + coefficient_bytes
    _check_workspace(peak, max_state_bytes, np, "numpy")  # before backend setup or allocations
    xp = _array_module(backend)
    _check_workspace(peak, max_state_bytes, xp, backend)
    operator = HamiltonianOperator(terms, backend, max_state_bytes=max_state_bytes)
    selected = path or AnnealPath()
    psi = _split_evolve_batch(operator, schedules, values, steps, selected, initial_states)
    errors = None
    if step_doubling:
        fine = _split_evolve_batch(operator, schedules, values, 2 * steps, selected, initial_states)
        errors = operator.to_numpy(xp.linalg.norm(fine - psi, axis=1) / 3)
        psi = fine
        steps *= 2
    norms = operator.to_numpy(xp.abs(xp.sum(xp.abs(psi) ** 2, axis=1) - 1))
    accepted = None if errors is None else (np.isfinite(errors) & np.isfinite(norms)
                                            & (errors <= state_tolerance) & (norms <= norm_tolerance))
    if require_accepted and not bool(accepted.all()):
        failed = np.flatnonzero(~accepted).tolist()
        raise RuntimeError(f"batch propagation numerical gate failed for rows {failed}; increase steps")
    return BatchPropagationResult(operator.to_numpy(psi), norms, int(steps), errors, accepted,
                                  backend, int(peak))


def dense_reference_propagate(terms: HamiltonianTerms, schedule: Schedule, runtime: float, *,
                              path: AnnealPath | None = None, initial_state: Any = None,
                              max_qubits: int = 8, rtol: float = 1e-10, atol: float = 1e-12) -> PropagationResult:
    """Independent DOP853 integration of dense matrices; small QA systems only."""
    from scipy.integrate import solve_ivp

    _validate_propagation(runtime, 1)
    if terms.n_qubits > max_qubits:
        raise MemoryError("dense propagation reference exceeds max_qubits")
    operator = HamiltonianOperator(terms)
    psi0 = _initial_state(operator, schedule, initial_state)
    selected = path or AnnealPath()
    matrices = [operator.dense_parts(coeff, max_qubits=max_qubits) for coeff in ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))]

    def rhs(tau: float, psi: np.ndarray) -> np.ndarray:
        coeffs = selected.coefficients(_path_coordinate(schedule(tau)))
        return -1j * runtime * sum(value * (matrix @ psi) for value, matrix in zip(coeffs, matrices))

    result = solve_ivp(rhs, (0., 1.), psi0, method="DOP853", rtol=rtol, atol=atol)
    if not result.success:
        raise RuntimeError(f"reference integration failed: {result.message}")
    psi = result.y[:, -1]
    return PropagationResult(psi, abs(float(np.vdot(psi, psi).real) - 1), len(result.t) - 1, method="dense_dop853")
