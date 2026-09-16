"""Explicit calibration units and an independent tiny-system Lindblad solver.

No calibrated-device fidelity, thermal bath, QPU execution, or GPU acceleration
is implied by this module. Density matrices require O(4**N) storage. Local
relaxation below means computational |1> -> |0>, NOT relaxation into the
instantaneous ground state of the interacting Hamiltonian.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.integrate import solve_ivp

from .physics import AnnealPath, HamiltonianTerms


@dataclass(frozen=True)
class CalibratedPath:
    """Piecewise-linear coefficients in angular units per declared time unit.

The JSON must explicitly give factors for the project convention
H_X=-sum(X), H_Z=sum(hZ)+sum(JZZ), H_XX=sum(KXX). Thus tables with a
one-half prefactor need factors {a: 0.5, b: 0.5, c: ...}; none is guessed.
Input GHz denotes H/h, so conversion for time in us is 2*pi*1000.
"""
    s: np.ndarray
    angular_coefficients: np.ndarray
    time_unit: str
    calibration_id: str
    source: str
    coefficient_unit: str
    component_factors: tuple[float, float, float]

    def __post_init__(self):
        s = np.asarray(self.s, dtype=float).copy()
        values = np.asarray(self.angular_coefficients, dtype=float).copy()
        if s.ndim != 1 or len(s) < 2 or not np.isfinite(s).all():
            raise ValueError("calibration s must be a finite vector")
        if s[0] != 0 or s[-1] != 1 or np.any(np.diff(s) <= 0):
            raise ValueError("calibration must cover [0,1] with strictly increasing s")
        if values.shape != (len(s), 3) or not np.isfinite(values).all():
            raise ValueError("calibrated coefficients must have shape (len(s),3)")
        if self.time_unit not in {"dimensionless", "us"}:
            raise ValueError("time_unit must be dimensionless or us")
        if not self.calibration_id or not self.source:
            raise ValueError("calibration_id and source must be explicitly supplied")
        if self.coefficient_unit not in {"dimensionless", "GHz", "rad/us"}:
            raise ValueError("unsupported coefficient_unit")
        if (self.coefficient_unit == "dimensionless") != (self.time_unit == "dimensionless"):
            raise ValueError("coefficient and time units are incompatible")
        factors = np.asarray(self.component_factors, dtype=float)
        if factors.shape != (3,) or not np.isfinite(factors).all():
            raise ValueError("three finite component factors are required")
        s.flags.writeable = values.flags.writeable = False
        object.__setattr__(self, "s", s)
        object.__setattr__(self, "angular_coefficients", values)

    @classmethod
    def from_mapping(cls, data: dict) -> "CalibratedPath":
        required = {"s", "a", "b", "c", "coefficient_unit", "time_unit",
                    "component_factors", "calibration_id", "source"}
        if required - data.keys():
            raise ValueError(f"missing calibration fields: {sorted(required - data.keys())}")
        unit, time = data["coefficient_unit"], data["time_unit"]
        conversions = {("dimensionless", "dimensionless"): 1.,
                       ("GHz", "us"): 2 * np.pi * 1000., ("rad/us", "us"): 1.}
        if (unit, time) not in conversions:
            raise ValueError("unsupported or inconsistent calibration energy/time units")
        factor_map = data["component_factors"]
        if not isinstance(factor_map, dict) or set(factor_map) != {"a", "b", "c"}:
            raise ValueError("component_factors must explicitly contain a,b,c")
        factors = tuple(float(factor_map[key]) for key in ("a", "b", "c"))
        try:
            values = np.column_stack([data[key] for key in ("a", "b", "c")]).astype(float)
            values *= np.asarray(factors) * conversions[(unit, time)]
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid calibration coefficient arrays") from exc
        return cls(data["s"], values, time, data["calibration_id"], data["source"], unit, factors)

    @classmethod
    def from_json(cls, filename: str | Path) -> "CalibratedPath":
        return cls.from_mapping(json.loads(Path(filename).read_text()))

    def _coordinate(self, s: float) -> float:
        value = float(s)
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("path coordinate outside calibrated domain [0,1]")
        return value

    def coefficients(self, s: float) -> tuple[float, float, float]:
        value = self._coordinate(s)
        return tuple(float(np.interp(value, self.s, self.angular_coefficients[:, i])) for i in range(3))

    def derivative(self, s: float) -> tuple[float, float, float]:
        """Right derivative at interior knots; left derivative at s=1."""
        value = self._coordinate(s)
        index = min(int(np.searchsorted(self.s, value, side="right")) - 1, len(self.s) - 2)
        return tuple((self.angular_coefficients[index + 1] - self.angular_coefficients[index]) /
                     (self.s[index + 1] - self.s[index]))


@dataclass(frozen=True)
class DensityResult:
    density: np.ndarray
    probabilities: np.ndarray
    diagnostics: dict[str, Any]


def _local_operator(n: int, i: int, local: np.ndarray) -> np.ndarray:
    # Independent Kronecker implementation: qubit i is bit i (LSB first).
    operator = np.ones((1, 1), dtype=complex)
    for site in reversed(range(n)):
        operator = np.kron(operator, local if site == i else np.eye(2))
    return operator


def _rate_vector(value: Any, n: int, name: str) -> np.ndarray:
    try:
        values = np.broadcast_to(np.asarray(0. if value is None else value, dtype=float), (n,)).copy()
    except ValueError as exc:
        raise ValueError(f"{name} must be scalar or one value per physical qubit") from exc
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError(f"{name} must be finite and nonnegative")
    return values


def _density_metrics(rho: np.ndarray) -> tuple[float, float, float]:
    trace = float(abs(np.trace(rho) - 1))
    hermitian = float(np.linalg.norm(rho - rho.conj().T, ord="fro"))
    # Symmetrization is only for diagnostics; the returned state is not repaired.
    minimum = float(np.linalg.eigvalsh((rho + rho.conj().T) / 2).min())
    return trace, hermitian, minimum


def simulate_lindblad(
    terms: HamiltonianTerms, schedule: Callable[[float], float], runtime: float, *,
    path: AnnealPath | CalibratedPath | None = None,
    dephasing_rates: Any = None, relaxation_rates: Any = None,
    rates_unit: str = "1/dimensionless", initial_state: Any = None,
    initial_density: Any = None, max_qubits: int = 6, check_points: int = 17,
    rtol: float = 1e-9, atol: float = 1e-11, physical_tolerance: float = 1e-7,
) -> DensityResult:
    """Solve dρ/dt=-i[H,ρ]+Σ(LρL†-{L†L,ρ}/2) by SciPy DOP853.

L_deph=sqrt(gamma_phi)*Z (coherence decay 2*gamma_phi); L_relax=
sqrt(gamma_1)*|0><1|. Rates must be inverse time units of the path.
Checks occur on an explicit grid, not continuously; no error certificate or
instantaneous-eigenbasis thermalization is claimed.
"""
    if terms.n_qubits > max_qubits or max_qubits > 8:
        raise MemoryError("density solver is capped at <=8 qubits (default 6), O(4**N) memory")
    if not np.isfinite(runtime) or runtime < 0:
        raise ValueError("runtime must be finite and nonnegative")
    if isinstance(check_points, bool) or not isinstance(check_points, (int, np.integer)) or check_points < 2:
        raise ValueError("check_points must be an integer >=2")
    if any(not np.isfinite(v) or v <= 0 for v in (rtol, atol, physical_tolerance)):
        raise ValueError("solver/check tolerances must be finite and positive")
    path = path or AnnealPath()
    unit = path.time_unit if isinstance(path, CalibratedPath) else "dimensionless"
    if rates_unit != "1/" + unit:
        raise ValueError(f"rates_unit must be 1/{unit}; time conversion is never inferred")
    n, dimension = terms.n_qubits, 1 << terms.n_qubits
    x = np.array([[0., 1.], [1., 0.]])
    z = np.diag([1., -1.])
    xs = [_local_operator(n, i, x) for i in range(n)]
    zs = [_local_operator(n, i, z) for i in range(n)]
    hx = -sum(xs)
    hz = sum(h * zi for h, zi in zip(terms.h, zs))
    for (i, j), weight in zip(terms.zz_edges, terms.zz_weights):
        hz += weight * (zs[i] @ zs[j])
    hxx = np.zeros_like(hx)
    for (i, j), weight in zip(terms.xx_edges, terms.xx_weights):
        hxx += weight * (xs[i] @ xs[j])
    start = float(schedule(0.))
    path.coefficients(start)
    path.coefficients(float(schedule(1.)))
    if initial_state is not None and initial_density is not None:
        raise ValueError("supply only initial_state or initial_density")
    if initial_density is None:
        if initial_state is None:
            a, b, c = path.coefficients(start)
            if start != 0 or a <= 0 or abs(b) > 1e-14 or abs(c) > 1e-14:
                raise ValueError("default |+> requires the canonical pure-X start; supply initial state")
            psi = np.full(dimension, 1 / np.sqrt(dimension), dtype=complex)
        else:
            psi = np.asarray(initial_state, dtype=complex)
        if psi.shape != (dimension,) or not np.isfinite(psi).all() or abs(np.vdot(psi, psi) - 1) > 1e-10:
            raise ValueError("initial_state must be finite, correctly sized, and normalized")
        rho0 = np.outer(psi, psi.conj())
    else:
        rho0 = np.asarray(initial_density, dtype=complex).copy()
    if rho0.shape != (dimension, dimension) or not np.isfinite(rho0).all():
        raise ValueError("initial_density must be a finite 2**N square matrix")
    tr, herm, minimum = _density_metrics(rho0)
    if tr > 1e-10 or herm > 1e-10 or minimum < -1e-10:
        raise ValueError("initial_density must be Hermitian, trace one, and positive semidefinite")
    dephasing = _rate_vector(dephasing_rates, n, "dephasing_rates")
    relaxation = _rate_vector(relaxation_rates, n, "relaxation_rates")
    collapses = [np.sqrt(rate) * zi for rate, zi in zip(dephasing, zs) if rate > 0]
    for i, rate in enumerate(relaxation):
        if rate > 0:
            collapses.append(np.sqrt(rate) * _local_operator(n, i, np.array([[0., 1.], [0., 0.]])))
    dissipators = [(op, op.conj().T, op.conj().T @ op) for op in collapses]

    def rhs(t, flat):
        rho = flat.reshape(dimension, dimension)
        a, b, c = path.coefficients(float(schedule(float(np.clip(t / runtime, 0., 1.)))))
        hamiltonian = a * hx + b * hz + c * hxx
        out = -1j * (hamiltonian @ rho - rho @ hamiltonian)
        for op, adjoint, square in dissipators:
            out += op @ rho @ adjoint - (square @ rho + rho @ square) / 2
        return out.ravel()

    if runtime == 0:
        states, evaluations = rho0[None, ...], 0
    else:
        tau = np.linspace(0., 1., check_points)
        if hasattr(schedule, "tau_knots"):
            tau = np.unique(np.concatenate((tau, schedule.tau_knots)))
        solution = solve_ivp(rhs, (0., runtime), rho0.ravel(), method="DOP853",
                             t_eval=tau * runtime, rtol=rtol, atol=atol, max_step=runtime / 64)
        if not solution.success or not np.isfinite(solution.y).all():
            raise ArithmeticError(f"Lindblad integration failed: {solution.message}")
        states = solution.y.T.reshape(-1, dimension, dimension)
        evaluations = int(solution.nfev)
    metrics = np.asarray([_density_metrics(state) for state in states])
    if metrics[:, 0].max() > physical_tolerance or metrics[:, 1].max() > physical_tolerance or metrics[:, 2].min() < -physical_tolerance:
        raise ArithmeticError("Lindblad physicality gate failed; tighten integration tolerances")
    final = states[-1].copy()
    return DensityResult(final, np.diag(final).real.copy(), {
        "method": "scipy_DOP853_density_matrix", "time_unit": unit, "runtime": float(runtime),
        "rates_unit": rates_unit, "dephasing_rates": dephasing.tolist(),
        "relaxation_rates": relaxation.tolist(), "rhs_evaluations": evaluations,
        "checked_states": len(states), "max_trace_error": float(metrics[:, 0].max()),
        "max_hermiticity_error": float(metrics[:, 1].max()),
        "minimum_eigenvalue": float(metrics[:, 2].min()), "physicality_passed": True,
        "rtol": rtol, "atol": atol, "physical_tolerance": physical_tolerance,
        "calibration_id": path.calibration_id if isinstance(path, CalibratedPath) else None,
        "calibration_source": path.source if isinstance(path, CalibratedPath) else None,
        "calibration_input_unit": path.coefficient_unit if isinstance(path, CalibratedPath) else "dimensionless",
        "calibration_component_factors": list(path.component_factors) if isinstance(path, CalibratedPath) else [1., 1., 1.],
        "scope": "phenomenological_local_noise_not_device_or_thermal_calibration",
        "state_repaired": False,
    })
