"""Opt-in, costly per-instance spectral scheduling baselines.

These generic exact-teacher baselines are NOT implementations of Tx-NQDT or
other named published methods. They are never inserted into the ML candidate
bank automatically. Charge teacher time and subsequent outcome evaluations.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from .physics import AnnealPath, HamiltonianTerms
from .schedules import Schedule, decode_durations
from .spectral import SpectralPoint, spectral_profile, spectral_teacher


@dataclass(frozen=True)
class SpectralBaselineResult:
    schedule: Schedule | None
    method: str
    status: str
    spectral_grid: np.ndarray
    density: np.ndarray
    unresolved_indices: tuple[int, ...]
    teacher_seconds: float
    teacher_evaluations: int
    diagnostics: dict[str, Any]
    privileged_per_instance: bool = True
    deployment_spectrum_required: bool = True


def bounded_density_schedule(s_grid, density, *, runtime: float, max_slope: float,
                             relative_density_floor: float = 1e-12) -> Schedule:
    """Trapezoid interval masses -> residual allocation -> feasible waveform.

    Every interval first receives Delta_s/max_slope; spare time is allocated
    proportionally to its positive density mass. Thus all returned ds/dt obey
    the bound. The relative floor is a declared numerical choice, not physics.
    An identically zero response uses a linear schedule. This is a discretized
    rule, not a proof that the underlying continuous inverse-CDF converged.
    """
    s, d = np.asarray(s_grid, dtype=float), np.asarray(density, dtype=float)
    if s.ndim != 1 or len(s) < 2 or s.shape != d.shape:
        raise ValueError("grid and density must be matching vectors")
    if not np.isfinite(s).all() or s[0] != 0 or s[-1] != 1 or np.any(np.diff(s) <= 0):
        raise ValueError("grid must strictly increase from 0 to 1")
    if not np.isfinite(d).all() or np.any(d < 0):
        raise ValueError("density must be finite and nonnegative")
    if not np.isfinite(relative_density_floor) or not 0 < relative_density_floor <= 1:
        raise ValueError("relative_density_floor must lie in (0,1]")
    maximum = float(d.max())
    relative = np.ones_like(d) if maximum == 0 else np.maximum(d / maximum, relative_density_floor)
    masses = 0.5 * (relative[:-1] + relative[1:]) * np.diff(s)
    result = decode_durations(np.log(masses), s, runtime=runtime, max_slope=max_slope)
    result.validate_slope(runtime, max_slope)
    return result


def _density(point: SpectralPoint, method: str, gap_epsilon: float) -> tuple[float, bool]:
    if method == "gap_inverse_square":
        # A floor must be explicit when the first gap is zero/unresolved.
        resolved = point.raw_gap > point.energy_resolution or gap_epsilon > 0
        denominator = point.raw_gap**2 + gap_epsilon**2
        return (1. / denominator if resolved else float("nan")), resolved
    resolved = point.moments_resolved
    value = point.d2 if gap_epsilon == 0 else point.regularized_d2
    return value, resolved and np.isfinite(value)


def exact_teacher_baseline(terms: HamiltonianTerms, method: str, *, runtime: float,
                           max_slope: float, path: AnnealPath | None = None,
                           s_grid=None, max_qubits: int = 10, gap_epsilon: float = 0.,
                           relative_density_floor: float = 1e-12, audit_points: int = 8,
                           audit_seed: int = 0, audit_relative_tolerance: float = 0.25) -> SpectralBaselineResult:
    """Build gap-only or D2 baseline with exact, capped spectral evaluations.

    ``gap_epsilon>0`` changes the named baseline to an explicitly regularized
    variant. Degenerate first-gap endpoints otherwise return schedule=None with
    unresolved status. D2 uses band-aware response, but changes of band rank are
    reported and must not be interpreted as one smooth adiabatic branch.

    Random held-out path-point audits check interpolation, not optimality or a
    rigorous uniform error bound. An audit failure returns the tentative
    waveform with status='interpolation_audit_failed'; callers must not silently
    claim it converged. No automatic refinement hides extra teacher expenditure.
    """
    if method not in {"gap_inverse_square", "d2"}:
        raise ValueError("method must be 'gap_inverse_square' or 'd2'")
    if not np.isfinite(gap_epsilon) or gap_epsilon < 0:
        raise ValueError("gap_epsilon must be finite and nonnegative")
    if isinstance(audit_points, bool) or not isinstance(audit_points, (int, np.integer)) or audit_points < 0:
        raise ValueError("audit_points must be a nonnegative integer")
    if not np.isfinite(audit_relative_tolerance) or audit_relative_tolerance <= 0:
        raise ValueError("audit_relative_tolerance must be positive")
    grid = np.linspace(0., 1., 33) if s_grid is None else np.asarray(s_grid, dtype=float)
    # Validate bounds/feasibility before expensive teacher work.
    bounded_density_schedule(grid, np.ones_like(grid), runtime=runtime, max_slope=max_slope,
                             relative_density_floor=relative_density_floor)
    started = perf_counter()
    points = spectral_profile(terms, grid, path=path, max_qubits=max_qubits,
                              regularization_epsilon=gap_epsilon)
    values_and_flags = [_density(point, method, gap_epsilon) for point in points]
    density = np.array([value for value, _ in values_and_flags])
    unresolved = tuple(i for i, (_, valid) in enumerate(values_and_flags) if not valid)
    diagnostics: dict[str, Any] = {
        "gap_epsilon": gap_epsilon, "relative_density_floor": relative_density_floor,
        "ground_ranks": [point.ground_rank for point in points],
        "rank_changes": any(a.ground_rank != b.ground_rank for a, b in zip(points[:-1], points[1:])),
        "max_eigenpair_residual": max(point.max_eigenpair_residual for point in points),
        "max_sum_rule_error": max(point.sum_rule_error for point in points),
        "uniform_interpolation_certificate": False,
        "control_optimality_certificate": False,
        "teacher_grid_count": len(points), "audit_count": 0,
        "outcome_evaluations": 0,
    }
    name = method if gap_epsilon == 0 else f"{method}_regularized"
    if unresolved:
        return SpectralBaselineResult(None, name, "unresolved_spectral_points", grid, density,
                                      unresolved, perf_counter() - started, len(points), diagnostics)
    schedule = bounded_density_schedule(grid, density, runtime=runtime, max_slope=max_slope,
                                         relative_density_floor=relative_density_floor)
    status = "sampled_profile_not_audited"
    if audit_points:
        queries = np.random.default_rng(audit_seed).uniform(0., 1., audit_points)
        evaluated = [spectral_teacher(terms, float(s), path=path, max_qubits=max_qubits,
                                      regularization_epsilon=gap_epsilon) for s in queries]
        audits = [_density(point, method, gap_epsilon) for point in evaluated]
        true = np.array([value for value, _ in audits])
        prediction = np.interp(queries, grid, density)
        valid = np.array([flag for _, flag in audits])
        denominator = np.maximum(np.maximum(np.abs(prediction), np.abs(true)), np.finfo(float).tiny)
        error = np.abs(prediction - true) / denominator
        maximum = float(np.max(error)) if valid.all() else float("inf")
        diagnostics.update(audit_count=audit_points, audit_s=queries.tolist(),
                           audit_max_relative_error=maximum, audit_relative_tolerance=audit_relative_tolerance,
                           audit_unresolved_count=int((~valid).sum()))
        status = "sampled_point_audit_passed" if maximum <= audit_relative_tolerance else "interpolation_audit_failed"
    diagnostics["zero_response_linear_fallback"] = bool(np.all(density == 0))
    return SpectralBaselineResult(schedule, name, status, grid, density, (),
                                  perf_counter() - started, len(points) + audit_points, diagnostics)
