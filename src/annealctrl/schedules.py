"""Feasible monotone controls; normalized time is tau=t/runtime.

These are monotone annealing controls, not bang--anneal--bang controls. Slope
bounds are always ds/dt in the same time unit as ``runtime``. Passing physical
microseconds therefore requires inverse-microsecond slope limits; dimensionless
simulation runtimes require dimensionless limits.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def _vector(value, name: str, minimum_size: int = 1) -> np.ndarray:
    result = np.asarray(value, dtype=float).copy()
    if result.ndim != 1 or len(result) < minimum_size or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite vector with >= {minimum_size} entries")
    return result


@dataclass(frozen=True)
class Schedule:
    """Piecewise-linear s(tau), tau in [0,1]; equal s values represent pauses."""

    tau_knots: np.ndarray
    s_knots: np.ndarray

    def __post_init__(self) -> None:
        tau = _vector(self.tau_knots, "tau_knots", 2)
        s = _vector(self.s_knots, "s_knots", 2)
        if tau.shape != s.shape:
            raise ValueError("tau and s knots must have the same shape")
        if tau[0] != 0 or tau[-1] != 1 or s[0] != 0 or s[-1] != 1:
            raise ValueError("forward schedules must start at (0,0) and end at (1,1)")
        if np.any(np.diff(tau) <= 0) or np.any(np.diff(s) < 0):
            raise ValueError("time knots must strictly increase; path knots must not decrease")
        tau.setflags(write=False)
        s.setflags(write=False)
        object.__setattr__(self, "tau_knots", tau)
        object.__setattr__(self, "s_knots", s)

    def __call__(self, tau):
        tau = np.asarray(tau, dtype=float)
        if not np.isfinite(tau).all() or np.any((tau < 0) | (tau > 1)):
            raise ValueError("normalized time must lie in [0,1]")
        value = np.interp(tau, self.tau_knots, self.s_knots)
        return float(value) if value.ndim == 0 else value

    @classmethod
    def linear(cls) -> "Schedule":
        return cls(np.array([0.0, 1.0]), np.array([0.0, 1.0]))

    def slopes(self, runtime: float = 1.0) -> np.ndarray:
        if not np.isfinite(runtime) or runtime <= 0:
            raise ValueError("runtime must be finite and positive")
        return np.diff(self.s_knots) / (runtime * np.diff(self.tau_knots))

    def validate_slope(self, runtime: float, max_slope: float) -> None:
        if not np.isfinite(max_slope) or max_slope <= 0:
            raise ValueError("max_slope must be finite and positive")
        if np.any(self.slopes(runtime) > max_slope * (1 + 1e-12)):
            raise ValueError("schedule violates maximum ds/dt")

    def to_dict(self) -> dict:
        return {"tau_knots": self.tau_knots.tolist(), "s_knots": self.s_knots.tolist()}


def decode_durations(
    logits: Sequence[float],
    s_knots: Sequence[float] | None = None,
    *,
    runtime: float = 1.0,
    max_slope: float | Sequence[float] = 10.0,
) -> Schedule:
    """Residual-softmax allocation, manuscript Eq. decoder.

    Delta t = Delta s/v + (T - sum(Delta s/v))*softmax(logits).
    Repeated s knots give pause segments. Zero-duration pauses (e.g. no spare
    time) are removed. The decoded waveform still needs hardware-specific point
    count and time-quantization validation before QPU submission.
    """
    logits = _vector(logits, "logits")
    s = np.linspace(0, 1, len(logits) + 1) if s_knots is None else _vector(s_knots, "s_knots", 2)
    if len(s) != len(logits) + 1 or s[0] != 0 or s[-1] != 1 or np.any(np.diff(s) < 0):
        raise ValueError("need K+1 nondecreasing path knots from 0 to 1 for K logits")
    if not np.isfinite(runtime) or runtime <= 0:
        raise ValueError("runtime must be finite and positive")
    slope = np.broadcast_to(np.asarray(max_slope, dtype=float), logits.shape)
    if not np.isfinite(slope).all() or np.any(slope <= 0):
        raise ValueError("all slope bounds must be finite and positive")
    minimum = np.diff(s) / slope
    required = float(minimum.sum())
    spare = runtime - required
    if spare < -1e-13 * max(runtime, required):
        raise ValueError(f"infeasible runtime: need at least {required:g}")
    spare = max(0.0, spare)
    # Subtraction first prevents overflow even for large finite network outputs.
    with np.errstate(over="ignore", under="ignore"):
        weight = np.exp(logits - np.max(logits))
    weight /= weight.sum()
    dt = minimum + spare * weight
    # Underflow can assign exactly zero to a pause. Such a segment has no effect.
    tau = np.concatenate(([0.0], np.cumsum(dt) / runtime))
    tau[-1] = 1.0
    keep = np.concatenate(([True], np.diff(tau) > 0))
    if not keep[-1]:
        # Numerical underflow at an endpoint pause: retain the correct endpoint.
        previous = np.flatnonzero(keep)[-1]
        if s[previous] != 1:
            raise FloatingPointError("segment duration below floating-point resolution")
    removed = np.flatnonzero(~keep)
    if any(s[i] != s[i - 1] for i in removed):
        raise FloatingPointError("traversal duration below floating-point resolution")
    return Schedule(tau[keep], s[keep])


def window_schedule(
    windows: Sequence[tuple[float, float]],
    weights: Sequence[float],
    *,
    runtime: float = 1.0,
    max_slope: float = 10.0,
) -> Schedule:
    """Exact piecewise-constant time density with one or more slow windows.

    Weights allocate fractions of *spare traversal time*. Their sum is <=1;
    the remaining spare time is spread uniformly. Overlap is permitted.
    """
    weights = _vector(weights, "weights")
    if len(windows) != len(weights) or np.any(weights < 0) or weights.sum() > 1 + 1e-13:
        raise ValueError("window weights must be nonnegative and sum to at most one")
    for a, b in windows:
        if not np.isfinite([a, b]).all() or not 0 <= a < b <= 1:
            raise ValueError("windows require 0 <= a < b <= 1")
    # This also checks feasibility and units.
    decode_durations([0.0], runtime=runtime, max_slope=max_slope)
    s = np.unique(np.array([0.0, 1.0, *[v for pair in windows for v in pair]]))
    midpoint = (s[:-1] + s[1:]) / 2
    residual_density = np.full(len(midpoint), max(0.0, 1.0 - weights.sum()))
    for (a, b), w in zip(windows, weights):
        residual_density += w * ((midpoint >= a) & (midpoint <= b)) / (b - a)
    base_fraction = 1 / (runtime * max_slope)
    dtau = np.diff(s) * (base_fraction + max(0.0, 1 - base_fraction) * residual_density)
    tau = np.concatenate(([0.0], np.cumsum(dtau)))
    tau[-1] = 1.0
    return Schedule(tau, s)


def slow_window_schedule(a: float, b: float, q: float, *, runtime: float = 1.0, max_slope: float = 10.0) -> Schedule:
    return window_schedule([(a, b)], [q], runtime=runtime, max_slope=max_slope)


def pause_schedule(location: float, pause_fraction: float, *, runtime: float = 1.0, max_slope: float = 10.0) -> Schedule:
    """An actual fixed-s dwell, not an approximation by a narrow slow window.

    ``pause_fraction`` is the fraction of total runtime spent at ``location``.
    """
    if not np.isfinite([location, pause_fraction]).all() or not 0 < location < 1 or not 0 <= pause_fraction < 1:
        raise ValueError("require interior location and pause_fraction in [0,1)")
    decode_durations([0.0], runtime=runtime, max_slope=max_slope)
    if (1 - pause_fraction) * runtime * max_slope < 1 - 1e-13:
        raise ValueError("pause leaves insufficient time for bounded-slope traversal")
    if pause_fraction == 0:
        return Schedule.linear()
    before = location * (1 - pause_fraction)
    return Schedule([0, before, before + pause_fraction, 1], [0, location, location, 1])


def inverse_density_schedule(s_grid: Sequence[float], density: Sequence[float]) -> Schedule:
    """Trapezoid-CDF discretization of inverse cumulative time density.

    Returns the piecewise-linear inverse of sampled cumulative mass. Positivity
    is required; zero densities would create instantaneous path jumps. This
    baseline does NOT enforce a slope bound; validate or project separately.
    """
    s = _vector(s_grid, "s_grid", 2)
    d = _vector(density, "density", 2)
    if s.shape != d.shape or s[0] != 0 or s[-1] != 1 or np.any(np.diff(s) <= 0) or np.any(d <= 0):
        raise ValueError("need matching strictly increasing [0,1] grid and positive density")
    d = d / d.max()  # normalize before integration to avoid overflow
    mass = 0.5 * (d[:-1] + d[1:]) * np.diff(s)
    tau = np.concatenate(([0.0], np.cumsum(mass)))
    tau /= tau[-1]
    return Schedule(tau, s)
