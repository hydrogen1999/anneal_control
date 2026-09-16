"""Capped full-spectrum privileged teacher, with explicit resolution metadata.

Dense diagonalization is O(4**N) memory and O(8**N) work. This reference does
not claim scalable labels; an approximate teacher must validate residuals,
response convergence, and omitted-mass bounds before replacing it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import numpy as np

from .physics import AnnealPath, HamiltonianOperator, HamiltonianTerms


@dataclass(frozen=True)
class LowEnergyResult:
    """Approximate eigenpairs, never a complete or symmetry-resolved teacher."""

    energies: np.ndarray
    states: np.ndarray
    residuals: np.ndarray
    orthogonality_error: float
    backend: str
    requested_k: int
    solver_tolerance: float
    truncated: bool = True
    full_response_resolved: bool = False
    ground_band_certified: bool = False


def sparse_low_energy(terms: HamiltonianTerms, s: float, k: int, *,
                      path: AnnealPath | None = None, backend: str = "numpy",
                      tolerance: float = 1e-10, maxiter: int | None = None,
                      ncv: int | None = None, seed: int = 0,
                      max_state_bytes: int = 512 * 2**20) -> LowEnergyResult:
    """Matrix-free smallest-algebraic Lanczos eigenpairs on CPU or CuPy GPU.

    O(ncv*2**N) storage remains exponential. The returned residuals are numerical
    diagnostics, not a proof that a tiny ground gap or all bright transitions
    have been resolved. In particular k=3 misses the bright state in the
    two-qubit dark-gap example; do not derive complete D2 labels from this API.
    An eigensolver failure propagates rather than silently using partial labels.
    """
    dimension = 1 << terms.n_qubits
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or not 1 <= k < dimension:
        raise ValueError("sparse k must be an integer in [1, 2**N - 1]")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    if maxiter is not None and (not isinstance(maxiter, (int, np.integer)) or maxiter < 1):
        raise ValueError("maxiter must be a positive integer")
    ncv = min(dimension, max(2 * k + 1, 20)) if ncv is None else ncv
    if not isinstance(ncv, (int, np.integer)) or not k < ncv <= dimension:
        raise ValueError("ncv must be an integer satisfying k < ncv <= 2**N")
    if dimension * (128 + 16 * ncv) > max_state_bytes:
        raise MemoryError("Lanczos work/basis estimate exceeds max_state_bytes")
    operator = HamiltonianOperator(terms, backend, max_state_bytes=max_state_bytes)
    xp = operator.xp
    if backend == "numpy":
        from scipy.sparse.linalg import LinearOperator, eigsh
    else:
        from cupyx.scipy.sparse.linalg import LinearOperator, eigsh
    selected = path or AnnealPath()
    # All supported X/ZZ/XX coefficients are real: real symmetric Lanczos.
    linear = LinearOperator((dimension, dimension), matvec=lambda state: operator.matvec(state, s, selected).real, dtype=xp.float64)
    v0 = xp.asarray(np.random.default_rng(seed).normal(size=dimension))
    energies, states = eigsh(linear, k=int(k), which="SA", tol=tolerance, maxiter=maxiter, ncv=int(ncv), v0=v0)
    order = xp.argsort(energies)
    energies, states = energies[order], states[:, order]
    residuals = xp.stack([xp.linalg.norm(operator.matvec(states[:, j], s, selected) - energies[j] * states[:, j]) for j in range(k)])
    orthogonality = float(xp.linalg.norm(states.conj().T @ states - xp.eye(k)))
    return LowEnergyResult(operator.to_numpy(energies), operator.to_numpy(states), operator.to_numpy(residuals),
                           orthogonality, backend, int(k), float(tolerance))


@dataclass(frozen=True)
class SpectralPoint:
    s: float
    energies: np.ndarray
    gaps: np.ndarray
    transition_strengths: np.ndarray
    ground_rank: int
    ground_band_spread: float
    near_degenerate_band: bool
    raw_gap: float
    accessible_gap: float
    mu0: float
    g_ss: float
    d2: float
    regularized_d2: float
    frequency_edges: np.ndarray
    bin_masses: np.ndarray
    normalized_bin_masses: np.ndarray
    low_frequency_mass: float
    high_frequency_mass: float
    unresolved_mass: float
    unresolved_transition_count: int
    omitted_response_mass: float
    sum_rule_error: float
    max_eigenpair_residual: float
    orthogonality_error: float
    energy_resolution: float
    moments_resolved: bool
    regularization_epsilon: float
    truncated: bool = False


def spectral_teacher(terms: HamiltonianTerms, s: float, *, path: AnnealPath | None = None,
                     frequency_edges: np.ndarray | None = None, max_qubits: int = 10,
                     ground_atol: float = 1e-10, ground_rtol: float = 1e-10,
                     ground_rank: int | None = None, regularization_epsilon: float = 1e-8,
                     bright_relative_tolerance: float = 1e-12) -> SpectralPoint:
    """Full-spectrum response of dH/ds from a low-energy band to its complement.

    With rank r, each |V_ba|^2 carries weight 1/r (uniform band population).
    This is a projector/band diagnostic, not the actual propagated population.
    The default rank includes states within declared degeneracy tolerances;
    nonzero band spread beyond numerical resolution is explicitly flagged.
    Rank-changing profiles must not be treated as smooth fixed-band geometry.
    Unresolved inverse moments return NaN, not a silently clipped gap.
    """
    if terms.n_qubits > max_qubits:
        raise MemoryError("dense spectral teacher exceeds max_qubits; physical, not logical, qubits count")
    if ground_atol < 0 or ground_rtol < 0 or regularization_epsilon < 0 or bright_relative_tolerance < 0:
        raise ValueError("spectral tolerances must be nonnegative")
    if not np.isfinite([ground_atol, ground_rtol, regularization_epsilon, bright_relative_tolerance]).all():
        raise ValueError("spectral tolerances must be finite")
    operator = HamiltonianOperator(terms)
    selected = path or AnnealPath()
    hamiltonian = operator.dense(s, selected, max_qubits=max_qubits)
    drive = operator.dense_parts(selected.derivative(s), max_qubits=max_qubits)
    energies, states = np.linalg.eigh(hamiltonian)
    residuals = np.linalg.norm(hamiltonian @ states - states * energies, axis=0)
    max_residual = float(np.max(residuals))
    scale = max(float(np.max(np.abs(energies))), np.finfo(float).tiny)
    resolution = max(4 * max_residual, 32 * np.finfo(float).eps * scale)
    automatic = ground_rank is None
    if automatic:
        tolerance = ground_atol + ground_rtol * scale
        rank = int(np.count_nonzero(energies - energies[0] <= tolerance))
    else:
        if isinstance(ground_rank, bool) or not isinstance(ground_rank, (int, np.integer)):
            raise ValueError("ground_rank must be an integer")
        rank = int(ground_rank)
    if not 1 <= rank <= len(energies):
        raise ValueError("ground_rank must lie in [1, 2**N]")
    band = states[:, :rank]
    outside = states[:, rank:]
    driven_band = drive @ band
    couplings = outside.conj().T @ driven_band
    gaps = (energies[rank:, None] - energies[None, :rank]).ravel()
    mass = (np.abs(couplings) ** 2 / rank).ravel()
    mu0 = float(mass.sum())
    # Generalizes Var_0(V): remove the entire in-band block, not one vector.
    independent_mass = (np.linalg.norm(driven_band, "fro") ** 2 - np.linalg.norm(band.conj().T @ driven_band, "fro") ** 2) / rank
    sum_rule_error = float(abs(independent_mass - mu0))
    unresolved = gaps <= resolution
    spread = float(energies[rank - 1] - energies[0])
    near_degenerate = rank > 1 and spread > resolution
    resolved = not np.any(unresolved) and not (automatic and near_degenerate)
    if resolved:
        g_ss = float(np.sum(mass / gaps**2))
        d2 = float(np.sqrt(np.sum(mass / gaps**4)))
    else:
        g_ss = d2 = float("nan")
    if regularization_epsilon == 0 and np.any(unresolved):
        regularized = float("nan")
    else:
        regularized = float(np.sqrt(np.sum(mass / (gaps**2 + regularization_epsilon**2)**2)))

    edges = np.geomspace(1e-4, 16., 9) if frequency_edges is None else np.array(frequency_edges, dtype=float, copy=True)
    if edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all() or np.any(edges <= 0) or np.any(np.diff(edges) <= 0):
        raise ValueError("frequency_edges must be positive, finite, strictly increasing")
    # Fixed bins are part of the dataset contract; no per-instance bin rescaling.
    known = ~unresolved
    within = known & (gaps >= edges[0]) & (gaps < edges[-1])
    bins = np.histogram(gaps[within], bins=edges, weights=mass[within])[0]
    low = float(mass[known & (gaps < edges[0])].sum())
    high = float(mass[known & (gaps >= edges[-1])].sum())
    uncertain_mass = float(mass[unresolved].sum())
    bright_threshold = bright_relative_tolerance * max(mu0, np.finfo(float).tiny)
    bright = known & (mass > bright_threshold)
    accessible = float(np.min(gaps[bright])) if np.any(bright) else float("inf")
    return SpectralPoint(
        s=float(s), energies=energies, gaps=gaps, transition_strengths=mass,
        ground_rank=rank, ground_band_spread=spread, near_degenerate_band=near_degenerate,
        raw_gap=float(energies[1] - energies[0]), accessible_gap=accessible,
        mu0=mu0, g_ss=g_ss, d2=d2, regularized_d2=regularized,
        frequency_edges=edges, bin_masses=bins, normalized_bin_masses=bins / mu0 if mu0 > 0 else bins.copy(),
        low_frequency_mass=low, high_frequency_mass=high, unresolved_mass=uncertain_mass,
        unresolved_transition_count=int(unresolved.sum()), omitted_response_mass=0.,
        sum_rule_error=sum_rule_error, max_eigenpair_residual=max_residual,
        orthogonality_error=float(np.linalg.norm(states.conj().T @ states - np.eye(len(energies)), "fro")),
        energy_resolution=resolution, moments_resolved=resolved, regularization_epsilon=regularization_epsilon,
    )


def spectral_profile(terms: HamiltonianTerms, grid: np.ndarray | None = None, **kwargs) -> list[SpectralPoint]:
    """Explicit fixed-grid pilot labels (default 33); NOT an adaptive certificate.

    Include independent random-point interpolation audits before trusting a
    profile for scheduling. ``ground_rank`` changes must be inspected by callers.
    """
    grid = np.linspace(0., 1., 33) if grid is None else np.asarray(grid, dtype=float)
    if grid.ndim != 1 or not len(grid) or not np.isfinite(grid).all() or np.any(np.diff(grid) <= 0):
        raise ValueError("grid must be nonempty, finite and strictly increasing")
    return [spectral_teacher(terms, float(s), **kwargs) for s in grid]


def _json_safe(value):
    """Strict-JSON conversion: unresolved NaN/Inf become null, never fake zero."""
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


@dataclass(frozen=True)
class AdaptiveSpectralProfile:
    """Queried labels and an independent audit, not a uniform certificate.

    ``points`` contains only deterministic adaptive queries; held-out random
    ``audit_points`` never influence refinement or enter these training labels.
    Each query is a complete capped dense diagonalization. Adaptivity reduces
    the number of path queries, not the exponential cost of each query.
    """

    points: list[SpectralPoint]
    audit_points: list[SpectralPoint]
    diagnostics: dict

    def to_dict(self) -> dict:
        return _json_safe(asdict(self))


def _response_vector(point: SpectralPoint) -> np.ndarray:
    # Include unnormalized masses: normalized spectra alone hide amplitude.
    return np.concatenate((np.asarray([point.raw_gap, point.mu0, point.g_ss,
                                       point.d2, point.low_frequency_mass,
                                       point.high_frequency_mass,
                                       point.unresolved_mass]), point.bin_masses))


def _interpolation_check(left: SpectralPoint, right: SpectralPoint,
                         actual: SpectralPoint, rtol: float, atol: float) -> dict:
    fraction = (actual.s - left.s) / (right.s - left.s)
    expected = (1 - fraction) * _response_vector(left) + fraction * _response_vector(right)
    observed = _response_vector(actual)
    finite = np.isfinite(expected) & np.isfinite(observed)
    scaled_error = np.abs(observed[finite] - expected[finite]) / (
        atol + rtol * np.maximum(np.abs(observed[finite]), np.abs(expected[finite])))
    score = float(np.max(scaled_error, initial=0.))
    reasons = []
    if len({left.ground_rank, actual.ground_rank, right.ground_rank}) > 1:
        reasons.append("ground_rank_change")
    if not all(point.moments_resolved for point in (left, actual, right)):
        reasons.append("unresolved_moments_or_numerics")
    if any(point.near_degenerate_band for point in (left, actual, right)):
        reasons.append("near_degenerate_band")
    if not finite.all():
        reasons.append("nonfinite_response")
    if score > 1:
        reasons.append("interpolation_tolerance")
    return {"left": left.s, "right": right.s, "query": actual.s,
            "scaled_error": score, "needs_refinement": bool(reasons), "reasons": reasons}


def adaptive_spectral_profile(terms: HamiltonianTerms, *, initial_grid=None,
                              max_queries: int = 65, audit_points: int = 8,
                              seed: int = 0, relative_tolerance: float = 0.05,
                              absolute_tolerance: float = 1e-4,
                              min_interval_width: float = 1e-4,
                              residual_tolerance: float = 1e-9,
                              orthogonality_tolerance: float = 1e-9,
                              **teacher_kwargs) -> AdaptiveSpectralProfile:
    """Budgeted midpoint refinement plus independent random interpolation audit.

    Compare raw gap, response moments and frequency masses against endpoint
    linear interpolation, using ``atol + rtol*max(abs(actual),abs(predicted))``.
    Rank changes and unresolved labels force refinement until the budget or
    minimum width is reached; they are never interpolated away. Eigensystem
    residuals are relative to max(1, max|energy|); failed residual/orthogonality
    gates mask inverse moments. The returned flags are sampled diagnostics.

    ``max_queries`` includes the reserved ``audit_points`` exact queries.
    Audit randomness is independent of adaptation; audit labels never feed
    back into refinement. An audit failure is reported, not repaired using the
    held-out sample. Even a passing result is NOT a uniform error certificate.
    """
    for name, value, minimum in (("max_queries", max_queries, 2),
                                 ("audit_points", audit_points, 0)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    for name, value in (("relative_tolerance", relative_tolerance),
                        ("absolute_tolerance", absolute_tolerance),
                        ("min_interval_width", min_interval_width),
                        ("residual_tolerance", residual_tolerance),
                        ("orthogonality_tolerance", orthogonality_tolerance)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if min_interval_width > 1:
        raise ValueError("min_interval_width cannot exceed the path length")
    if min_interval_width < 4 * np.finfo(float).eps:
        raise ValueError("min_interval_width must exceed double-precision path resolution")
    grid = np.linspace(0., 1., 5) if initial_grid is None else np.asarray(initial_grid, dtype=float)
    if (grid.ndim != 1 or len(grid) < 2 or not np.isfinite(grid).all()
            or np.any(np.diff(grid) <= 0) or grid[0] != 0 or grid[-1] != 1):
        raise ValueError("initial_grid must increase strictly from 0 to 1")
    if np.min(np.diff(grid)) < 4 * np.finfo(float).eps:
        raise ValueError("initial_grid intervals must exceed double-precision path resolution")
    main_budget = int(max_queries - audit_points)
    if main_budget < len(grid):
        raise ValueError("max_queries must cover initial_grid plus reserved audit_points")
    cache: dict[float, SpectralPoint] = {}
    numerical_failures: list[dict] = []

    def query(s: float, *, audit: bool = False) -> SpectralPoint:
        point = spectral_teacher(terms, float(s), **teacher_kwargs)
        energy_scale = max(1., float(np.max(np.abs(point.energies))))
        residual_ok = point.max_eigenpair_residual <= residual_tolerance * energy_scale
        orthogonality_ok = point.orthogonality_error <= orthogonality_tolerance
        if not residual_ok or not orthogonality_ok:
            numerical_failures.append({"s": float(s), "audit": audit,
                                       "residual_ok": bool(residual_ok),
                                       "orthogonality_ok": bool(orthogonality_ok)})
            point = replace(point, moments_resolved=False, g_ss=float("nan"), d2=float("nan"))
        return point

    for s in grid:
        cache[float(s)] = query(float(s))
    leaves = [{"left": float(a), "right": float(b), "check": None}
              for a, b in zip(grid[:-1], grid[1:])]
    stop_reason = "sampled_tolerance"
    while True:
        pending = [leaf for leaf in leaves if leaf["check"] is None]
        if pending:
            if len(cache) >= main_budget:
                stop_reason = "query_budget"
                break
            # Widest unchecked interval first avoids exhausting one branch
            # before ever inspecting the remaining initial path intervals.
            leaf = max(pending, key=lambda item: item["right"] - item["left"])
            midpoint = (leaf["left"] + leaf["right"]) / 2
            cache[midpoint] = query(midpoint)
            leaf["check"] = _interpolation_check(cache[leaf["left"]], cache[leaf["right"]],
                                                  cache[midpoint], relative_tolerance, absolute_tolerance)
            continue
        failing = [leaf for leaf in leaves if leaf["check"]["needs_refinement"]]
        refinable = [leaf for leaf in failing if leaf["right"] - leaf["left"] > min_interval_width]
        if not refinable:
            if failing:
                stop_reason = "minimum_interval_width"
            break
        if len(cache) >= main_budget:
            stop_reason = "query_budget"
            break
        # Discontinuity/unresolved status outranks curvature; among equivalent
        # failures choose widest first to avoid starving a second rank change.
        def priority(leaf):
            check = leaf["check"]
            unresolved = any(reason != "interpolation_tolerance" for reason in check["reasons"])
            return (unresolved, (leaf["right"] - leaf["left"]) if unresolved else check["scaled_error"])
        leaf = max(refinable, key=priority)
        midpoint = (leaf["left"] + leaf["right"]) / 2
        leaves.remove(leaf)
        leaves.extend(({"left": leaf["left"], "right": midpoint, "check": None},
                       {"left": midpoint, "right": leaf["right"], "check": None}))

    points = [cache[s] for s in sorted(cache)]
    locations = np.asarray([point.s for point in points])
    rng = np.random.default_rng(seed)
    audits: list[SpectralPoint] = []
    audit_checks: list[dict] = []
    # Continuous draws are independent of the adaptive queries. Guard exact
    # coincidences for unusual PRNG inputs without changing the query budget.
    for _ in range(int(audit_points)):
        s = float(rng.uniform(0., 1.))
        while s in cache or any(point.s == s for point in audits):
            s = float(rng.uniform(0., 1.))
        actual = query(s, audit=True)
        index = int(np.clip(np.searchsorted(locations, s, side="right") - 1, 0, len(points) - 2))
        audit_checks.append(_interpolation_check(points[index], points[index + 1], actual,
                                                  relative_tolerance, absolute_tolerance))
        audits.append(actual)
    interval_diagnostics = []
    for leaf in sorted(leaves, key=lambda item: item["left"]):
        check = leaf["check"]
        interval_diagnostics.append(check if check is not None else {
            "left": leaf["left"], "right": leaf["right"], "query": None,
            "scaled_error": None, "needs_refinement": True, "reasons": ["unchecked_budget"]})
    converged = all(not interval["needs_refinement"] for interval in interval_diagnostics)
    audit_passed = all(not check["needs_refinement"] for check in audit_checks) if audits else None
    diagnostics = _json_safe({
        "method": "dense_midpoint_adaptive_v1", "uniform_certificate": False,
        "sparse_teacher": False, "max_queries": max_queries,
        "query_count": len(points) + len(audits), "profile_query_count": len(points),
        "audit_query_count": len(audits), "audit_seed": seed,
        "stop_reason": stop_reason, "budget_exhausted": stop_reason == "query_budget",
        "sampled_refinement_converged": converged, "audit_passed": audit_passed,
        "sampled_checks_passed": converged and audit_passed is True,
        "relative_tolerance": relative_tolerance, "absolute_tolerance": absolute_tolerance,
        "min_interval_width": min_interval_width,
        "residual_tolerance": residual_tolerance, "orthogonality_tolerance": orthogonality_tolerance,
        "numerical_failures": numerical_failures,
        "unresolved_profile_points": [point.s for point in points if not point.moments_resolved],
        "ground_ranks": [point.ground_rank for point in points],
        "intervals": interval_diagnostics, "audit_checks": audit_checks,
        "max_audit_scaled_error": max((item["scaled_error"] for item in audit_checks), default=None),
    })
    return AdaptiveSpectralProfile(points, audits, diagnostics)
