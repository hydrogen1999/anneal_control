"""Shared best-found teacher banks and strictly accounted local search.

None of the routines certifies a globally optimal control. Keep every outcome;
candidate bank definitions, seeds, families and budgets are experiment metadata.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable, Sequence

import numpy as np
from scipy.special import ndtri, ndtr
from scipy.stats import qmc

from .schedules import Schedule, decode_durations, pause_schedule, slow_window_schedule, window_schedule


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    family: str
    schedule: Schedule
    parameters: dict = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationRecord:
    candidate: Candidate
    loss: float
    evaluation_index: int
    elapsed_seconds: float


@dataclass(frozen=True)
class SearchResult:
    records: tuple[EvaluationRecord, ...]
    split: str
    online_adaptation: bool
    reference_status: str = "best_found_within_evaluated_candidates"

    @property
    def best(self) -> EvaluationRecord:
        if not self.records:
            raise ValueError("no candidates evaluated")
        return min(self.records, key=lambda r: r.loss)

    @property
    def n_evaluations(self) -> int:
        return len(self.records)

    @property
    def elapsed_seconds(self) -> float:
        return sum(record.elapsed_seconds for record in self.records)


def capped_simplex_samples(logits, *, max_ds_dtau: float = 4.0) -> np.ndarray:
    """NumPy twin of ``models.monotone_samples``: the policy's own decoder.

    The original bank and policy use different parameterisations. Random logits
    through this decoder provide a control for that difference; they do not
    reproduce the learned distribution. Geometric distance alone does not show
    that critic error is caused by distribution shift or that one acquisition
    rule improves control. See ``acquisition_study`` for matched comparisons.

    Kept in numpy, and pinned to the torch decoder by test, so callers can build
    waveforms on the policy manifold without ``search.py`` depending on torch.
    """
    logits = np.atleast_2d(np.asarray(logits, dtype=float))
    bins = logits.shape[-1]
    if bins < 1 or not np.isfinite(max_ds_dtau) or max_ds_dtau < 1 or not np.isfinite(logits).all():
        raise ValueError("finite logits and max_ds_dtau >= 1 required")
    if max_ds_dtau == 1:
        increments = np.ones_like(logits) / bins
    else:
        cap = min(float(max_ds_dtau) / bins, 1.0)
        weights = np.exp(np.clip(logits - logits.max(axis=-1, keepdims=True), -30, None))
        saturated = np.zeros_like(logits, dtype=bool)
        increments = weights / weights.sum(axis=-1, keepdims=True)
        for _ in range(bins):
            free_weights = weights * ~saturated
            remaining = np.clip(1 - cap * saturated.sum(axis=-1, keepdims=True), 0, None)
            free = remaining * free_weights / np.maximum(
                free_weights.sum(axis=-1, keepdims=True), 1e-30)
            increments = np.where(saturated, cap, free)
            new_saturated = saturated | (free > cap)
            if np.array_equal(new_saturated, saturated):
                break
            saturated = new_saturated
    cumulative = np.cumsum(increments, axis=-1)
    return np.concatenate((np.zeros_like(cumulative[..., :1]), cumulative[..., :-1],
                           np.ones_like(cumulative[..., :1])), axis=-1)


def shared_candidate_bank(
    n: int = 64,
    n_segments: int = 8,
    *,
    seed: int = 0,
    runtime: float = 1.0,
    max_slope: float = 10.0,
    include_pauses: bool = True,
) -> list[Candidate]:
    """A fixed deterministic control bank shared across instances at a runtime.

    Prefix: linear, three one-window controls, two two-window controls, and two
    true pauses if requested/feasible. Remaining slots use scrambled Sobol points
    mapped to zero-mean normal duration logits. A small n truncates this prefix.
    Physics-derived candidates can be appended by callers, who must charge their
    spectrum construction and all resulting outcome evaluations separately.
    """
    if not isinstance(n, int) or n < 1 or not isinstance(n_segments, int) or n_segments < 1:
        raise ValueError("n and n_segments must be positive integers")
    decode_durations([0.0], runtime=runtime, max_slope=max_slope)
    result = [Candidate("linear", "linear", Schedule.linear())]
    for i, (a, b, q) in enumerate([(0.2, 0.45, 0.7), (0.4, 0.65, 0.7), (0.65, 0.9, 0.7)]):
        result.append(Candidate(f"one_window_{i}", "one_window", slow_window_schedule(a, b, q, runtime=runtime, max_slope=max_slope), {"a": a, "b": b, "q": q}))
    for i, windows in enumerate([[(0.2, 0.35), (0.65, 0.8)], [(0.35, 0.45), (0.8, 0.9)]]):
        result.append(Candidate(f"two_window_{i}", "two_window", window_schedule(windows, [0.4, 0.4], runtime=runtime, max_slope=max_slope), {"windows": windows, "weights": [0.4, 0.4]}))
    spare_fraction = max(0.0, 1 - 1 / (runtime * max_slope))
    if include_pauses and spare_fraction > 1e-14:
        for i, location in enumerate([0.45, 0.7]):
            fraction = min(0.2, 0.5 * spare_fraction)
            result.append(Candidate(f"pause_{i}", "pause", pause_schedule(location, fraction, runtime=runtime, max_slope=max_slope), {"location": location, "pause_fraction": fraction}))
    remaining = max(0, n - len(result))
    if remaining:
        points = qmc.Sobol(n_segments, scramble=True, seed=seed).random_base2(int(np.ceil(np.log2(remaining))))[:remaining]
        logits_bank = 1.5 * ndtri(np.clip(points, 1e-12, 1 - 1e-12))
        logits_bank -= logits_bank.mean(axis=1, keepdims=True)
        for i, logits in enumerate(logits_bank):
            result.append(Candidate(f"sobol_{i:04d}", "duration_logits", decode_durations(logits, runtime=runtime, max_slope=max_slope), {"logits": logits.tolist()}))
    return result[:n]


def _check_split(split: str) -> None:
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")


def _evaluate(loss_fn: Callable[[Schedule], float], candidate: Candidate, index: int) -> EvaluationRecord:
    start = perf_counter()
    loss = float(loss_fn(candidate.schedule))
    elapsed = perf_counter() - start
    if not np.isfinite(loss):
        raise ValueError(f"nonfinite loss for {candidate.candidate_id}")
    return EvaluationRecord(candidate, loss, index, elapsed)


def evaluate_candidates(
    loss_fn: Callable[[Schedule], float],
    candidates: Sequence[Candidate],
    *,
    budget: int | None = None,
    split: str = "train",
    online_adaptation: bool = False,
) -> SearchResult:
    """Evaluate a prefix once, retaining every loss and its measured wall time.

    Test-bank evaluation is permitted as a *reference* (online_adaptation=False).
    Selecting among it to deploy a method on that test instance is online search:
    callers must set online_adaptation=True and charge all evaluations. This API
    does not perform across-instance fitting or validation-set hyperparameter
    selection; test records must never be reused as training examples.
    """
    _check_split(split)
    if not candidates:
        raise ValueError("candidate sequence must be nonempty")
    if budget is None:
        budget = len(candidates)
    if not isinstance(budget, int) or budget < 1 or budget > len(candidates):
        raise ValueError("budget must be an integer in [1, number of candidates]")
    ids = [candidate.candidate_id for candidate in candidates[:budget]]
    if len(set(ids)) != len(ids):
        raise ValueError("candidate IDs must be unique")
    records = tuple(_evaluate(loss_fn, candidate, i) for i, candidate in enumerate(candidates[:budget]))
    return SearchResult(records, split, online_adaptation)


def equal_budget_refinement(
    loss_fn: Callable[[Schedule], float],
    initial_logits: Sequence[float],
    *,
    budget: int,
    s_knots: Sequence[float] | None = None,
    runtime: float = 1.0,
    max_slope: float = 10.0,
    seed: int = 0,
    step_scale: float = 0.75,
    split: str = "train",
    allow_test_adaptation: bool = False,
) -> SearchResult:
    """Simple fixed-budget stochastic hill-climbing baseline in feasible logits.

    The initial evaluation consumes one unit, so ``budget`` is TOTAL objective
    calls, not extra calls after a free seed evaluation. Rejected trials also
    count. This is random local search, not a Bayesian-optimization baseline.
    Test-instance refinement requires explicit opt-in and is always marked as
    online adaptation. Do not fit global model/hyperparameters on its results.
    """
    _check_split(split)
    if split == "test" and not allow_test_adaptation:
        raise ValueError("test refinement must be explicitly charged as online adaptation")
    if not isinstance(budget, int) or budget < 1 or not np.isfinite(step_scale) or step_scale <= 0:
        raise ValueError("positive integer budget and positive finite step_scale required")
    best_logits = np.asarray(initial_logits, dtype=float).copy()
    initial = decode_durations(best_logits, s_knots, runtime=runtime, max_slope=max_slope)
    best_logits -= best_logits.mean()
    first = Candidate("refine_0000", "duration_logits", initial, {"logits": best_logits.tolist()})
    records = [_evaluate(loss_fn, first, 0)]
    best_loss = records[0].loss
    rng = np.random.default_rng(seed)
    for i in range(1, budget):
        direction = rng.normal(size=best_logits.shape)
        direction -= direction.mean()
        proposal = best_logits + step_scale * direction
        schedule = decode_durations(proposal, s_knots, runtime=runtime, max_slope=max_slope)
        record = _evaluate(loss_fn, Candidate(f"refine_{i:04d}", "duration_logits", schedule, {"logits": proposal.tolist()}), i)
        records.append(record)
        if record.loss < best_loss:
            best_logits, best_loss = proposal, record.loss
    return SearchResult(tuple(records), split, split == "test")


CONTROL_FAMILIES = ("linear", "one_window", "two_window", "eight_bin", "pause")


STRATEGIES = ("sobol_local", "bayesian", "policy_gradient", "cem")


def decode_eight_bin(parameters, *, runtime=1., max_slope=4.) -> Schedule:
    """The identical unit-box decoder used by all budget-study optimizers."""
    p = np.asarray(parameters, dtype=float)
    if p.shape != (8,) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("eight_bin needs eight finite parameters in [0,1]")
    return decode_durations(2 * ndtri(np.clip(p, 1e-6, 1 - 1e-6)),
                            runtime=runtime, max_slope=max_slope)


def eight_bin_hint(schedule: Schedule, *, runtime=1., max_slope=4.,
                   mode="reject", tolerance=1e-7) -> dict:
    """Invert fixed-s duration controls; explicitly project other waveforms.

    A policy's fixed-time knots generally do NOT belong to this family. Sampling
    the inverse t(s) at the family's nine s knots defines a deterministic
    projection. Pauses use the midpoint of their inverse interval. Residual
    durations are floored for the finite unit-box decoder. Both changes are
    included in the exact piecewise-linear sup-norm error (union of knots).
    This is not an optimal approximation. The projected control must be scored
    as a charged query; the original waveform's outcome cannot be reused.
    """
    if mode not in {"reject", "project"}:
        raise ValueError("hint mode must be reject or project")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    # Only float32 construction tolerance; projection retains exact family bound.
    schedule.validate_slope(runtime=runtime, max_slope=max_slope * (1 + 1e-6))
    grid = np.linspace(0., 1., 9)
    # Exact mid-inverse at repeated s knots; interpolation elsewhere.
    values = np.unique(schedule.s_knots)
    times = np.array([schedule.tau_knots[schedule.s_knots == s].mean() for s in values])
    inverse = np.interp(grid, values, times)
    inverse[0], inverse[-1] = 0., 1.
    minimum = np.diff(grid) / (runtime * max_slope)
    spare = 1. - minimum.sum()
    if spare < -1e-12:
        raise ValueError("infeasible slope/runtime")
    if spare <= 1e-12:
        point = np.full(8, .5)
    else:
        residual = np.maximum((np.diff(inverse) - minimum) / spare, 1e-12)
        logits = np.log(residual)
        # Common offset is a gauge; centering extremes uses finite range best.
        logits -= .5 * (logits.max() + logits.min())
        point = np.clip(ndtr(logits / 2), 1e-6, 1 - 1e-6)
    projected = decode_eight_bin(point, runtime=runtime, max_slope=max_slope)
    knots = np.union1d(schedule.tau_knots, projected.tau_knots)
    error = float(np.max(np.abs(schedule(knots) - projected(knots))))
    if mode == "reject" and error > tolerance:
        raise ValueError(f"waveform is not exactly representable by eight_bin (sup error {error:g}); use explicit project mode")
    return {"unit_parameters": point.tolist(), "waveform": projected.to_dict(),
            "original_waveform": schedule.to_dict(), "sup_waveform_error": error,
            "exact_within_tolerance": error <= tolerance, "tolerance": tolerance,
            "mapping": "inverse_grid_midpoint_pause_residual_duration_projection",
            "mode": mode, "evaluation_required": True}


def bank_candidate_to_unit_parameters(candidate, *, runtime: float = 1.0,
                                      max_slope: float = 4.0) -> tuple[str, np.ndarray] | None:
    """Invert a bank candidate into the search's unit-cube parameters.

    Warm-starting a search from a learned selection needs this to be *exact*.
    The bank stores construction parameters -- window edges, pause fractions,
    duration logits -- while ``optimize_control_family`` works in a unit cube and
    decodes. If the inverse were approximate the experiment would be measuring
    the conversion error rather than the hint, so every family here round-trips
    to the same waveform and an unrecognised family is refused rather than
    guessed at.

    ``runtime`` and ``max_slope`` are required because one inverse genuinely
    depends on them: the search encodes a pause as a fraction of the spare time
    ``1 - 1/(runtime * max_slope)``, while the bank stores the fraction itself.
    Ignoring that produced a waveform 0.032 away from the candidate it claimed to
    reproduce.

    Returns ``None`` for ``linear``, which has no free parameters, and the search
    evaluates it as trial 0 regardless.
    """
    family, parameters = candidate.family, dict(candidate.parameters or {})
    if family == "linear":
        return None
    if family == "duration_logits":
        # decode() for eight_bin applies 2 * ndtri(p), so p = ndtr(logits / 2)
        # reproduces the stored logits exactly.
        logits = np.asarray(parameters["logits"], dtype=float)
        return "eight_bin", ndtr(logits / 2.0)
    if family == "one_window":
        a, b, q = (float(parameters[k]) for k in ("a", "b", "q"))
        p0 = a / 0.98
        p1 = (((b - a) / (1.0 - a)) - 0.01) / 0.99
        return "one_window", np.clip(np.array([p0, p1, q]), 1e-6, 1 - 1e-6)
    if family == "two_window":
        windows = parameters["windows"]
        weights = list(parameters["weights"])
        values = []
        for a, b in windows:
            a, b = float(a), float(b)
            values.append(a / 0.98)
            values.append((((b - a) / (1.0 - a)) - 0.01) / 0.99)
        # decode() reads weights as [p4, (1 - p4) * p5].
        w0, w1 = float(weights[0]), float(weights[1])
        values.append(w0)
        values.append(w1 / (1.0 - w0) if w0 < 1.0 else 0.0)
        return "two_window", np.clip(np.array(values), 1e-6, 1 - 1e-6)
    if family == "pause":
        location = float(parameters["location"])
        fraction = float(parameters["pause_fraction"])
        spare = max(0.0, 1.0 - 1.0 / (runtime * max_slope))
        if spare <= 0:
            raise ValueError("no spare time at this runtime and slope cap; a pause candidate "
                             "cannot be expressed in the search's parameterisation")
        return "pause", np.clip(np.array([location, fraction / spare]), 1e-6, 1 - 1e-6)
    raise ValueError(f"cannot map candidate family {family!r} to search parameters; "
                     "add an explicit inverse rather than approximating one")


def optimize_control_family(
    loss_fn: Callable[[Schedule], float], family: str, *, budget: int = 32,
    runtime: float = 1.0, max_slope: float = 4.0, seed: int = 0,
    split: str = "train", allow_test_adaptation: bool = False,
    strategy: str = "sobol_local", warm_start=None,
) -> SearchResult:
    """Exact-waveform Sobol exploration plus incumbent-centered random search.

    All tunable families get exactly ``budget`` objective calls, including their
    linear incumbent. Odd trials explore a fixed Sobol sequence; even trials
    perturb the best parameters found so far. This is a reproducible classical
    baseline, not Bayesian optimization or a certificate of family optimality.
    Linear has no free parameters and is evaluated ONCE (never padded with fake
    optimization calls). Returned controls retain their actual switching knots;
    they are NOT resampled onto the learned policy's nine-point representation.
    """
    _check_split(split)
    if split == "test" and not allow_test_adaptation:
        raise ValueError("test control search requires explicit allow_test_adaptation=True")
    if family not in CONTROL_FAMILIES:
        raise ValueError(f"unknown control family {family!r}")
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown search strategy {strategy!r}; available: {list(STRATEGIES)}")
    if warm_start is not None and strategy not in {"sobol_local", "bayesian"}:
        raise ValueError(f"warm_start is only implemented for sobol_local and bayesian, not {strategy!r}; "
                         "dropping the hint silently would make a warm-start experiment "
                         "measure nothing")
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        raise ValueError("budget must be a positive integer")
    decode_durations([0.0], runtime=runtime, max_slope=max_slope)
    dimension = {"linear": 0, "one_window": 3, "two_window": 6, "eight_bin": 8, "pause": 2}[family]
    hint = None
    if warm_start is not None:
        hint = np.asarray(warm_start, dtype=float)
        if family == "linear" or budget < 2:
            raise ValueError("warm_start needs a tunable family and budget >= 2; linear and hint are charged")
        if hint.shape != (dimension,) or not np.isfinite(hint).all() or np.any((hint < 1e-6) | (hint > 1 - 1e-6)):
            raise ValueError(f"warm_start must be {dimension} finite parameters in [1e-6, 1-1e-6]")
    first = _evaluate(loss_fn, Candidate(f"{family}_0000", family, Schedule.linear(),
                                      {"initial_incumbent": "linear"}), 0)
    if family == "linear" or budget == 1:
        return SearchResult((first,), split, split == "test")
    rng = np.random.default_rng(seed)
    points = qmc.Sobol(dimension, scramble=True, seed=seed).random_base2(
        int(np.ceil(np.log2(budget - 1))))[:budget - 1]
    best_parameters, best_loss = None, first.loss
    records = [first]

    warm_offset = int(hint is not None)

    def decode(parameters):
        p = np.clip(parameters, 1e-6, 1 - 1e-6)
        if family == "eight_bin":
            return decode_durations(2 * ndtri(p), runtime=runtime, max_slope=max_slope)
        if family == "pause":
            spare = max(0., 1 - 1 / (runtime * max_slope))
            return pause_schedule(float(p[0]), float(p[1] * spare), runtime=runtime, max_slope=max_slope)
        if family == "one_window":
            a = 0.98 * p[0]
            b = a + (1 - a) * (0.01 + 0.99 * p[1])
            return slow_window_schedule(a, b, p[2], runtime=runtime, max_slope=max_slope)
        windows = []
        for k in (0, 2):
            a = 0.98 * p[k]
            windows.append((a, a + (1 - a) * (0.01 + 0.99 * p[k + 1])))
        # Third stick is the uniform residual; overlap is allowed and explicit.
        weights = [p[4], (1 - p[4]) * p[5]]
        return window_schedule(windows, weights, runtime=runtime, max_slope=max_slope)

    if strategy == "bayesian":
        # The linear incumbent is already charged as trial 0; the optimiser gets
        # the rest of the budget and never sees the loss before it proposes.
        from .bayesopt import minimise

        index = [1]

        def objective(parameters: np.ndarray) -> float:
            schedule = decode(np.clip(parameters, 1e-6, 1 - 1e-6))
            schedule.validate_slope(runtime=runtime, max_slope=max_slope)
            position = index[0]
            index[0] += 1
            candidate = Candidate(f"{family}_{position:04d}", family, schedule,
                                  {"unit_parameters": np.asarray(parameters).tolist(),
                                   "proposal": ("warm_start" if hint is not None and position == 1 else
                                                "bayesian_design" if position <= design_marker[0]
                                                else "expected_improvement")})
            record = _evaluate(loss_fn, candidate, position)
            records.append(record)
            return record.loss

        design_marker = [min(max(4, 2 * dimension), budget - 1)]
        minimise(objective, dimension=dimension, budget=budget - 1, seed=seed,
                 initial=None if hint is None else [hint])
        return SearchResult(tuple(records), split, split == "test")

    if hint is not None:
        # Charged like any other call, and it becomes the incumbent that the
        # local moves refine from.
        schedule = decode(hint)
        schedule.validate_slope(runtime=runtime, max_slope=max_slope)
        record = _evaluate(loss_fn, Candidate(f"{family}_0001", family, schedule,
                                              {"unit_parameters": hint.tolist(),
                                               "proposal": "warm_start"}), 1)
        records.append(record)
        if record.loss < best_loss:
            best_parameters, best_loss = hint.copy(), record.loss

    if strategy == "policy_gradient":
        # REINFORCE on a diagonal Gaussian over the unconstrained parameter
        # vector, squashed to the unit cube. This is the learned counterpart of
        # the other two strategies and exists so the comparison contains a
        # method that *learns* rather than only searches: a reviewer asking
        # "against which learned baseline?" has an answer at equal budget.
        #
        # Batch size grows with dimension because a policy gradient estimated
        # from fewer samples than parameters is mostly noise, and sigma decays
        # so late trials exploit what early ones found. The linear incumbent is
        # already charged as trial 0 and the policy never sees a loss before it
        # proposes.
        batch = int(max(4, min(dimension + 2, budget - 1)))
        learning_rate, sigma, decay = 0.4, 1.0, 0.92
        mean = np.zeros(dimension)
        index = 1
        while index < budget:
            size = min(batch, budget - index)
            draws = mean + sigma * rng.normal(size=(size, dimension))
            losses = np.empty(size)
            for offset in range(size):
                parameters = 1.0 / (1.0 + np.exp(-draws[offset]))
                schedule = decode(parameters)
                schedule.validate_slope(runtime=runtime, max_slope=max_slope)
                candidate = Candidate(f"{family}_{index + offset:04d}", family, schedule,
                                      {"unit_parameters": parameters.tolist(),
                                       "proposal": "policy_gradient",
                                       "batch_size": batch,
                                       "learning_rate": learning_rate,
                                       "sigma": float(sigma)})
                record = _evaluate(loss_fn, candidate, index + offset)
                records.append(record)
                losses[offset] = record.loss
            index += size
            # Standardised advantage: lower loss is better, so the sign is
            # flipped. A degenerate batch carries no gradient and is skipped
            # rather than dividing by a vanishing standard deviation.
            spread = float(losses.std())
            if size > 1 and spread > 1e-12:
                advantage = -(losses - losses.mean()) / spread
                # d/dmu log N(z; mu, sigma^2 I) = (z - mu) / sigma^2. An earlier
                # version divided by sigma once, which shrank every step as sigma
                # decayed and left the policy almost stationary.
                mean = mean + learning_rate * (
                    advantage[:, None] * (draws - mean)).mean(axis=0) / (sigma * sigma)
            sigma *= decay
        return SearchResult(tuple(records), split, split == "test")

    if strategy == "cem":
        # Cross-entropy method: sample, keep the best fraction, refit the Gaussian
        # to those elites. It exists here because the first learned baseline in
        # this project was REINFORCE, and at a campaign budget of 64 calls that
        # gets seven gradient updates on eight parameters -- a test of the budget,
        # not of learned search. CEM is the standard choice in exactly this
        # regime: it needs no gradient estimate and improves from the first
        # generation.
        batch = int(max(6, min(4 * dimension, max(6, (budget - 1) // 4))))
        elite_fraction = 0.25
        mean, sigma = np.zeros(dimension), np.ones(dimension)
        index = 1
        while index < budget:
            size = min(batch, budget - index)
            draws = mean + sigma * rng.normal(size=(size, dimension))
            losses = np.empty(size)
            for offset in range(size):
                parameters = 1.0 / (1.0 + np.exp(-draws[offset]))
                schedule = decode(parameters)
                schedule.validate_slope(runtime=runtime, max_slope=max_slope)
                candidate = Candidate(f"{family}_{index + offset:04d}", family, schedule,
                                      {"unit_parameters": parameters.tolist(),
                                       "proposal": "cem",
                                       "batch_size": batch,
                                       "elite_fraction": elite_fraction})
                record = _evaluate(loss_fn, candidate, index + offset)
                records.append(record)
                losses[offset] = record.loss
            index += size
            n_elite = max(2, int(round(elite_fraction * size)))
            if size >= 2:
                elite = draws[np.argsort(losses)[:min(n_elite, size)]]
                mean = elite.mean(axis=0)
                # A floor on sigma keeps the search from collapsing onto one
                # point when a generation happens to be tightly clustered.
                sigma = np.maximum(elite.std(axis=0), 0.05)
        return SearchResult(tuple(records), split, split == "test")

    for index in range(1 + warm_offset, budget):
        # The linear closure incumbent has no interior parameter vector for
        # window/pause families. Explore until a genuine parameter incumbent
        # exists, instead of pretending arbitrary 0.5 coordinates encode it.
        exploratory = index % 2 == 1 or best_parameters is None
        parameters = points[index - 1 - warm_offset] if exploratory else np.clip(
            best_parameters + rng.normal(0., 0.15, dimension), 1e-6, 1 - 1e-6)
        schedule = decode(parameters)
        schedule.validate_slope(runtime=runtime, max_slope=max_slope)
        candidate = Candidate(f"{family}_{index:04d}", family, schedule,
                              {"unit_parameters": parameters.tolist(),
                               "proposal": "sobol" if exploratory else "incumbent_local"})
        record = _evaluate(loss_fn, candidate, index)
        records.append(record)
        if record.loss < best_loss:
            best_parameters, best_loss = parameters.copy(), record.loss
    return SearchResult(tuple(records), split, split == "test")


@dataclass(frozen=True)
class InterventionLabels:
    baseline_logits: np.ndarray
    baseline_schedule: Schedule
    directions: np.ndarray
    derivatives: np.ndarray
    plus_losses: np.ndarray
    minus_losses: np.ndarray
    epsilon: float
    n_evaluations: int
    elapsed_seconds: float
    runtime: float
    max_slope: float


def finite_difference_interventions(
    loss_fn: Callable[[Schedule], float],
    baseline_logits: Sequence[float],
    *,
    directions: np.ndarray | None = None,
    epsilon: float = 1e-3,
    s_knots: Sequence[float] | None = None,
    runtime: float = 1.0,
    max_slope: float = 10.0,
) -> InterventionLabels:
    """Central differences along feasible controls; labels retain their baseline.

    Default directions are centered coordinate vectors, removing the irrelevant
    common-logit offset. Every plus/minus propagation is charged: 2*directions.
    Derivatives are finite-difference labels, not global-optimality certificates.
    Repeat at epsilon/2 to estimate discretization error before sharp supervision.
    """
    logits = np.asarray(baseline_logits, dtype=float).copy()
    baseline = decode_durations(logits, s_knots, runtime=runtime, max_slope=max_slope)
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    if directions is None:
        directions = np.eye(len(logits)) - np.ones((len(logits), len(logits))) / len(logits)
    directions = np.asarray(directions, dtype=float)
    if directions.ndim != 2 or directions.shape[1] != len(logits) or len(directions) < 1 or not np.isfinite(directions).all():
        raise ValueError("directions must have shape (n_directions, n_logits) and be finite")
    plus, minus = [], []
    start = perf_counter()
    for direction in directions:
        plus.append(float(loss_fn(decode_durations(logits + epsilon * direction, s_knots, runtime=runtime, max_slope=max_slope))))
        minus.append(float(loss_fn(decode_durations(logits - epsilon * direction, s_knots, runtime=runtime, max_slope=max_slope))))
    plus, minus = np.asarray(plus), np.asarray(minus)
    if not np.isfinite(plus).all() or not np.isfinite(minus).all():
        raise ValueError("nonfinite intervention loss")
    return InterventionLabels(logits, baseline, directions.copy(), (plus - minus) / (2 * epsilon), plus, minus, epsilon, 2 * len(directions), perf_counter() - start, runtime, max_slope)
