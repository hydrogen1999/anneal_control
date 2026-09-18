"""Shared best-found teacher banks and strictly accounted local search.

None of the routines certifies a globally optimal control. Keep every outcome;
candidate bank definitions, seeds, families and budgets are experiment metadata.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable, Sequence

import numpy as np
from scipy.special import ndtri
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

    The critic is trained on the candidate bank and deployed on the policy's
    proposals. Those came from a different parameterisation - the bank uses
    residual-softmax durations and window/pause closures, the policy uses this
    capped-simplex water filling - and the manifolds do not coincide. Measured on
    P16-scale banks, 92% of policy waveforms sat further from the bank than a
    typical bank waveform sits from its own nearest neighbour (0.133 against
    0.079 in max-norm over nine knots). That gap is where a Spearman correlation
    of 0.55 between predicted and true proposal losses comes from.

    Adding random candidates from this family to the shared bank was measured and
    does NOT fix it: sixteen extra candidates in a sixty-four candidate bank moved
    the mean policy-to-bank distance only from 0.133 to 0.127, because an
    eight-dimensional waveform space is not coverable by a bank of that size. The
    fix that works has to target the model's *actual* proposals, which is what
    ``policy_diagnostics.aggregate_proposals`` collects for a DAgger round.

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


STRATEGIES = ("sobol_local", "bayesian", "policy_gradient")


def optimize_control_family(
    loss_fn: Callable[[Schedule], float], family: str, *, budget: int = 32,
    runtime: float = 1.0, max_slope: float = 4.0, seed: int = 0,
    split: str = "train", allow_test_adaptation: bool = False,
    strategy: str = "sobol_local",
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
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        raise ValueError("budget must be a positive integer")
    decode_durations([0.0], runtime=runtime, max_slope=max_slope)
    dimension = {"linear": 0, "one_window": 3, "two_window": 6, "eight_bin": 8, "pause": 2}[family]
    first = _evaluate(loss_fn, Candidate(f"{family}_0000", family, Schedule.linear(),
                                      {"initial_incumbent": "linear"}), 0)
    if family == "linear" or budget == 1:
        return SearchResult((first,), split, split == "test")
    rng = np.random.default_rng(seed)
    points = qmc.Sobol(dimension, scramble=True, seed=seed).random_base2(
        int(np.ceil(np.log2(budget - 1))))[:budget - 1]
    best_parameters, best_loss = None, first.loss
    records = [first]

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
                                   "proposal": "bayesian_design" if position <= design_marker[0]
                                   else "expected_improvement"})
            record = _evaluate(loss_fn, candidate, position)
            records.append(record)
            return record.loss

        design_marker = [min(max(4, 2 * dimension), budget - 1)]
        minimise(objective, dimension=dimension, budget=budget - 1, seed=seed)
        return SearchResult(tuple(records), split, split == "test")

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
                mean = mean + learning_rate * (advantage[:, None] * (draws - mean)).mean(axis=0) / sigma
            sigma *= decay
        return SearchResult(tuple(records), split, split == "test")

    for index in range(1, budget):
        # The linear closure incumbent has no interior parameter vector for
        # window/pause families. Explore until a genuine parameter incumbent
        # exists, instead of pretending arbitrary 0.5 coordinates encode it.
        exploratory = index % 2 == 1 or best_parameters is None
        parameters = points[index - 1] if exploratory else np.clip(
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
