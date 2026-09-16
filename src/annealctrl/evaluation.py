"""Leakage-aware experimental accounting and paired parent-level statistics."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import beta


def teacher_relative_regret(policy_loss, teacher_loss) -> np.ndarray:
    """Loss difference, NOT clipped: imperfect best-found teachers can be beaten."""
    policy, teacher = np.broadcast_arrays(np.asarray(policy_loss, dtype=float), np.asarray(teacher_loss, dtype=float))
    if not np.isfinite(policy).all() or not np.isfinite(teacher).all():
        raise ValueError("losses must be finite")
    return policy - teacher


def audit_parent_splits(parent_ids: Sequence[str], splits: Sequence[str]) -> dict[str, int]:
    """Reject leakage of any logical parent into multiple dataset splits.

    All embeddings, strengths, gauges, permutations and runtime variants inherit
    their parent's split. This audit assumes the provided lineage IDs are correct;
    it cannot detect two different IDs for the same original parent.
    """
    if len(parent_ids) != len(splits) or not len(parent_ids):
        raise ValueError("equal nonempty parent_ids and splits required")
    seen: dict[str, str] = {}
    for parent, split in zip(parent_ids, splits):
        if not isinstance(parent, str) or not parent or split not in {"train", "validation", "test"}:
            raise ValueError("nonempty string parent IDs and canonical splits required")
        if parent in seen and seen[parent] != split:
            raise ValueError(f"parent leakage: {parent!r} occurs in {seen[parent]} and {split}")
        seen[parent] = split
    return {split: sum(value == split for value in seen.values()) for split in ("train", "validation", "test")}


@dataclass(frozen=True)
class PairedBootstrapResult:
    mean_difference: float
    ci_low: float
    ci_high: float
    confidence: float
    n_parents: int
    n_observations: int
    n_resamples: int
    weighting: str = "equal_parent_mean"


def paired_parent_bootstrap(
    method_loss: Sequence[float],
    baseline_loss: Sequence[float],
    parent_ids: Sequence[str],
    *,
    confidence: float = 0.95,
    n_resamples: int = 10000,
    seed: int = 0,
) -> PairedBootstrapResult:
    """Paired percentile CI resampling logical parents, not their correlated variants.

    Estimand is the mean of within-parent mean differences (equal parent weight).
    A negative difference favors the method. Training seeds should be analyzed
    separately or with a prespecified multi-level design, not silently pooled as
    independent parents. This function does not bootstrap training uncertainty.
    """
    method = np.asarray(method_loss, dtype=float)
    baseline = np.asarray(baseline_loss, dtype=float)
    parents = np.asarray(parent_ids, dtype=str)
    if method.ndim != 1 or method.shape != baseline.shape or method.shape != parents.shape or not len(method):
        raise ValueError("need matching nonempty one-dimensional observations")
    if not np.isfinite(method).all() or not np.isfinite(baseline).all():
        raise ValueError("losses must be finite")
    if not 0 < confidence < 1 or not isinstance(n_resamples, int) or n_resamples < 1:
        raise ValueError("confidence in (0,1) and positive n_resamples required")
    unique, inverse = np.unique(parents, return_inverse=True)
    if len(unique) < 2:
        raise ValueError("at least two independent parents required for bootstrap")
    parent_diffs = np.bincount(inverse, weights=method - baseline) / np.bincount(inverse)
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples)
    # Bound memory for large parent collections.
    chunk_size = max(1, min(512, 1000000 // len(unique)))
    for start in range(0, n_resamples, chunk_size):
        count = min(chunk_size, n_resamples - start)
        indices = rng.integers(len(unique), size=(count, len(unique)))
        means[start:start + count] = parent_diffs[indices].mean(axis=1)
    alpha = (1 - confidence) / 2
    lower, upper = np.quantile(means, [alpha, 1 - alpha])
    return PairedBootstrapResult(float(parent_diffs.mean()), float(lower), float(upper), confidence, len(unique), len(method), n_resamples)


@dataclass(frozen=True)
class TimeToSolutionResult:
    probability_estimate: float
    probability_ci: tuple[float, float]
    target_success: float
    nominal_reads: float
    nominal_seconds: float
    seconds_ci: tuple[float, float]
    censored: bool
    assumption: str = "independent identically distributed trials; fixed per-read cost"


def _reads_for_probability(p: float, target: float) -> float:
    if p <= 0:
        return math.inf
    if p >= 1:
        return 1.0
    return float(math.ceil(math.log1p(-target) / math.log1p(-p)))


def time_to_solution(
    successes: int,
    reads: int,
    seconds_per_read: float,
    *,
    target_success: float = 0.99,
    confidence: float = 0.95,
) -> TimeToSolutionResult:
    """Binomial Clopper--Pearson uncertainty and honest zero-success censoring.

    ``seconds_per_read`` must include all recurring costs assigned to a read.
    Fixed programming, inference and search costs belong in separate accounting.
    A zero-success observation produces infinite nominal TTS and an unbounded CI,
    not a pseudocount-derived finite estimate. This is time-to-target instead when
    'success' denotes a prespecified energy target rather than certified optimum.
    """
    if not isinstance(successes, int) or not isinstance(reads, int) or reads < 1 or not 0 <= successes <= reads:
        raise ValueError("integer counts satisfying 0 <= successes <= reads required")
    if not np.isfinite(seconds_per_read) or seconds_per_read <= 0 or not 0 < target_success < 1 or not 0 < confidence < 1:
        raise ValueError("positive read duration and probabilities in (0,1) required")
    alpha = (1 - confidence) / 2
    low = 0.0 if successes == 0 else float(beta.ppf(alpha, successes, reads - successes + 1))
    high = 1.0 if successes == reads else float(beta.ppf(1 - alpha, successes + 1, reads - successes))
    p = successes / reads
    needed = _reads_for_probability(p, target_success)
    interval = (_reads_for_probability(high, target_success) * seconds_per_read, _reads_for_probability(low, target_success) * seconds_per_read)
    return TimeToSolutionResult(p, (low, high), target_success, needed, needed * seconds_per_read, interval, successes == 0)


def amortized_cost(
    *,
    data_seconds: float,
    training_seconds: float,
    deployment_instances: int,
    inference_seconds: float,
    online_search_seconds: float,
    execution_seconds: float,
    embedding_seconds: float = 0.0,
    programming_seconds: float = 0.0,
) -> dict[str, float]:
    """Report upfront, online, one-instance total, and deployment-amortized costs.

    All online inputs are per new instance. Components must not overlap: e.g.
    do not count online-search execution inside execution_seconds a second time.
    """
    parts = np.array([data_seconds, training_seconds, inference_seconds, online_search_seconds, execution_seconds, embedding_seconds, programming_seconds])
    if not np.isfinite(parts).all() or np.any(parts < 0) or not isinstance(deployment_instances, int) or deployment_instances < 1:
        raise ValueError("finite nonnegative costs and positive integer deployment count required")
    upfront = float(data_seconds + training_seconds)
    online = float(inference_seconds + online_search_seconds + execution_seconds + embedding_seconds + programming_seconds)
    return {"upfront_seconds": upfront, "online_seconds_per_instance": online, "total_for_one_instance_seconds": upfront + online, "amortized_seconds_per_instance": upfront / deployment_instances + online, "deployment_total_seconds": upfront + deployment_instances * online}
