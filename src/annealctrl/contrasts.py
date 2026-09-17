"""Method against method, corrected for the family of comparisons.

The per-method results table reports each method's difference from the global
baseline. That is the wrong instrument for the question the paper has to answer,
which is not "does this method beat a fixed schedule?" but "does the proposed
architecture beat the simpler encoders?". Two intervals that both exclude zero
say nothing about whether they differ from each other, and reading overlap off a
plot is not a test.

So the contrast is computed directly: pair on the logical parent, take the mean
difference of parent means, bootstrap the parents, and then correct across every
pair examined. With five encoders there are ten comparisons, and at 95% nominal
coverage the chance of at least one spurious separation is not 5%. Holm's step
down correction is applied to the family, and a pair is only called separated
after correction. Uncorrected numbers are reported alongside, because hiding
them would make the correction unauditable.

``encoder_information_contrast`` asks the narrower question that the embedding
interventions already answered in the search setting: does an encoder that can
see the embedding beat one that cannot? That is a claim about information, not
about architecture, and the two must not be conflated in the write-up.
"""
from __future__ import annotations

from itertools import combinations
from typing import Any, Mapping, Sequence

import numpy as np

from .telemetry import _safe


def _parent_table(rows: Sequence[Mapping[str, Any]], method: str, mode: str,
                  key: str) -> dict[str, float]:
    """Mean of ``key`` per parent, for one method in one mode."""
    buckets: dict[str, list[float]] = {}
    for row in rows:
        if row.get("method") != method or row.get("mode") != mode:
            continue
        value = row.get(key)
        if value is None:
            continue
        buckets.setdefault(str(row["parent_id"]), []).append(float(value))
    if not buckets:
        raise ValueError(f"no rows for method {method!r} in mode {mode!r}")
    return {parent: float(np.mean(values)) for parent, values in buckets.items()}


def _paired_bootstrap(differences: np.ndarray, *, n_resamples: int, seed: int,
                      confidence: float = 0.95) -> dict:
    """Percentile CI and a two-sided bootstrap p-value for a mean difference.

    The p-value inverts the bootstrap distribution about zero rather than
    assuming normality; with few parents the difference matters.
    """
    if differences.size < 2:
        return {"ci_low": None, "ci_high": None, "p_value": None,
                "status": "insufficient_independent_parents"}
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples)
    chunk = max(1, min(512, 1_000_000 // differences.size))
    for start in range(0, n_resamples, chunk):
        count = min(chunk, n_resamples - start)
        means[start:start + count] = differences[
            rng.integers(differences.size, size=(count, differences.size))].mean(axis=1)
    alpha = (1 - confidence) / 2
    low, high = np.quantile(means, [alpha, 1 - alpha])
    # Two-sided, with the conventional +1 so a p-value is never exactly zero.
    tail = min((means <= 0).sum(), (means >= 0).sum())
    p_value = min(1.0, 2.0 * (tail + 1) / (n_resamples + 1))
    return {"ci_low": float(low), "ci_high": float(high), "p_value": float(p_value),
            "status": "ok"}


def paired_method_contrast(rows: Sequence[Mapping[str, Any]], method_a: str, method_b: str, *,
                           mode: str = "bank", key: str = "loss",
                           bootstrap_resamples: int = 2000, seed: int = 0,
                           confidence: float = 0.95) -> dict:
    """``method_b`` minus ``method_a``, paired on the parent. Negative favours b.

    Refuses methods that were not evaluated on the same parents: an unpaired
    difference of two different populations is not the quantity anyone means.
    """
    table_a = _parent_table(rows, method_a, mode, key)
    table_b = _parent_table(rows, method_b, mode, key)
    if set(table_a) != set(table_b):
        missing = sorted(set(table_a) ^ set(table_b))[:5]
        raise ValueError(f"{method_a!r} and {method_b!r} must cover the same parents; "
                         f"{len(set(table_a) ^ set(table_b))} differ, e.g. {missing}")

    parents = sorted(table_a)
    differences = np.array([table_b[p] - table_a[p] for p in parents])
    statistics = _paired_bootstrap(differences, n_resamples=bootstrap_resamples, seed=seed,
                                   confidence=confidence)
    mean_difference = float(differences.mean())
    separated = bool(statistics["ci_low"] is not None
                     and (statistics["ci_low"] > 0 or statistics["ci_high"] < 0))
    return _safe({
        "method_a": method_a, "method_b": method_b, "mode": mode, "key": key,
        "n_parents": len(parents), "mean_a": float(np.mean([table_a[p] for p in parents])),
        "mean_b": float(np.mean([table_b[p] for p in parents])),
        "mean_difference": mean_difference,
        "std_difference": float(differences.std(ddof=1)) if differences.size > 1 else 0.0,
        "parents_favouring_b": int((differences < 0).sum()),
        **statistics,
        "confidence": confidence, "resamples": bootstrap_resamples,
        "separated": separated,
        "better": (method_b if mean_difference < 0 else method_a) if separated else None,
        "unit_of_independence": "logical_parent",
        "scope": ("paired on the parent; uncorrected for multiplicity, see contrast_matrix "
                  "for the family-corrected decision"),
    })


def _holm(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values, kept monotone in the sorted order."""
    order = np.argsort(p_values)
    adjusted, running, total = [0.0] * len(p_values), 0.0, len(p_values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (total - rank) * float(p_values[index])))
        adjusted[index] = running
    return adjusted


def contrast_matrix(rows: Sequence[Mapping[str, Any]], *, mode: str = "bank", key: str = "loss",
                    methods: Sequence[str] | None = None, bootstrap_resamples: int = 2000,
                    seed: int = 0, confidence: float = 0.95, alpha: float = 0.05) -> dict:
    """Every unordered pair, with Holm correction across the whole family.

    ``indistinguishable_groups`` lists maximal sets of methods no comparison
    separated. It is a statement about this experiment's power, not proof that
    the methods are identical, and the group sizes should be read with the
    parent count in mind.
    """
    if methods is None:
        methods = sorted({str(row["method"]) for row in rows if row.get("mode") == mode})
    if len(methods) < 2:
        raise ValueError(f"contrast_matrix needs at least two methods in mode {mode!r}")

    pairs = [paired_method_contrast(rows, a, b, mode=mode, key=key,
                                    bootstrap_resamples=bootstrap_resamples,
                                    seed=seed + index, confidence=confidence)
             for index, (a, b) in enumerate(combinations(methods, 2))]
    adjusted = _holm([pair["p_value"] if pair["p_value"] is not None else 1.0 for pair in pairs])
    for pair, value in zip(pairs, adjusted):
        pair["p_value_holm"] = float(value)
        pair["separated_after_correction"] = bool(value < alpha and pair["separated"])

    separated = {tuple(sorted((p["method_a"], p["method_b"])))
                 for p in pairs if p["separated_after_correction"]}
    groups, remaining = [], list(methods)
    while remaining:
        head = remaining.pop(0)
        group = [head] + [m for m in remaining if tuple(sorted((head, m))) not in separated]
        groups.append(sorted(group))
        remaining = [m for m in remaining if m not in group]

    ranked = sorted(methods, key=lambda m: float(np.mean(list(_parent_table(rows, m, mode, key).values()))))
    return _safe({
        "schema_version": 1, "mode": mode, "key": key, "methods": list(methods),
        "n_comparisons": len(pairs), "alpha": alpha, "correction": "holm",
        "pairs": pairs,
        "ranked_by_mean": ranked,
        "indistinguishable_groups": groups,
        "n_separated_after_correction": len(separated),
        "scope": ("a pair is called separated only after Holm correction over all "
                  f"{len(pairs)} comparisons; failing to separate is not evidence of equality"),
    })


def encoder_information_contrast(rows: Sequence[Mapping[str, Any]], *,
                                 blind: Sequence[str] = ("logical",), mode: str = "bank",
                                 key: str = "loss", bootstrap_resamples: int = 2000,
                                 seed: int = 0, confidence: float = 0.95) -> dict:
    """Embedding-aware encoders pooled against embedding-blind ones.

    This is the information question, and it is the one the G3 interventions
    already answered for simulator search. Pooling the aware encoders is
    deliberate: it asks whether *seeing the embedding* helps, and deliberately
    does not ask which aware architecture is best. That second question is
    ``contrast_matrix``'s, and the two answers must be reported separately.
    """
    present = {str(row["method"]) for row in rows if row.get("mode") == mode}
    blind_methods = sorted(present & set(blind))
    aware_methods = sorted(present - set(blind))
    if not blind_methods:
        raise ValueError(f"none of the blind encoders {tuple(blind)!r} appear in mode {mode!r}")
    if not aware_methods:
        raise ValueError(f"no embedding-aware encoders appear in mode {mode!r}")

    def pooled(names):
        tables = [_parent_table(rows, name, mode, key) for name in names]
        parents = set(tables[0])
        for table in tables[1:]:
            parents &= set(table)
        return {p: float(np.mean([t[p] for t in tables])) for p in parents}

    aware, blind_table = pooled(aware_methods), pooled(blind_methods)
    parents = sorted(set(aware) & set(blind_table))
    if len(parents) < 2:
        raise ValueError("aware and blind encoders share fewer than two parents")

    differences = np.array([aware[p] - blind_table[p] for p in parents])
    statistics = _paired_bootstrap(differences, n_resamples=bootstrap_resamples, seed=seed,
                                   confidence=confidence)
    return _safe({
        "schema_version": 1, "mode": mode, "key": key,
        "aware_methods": aware_methods, "blind_methods": blind_methods,
        "n_parents": len(parents),
        "mean_aware": float(np.mean([aware[p] for p in parents])),
        "mean_blind": float(np.mean([blind_table[p] for p in parents])),
        "mean_difference": float(differences.mean()),
        "parents_favouring_aware": int((differences < 0).sum()),
        **statistics, "confidence": confidence, "resamples": bootstrap_resamples,
        "separated": bool(statistics["ci_low"] is not None
                          and (statistics["ci_low"] > 0 or statistics["ci_high"] < 0)),
        "unit_of_independence": "logical_parent",
        "scope": ("whether embedding information helps, pooled over aware encoders; "
                  "not a claim about which aware architecture is best"),
    })
