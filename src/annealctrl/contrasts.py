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
        if not np.isfinite(float(value)):
            raise ValueError(f"{key!r} must contain finite values")
        buckets.setdefault(str(row["parent_id"]), []).append(float(value))
    if not buckets:
        raise ValueError(f"no rows for method {method!r} in mode {mode!r}")
    return {parent: float(np.mean(values)) for parent, values in buckets.items()}


def _paired_bootstrap(differences: np.ndarray, *, n_resamples: int, seed: int,
                      confidence: float = 0.95) -> dict:
    """Percentile CI and centered-null bootstrap test of a mean difference.

    The test resamples differences shifted to have mean zero and compares the
    absolute null mean to the observed absolute mean. This is an approximate
    test under independent, representative parent sampling, not an exact
    randomization test. The percentile interval and test are distinct summaries.
    """
    _validate_bootstrap(n_resamples, confidence)
    differences = np.asarray(differences, dtype=float)
    if differences.ndim != 1 or not np.isfinite(differences).all():
        raise ValueError("differences must be a finite one-dimensional array")
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
    observed = float(differences.mean())
    # Centering the resampled means is equivalent to resampling the centered
    # parent differences. The +1 correction prevents a zero Monte Carlo p-value.
    tail = int((np.abs(means - observed) >= abs(observed)).sum())
    p_value = (tail + 1) / (n_resamples + 1)
    return {"ci_low": float(low), "ci_high": float(high), "p_value": float(p_value),
            "p_value_method": "centered_null_bootstrap_absolute_mean",
            "p_value_scope": "approximate; conditional on observed training seeds",
            "status": "ok"}


def _validate_bootstrap(n_resamples: int, confidence: float) -> None:
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, (int, np.integer)) \
            or n_resamples < 1:
        raise ValueError("bootstrap_resamples must be a positive integer")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between zero and one")


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
        "schema_version": 2,
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
        "difference_definition": "method_b minus method_a; negative favours method_b for loss",
        "correction": "none",
        "scope": ("paired on the parent, conditional on observed training seeds; "
                  "CI is pointwise and uncorrected for multiplicity; see contrast_matrix "
                  "for the family-corrected decision; failure to separate is not equivalence"),
    })


def paired_parent_seed_contrast(rows: Sequence[Mapping[str, Any]], method_a: str,
                                method_b: str, *, mode: str = "direct", key: str = "loss",
                                seed_key: str = "seed", bootstrap_resamples: int = 2000,
                                seed: int = 0, confidence: float = 0.95) -> dict:
    """Descriptive crossed parent/seed bootstrap of ``method_b - method_a``.

    Both methods must cover the same complete parent-by-training-seed panel,
    with matching record IDs (when present) within each cell. Interventions are
    averaged within a cell, then parents and seeds receive equal weight. Each
    bootstrap draw independently samples parent and seed indices, preserving
    all dependence along both axes. This is the two-factor pigeonhole bootstrap
    (Owen, 2007), not a nested seed-within-parent bootstrap. Its percentile CI is
    an approximate sensitivity diagnostic; a few seeds cannot establish precise
    uncertainty over optimization randomness. No p-value or equivalence claim
    is attached to this diagnostic.
    """
    _validate_bootstrap(bootstrap_resamples, confidence)

    def table(method):
        cells: dict[tuple[str, str], list[float]] = {}
        identities: dict[tuple[str, str], list[str]] = {}
        for row in rows:
            if row.get("method") != method or row.get("mode") != mode:
                continue
            if seed_key not in row or row.get(key) is None:
                raise ValueError(f"each selected row needs {seed_key!r} and {key!r}")
            value = float(row[key])
            if not np.isfinite(value):
                raise ValueError(f"{key!r} must contain finite values")
            cell = (str(row["parent_id"]), str(row[seed_key]))
            cells.setdefault(cell, []).append(value)
            if "record_id" in row:
                identities.setdefault(cell, []).append(str(row["record_id"]))
        if not cells:
            raise ValueError(f"no rows for method {method!r} in mode {mode!r}")
        return cells, identities

    a, ids_a = table(method_a)
    b, ids_b = table(method_b)
    if set(a) != set(b):
        raise ValueError("methods must cover the same parent-by-seed cells")
    parents = sorted({p for p, _ in a})
    seeds = sorted({s for _, s in a})
    if len(a) != len(parents) * len(seeds):
        raise ValueError("each method must cover a complete parent-by-seed panel")
    if any(len(a[cell]) != len(b[cell]) for cell in a):
        raise ValueError("methods must cover matching records within each parent-by-seed cell")
    if (ids_a or ids_b) and (set(ids_a) != set(a) or set(ids_b) != set(b)
                            or any(sorted(ids_a[c]) != sorted(ids_b[c]) for c in a)):
        raise ValueError("methods must have matching record IDs within each parent-by-seed cell")
    differences = np.array([[np.mean(b[p, s]) - np.mean(a[p, s]) for s in seeds]
                            for p in parents])
    result = {
        "schema_version": 1, "method_a": method_a, "method_b": method_b,
        "mode": mode, "key": key, "n_parents": len(parents), "n_seeds": len(seeds),
        "mean_difference": float(differences.mean()),
        "per_seed_mean_difference": {s: float(differences[:, i].mean())
                                     for i, s in enumerate(seeds)},
        "ci_low": None, "ci_high": None, "confidence": confidence,
        "resamples": bootstrap_resamples, "correction": "none", "p_value": None,
        "unit_of_resampling": "crossed_logical_parent_and_training_seed",
        "difference_definition": "method_b minus method_a; negative favours method_b for loss",
        "reference": "https://arxiv.org/abs/0712.1111",
        "scope": ("descriptive crossed-factor percentile interval; parents and training seeds "
                  "resampled independently; conditional on the fixed training dataset and recipe; "
                  "not a multiplicity-adjusted test or evidence of equivalence; uncertainty with "
                  "few seeds is poorly resolved"),
    }
    if len(parents) < 2 or len(seeds) < 2:
        return _safe({**result, "status": "insufficient_parents_or_seeds"})
    rng = np.random.default_rng(seed)
    means = np.empty(bootstrap_resamples)
    for index in range(bootstrap_resamples):
        parent_draw = rng.integers(len(parents), size=len(parents))
        seed_draw = rng.integers(len(seeds), size=len(seeds))
        means[index] = differences[np.ix_(parent_draw, seed_draw)].mean()
    tail = (1 - confidence) / 2
    low, high = np.quantile(means, [tail, 1 - tail])
    return _safe({**result, "ci_low": float(low), "ci_high": float(high), "status": "ok"})


def _holm(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values, kept monotone in the sorted order."""
    order = np.argsort(p_values)
    adjusted, running, total = [0.0] * len(p_values), 0.0, len(p_values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (total - rank) * float(p_values[index])))
        adjusted[index] = running
    return adjusted


def _nonseparated_maximal_sets(methods: Sequence[str],
                              separated: set[tuple[str, str]]) -> list[list[str]]:
    """Maximal cliques of pairwise non-rejection; these may overlap.

    Non-rejection is not transitive: a~b and a~c cannot imply b~c. A greedy
    partition can therefore contain a rejected pair. Bron--Kerbosch enumerates
    actual maximal cliques instead (the encoder comparison family is small).
    """
    neighbours = {a: {b for b in methods if b != a and tuple(sorted((a, b))) not in separated}
                  for a in methods}
    groups: list[list[str]] = []

    def visit(current: set[str], possible: set[str], excluded: set[str]) -> None:
        if not possible and not excluded:
            groups.append(sorted(current))
            return
        for node in sorted(possible):
            visit(current | {node}, possible & neighbours[node], excluded & neighbours[node])
            possible.remove(node)
            excluded.add(node)

    visit(set(), set(methods), set())
    return sorted(groups)


def contrast_matrix(rows: Sequence[Mapping[str, Any]], *, mode: str = "bank", key: str = "loss",
                    methods: Sequence[str] | None = None, bootstrap_resamples: int = 2000,
                    seed: int = 0, confidence: float = 0.95, alpha: float = 0.05) -> dict:
    """Every unordered pair, with Holm correction across the whole family.

    ``nonseparated_maximal_sets`` lists possibly overlapping maximal sets whose
    every pair fails to separate. This is not evidence of equality/equivalence.
    ``indistinguishable_groups`` remains only as a deprecated compatibility alias.
    """
    if methods is None:
        methods = sorted({str(row["method"]) for row in rows if row.get("mode") == mode})
    if len(methods) < 2:
        raise ValueError(f"contrast_matrix needs at least two methods in mode {mode!r}")
    if len(set(methods)) != len(methods):
        raise ValueError("methods must be unique")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")

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
    groups = _nonseparated_maximal_sets(methods, separated)

    ranked = sorted(methods, key=lambda m: float(np.mean(list(_parent_table(rows, m, mode, key).values()))))
    return _safe({
        "schema_version": 2, "mode": mode, "key": key, "methods": list(methods),
        "n_comparisons": len(pairs), "alpha": alpha, "correction": "holm",
        "pairs": pairs,
        "ranked_by_mean": ranked,
        "nonseparated_maximal_sets": groups,
        "indistinguishable_groups": groups,
        "deprecated_fields": {"indistinguishable_groups": "use nonseparated_maximal_sets; not equivalence"},
        "n_separated_after_correction": len(separated),
        "scope": ("a pair is called separated only after Holm correction of approximate "
                  f"centered-null bootstrap p-values over these {len(pairs)} comparisons "
                  "and a pointwise CI excluding zero; CIs are not simultaneous; conditional "
                  "on observed training seeds; failing to separate is not evidence of equality; "
                  "the pooled embedding-information contrast is outside this correction family"),
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

    tables = {name: _parent_table(rows, name, mode, key)
              for name in aware_methods + blind_methods}
    expected_parents = set(tables[aware_methods[0]])
    if any(set(table) != expected_parents for table in tables.values()):
        raise ValueError("all aware and blind encoders must cover the same parents")

    def pooled(names):
        return {p: float(np.mean([tables[name][p] for name in names])) for p in expected_parents}

    aware, blind_table = pooled(aware_methods), pooled(blind_methods)
    parents = sorted(set(aware) & set(blind_table))
    if len(parents) < 2:
        raise ValueError("aware and blind encoders share fewer than two parents")

    differences = np.array([aware[p] - blind_table[p] for p in parents])
    statistics = _paired_bootstrap(differences, n_resamples=bootstrap_resamples, seed=seed,
                                   confidence=confidence)
    return _safe({
        "schema_version": 2, "mode": mode, "key": key,
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
        "correction": "none", "inference_role": "exploratory_pooled_information_effect",
        "difference_definition": "mean_aware minus mean_blind; negative favours aware for loss",
        "scope": ("exploratory pooled embedding-information contrast, unadjusted for "
                  "multiplicity and NOT part of the pairwise Holm family; conditional on "
                  "observed training seeds; not a claim about architecture or equivalence"),
    })
