"""Is the local-adiabatic schedule the thing a learned method should approximate?

``gap_inverse_square`` is the textbook answer to how one should anneal: move
slowly where the instantaneous gap is small, ds/dt proportional to the square of
that gap. Computing it needs the exact spectrum at every point along the path,
which is exponentially expensive and which no deployed method has. The usual
framing of learned control is therefore "a cheap approximation to the spectral
schedule", and under that framing the spectral schedule is an upper bound the
learner chases.

That framing is an assumption, and this module measures it instead. Three
outcomes are possible and the instrument reports whichever occurs: the oracle
beats the search, so learning is approximation; the oracle beats linear but
loses to search, so it is a weak reference; or the oracle loses to linear, in
which case it is not a target at all and a paper that treats it as one is
motivated wrongly.

Two things make this measurement easy to get wrong, and both are handled
explicitly rather than by filtering.

**Resolution.** The spectral teacher returns no schedule when the first gap is
degenerate or unresolved at some grid point. Those instances are not missing at
random -- a vanishing gap is exactly what makes an instance hard -- so dropping
them silently would compare the oracle against search on the subset where the
oracle happened to work. The resolved and unresolved populations are described
side by side so the conditioning is visible.

**Audits.** An interpolation audit failure means the waveform was built from a
grid that did not reproduce held-out spectral points. Such a waveform is still
evaluated, and its loss is still a real simulator outcome, but it is not the
schedule the method specifies. Headline numbers use audited waveforms only, and
the pooled number is reported beside them with a flag when including the
failures would change the conclusion.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np

from .headroom import _bootstrap, _describe, _parent_means
from .telemetry import _safe

AUDIT_PASSED = "sampled_point_audit_passed"
AUDIT_FAILED = "interpolation_audit_failed"


def _paired(rows: Sequence[Mapping[str, Any]], teacher_key: str, reference: str, *,
            bootstrap_resamples: int, seed: int) -> dict:
    """Teacher minus reference, paired on the parent. Positive means worse."""
    differences = [{"parent_id": row["parent_id"],
                    "d": float(row["privileged_teachers"][teacher_key]["loss"]) - float(row[reference])}
                   for row in rows]
    values, parents = _parent_means(differences, lambda row: row["d"])
    block = _describe(values)
    block["mean_difference"] = block.pop("mean")
    block["parent_bootstrap_ci"] = _bootstrap(values, n_resamples=bootstrap_resamples, seed=seed)
    block["n_parents"] = len(parents)
    block["reference"] = reference
    block["sign_convention"] = "positive means the privileged teacher is worse"
    return block


def _method_block(rows: Sequence[Mapping[str, Any]], method: str, *, bootstrap_resamples: int,
                  seed: int, stratify: bool = True) -> dict:
    present = [row for row in rows if method in row.get("privileged_teachers", {})]
    entries = [(row, row["privileged_teachers"][method]) for row in present]
    statuses = Counter(str(entry.get("status")) for _, entry in entries)
    resolved = [row for row, entry in entries if entry.get("loss") is not None]
    passed = [row for row, entry in entries
              if entry.get("loss") is not None and entry.get("status") == AUDIT_PASSED]
    failed = [row for row, entry in entries
              if entry.get("loss") is not None and entry.get("status") == AUDIT_FAILED]
    unresolved = [row for row, entry in entries if entry.get("loss") is None]

    def mean_loss(subset):
        return (float(np.mean([float(row["privileged_teachers"][method]["loss"]) for row in subset]))
                if subset else None)

    def mean_of(subset, key):
        return float(np.mean([float(row[key]) for row in subset])) if subset else None

    block = {
        "method": method,
        "n_rows": len(present),
        "n_resolved": len(resolved),
        "resolution_rate": len(resolved) / len(present) if present else 0.0,
        "n_audit_passed": len(passed),
        "n_audit_failed": len(failed),
        "status_counts": dict(sorted(statuses.items())),
        # Instances the teacher could not schedule are not missing at random.
        "comparison_population_is_conditional": len(resolved) < len(present),
        "resolved_mean_linear_loss": mean_of(resolved, "linear_loss"),
        "unresolved_mean_linear_loss": mean_of(unresolved, "linear_loss"),
        "teacher_seconds_total": float(sum(float(entry.get("teacher_seconds", 0.0))
                                           for _, entry in entries)),
    }
    if not passed:
        block.update({"mean_loss": None, "status": "no_audited_waveforms"})
        return _safe(block)

    block["mean_loss"] = mean_loss(passed)
    block["mean_loss_including_audit_failures"] = mean_loss(passed + failed)
    block["mean_linear_loss"] = mean_of(passed, "linear_loss")
    block["mean_best_found_loss"] = mean_of(passed, "best_found_loss")
    block["vs_linear"] = _paired(passed, method, "linear_loss",
                                 bootstrap_resamples=bootstrap_resamples, seed=seed)
    block["vs_best_found"] = _paired(passed, method, "best_found_loss",
                                     bootstrap_resamples=bootstrap_resamples, seed=seed)
    block["beats_linear_fraction"] = sum(
        1 for row in passed
        if float(row["privileged_teachers"][method]["loss"]) < float(row["linear_loss"])) / len(passed)
    block["beats_best_found_fraction"] = sum(
        1 for row in passed
        if float(row["privileged_teachers"][method]["loss"]) < float(row["best_found_loss"])) / len(passed)

    pooled = mean_loss(passed + failed)
    headline_loses = block["vs_linear"]["mean_difference"] > 0
    block["audit_failures_change_the_sign"] = bool(
        failed and headline_loses != (pooled > block["mean_linear_loss"]))

    if block["vs_best_found"]["mean_difference"] < 0:
        block["verdict"] = "beats_search"
    elif block["vs_linear"]["mean_difference"] < 0:
        block["verdict"] = "beats_linear_loses_to_search"
    else:
        block["verdict"] = "loses_to_linear"

    if stratify:
        runtimes = sorted({float(row["runtime"]) for row in passed})
        block["by_runtime"] = {
            str(runtime): _method_block([row for row in rows
                                           if float(row.get("runtime", float("nan"))) == runtime],
                                          method, bootstrap_resamples=bootstrap_resamples,
                                          seed=seed, stratify=False)
            for runtime in runtimes}
    return _safe(block)


def aggregate_privileged_teachers(rows: Sequence[Mapping[str, Any]], *,
                                  bootstrap_resamples: int = 2000, seed: int = 0) -> dict:
    """Every privileged spectral baseline against linear and against the search.

    These baselines are excluded from any equal-budget claim by construction:
    they consume exact spectral evaluations that the compared methods never get.
    That is the point -- they are an upper reference, and the question is whether
    the upper reference is actually up.
    """
    rows = list(rows)
    if not rows:
        raise ValueError("aggregate_privileged_teachers requires a nonempty row set")
    methods = sorted({method for row in rows for method in row.get("privileged_teachers", {})})
    if not methods:
        raise ValueError("no row carries privileged_teachers; nothing to aggregate")

    blocks = {method: _method_block(rows, method, bootstrap_resamples=bootstrap_resamples, seed=seed)
              for method in methods}
    beaten = [method for method, block in blocks.items()
              if block.get("verdict") == "loses_to_linear"]
    return _safe({
        "schema_version": 1,
        "n_rows": len(rows),
        "n_parents": len({str(row["parent_id"]) for row in rows}),
        "methods": blocks,
        "methods_losing_to_linear": beaten,
        "scope": ("privileged per-instance spectral baselines, excluded from every equal-budget "
                  "claim; losses are true simulator outcomes of the waveform each rule specifies, "
                  "and the resolved subset is conditional on the teacher resolving at all"),
    })
