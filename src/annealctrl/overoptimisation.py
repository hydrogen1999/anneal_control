"""Does searching longer against a noiseless simulator make the result worse?

The erosion measurement says a control's closed-system quality predicts how
much an environment takes back from it. If that is a real property of
optimisation and not a curiosity about a bank, it has a consequence a
practitioner can act on: **a search that runs longer against a noiseless
simulator should, past some budget, return a control that performs worse once
the environment is present.**

That is a curve, not a correlation, and it either turns up or it does not.

The construction is deliberately plain. A search's incumbent at budget b is the
best candidate it had seen after b true evaluations, which is monotone in the
closed system by definition. Evaluating those same incumbents under a Lindblad
channel gives a second curve over the same budgets, and the two are directly
comparable because they are the same controls.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .telemetry import _safe


def incumbent_at_budgets(losses: Sequence[float], budgets: Sequence[int]) -> list[int]:
    """Index of the best candidate seen within each budget.

    ``losses`` is in evaluation order. A budget larger than the trace is
    refused rather than silently clamped, because reporting budget 64 from a
    48-evaluation trace would make the curve flat where it was simply absent.
    """
    losses = np.asarray(losses, dtype=float)
    if losses.ndim != 1 or not losses.size:
        raise ValueError("losses must be a nonempty 1-D sequence in evaluation order")
    if not np.isfinite(losses).all():
        raise ValueError("losses must be finite")
    out = []
    for budget in budgets:
        if isinstance(budget, bool) or not isinstance(budget, (int, np.integer)) or budget < 1:
            raise ValueError("budgets must be positive integers")
        if budget > losses.size:
            raise ValueError(f"budget {budget} exceeds the {losses.size}-evaluation trace")
        out.append(int(np.argmin(losses[:budget])))
    return out


def overoptimisation_curve(rows: Sequence[Mapping[str, Any]], *, budgets: Sequence[int],
                           bootstrap_resamples: int = 4000, seed: int = 0) -> dict:
    """Parent-level noiseless and noisy loss against search budget.

    ``rows`` carry ``parent_id`` and, per budget, ``closed`` and ``open``
    losses of that budget's incumbent. The noiseless curve is monotone by
    construction; whether the noisy one is, is the question.
    """
    from .headroom import _bootstrap, _describe, _parent_means

    rows = list(rows)
    if not rows:
        raise ValueError("overoptimisation_curve requires a nonempty row set")
    budgets = [int(b) for b in budgets]

    by_budget = {}
    for budget in budgets:
        block = {}
        for kind in ("closed", "open"):
            values, parents = _parent_means(
                rows, lambda r, b=budget, k=kind: r[k][str(b)])
            described = _describe(values)
            described["parent_bootstrap_ci"] = _bootstrap(
                values, n_resamples=bootstrap_resamples, seed=seed)
            block[kind] = described
            block["n_parents"] = len(parents)
        by_budget[str(budget)] = block

    closed = [by_budget[str(b)]["closed"]["mean"] for b in budgets]
    open_ = [by_budget[str(b)]["open"]["mean"] for b in budgets]
    best_open = int(np.argmin(open_))
    # Paired contrast between the budget that is best under noise and the
    # largest budget: the whole claim in one interval.
    largest = len(budgets) - 1
    paired, _ = _parent_means(
        rows, lambda r: r["open"][str(budgets[largest])] - r["open"][str(budgets[best_open])])
    paired_ci = _bootstrap(paired, n_resamples=bootstrap_resamples, seed=seed)

    return _safe({
        "schema_version": 1,
        "n_records": len(rows),
        "n_parents": len({str(r["parent_id"]) for r in rows}),
        "budgets": budgets,
        "by_budget": by_budget,
        "closed_curve": closed,
        "open_curve": open_,
        "budget_minimising_open_loss": budgets[best_open],
        "open_loss_is_monotone_in_budget": bool(
            all(a >= b - 1e-12 for a, b in zip(open_, open_[1:]))),
        "cost_of_the_largest_budget_against_the_best": {
            "mean": float(paired.mean()),
            "parent_bootstrap_ci": paired_ci,
            "worse_in_parents": int((paired > 0).sum()),
            "n_parents": int(paired.size)},
        "scope": ("incumbents of one search trace per record, re-evaluated under a "
                  "declared dephasing rate; the closed curve is monotone by construction "
                  "and only the open curve carries information"),
    })
