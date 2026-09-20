"""How large is the effect, and against which reference.

A "share of achievable headroom" is a ratio, and a ratio is only as meaningful
as its denominator. This project has two denominators and they differ by more
than a factor of two:

**The bank oracle.** The best of the 64 stored candidates, scored once when the
dataset was built. It is the ceiling on the *same menu the selector chooses
from*, so a high share against it says the critic ranks its own options well.
It does not say the chosen control is near optimal.

**The frontier search.** An instance-specific Sobol-local search over five
control families under a declared budget -- 257 propagations in the published
configuration. This is the reference that answers "how much of what is findable
did we find?", and it is strictly the harder bar.

A published table once put one of each in a single column and read the
resulting gap as a device-connectivity effect. It was an artifact of the
denominators: scored against the bank oracle both topologies land within a
point of each other. Three properties of this module exist to make that
specific mistake impossible to repeat.

1. ``reference`` is required and must name one of the two. There is no default.
2. The reference's per-instance cost is **read from the artifacts** -- from
   ``total_objective_calls`` on the frontier rows, from ``candidate_count`` on
   the evaluation rows. A 97-call sweep cannot describe itself as 257-call,
   and a sweep of mixed budget refuses to carry a single label at all.
3. A frontier reference must cover every parent it is the denominator for.
   Using validation-parent headroom against test-parent gain is the error this
   catches; a frontier over a superset is fine, and the surplus is dropped
   rather than averaged in.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .headroom import _bootstrap, _describe, _parent_means
from .telemetry import _safe

REFERENCES = {
    "bank_oracle": ("best of the stored candidate bank -- the ceiling on the same menu "
                    "the selector chooses from, scored once at dataset build"),
    "frontier": ("best found by an instance-specific Sobol-local search over the control "
                 "families, under a declared budget; not a global control optimum"),
}


def _single(values: set, what: str, where: str) -> Any:
    if not values:
        raise ValueError(f"{where} carry no {what}")
    if len(values) > 1:
        raise ValueError(
            f"{where} mix {len(values)} {what} values ({sorted(values)}); a single budget "
            f"label would misdescribe them. Split the report by budget.")
    return values.pop()


def effect_size_report(evaluation_rows: Sequence[Mapping[str, Any]], *, reference: str,
                       frontier_rows: Sequence[Mapping[str, Any]] | None = None,
                       bootstrap_resamples: int = 10000, seed: int = 0) -> dict:
    """Gain, effect size, win rate and share of headroom against a named reference.

    ``evaluation_rows`` are the per-record rows of a completed evaluation.
    ``frontier_rows`` are the per-record rows of a control-sweep, required for
    and only for ``reference="frontier"``.
    """
    rows = list(evaluation_rows)
    if not rows:
        raise ValueError("effect_size_report requires a nonempty set of evaluation records")
    if reference not in REFERENCES:
        raise ValueError(
            f"reference must be named, one of {sorted(REFERENCES)}; got {reference!r}. "
            "A share of headroom has no meaning without the denominator it is a share of.")
    if reference == "bank_oracle" and frontier_rows is not None:
        raise ValueError(
            "reference='bank_oracle' does not read frontier_rows; passing them would "
            "silently ignore the harder reference. Pass reference='frontier' to use them.")
    if reference == "frontier" and not frontier_rows:
        raise ValueError("reference='frontier' requires frontier_rows from a control-sweep")

    gain, parents = _parent_means(rows, lambda r: float(r["linear_loss"]) - float(r["selected_loss"]))
    if not parents:
        raise ValueError("no records carried both a linear and a selected loss")

    if reference == "bank_oracle":
        headroom, headroom_parents = _parent_means(
            rows, lambda r: float(r["linear_loss"]) - float(r["bank_best_loss"]))
        cost = int(_single({int(r["candidate_count"]) for r in rows},
                           "candidate_count", "evaluation rows"))
        reference_block = {"name": reference, "objective_calls_per_instance": cost,
                           "description": REFERENCES[reference],
                           "cost_read_from": "evaluation rows' candidate_count"}
    else:
        wanted = set(parents)
        kept = [r for r in frontier_rows if str(r["parent_id"]) in wanted]
        covered = {str(r["parent_id"]) for r in kept}
        missing = sorted(wanted - covered)
        if missing:
            raise ValueError(
                f"the frontier reference does not cover {len(missing)} of {len(wanted)} "
                f"evaluated parents (missing {missing[:5]}...). Headroom measured on other "
                "parents is not a denominator for these ones.")
        headroom, headroom_parents = _parent_means(kept, lambda r: float(r["headroom"]))
        cost = int(_single({int(r["total_objective_calls"]) for r in kept},
                           "total_objective_calls", "frontier rows"))
        families = sorted({f for r in kept for f in r.get("evaluated_families", ())})
        reference_block = {"name": reference, "objective_calls_per_instance": cost,
                           "description": REFERENCES[reference],
                           "cost_read_from": "frontier rows' total_objective_calls",
                           "evaluated_families": families}

    if headroom_parents != parents:
        raise ValueError("gain and headroom resolved different parent sets; refusing to divide")

    gain_block = _describe(gain)
    gain_block["parent_bootstrap_ci"] = _bootstrap(gain, n_resamples=bootstrap_resamples, seed=seed)
    gain_block["parents_won"] = int((gain > 0).sum())
    spread = float(gain.std(ddof=1)) if gain.size > 1 else 0.0
    gain_block["cohens_d"] = float(gain.mean() / spread) if spread > 0 else None

    headroom_block = _describe(headroom)
    headroom_block["parent_bootstrap_ci"] = _bootstrap(headroom, n_resamples=bootstrap_resamples,
                                                       seed=seed)

    mean_headroom = float(headroom.mean())
    if mean_headroom > 0:
        share, status = float(gain.mean() / mean_headroom), "ok"
    else:
        share = None
        status = ("mean headroom is not positive, so the reference found nothing to take a "
                  "share of; no percentage is reported")

    return _safe({
        "schema_version": 1,
        "reference": reference_block,
        "n_parents": len(parents),
        "n_records": len(rows),
        "gain_vs_linear": gain_block,
        "parent_gains": {parent: float(value) for parent, value in zip(parents, gain)},
        "headroom": headroom_block,
        "share_of_headroom": share,
        "share_status": status,
        "scope": ("parent-level means throughout; the logical parent is the unit of "
                  "independence. The share is meaningful only against the named reference "
                  "and must never be compared across references."),
    })


def pooled_across_seeds(reports: Sequence[Mapping[str, Any]]) -> dict:
    """Pool per-parent gains over training seeds, and say what the worst seed did.

    The published win rates and effect sizes are pooled: each parent's gain is
    averaged over training seeds first, then the statistics are taken. That is
    the right summary of what the method does in expectation, but on its own it
    can read as stronger than any single training run -- a parent whose gain is
    positive on average may be lost by one seed. Both are reported here so the
    headline can name which one it is quoting.
    """
    reports = list(reports)
    if not reports:
        raise ValueError("pooled_across_seeds requires at least one report")

    names = {r["reference"]["name"] for r in reports}
    if len(names) > 1:
        raise ValueError(
            f"cannot pool reports built on different references ({sorted(names)}); "
            "shares and headrooms against different denominators are not comparable")
    costs = {r["reference"]["objective_calls_per_instance"] for r in reports}
    if len(costs) > 1:
        raise ValueError(
            f"cannot pool reports whose reference cost differs ({sorted(costs)})")

    parent_sets = {frozenset(r["parent_gains"]) for r in reports}
    if len(parent_sets) > 1:
        raise ValueError("cannot pool reports over different parent sets")

    # Headroom is a property of the records and the reference, never of the training
    # seed. If it moves between reports they are not seeds of one evaluation, and
    # pooling them would silently average two different denominators -- the same
    # class of mistake this module exists to prevent.
    headrooms = {round(float(r["headroom"]["mean"]), 12) for r in reports}
    if len(headrooms) > 1:
        raise ValueError(
            f"reports disagree on headroom ({sorted(headrooms)}); headroom cannot depend "
            "on the training seed, so these are not seeds of one evaluation")

    parents = sorted(next(iter(parent_sets)))
    matrix = np.array([[float(r["parent_gains"][p]) for p in parents] for r in reports])
    pooled = matrix.mean(axis=0)
    spread = float(pooled.std(ddof=1)) if pooled.size > 1 else 0.0
    per_seed_won = [int((row > 0).sum()) for row in matrix]

    # The returned shape is deliberately decomposable: it carries the same
    # reference / parent_gains / headroom / gain_vs_linear keys an individual
    # report does, so decompose_share accepts a pooled result unchanged and the
    # factorisation is taken on seed-averaged gains rather than on one seed.
    mean_headroom = float(reports[0]["headroom"]["mean"])
    return _safe({
        "schema_version": 2,
        "reference": reports[0]["reference"],
        "n_seeds": len(reports),
        "n_parents": len(parents),
        "mean_gain": float(pooled.mean()),
        "gain_vs_linear": {"mean": float(pooled.mean()),
                           "cohens_d": float(pooled.mean() / spread) if spread > 0 else None,
                           "parents_won": int((pooled > 0).sum()),
                           "n_parents": len(parents)},
        "parent_gains": {p: float(v) for p, v in zip(parents, pooled)},
        "headroom": {"mean": mean_headroom, "n_parents": len(parents)},
        "cohens_d": float(pooled.mean() / spread) if spread > 0 else None,
        "parents_won": int((pooled > 0).sum()),
        "per_seed_parents_won": per_seed_won,
        "worst_seed_parents_won": min(per_seed_won),
        "scope": ("gains are averaged over training seeds before the statistics are taken; "
                  "worst_seed_parents_won is what the least favourable single training run "
                  "achieved and is the number to quote when a single run is what ships"),
    })


def decompose_share(bank_report: Mapping[str, Any], frontier_report: Mapping[str, Any]) -> dict:
    """Split the share of findable improvement into the critic and the menu.

    The two references are not rivals; they compose. Writing ``g`` for the
    learned gain, ``H_bank`` for the bank oracle's headroom and ``H_front`` for
    the frontier search's::

        g / H_front  =  (g / H_bank)  x  (H_bank / H_front)
        share of     =  selector      x  bank
        findable        efficiency       coverage

    The left factor asks whether the critic picks well from the menu it has.
    The right asks whether the menu is worth picking from. A table that reports
    only the first can look excellent while the menu throws away most of what
    is findable -- which is precisely how a bank-oracle share of 83% was once
    read as a near-optimal control.

    They point at different work. A low ``selector_efficiency`` is a modelling
    problem; a low ``bank_coverage`` is a dataset problem, and no encoder can
    fix it.
    """
    reports = {r["reference"]["name"]: r for r in (bank_report, frontier_report)}
    if set(reports) != {"bank_oracle", "frontier"}:
        raise ValueError(
            "decompose_share needs one of each reference, one bank_oracle and one "
            f"frontier; got {sorted(r['reference']['name'] for r in (bank_report, frontier_report))}")
    bank, frontier = reports["bank_oracle"], reports["frontier"]

    if set(bank["parent_gains"]) != set(frontier["parent_gains"]):
        raise ValueError("the two reports cover different parents; they must be the "
                         "same evaluation scored against two references")
    gains = [(bank["parent_gains"][p], frontier["parent_gains"][p]) for p in bank["parent_gains"]]
    if any(abs(a - b) > 1e-12 for a, b in gains):
        raise ValueError("the two reports disagree on the learned gain; they must be the "
                         "same evaluation scored against two references")

    h_bank = float(bank["headroom"]["mean"])
    h_front = float(frontier["headroom"]["mean"])
    if h_bank <= 0 or h_front <= 0:
        raise ValueError("both references must find positive headroom to be decomposed")
    if h_bank > h_front:
        raise ValueError(
            f"bank headroom {h_bank:.5f} exceeds frontier headroom {h_front:.5f}. The search "
            "scored the bank's own families, so it cannot find less; check that both "
            "references cover the same records.")

    gain = float(bank["gain_vs_linear"]["mean"])
    selector = gain / h_bank
    coverage = h_bank / h_front
    return _safe({
        "schema_version": 1,
        "n_parents": bank["n_parents"],
        "bank_calls_per_instance": bank["reference"]["objective_calls_per_instance"],
        "frontier_calls_per_instance": frontier["reference"]["objective_calls_per_instance"],
        "learned_gain": gain,
        "bank_headroom": h_bank,
        "frontier_headroom": h_front,
        "selector_efficiency": selector,
        "bank_coverage": coverage,
        "share_of_findable": gain / h_front,
        "binding_factor": "selector_efficiency" if selector < coverage else "bank_coverage",
        "scope": ("selector_efficiency is a modelling result and bank_coverage is a dataset "
                  "result; the binding factor names which one limits the share, not which "
                  "one is cheaper to improve"),
    })


def search_call_equivalent(curves: Mapping[str, Mapping[int, float]],
                           gains: Mapping[str, float], *, censor_at: int,
                           bootstrap_resamples: int = 10000, seed: int = 0) -> dict:
    """How many instance-specific simulator calls does one forward pass buy?

    A share of headroom is a ratio against a budget someone chose, and the
    budget is arbitrary: this project's frontier ran 257 calls, and at 129 it
    had already found 96.7% of that. Expressing the same result as a call count
    removes the arbitrary denominator -- "one forward pass is worth about N
    calls" needs no reference budget to be meaningful, and a reader can price
    it directly.

    ``curves`` maps each parent to its incumbent headroom by budget; the budget
    is the total objective calls, so it already accounts for how the search
    split its budget across families. The equivalent is the **smallest budget
    whose headroom reaches the gain**, which is an upper bound: the search may
    have passed the gain between two grid points. ``per_parent_lower_bound``
    gives the last budget still short, so the pair brackets the truth.

    A parent whose gain the search never reaches is **censored**, never
    extrapolated. Censoring here means the learned selector beat the search at
    its full budget on that parent, which is a result, not a missing value.
    """
    if set(curves) != set(gains):
        missing = sorted(set(gains) ^ set(curves))
        raise ValueError(f"curves and gains must cover the same parents; differ on {missing}")
    if not curves:
        raise ValueError("search_call_equivalent requires at least one parent")

    reached: dict[str, int | None] = {}
    lower: dict[str, int | None] = {}
    for parent, curve in curves.items():
        budgets = sorted(curve)
        values = [float(curve[b]) for b in budgets]
        if any(b - a < -1e-12 for a, b in zip(values, values[1:])):
            raise ValueError(
                f"the headroom curve for {parent} is not monotone; an incumbent trace "
                "cannot get worse, so this is not one")
        target = float(gains[parent])
        hit, last_short = None, None
        for budget, value in zip(budgets, values):
            if value >= target:
                hit = int(budget)
                break
            last_short = int(budget)
        reached[parent] = hit
        lower[parent] = last_short

    resolved = np.array([v for v in reached.values() if v is not None], dtype=float)
    censored = [p for p, v in reached.items() if v is None]

    block = _describe(resolved)
    block["parent_bootstrap_ci"] = _bootstrap(resolved, n_resamples=bootstrap_resamples, seed=seed)

    return _safe({
        "schema_version": 1,
        "n_parents": len(curves),
        "n_resolved": int(resolved.size),
        "n_censored": len(censored),
        "censored_parents": sorted(censored),
        "median_calls": int(np.median(resolved)) if resolved.size else None,
        "mean_calls": float(resolved.mean()) if resolved.size else None,
        "per_parent_calls": reached,
        "per_parent_lower_bound": lower,
        "distribution": block,
        "parent_bootstrap_ci": block["parent_bootstrap_ci"],
        "censor_at": int(censor_at),
        "censoring_note": (
            f"{len(censored)} of {len(curves)} parents are censored: the search never "
            f"reached the learned gain within {censor_at} calls, so the equivalent is "
            "reported as greater than the budget rather than extrapolated"),
        "scope": ("the equivalent is the smallest grid budget reaching the gain, an upper "
                  "bound; per_parent_lower_bound is the last budget still short. It prices "
                  "the online cost only -- training and data generation are offline and "
                  "accounted separately in costs.py"),
    })
