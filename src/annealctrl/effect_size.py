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

    parents = sorted(next(iter(parent_sets)))
    matrix = np.array([[float(r["parent_gains"][p]) for p in parents] for r in reports])
    pooled = matrix.mean(axis=0)
    spread = float(pooled.std(ddof=1)) if pooled.size > 1 else 0.0
    per_seed_won = [int((row > 0).sum()) for row in matrix]

    return _safe({
        "schema_version": 1,
        "reference": reports[0]["reference"],
        "n_seeds": len(reports),
        "n_parents": len(parents),
        "mean_gain": float(pooled.mean()),
        "cohens_d": float(pooled.mean() / spread) if spread > 0 else None,
        "parents_won": int((pooled > 0).sum()),
        "per_seed_parents_won": per_seed_won,
        "worst_seed_parents_won": min(per_seed_won),
        "scope": ("gains are averaged over training seeds before the statistics are taken; "
                  "worst_seed_parents_won is what the least favourable single training run "
                  "achieved and is the number to quote when a single run is what ships"),
    })
