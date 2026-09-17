"""One table, every method on the same held-out records, each with its cost class.

The temptation is a single column of losses sorted best-first. That column would
put a linear ramp, a spectral oracle whose construction is exponential in the
number of qubits, an amortised network that consults no outcome, and a search
that spends 257 true simulator evaluations per instance into one ordering, and
whichever row came first would read as "the best method". They are not competing
for the same prize.

So every row carries what it had to consume:

``fixed``
    One waveform for every instance. No per-instance work at all.
``privileged_spectrum``
    Needs the exact instantaneous spectrum along the path. Exponential, and
    unavailable to anything deployed. Consults no outcome.
``amortised``
    Offline training, then inference. Consults no outcome at deployment.
``online_adaptation``
    Consults true outcomes per instance. A different cost class from everything
    above, and the one an amortised method is trying to replace.

Ordering happens *within* a class. The assembler deliberately does not emit a
global rank, because there is no question to which a global rank is the answer.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .headroom import _bootstrap, _parent_means
from .telemetry import _safe

COST_CLASSES = ("fixed", "privileged_spectrum", "amortised", "online_adaptation")


def _block(rows: Sequence[Mapping[str, Any]], method: str, cost_class: str, *,
           loss_key, reference_total: int, bootstrap_resamples: int, seed: int,
           consults_outcomes: bool, note: str = "") -> dict:
    usable = [row for row in rows if loss_key(row) is not None]
    if not usable:
        return None
    values, parents = _parent_means(usable, loss_key)
    against = {}
    for reference, name in (("linear_loss", "vs_linear"), ("global_loss", "vs_global")):
        pairs = [{"parent_id": row["parent_id"], "d": float(loss_key(row)) - float(row[reference])}
                 for row in usable if row.get(reference) is not None]
        if not pairs:
            continue
        differences, _ = _parent_means(pairs, lambda row: row["d"])
        against[name] = {
            "mean_difference": float(differences.mean()),
            "parent_bootstrap_ci": _bootstrap(differences, n_resamples=bootstrap_resamples,
                                              seed=seed),
            "reference": reference,
            "sign_convention": "negative means this method is better"}
    return _safe({
        "method": method,
        "cost_class": cost_class,
        "consults_true_outcomes": consults_outcomes,
        "n_records": len(usable),
        "n_parents": len(parents),
        # A method measured on fewer records than the rest is measured on a
        # different population, and the comparison is conditional on it.
        "measured_on_full_population": len(usable) == reference_total,
        "mean_loss": float(values.mean()),
        "std_parent_loss": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "parent_bootstrap_ci": _bootstrap(values, n_resamples=bootstrap_resamples, seed=seed),
        **against,
        **({"note": note} if note else {}),
    })


def assemble_comparison(ml_rows: Sequence[Mapping[str, Any]],
                        frontier_rows: Sequence[Mapping[str, Any]], *,
                        bootstrap_resamples: int = 20000, seed: int = 0) -> dict:
    """Join the learned evaluation and the test-split reference sweep by record.

    Both sources must describe the same held-out records. Anything present in
    only one of them is dropped and counted, because a method evaluated on a
    different record set is not in the same table however similar the numbers
    look.
    """
    ml_rows, frontier_rows = list(ml_rows), list(frontier_rows)
    splits = {str(row.get("split")) for row in ml_rows}
    if splits != {"test"}:
        raise ValueError(f"comparison table is a held-out measurement; got splits {sorted(splits)}")

    frontier_by_id = {str(row["record_id"]): row for row in frontier_rows}
    ml_ids = {str(row["record_id"]) for row in ml_rows}
    shared = ml_ids & set(frontier_by_id)
    if not shared:
        raise ValueError("the learned evaluation and the reference sweep have no records in common")

    joined = []
    for row in ml_rows:
        record = str(row["record_id"])
        if record in shared:
            joined.append({**row, "_frontier": frontier_by_id[record]})
    reference = [frontier_by_id[record] for record in sorted(shared)]
    total = len(shared)

    # Fixed and privileged rows come from the reference sweep, one row per record.
    base = [{"parent_id": row["parent_id"], "record_id": row["record_id"],
             "linear_loss": row.get("linear_loss"), "global_loss": None,
             "best_found_loss": row.get("best_found_loss"),
             "teachers": row.get("privileged_teachers", {}),
             "objective_calls": row.get("total_objective_calls")} for row in reference]

    rows = []
    rows.append(_block(base, "linear", "fixed", loss_key=lambda r: r["linear_loss"],
                       reference_total=total, bootstrap_resamples=bootstrap_resamples, seed=seed,
                       consults_outcomes=False))

    # The global baseline is a property of the learned evaluation, not the sweep.
    global_rows = [{"parent_id": row["parent_id"], "global_loss": row.get("global_loss"),
                    "linear_loss": row.get("linear_loss")}
                   for row in {str(r["record_id"]): r for r in joined}.values()]
    if any(row["global_loss"] is not None for row in global_rows):
        rows.append(_block(global_rows, "global", "fixed", loss_key=lambda r: r["global_loss"],
                           reference_total=total, bootstrap_resamples=bootstrap_resamples,
                           seed=seed, consults_outcomes=False))

    for teacher in sorted({name for row in base for name in row["teachers"]}):
        rows.append(_block(
            base, teacher, "privileged_spectrum",
            loss_key=lambda r, t=teacher: (r["teachers"].get(t) or {}).get("loss"),
            reference_total=total, bootstrap_resamples=bootstrap_resamples, seed=seed,
            consults_outcomes=False,
            note="requires the exact instantaneous spectrum; not available to a deployed method"))

    for method in sorted({str(row["method"]) for row in ml_rows}):
        for mode in sorted({str(row["mode"]) for row in ml_rows if str(row["method"]) == method}):
            subset = [row for row in joined
                      if str(row["method"]) == method and str(row["mode"]) == mode]
            if subset:
                rows.append(_block(subset, f"{method}/{mode}", "amortised",
                                   loss_key=lambda r: r.get("loss"), reference_total=total,
                                   bootstrap_resamples=bootstrap_resamples, seed=seed,
                                   consults_outcomes=False))

    calls = [float(row["objective_calls"]) for row in base if row["objective_calls"] is not None]
    search = _block(base, "search_best_found", "online_adaptation",
                    loss_key=lambda r: r["best_found_loss"], reference_total=total,
                    bootstrap_resamples=bootstrap_resamples, seed=seed, consults_outcomes=True,
                    note="online adaptation: true outcomes are consulted per instance")
    if search is not None:
        search["objective_calls_per_instance"] = float(np.mean(calls)) if calls else None
        rows.append(search)

    rows = [row for row in rows if row is not None]
    ranked = {}
    for cost_class in COST_CLASSES:
        members = [row for row in rows if row["cost_class"] == cost_class]
        if members:
            ranked[cost_class] = [row["method"] for row in
                                  sorted(members, key=lambda row: row["mean_loss"])]

    return _safe({
        "schema_version": 1,
        "n_records": total,
        "n_parents": len({str(row["parent_id"]) for row in reference}),
        "n_dropped_no_frontier_row": len(ml_ids - shared),
        "n_dropped_no_ml_row": len(set(frontier_by_id) - shared),
        "rows": rows,
        "ranked_within_cost_class": ranked,
        "cost_classes": list(COST_CLASSES),
        "scope": ("every row is measured on the same held-out records; rows are ordered only "
                  "within a cost class, because a method that consults true outcomes per "
                  "instance and one that consults none are not competing for the same prize"),
    })
