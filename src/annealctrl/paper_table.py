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

from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np

from .headroom import _bootstrap, _parent_means
from .telemetry import _safe

COST_CLASSES = ("fixed", "privileged_spectrum", "amortised", "online_adaptation")


def _audited_teacher_loss(row, teacher):
    entry = row["teachers"].get(teacher) or {}
    value = entry.get("loss")
    if entry.get("status") != "sampled_point_audit_passed" or value is None:
        return None
    return float(value) if np.isfinite(value) else None


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
                        searches: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
                        bootstrap_resamples: int = 20000, seed: int = 0) -> dict:
    """Join the learned evaluation and the test-split reference sweep by record.

    Both sources must describe the same held-out records. Anything present in
    only one of them is dropped and counted, because a method evaluated on a
    different record set is not in the same table however similar the numbers
    look.

    ``searches`` adds further search strategies run at the same budget -- a
    Bayesian-optimisation arm beside the quasi-random one, say. Each becomes its
    own ``online_adaptation`` row. They are restricted to the records the primary
    join already covers, so a strategy that happened to run on more instances
    cannot look better by being measured somewhere easier. The linear reference
    and the privileged teachers are taken from the primary sweep alone, because
    neither depends on which search was used and reading them twice would invite
    two slightly different values for the same quantity.
    """
    ml_rows, frontier_rows = list(ml_rows), list(frontier_rows)
    splits = {str(row.get("split")) for row in ml_rows}
    if splits != {"test"}:
        raise ValueError(f"comparison table is a held-out measurement; got splits {sorted(splits)}")

    frontier_by_id = {str(row["record_id"]): row for row in frontier_rows}
    if len(frontier_by_id) != len(frontier_rows):
        raise ValueError("duplicate reference record_id")
    if any(str(row.get("split")) != "test" for row in frontier_rows):
        raise ValueError("reference rows must use the test split")
    ml_ids = {str(row["record_id"]) for row in ml_rows}
    shared = ml_ids & set(frontier_by_id)
    if not shared:
        raise ValueError("the learned evaluation and the reference sweep have no records in common")

    joined = []
    for row in ml_rows:
        record = str(row["record_id"])
        if record in shared:
            if str(row["parent_id"]) != str(frontier_by_id[record]["parent_id"]):
                raise ValueError("learned/reference parent identity mismatch")
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

    teacher_populations, teacher_contrasts = {}, []
    for teacher in sorted({name for row in base for name in row["teachers"]}):
        eligible = {str(row["record_id"]): row for row in base
                    if _audited_teacher_loss(row, teacher) is not None}
        teacher_populations[teacher] = {
            "eligible_record_ids": sorted(eligible), "n_eligible_records": len(eligible),
            "n_excluded_records": total - len(eligible),
            "status_counts": dict(Counter((row["teachers"].get(teacher) or {}).get("status", "missing")
                                          for row in base)),
            "eligibility": "sampled_point_audit_passed and finite non-null loss"}
        block = _block(
            base, teacher, "privileged_spectrum",
            loss_key=lambda r, t=teacher: _audited_teacher_loss(r, t),
            reference_total=total, bootstrap_resamples=bootstrap_resamples, seed=seed,
            consults_outcomes=False,
            note="conditional on this teacher's passed sampled-point audit; requires exact spectrum; audit is not a certificate")
        if block is not None:
            block["eligible_record_ids"] = sorted(eligible)
            rows.append(block)
        for method, mode in sorted({(str(row["method"]), str(row["mode"])) for row in joined}):
            subset = [row for row in joined if str(row["method"]) == method and str(row["mode"]) == mode
                      and str(row["record_id"]) in eligible and row.get("loss") is not None]
            if not subset:
                continue
            ids = [str(row["record_id"]) for row in subset]
            if len(set(ids)) != len(ids):
                raise ValueError("matched teacher comparison requires one learned mean per method/mode/record")
            pairs = [{"parent_id": row["parent_id"], "learned": float(row["loss"]),
                      "teacher": _audited_teacher_loss(eligible[str(row["record_id"])], teacher)}
                     for row in subset]
            differences, parents = _parent_means(pairs, lambda row: row["learned"] - row["teacher"])
            learned, _ = _parent_means(pairs, lambda row: row["learned"])
            target, _ = _parent_means(pairs, lambda row: row["teacher"])
            teacher_contrasts.append({
                "method": f"{method}/{mode}", "teacher": teacher,
                "method_cost_class": "amortised", "teacher_cost_class": "privileged_spectrum",
                "n_records": len(pairs), "n_parents": len(parents), "record_ids": sorted(ids),
                "covers_all_teacher_eligible_records": set(ids) == set(eligible),
                "learned_mean_loss": float(learned.mean()), "teacher_mean_loss": float(target.mean()),
                "mean_difference": float(differences.mean()),
                "parent_bootstrap_ci": _bootstrap(differences, n_resamples=bootstrap_resamples, seed=seed),
                "difference_definition": "learned minus teacher; negative favours learned",
                "scope": "paired on teacher-specific audited records; parent bootstrap conditional on supplied learned seed means; unadjusted descriptive interval; different cost classes"})

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

    for name, extra_rows in sorted((searches or {}).items()):
        by_id = {str(row["record_id"]): row for row in extra_rows if str(row["record_id"]) in shared}
        entries = [{"parent_id": row["parent_id"], "record_id": record,
                    "linear_loss": row.get("linear_loss"), "global_loss": None,
                    "best_found_loss": row.get("best_found_loss"),
                    "teachers": {}, "objective_calls": row.get("total_objective_calls")}
                   for record, row in sorted(by_id.items())]
        if not entries:
            continue
        block = _block(entries, f"search_{name}", "online_adaptation",
                       loss_key=lambda r: r["best_found_loss"], reference_total=total,
                       bootstrap_resamples=bootstrap_resamples, seed=seed,
                       consults_outcomes=True,
                       note=f"online adaptation, {name} strategy at the same budget")
        if block is None:
            continue
        extra_calls = [float(entry["objective_calls"]) for entry in entries
                       if entry["objective_calls"] is not None]
        block["objective_calls_per_instance"] = float(np.mean(extra_calls)) if extra_calls else None
        rows.append(block)

    rows = [row for row in rows if row is not None]
    ranked = {}
    for cost_class in COST_CLASSES:
        members = [row for row in rows if row["cost_class"] == cost_class
                   and row["measured_on_full_population"]]
        if members:
            ranked[cost_class] = [row["method"] for row in
                                  sorted(members, key=lambda row: row["mean_loss"])]

    return _safe({
        "schema_version": 2,
        "n_records": total,
        "n_parents": len({str(row["parent_id"]) for row in reference}),
        "n_dropped_no_frontier_row": len(ml_ids - shared),
        "n_dropped_no_ml_row": len(set(frontier_by_id) - shared),
        "rows": rows,
        "teacher_populations": teacher_populations,
        "matched_teacher_contrasts": teacher_contrasts,
        "ranked_within_cost_class": ranked,
        "ranking_scope": "full common population only; conditional rows are not ranked against another population",
        "cost_classes": list(COST_CLASSES),
        "scope": ("common held-out join with explicitly conditional teacher populations; compare learned and "
                  "teacher losses using matched_teacher_contrasts, not unpaired row means; ordering within "
                  "a cost class includes only rows covering the full common population"),
    })
