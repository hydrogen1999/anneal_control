"""G2b: qualify a stress subset by *measured* control gain, using no test information.

`docs/paper_protocol.md` §3.3 is explicit that hardness is an attribute you
measure, not one you assert from a gap quantile, and that the screening rule must
be fitted on training/validation parents only. Two failure modes follow, and both
are enforced here rather than left to discipline:

1. **Fitting on test.** A threshold chosen after seeing test headroom turns a
   conditional subset into a claim about deployment frequency. ``fit_threshold``
   refuses any row whose split is ``test``.
2. **Filtering on the proposed method's advantage.** Selecting instances where
   the model happens to win is circular. The screening quantity is therefore
   restricted to measured control gain — headroom, its normalised form, or a
   family restriction loss — and any other name is rejected.

A selected subset is a **conditional stress benchmark**. Every result carries
that label, the unscreened population, the rule, the selected fraction and the
objective-call cost of screening itself.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .telemetry import _safe

ALLOWED_QUANTITIES = ("headroom", "relative_headroom")
FIT_SPLITS = ("train", "validation")


def _quantity(row: Mapping[str, Any], quantity: str):
    if quantity.startswith("family_restriction_loss:"):
        return (row.get("family_restriction_loss") or {}).get(quantity.split(":", 1)[1])
    return row.get(quantity)


def _check_quantity(quantity: str) -> None:
    if quantity in ALLOWED_QUANTITIES or quantity.startswith("family_restriction_loss:"):
        return
    raise ValueError(
        f"screening quantity {quantity!r} is not a measured control gain. Allowed: "
        f"{list(ALLOWED_QUANTITIES)} or 'family_restriction_loss:<family>'. Screening on a "
        "model's advantage would select the instances the method already wins.")


def _parent_values(rows: Sequence[Mapping[str, Any]], quantity: str) -> tuple[list[str], np.ndarray, int]:
    """Parent means over resolved rows only; censored rows counted, never used."""
    buckets: dict[str, list[float]] = {}
    censored = 0
    for row in rows:
        status = row.get("resolution_status")
        if status == "censored_numerical":
            censored += 1
            continue
        if status != "resolved":
            raise ValueError(f"row {row.get('record_id')!r} has resolution_status {status!r}; "
                             "expected 'resolved' or 'censored_numerical'")
        value = _quantity(row, quantity)
        if value is None:
            continue
        buckets.setdefault(str(row["parent_id"]), []).append(float(value))
    parents = sorted(buckets)
    return parents, np.array([float(np.mean(buckets[p])) for p in parents]), censored


def fit_threshold(rows: Sequence[Mapping[str, Any]], *, quantity: str = "headroom",
                  quantile: float = 0.75, min_headroom: float | None = None) -> dict:
    """Fit a selection threshold on training/validation parents only.

    ``quantile`` is taken over parent-level means of the measured quantity;
    ``min_headroom`` is an absolute floor applied afterwards, so a distribution
    whose 75th percentile is still numerically trivial cannot qualify anything.
    """
    _check_quantity(quantity)
    rows = list(rows)
    if not rows:
        raise ValueError("fit_threshold requires a nonempty row set")
    if isinstance(quantile, bool) or not np.isfinite(quantile) or not 0.0 <= float(quantile) <= 1.0:
        raise ValueError("quantile must be a real number in [0,1]")
    splits = sorted({str(row.get("split")) for row in rows})
    if "test" in splits:
        raise ValueError("refusing to fit a screening threshold on test rows: a threshold chosen "
                         "after seeing test headroom is not a predeclared selection rule")
    invalid = set(splits) - set(FIT_SPLITS)
    if invalid:
        raise ValueError(f"screening may only be fitted on {list(FIT_SPLITS)} rows; got {sorted(invalid)}")

    parents, values, censored = _parent_values(rows, quantity)
    if not values.size:
        raise ValueError("no resolved rows available to fit a screening threshold; the population "
                         "offers no control gain above its own numerical resolution")
    threshold = float(np.quantile(values, float(quantile)))
    if min_headroom is not None:
        if not np.isfinite(min_headroom) or min_headroom < 0:
            raise ValueError("min_headroom must be finite and nonnegative")
        threshold = max(threshold, float(min_headroom))
    return _safe({
        "schema_version": 1, "quantity": quantity, "quantile": float(quantile),
        "min_headroom": None if min_headroom is None else float(min_headroom),
        "threshold": threshold, "fit_splits": splits,
        "n_fit_parents": len(parents), "n_fit_records": len(rows),
        "n_fit_censored_records": censored,
        "fit_parent_ids": parents,
        "fit_distribution": {"mean": float(values.mean()), "min": float(values.min()),
                             "max": float(values.max()),
                             "quantiles": {f"p{q}": float(np.percentile(values, q))
                                           for q in (10, 25, 50, 75, 90)}},
        "fit_objective_calls": int(sum(int(row.get("total_objective_calls", 0)) for row in rows)),
        "rule": f"select parents whose mean {quantity} exceeds {threshold:.6g}",
        "fitted_on_test": False,
        "selects_on_model_advantage": False,
    })


def apply_threshold(rule: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict:
    """Apply a fitted rule to any split, including test, and label the result.

    Applying to the split the rule was fitted on is legitimate for diagnostics
    but is in-sample; the overlap is measured and reported rather than assumed
    away.
    """
    rows = list(rows)
    if not rows:
        raise ValueError("apply_threshold requires a nonempty row set")
    quantity = str(rule["quantity"])
    _check_quantity(quantity)
    threshold = float(rule["threshold"])

    parents, values, censored_records = _parent_values(rows, quantity)
    censored_parents = len({str(row["parent_id"]) for row in rows
                            if row.get("resolution_status") == "censored_numerical"}
                           - set(parents))
    all_parents = sorted({str(row["parent_id"]) for row in rows})
    selected = [parent for parent, value in zip(parents, values) if value > threshold]
    overlap = sorted(set(all_parents) & set(rule.get("fit_parent_ids") or []))

    return _safe({
        "schema_version": 1, "quantity": quantity, "threshold": threshold,
        "apply_splits": sorted({str(row.get("split")) for row in rows}),
        "unscreened_parents": len(all_parents), "unscreened_records": len(rows),
        "censored_parents": censored_parents, "censored_records": censored_records,
        "selected_parent_ids": selected,
        "selected_parents": len(selected),
        "selected_fraction": len(selected) / len(all_parents),
        "selected_record_ids": sorted(str(row["record_id"]) for row in rows
                                      if str(row["parent_id"]) in set(selected)),
        "in_sample": bool(overlap), "parent_overlap": len(overlap),
        "apply_objective_calls": int(sum(int(row.get("total_objective_calls", 0)) for row in rows)),
        "total_screening_objective_calls": int(rule.get("fit_objective_calls", 0))
        + int(sum(int(row.get("total_objective_calls", 0)) for row in rows)),
        "verdict": "no_parent_qualifies" if not selected else "stress_subset_selected",
        "interpretation": ("conditional stress benchmark: results on this subset describe behaviour "
                           "given measured control headroom above the declared threshold, and say "
                           "nothing about how often such instances occur in deployment"),
        "is_unbiased_deployment_sample": False,
    })


def _sweep_rows(directories: Sequence[str | Path] | str | Path) -> tuple[list[dict], list[Path]]:
    """Successful rows from one sweep directory or several shards of one sweep."""
    from .sweeps import ROWS, load_rows

    paths = [Path(directories)] if isinstance(directories, (str, Path)) else [Path(p) for p in directories]
    if not paths:
        raise ValueError("at least one sweep directory is required")
    rows: list[dict] = []
    for path in paths:
        if not (path / ROWS).exists():
            raise ValueError(f"{path} is not a sweep directory: no {ROWS} in it")
        rows.extend(row["result"] for row in load_rows(path) if row.get("status") == "ok")
    if not rows:
        raise ValueError(f"{[str(p) for p in paths]} contain no successful sweep rows")
    return rows, paths


def screen_records(fit_sweeps: Sequence[str | Path], apply_sweep: Sequence[str | Path] | str | Path, *,
                   quantity: str = "headroom", quantile: float = 0.75,
                   min_headroom: float | None = None) -> dict:
    """Fit on one or more train/validation sweeps, then apply to one or more sweeps.

    Both sides accept several directories because a sharded campaign is several
    directories; requiring a single one on the apply side would force a merge
    step that has no scientific meaning.
    """
    fit, fit_paths = _sweep_rows(fit_sweeps)
    rule = fit_threshold(fit, quantity=quantity, quantile=quantile, min_headroom=min_headroom)
    rule["fit_sweeps"] = [str(path) for path in fit_paths]
    applied, apply_paths = _sweep_rows(apply_sweep)
    selection = apply_threshold(rule, applied)
    selection["apply_sweep"] = [str(path) for path in apply_paths]
    return {"schema_version": 1, "rule": rule, "selection": selection,
            "scope": "screening is measured control gain only; it never consults a learned model"}
