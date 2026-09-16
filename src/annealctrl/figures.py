"""Camera-ready Figure 3 (G2 frontier) and Figure 4 (G3 interventions).

Matplotlib's default ``pdf.fonttype`` is 3. Type 3 fonts are rejected outright by
IEEE PDF eXpress and flagged by NeurIPS/ICML/CVPR submission systems, so a figure
can pass every review pass and still be stopped by the submission system. These
figures therefore render through ``~/research-os/scripts/plot_utils.py``, which
sets ``fonttype = 42``, supplies per-venue column widths and a colourblind-safe
palette, and refuses to save a PDF that still embeds a Type 3 font.

If that helper is missing this module raises and names the path it looked in. It
never falls back to raw matplotlib defaults: a silent fallback produces a figure
that looks fine and fails at submission, which is the failure this exists to
prevent. See ``docs/decisions/ADR-0006-figures-through-plot-utils.md``.

Both figures draw censored and unresolved cases rather than dropping them. A
panel that quietly omits the instances where nothing was measurable would make a
conditional result look unconditional.
"""
from __future__ import annotations

from collections import defaultdict
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

DEFAULT_PLOT_UTILS = Path.home() / "research-os" / "scripts"
VENUES = ("neurips", "icml", "iclr", "acl", "cvpr", "aaai")


def load_plot_utils():
    """Import ``plot_utils``, or raise naming the directory that was searched."""
    root = Path(os.environ.get("ANNEALCTRL_PLOT_UTILS", DEFAULT_PLOT_UTILS))
    module = sys.modules.get("plot_utils")
    if module is not None and Path(getattr(module, "__file__", "")).parent == root:
        return module
    if not (root / "plot_utils.py").exists():
        raise RuntimeError(
            f"plot_utils.py not found in {root}. Camera-ready figures must be rendered through it: "
            "matplotlib's default pdf.fonttype is 3, and Type 3 fonts are rejected by IEEE PDF "
            "eXpress and flagged by NeurIPS/ICML/CVPR. Set ANNEALCTRL_PLOT_UTILS to the directory "
            "containing it, or vendor a copy with its upstream revision recorded.")
    sys.modules.pop("plot_utils", None)
    sys.path.insert(0, str(root))
    try:
        import plot_utils  # noqa: PLC0415
    finally:
        sys.path.remove(str(root))
    return plot_utils


def _check_venue(venue: str) -> None:
    if venue not in VENUES:
        raise ValueError(f"unknown venue {venue!r}; known: {sorted(VENUES)}")


def _parent_mean(rows: Sequence[Mapping[str, Any]], value) -> np.ndarray:
    buckets: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        item = value(row)
        if item is not None and np.isfinite(float(item)):
            buckets[str(row["parent_id"])].append(float(item))
    return np.array([float(np.mean(values)) for _, values in sorted(buckets.items())])


def figure_frontier(rows: Sequence[Mapping[str, Any]], stem: str | Path, *,
                    venue: str = "neurips", column: str = "double") -> dict:
    """Figure 3: what restricting the control family costs, and at what budget.

    Left panel: parent-averaged family restriction loss, one box per family, with
    the censored-record count stated on the axis so a reader cannot mistake the
    resolved subset for the whole population.

    Right panel: mean incumbent regret to each record's own best found against
    the objective-call index. A trace still falling at the budget boundary means
    the reference is budget-limited and the headroom on the left is a loose lower
    bound.
    """
    _check_venue(venue)
    rows = list(rows)
    if not rows:
        raise ValueError("figure_frontier requires a nonempty row set")
    plot_utils = load_plot_utils()
    import matplotlib.pyplot as plt

    resolved = [row for row in rows if row.get("resolution_status") == "resolved"]
    censored = [row for row in rows if row.get("resolution_status") == "censored_numerical"]
    families = sorted({name for row in rows for name in (row.get("family_restriction_loss") or {})})

    width = plot_utils.use_venue(venue, column)
    fig, (left, right) = plt.subplots(1, 2, figsize=(width, width * 0.38))

    data, labels = [], []
    for name in families:
        values = _parent_mean(resolved, lambda row, name=name: (row.get("family_restriction_loss") or {}).get(name))
        if values.size:
            data.append(values)
            labels.append(name.replace("_", "\n"))
    if data:
        parts = left.boxplot(data, tick_labels=labels, widths=0.6, showfliers=False,
                             medianprops={"linewidth": 1.2})
        for index, values in enumerate(data, start=1):
            jitter = np.random.default_rng(0).normal(0, 0.05, values.size)
            left.plot(index + jitter, values, ".", alpha=0.55, markersize=3)
        del parts
    left.set_ylabel("family restriction loss")
    left.set_xlabel(f"control family  ({len(censored)}/{len(rows)} records censored)")
    left.axhline(0.0, linewidth=0.6, linestyle=":", color="0.4")

    traces = 0
    for name in families:
        curves = [np.asarray(row["incumbent_trace"][name], dtype=float) - float(row["best_found_loss"])
                  for row in resolved if (row.get("incumbent_trace") or {}).get(name)]
        if not curves:
            continue
        longest = max(len(curve) for curve in curves)
        padded = np.full((len(curves), longest), np.nan)
        for index, curve in enumerate(curves):
            padded[index, :len(curve)] = curve
            padded[index, len(curve):] = curve[-1]   # a finished search holds its incumbent
        right.plot(np.arange(1, longest + 1), np.nanmean(padded, axis=0), marker="o",
                   label=name.replace("_", " "))
        traces += 1
    right.set_xlabel("objective calls within family")
    right.set_ylabel("mean incumbent regret\nto best found")
    if traces:
        right.legend(ncol=2, loc="best")
    fig.tight_layout()
    written = plot_utils.save(fig, stem)
    plt.close(fig)

    return {"figure": "frontier", "venue": venue, "column": column,
            "panels": ["family_restriction", "budget_sensitivity"],
            "families": families, "n_records": len(rows), "n_resolved": len(resolved),
            "n_censored": len(censored), "censored_shown": True,
            "n_budget_traces": traces, "files": [str(path) for path in written],
            "note": "censored records are counted on the axis label, never dropped silently"}


def figure_interventions(rows: Sequence[Mapping[str, Any]], stem: str | Path, *,
                         venue: str = "neurips", column: str = "double") -> dict:
    """Figure 4: transfer penalties by factor and scale arm, and the 2x2 matrices.

    Left panel: parent-averaged transfer penalty per factor, with the two scale
    arms drawn as separate series. They answer different questions and are never
    pooled into one distribution.

    Right panel: every pair's cross-control matrix as two points against the
    y = x diagonal — the own best-found loss on the x axis, the imported
    control's loss on the y axis. Distance above the diagonal is the transfer
    penalty; censored pairs are drawn hollow so a reader sees how much of the
    population sat on the line.
    """
    _check_venue(venue)
    rows = list(rows)
    if not rows:
        raise ValueError("figure_interventions requires a nonempty row set")
    plot_utils = load_plot_utils()
    import matplotlib.pyplot as plt

    censored = [row for row in rows if row.get("resolution_status") == "censored_numerical"]
    measured = [row for row in rows if row.get("resolution_status") != "censored_numerical"]
    factors = sorted({str(row.get("factor")) for row in rows})
    arms = sorted({str(row.get("scale_arm")) for row in rows})

    width = plot_utils.use_venue(venue, column)
    fig, (left, right) = plt.subplots(1, 2, figsize=(width, width * 0.38))

    offsets = np.linspace(-0.18, 0.18, max(len(arms), 1))
    for arm_index, arm in enumerate(arms):
        centres, values, spreads = [], [], []
        for position, factor in enumerate(factors):
            subset = [row for row in measured
                      if row.get("scale_arm") == arm and str(row.get("factor")) == factor]
            parent_values = _parent_mean(subset, lambda row: row.get("mean_transfer_penalty"))
            if not parent_values.size:
                continue
            centres.append(position + offsets[arm_index])
            values.append(float(parent_values.mean()))
            spreads.append(float(parent_values.std(ddof=1)) if parent_values.size > 1 else 0.0)
        if centres:
            left.errorbar(centres, values, yerr=spreads, fmt="o", capsize=2,
                          label=arm.replace("_", " "))
    left.set_xticks(range(len(factors)))
    left.set_xticklabels([f.replace("_", "\n") for f in factors])
    left.set_ylabel("transfer penalty")
    left.set_xlabel(f"intervened factor  ({len(censored)}/{len(rows)} pairs censored)")
    left.axhline(0.0, linewidth=0.6, linestyle=":", color="0.4")
    if arms:
        left.legend(loc="best")

    own, imported, was_censored = [], [], []
    for row in rows:
        matrix = row.get("loss_matrix") or {}
        for base, cross in (("A_on_A", "B_on_A"), ("B_on_B", "A_on_B")):
            if matrix.get(base) is None or matrix.get(cross) is None:
                continue
            own.append(float(matrix[base]))
            imported.append(float(matrix[cross]))
            was_censored.append(row.get("resolution_status") == "censored_numerical")
    own, imported = np.asarray(own), np.asarray(imported)
    was_censored = np.asarray(was_censored, dtype=bool)
    if own.size:
        limits = [float(min(own.min(), imported.min())), float(max(own.max(), imported.max()))]
        pad = 0.02 * max(limits[1] - limits[0], 1e-6)
        right.plot([limits[0] - pad, limits[1] + pad], [limits[0] - pad, limits[1] + pad],
                   linestyle="--", linewidth=0.8, color="0.4", label="no transfer penalty")
        right.plot(own[~was_censored], imported[~was_censored], "o", markersize=3.5,
                   label="measurable")
        if was_censored.any():
            right.plot(own[was_censored], imported[was_censored], "o", markersize=3.5,
                       markerfacecolor="none", label="censored")
        right.legend(loc="best")
    right.set_xlabel("own best-found loss")
    right.set_ylabel("imported control's loss")
    fig.tight_layout()
    written = plot_utils.save(fig, stem)
    plt.close(fig)

    return {"figure": "interventions", "venue": venue, "column": column,
            "panels": ["penalty_by_factor", "cross_control_scatter"],
            "factors": factors, "scale_arms": arms, "scale_arms_pooled": False,
            "n_pairs": len(rows), "n_measurable": len(measured), "n_censored": len(censored),
            "censored_shown": True, "files": [str(path) for path in written],
            "note": "causal scope is inside the declared closed-system simulator"}
