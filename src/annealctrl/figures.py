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
# A copy travels with the repository so a clean checkout can regenerate figures.
# Upstream stays canonical; this is only reached when upstream is absent.
VENDORED_PLOT_UTILS = Path(__file__).resolve().parent / "_vendor"
VENUES = ("neurips", "icml", "iclr", "acl", "cvpr", "aaai")


def plot_utils_source() -> Path:
    """The directory ``load_plot_utils`` will import from, without importing it.

    An explicit ``ANNEALCTRL_PLOT_UTILS`` is honoured strictly: if it names a
    directory with no ``plot_utils.py``, that is an error rather than a cue to
    fall back, because silently ignoring an explicit pointer is how figures get
    rendered by something other than what the author asked for.
    """
    override = os.environ.get("ANNEALCTRL_PLOT_UTILS")
    if override is not None:
        root = Path(override)
        if not (root / "plot_utils.py").exists():
            raise RuntimeError(
                f"plot_utils.py not found in {root}. Camera-ready figures must be rendered "
                "through it: matplotlib's default pdf.fonttype is 3, and Type 3 fonts are "
                "rejected by IEEE PDF eXpress and flagged by NeurIPS/ICML/CVPR. Point "
                "ANNEALCTRL_PLOT_UTILS at the directory containing it, or unset it to use the "
                f"copy vendored at {VENDORED_PLOT_UTILS}.")
        return root
    if (DEFAULT_PLOT_UTILS / "plot_utils.py").exists():
        return DEFAULT_PLOT_UTILS
    if (VENDORED_PLOT_UTILS / "plot_utils.py").exists():
        return VENDORED_PLOT_UTILS
    raise RuntimeError(
        f"plot_utils.py not found in {DEFAULT_PLOT_UTILS} and no vendored copy at "
        f"{VENDORED_PLOT_UTILS}. Camera-ready figures must be rendered through it: "
        "matplotlib's default pdf.fonttype is 3, and Type 3 fonts are rejected by IEEE PDF "
        "eXpress and flagged by NeurIPS/ICML/CVPR.")


def load_plot_utils():
    """Import ``plot_utils`` from upstream if present, else the vendored copy."""
    root = plot_utils_source()
    module = sys.modules.get("plot_utils")
    if module is not None and Path(getattr(module, "__file__", "")).parent == root:
        return module
    sys.modules.pop("plot_utils", None)
    sys.path.insert(0, str(root))
    try:
        import plot_utils  # noqa: PLC0415
    finally:
        sys.path.remove(str(root))
    plot_utils.__annealctrl_vendored__ = root == VENDORED_PLOT_UTILS
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
    fig, (left, right) = plt.subplots(1, 2, figsize=(width, width * 0.50))

    data, labels = [], []
    for name in families:
        values = _parent_mean(resolved, lambda row, name=name: (row.get("family_restriction_loss") or {}).get(name))
        if values.size:
            data.append(values)
            labels.append(name.replace("_", " "))
    if data:
        parts = left.boxplot(data, tick_labels=labels, widths=0.6, showfliers=False,
                             medianprops={"linewidth": 1.2})
        for index, values in enumerate(data, start=1):
            jitter = np.random.default_rng(0).normal(0, 0.05, values.size)
            left.plot(index + jitter, values, ".", alpha=0.55, markersize=3)
        del parts
    left.set_ylabel("family restriction loss")
    left.set_xlabel(f"control family  ({len(censored)}/{len(rows)} records censored)")
    left.tick_params(axis="x", labelrotation=22)
    for label in left.get_xticklabels():
        label.set_horizontalalignment("right")
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
        # Above the axes, never inside: an in-axes legend marker sits at a data
        # coordinate and can be read as a measurement.
        right.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0),
                     borderaxespad=0.0, handletextpad=0.3, columnspacing=0.8,
                     fontsize="small")
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
    fig, (left, right) = plt.subplots(1, 2, figsize=(width, width * 0.50))

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
    left.set_xticklabels([f.replace("_", " ") for f in factors], rotation=22, ha="right")
    left.set_ylabel("transfer penalty")
    left.set_xlabel(f"intervened factor  ({len(censored)}/{len(rows)} pairs censored)")
    left.axhline(0.0, linewidth=0.6, linestyle=":", color="0.4")
    if arms:
        left.legend(ncol=1, loc="upper left", bbox_to_anchor=(0.0, 1.02),
                    borderaxespad=0.0, handletextpad=0.3, fontsize="small")

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
        right.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0),
                     borderaxespad=0.0, handletextpad=0.4, columnspacing=1.0)
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


def figure_representation(rows, stem: str | Path, *, baseline: str = "logical",
                          swap_pairs_only: bool = False, venue: str = "neurips",
                          column: str = "double") -> dict:
    """Figure 5: does seeing the embedding buy better decisions on swap pairs?

    Left panel: excess loss against each arm's own best-found control, one box per
    method, parent-averaged. Methods that are embedding-blind by construction are
    hatched — a logical or summary encoder receives identical input on both arms
    of a pair and cannot propose different controls, so its excess loss is the
    price of that blindness rather than a training shortfall.

    Right panel: the paired contrast against the baseline, per parent, with a
    bootstrap interval. Negative favours the method. A zero-crossing interval is
    the honest outcome when physical information does not pay, and the panel is
    drawn so that outcome is as visible as the favourable one.
    """
    from .representation import aggregate_model_interventions

    _check_venue(venue)
    rows = list(rows)
    if not rows:
        raise ValueError("figure_representation requires a nonempty row set")
    plot_utils = load_plot_utils()
    import matplotlib.pyplot as plt

    summary = aggregate_model_interventions(rows, baseline=baseline,
                                            swap_pairs_only=swap_pairs_only)
    methods = summary["methods"]
    blind = [m for m in methods if summary["by_method"][m]["embedding_blind_by_construction"]]

    if swap_pairs_only:
        swaps = {str(r["pair_id"]) for r in rows if r.get("preferred_control_swapped")}
        rows = [r for r in rows if str(r["pair_id"]) in swaps]

    width = plot_utils.use_venue(venue, column)
    fig, (left, right) = plt.subplots(1, 2, figsize=(width, width * 0.50))

    data, labels, hatches = [], [], []
    for method in methods:
        values = _parent_mean([r for r in rows if r["method"] == method],
                              lambda row: row.get("mean_excess_loss"))
        if not values.size:
            continue
        data.append(values)
        labels.append(method.replace("_", " "))
        hatches.append(method in blind)
    if data:
        boxes = left.boxplot(data, tick_labels=labels, widths=0.6, showfliers=False,
                             patch_artist=True, medianprops={"linewidth": 1.2})
        for patch, is_blind in zip(boxes["boxes"], hatches):
            patch.set_facecolor("none")
            if is_blind:
                patch.set_hatch("////")
    left.set_ylabel("excess loss vs arm best-found")
    left.set_xlabel("method  (hatched = embedding-blind by construction)")
    left.tick_params(axis="x", labelrotation=22)
    for label in left.get_xticklabels():
        label.set_horizontalalignment("right")
    left.axhline(0.0, linewidth=0.6, linestyle=":", color="0.4")

    contrasts = summary["decision_value"]
    names = [m for m in methods if m in contrasts]
    if names:
        centres = np.arange(len(names))
        means = [contrasts[m]["mean_difference"] for m in names]
        lows = [contrasts[m]["parent_bootstrap_ci"]["low"] for m in names]
        highs = [contrasts[m]["parent_bootstrap_ci"]["high"] for m in names]
        errors = np.array([[m - (l if l is not None else m) for m, l in zip(means, lows)],
                           [(h if h is not None else m) - m for m, h in zip(means, highs)]])
        right.errorbar(centres, means, yerr=np.abs(errors), fmt="o", capsize=3)
        right.set_xticks(centres)
        right.set_xticklabels([m.replace("_", " ") for m in names], rotation=22, ha="right")
    right.axhline(0.0, linewidth=0.8, linestyle="--", color="0.4")
    right.set_ylabel(f"excess loss minus\n{baseline.replace('_', ' ')} (negative favours method)")
    right.set_xlabel("method")
    fig.tight_layout()
    written = plot_utils.save(fig, stem)
    plt.close(fig)

    return {"figure": "representation", "venue": venue, "column": column,
            "panels": ["excess_loss_by_method", "paired_decision_value"],
            "methods": methods, "baseline": baseline, "blind_methods": blind,
            "blindness_shown": True,
            "n_pairs_per_method": summary["n_pairs_per_method"],
            "n_parents": summary["n_parents"],
            "restricted_to_swap_pairs": bool(swap_pairs_only),
            "blindness_violations": summary["blindness_violations"],
            "files": [str(path) for path in written],
            "note": "excess loss is against a finite-budget best-found reference, not a global optimum"}
