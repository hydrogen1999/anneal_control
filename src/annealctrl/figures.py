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


def figure_teacher_baselines(summary: Mapping[str, Any], stem: str | Path, *,
                             venue: str = "neurips", column: str = "double") -> dict:
    """Figure: the privileged spectral schedule is not the ceiling it is assumed to be.

    Left panel: each privileged baseline's paired difference against a linear
    ramp, stratified by runtime, with parent bootstrap intervals. Zero is drawn
    plainly; a rule that loses sits above the line. The monotone approach to zero
    with runtime is the adiabatic theorem behaving as advertised, and drawing it
    is more honest than quoting a single pooled number that hides it.

    Right panel: mean loss per method on the same records, so the gap between
    both oracles and the equal-budget search is visible at once. The resolution
    rate is printed on the axis because ``gap_inverse_square`` is measured on the
    subset where its first gap resolves, which is not a random subset.
    """
    _check_venue(venue)
    methods = dict(summary.get("methods") or {})
    usable = {name: block for name, block in methods.items() if block.get("mean_loss") is not None}
    if not usable:
        raise ValueError("figure_teacher_baselines requires at least one resolved baseline")
    plot_utils = load_plot_utils()
    import matplotlib.pyplot as plt

    width = plot_utils.use_venue(venue, column)
    fig, (left, right) = plt.subplots(1, 2, figsize=(width, width * 0.38))

    for offset, (name, block) in enumerate(sorted(usable.items())):
        by_runtime = block.get("by_runtime") or {}
        runtimes = sorted(by_runtime, key=float)
        if not runtimes:
            continue
        x = np.arange(len(runtimes), dtype=float) + 0.16 * (offset - 0.5)
        centres, low, high = [], [], []
        for runtime in runtimes:
            entry = by_runtime[runtime]["vs_linear"]
            interval = entry.get("parent_bootstrap_ci") or {}
            centre = float(entry["mean_difference"])
            centres.append(centre)
            low.append(centre - float(interval["low"]) if interval.get("low") is not None else 0.0)
            high.append(float(interval["high"]) - centre if interval.get("high") is not None else 0.0)
        left.errorbar(x, centres, yerr=[low, high], marker="o", linestyle="-",
                      capsize=2, label=name)
        left.set_xticks(np.arange(len(runtimes), dtype=float))
        left.set_xticklabels([f"{float(r):g}" for r in runtimes])

    left.axhline(0.0, linewidth=0.8, linestyle="--", color="0.35")
    left.set_xlabel("runtime")
    left.set_ylabel("loss vs linear ramp\n(positive = oracle is worse)")
    left.legend(loc="best")

    # Each baseline is drawn beside ITS OWN linear and search references. The two
    # baselines resolve on different subsets -- d2 on almost everything,
    # gap_inverse_square on under half -- so a single shared "linear" bar would
    # compare one baseline's loss against another baseline's population. That is
    # the same composition error the G3 scale-arm ratio made, and it is not worth
    # repeating in a figure because the bars happen to look tidier.
    ordered = sorted(usable.items())
    group = np.arange(len(ordered), dtype=float)
    series = [("linear", lambda b: float(b["mean_linear_loss"])),
              ("privileged teacher", lambda b: float(b["mean_loss"])),
              ("search (equal budget)", lambda b: float(b["mean_best_found_loss"]))]
    bar_width = 0.26
    peak = 0.0
    for index, (label, extract) in enumerate(series):
        values = [extract(block) for _, block in ordered]
        peak = max(peak, max(values))
        right.bar(group + (index - 1) * bar_width, values, width=bar_width, label=label)
    for index, (name, block) in enumerate(ordered):
        counted = block.get("n_audit_passed", block.get("n_resolved"))
        right.annotate(f"{100 * block['resolution_rate']:.0f}% resolved\n(n={counted})",
                       (group[index], peak * 1.04), ha="center", fontsize=6)
    right.set_xticks(group)
    right.set_xticklabels([name for name, _ in ordered], rotation=12, ha="right")
    right.set_ylabel("mean loss (lower is better)")
    right.set_xlabel("each baseline against its own population")
    right.set_ylim(0.0, peak * 1.42)
    right.legend(loc="upper center", ncol=3, fontsize=5.5, columnspacing=1.0,
                 handlelength=1.2, borderaxespad=0.2)

    fig.tight_layout()
    written = plot_utils.save(fig, stem)
    plt.close(fig)
    return {"figure": "teacher_baselines", "files": [str(path) for path in written],
            "methods": sorted(usable),
            "vendored_plot_utils": bool(getattr(plot_utils, "__annealctrl_vendored__", False)),
            "scope": ("privileged spectral baselines against linear and against the equal-budget "
                      "search; resolution rates are drawn because the resolved subset is not random")}


COST_CLASS_LABELS = {
    "fixed": "fixed\none waveform for every instance",
    "privileged_spectrum": "privileged spectrum\nexact gap at every path point",
    "amortised": "amortised\ntrain offline, forward pass at deployment",
    "online_adaptation": "online adaptation\ntrue outcomes consulted per instance",
}


def figure_comparison(table: Mapping[str, Any], stem: str | Path, *,
                      venue: str = "neurips", column: str = "double",
                      max_rows_per_class: int = 4) -> dict:
    """Figure 1: every method on one held-out population, grouped by what it consumes.

    Horizontal bars with parent bootstrap intervals, blocked by cost class and
    ordered only inside a block. There is deliberately no shared cost axis: an
    exact spectral evaluation, a forward pass and a simulator call are not the
    same unit, and a scatter against "cost" would invent a common currency in
    order to draw a trade-off curve. The class label carries the cost in words
    instead, which is less pretty and does not mislead.

    Rows measured on a subset are hatched and annotated with their record count,
    because a bar drawn the same way as its neighbours implies the same
    population.
    """
    _check_venue(venue)
    rows = [row for row in (table.get("rows") or []) if row.get("mean_loss") is not None]
    if not rows:
        raise ValueError("figure_comparison requires at least one measured row")
    plot_utils = load_plot_utils()
    import matplotlib.pyplot as plt

    # Selecting the lowest-loss rows outright drops whole modes: the amortised
    # class is four bank rows within 0.0015 of each other, and the direct mode --
    # the one the paper says is weak -- disappears entirely. Keep the best row of
    # each variant instead, where a variant is the part after "/" when a method
    # has one.
    ordered, blocks = [], []
    position = 0.0
    for cost_class in table.get("cost_classes", COST_CLASS_LABELS):
        members = sorted((row for row in rows if row["cost_class"] == cost_class),
                         key=lambda row: row["mean_loss"])
        best_of_variant: dict[str, Any] = {}
        for row in members:
            variant = row["method"].split("/")[-1] if "/" in row["method"] else row["method"]
            best_of_variant.setdefault(variant, row)
        members = sorted(best_of_variant.values(), key=lambda row: row["mean_loss"])
        members = members[:max_rows_per_class]
        if not members:
            continue
        position += 1.0            # a blank slot carries the class label
        blocks.append({"label": COST_CLASS_LABELS.get(cost_class, cost_class),
                       "label_position": position - 0.46,
                       "rows": members,
                       "positions": [position + index for index in range(len(members))]})
        ordered.extend(members)
        position += len(members)

    width = plot_utils.use_venue(venue, column)
    fig, axis = plt.subplots(figsize=(width, width * 0.5))
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    ceiling = max(float(row["mean_loss"]) for row in ordered)
    upper = 0.0

    for index, block in enumerate(blocks):
        colour = palette[index % len(palette)]
        for row, y in zip(block["rows"], block["positions"]):
            centre = float(row["mean_loss"])
            interval = row.get("parent_bootstrap_ci") or {}
            low = centre - float(interval["low"]) if interval.get("low") is not None else 0.0
            high = float(interval["high"]) - centre if interval.get("high") is not None else 0.0
            partial = not row.get("measured_on_full_population", True)
            axis.barh(y, centre, height=0.66, color=colour,
                      hatch="///" if partial else None,
                      edgecolor="white" if partial else "none", linewidth=0.0)
            axis.errorbar(centre, y, xerr=[[low], [high]], fmt="none", ecolor="0.25",
                          elinewidth=0.9, capsize=2)
            upper = max(upper, centre + high)
            if partial:
                # Past the whisker, never on top of it.
                axis.annotate(f"n={row['n_records']}", (centre + high, y),
                              textcoords="offset points", xytext=(5, 0), va="center",
                              fontsize=5.5, color="0.3")
        # The label lives in the blank slot above the block, so it cannot sit on a bar.
        # Anchored at the bottom so a two-line label grows upward into the blank
        # slot instead of downward onto the first bar of its own block.
        axis.annotate(block["label"], (0.0, block["label_position"]),
                      xytext=(2, 1), textcoords="offset points", fontsize=5.5,
                      va="bottom", ha="left", style="italic", color="0.3")
        if index:
            axis.axhline(block["label_position"] - 0.62, linewidth=0.5, color="0.8")

    positions = [y for block in blocks for y in block["positions"]]
    axis.set_yticks(positions)
    axis.set_yticklabels([row["method"] for row in ordered], fontsize=6)
    axis.set_ylim(positions[-1] + 0.8, blocks[0]["label_position"] - 1.05)
    axis.set_xlabel("mean loss on held-out records (lower is better)")
    axis.set_xlim(0.0, max(ceiling, upper) * 1.14)
    axis.grid(axis="y", visible=False)

    fig.tight_layout()
    written = plot_utils.save(fig, stem)
    plt.close(fig)
    return {"figure": "comparison", "files": [str(path) for path in written],
            "methods": [row["method"] for row in ordered],
            "hatched_partial_population": [row["method"] for row in ordered
                                           if not row.get("measured_on_full_population", True)],
            "vendored_plot_utils": bool(getattr(plot_utils, "__annealctrl_vendored__", False)),
            "scope": ("one held-out population; bars are ordered within a cost class and never "
                      "across one, and no shared cost axis is drawn because the units differ")}
