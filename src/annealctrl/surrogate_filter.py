"""Can the critic rank candidates it did not generate, and does that buy budget?

The critic picks well from the bank it was trained on -- rank correlation 0.834,
the true best inside its top half 95.5% of the time. That is the in-distribution
number and it is not the interesting one. The interesting one is whether the
same ranking holds on waveforms a *search* produced, because that is what would
turn the model from a picker into a budget multiplier: propose many, score them
all for free, and pay the simulator only for the ones the critic likes.

Two things make this measurable honestly rather than approximately.

**The grid.** The critic requires ``schedule_points`` values on a common uniform
tau grid; search families put their knots elsewhere. Re-expressing a search
waveform on the bank grid moves it by a sup norm of 0.04-0.07 on these sweeps,
which is the same size as the spacing between bank candidates -- large enough to
dominate a ranking signal. So the archived search loss is *not* a valid label
for the regridded waveform. ``regrid`` returns the representation error beside
the waveform so a caller cannot quietly ignore it, and the study re-scores the
regridded waveform with the true simulator, making the critic and the label
describe the same object.

**The split.** A checkpoint trained on a split must not be ranked on it. The
study takes the record identities from the dataset split, not from the sweep.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .telemetry import _safe


def regrid(tau_knots: Sequence[float], s_knots: Sequence[float], *, points: int,
           samples: int = 513) -> dict:
    """A piecewise-linear waveform re-expressed on a uniform tau grid.

    ``representation_error`` is the sup norm between the original function and
    the regridded one, densely sampled. It is returned, not logged, because the
    whole study is invalid if it is large and silently ignored.
    """
    tau = np.asarray(tau_knots, dtype=float)
    s = np.asarray(s_knots, dtype=float)
    if tau.ndim != 1 or tau.shape != s.shape or tau.size < 2:
        raise ValueError("tau_knots and s_knots must be 1-D, equal length, at least 2 points")
    if not (np.isfinite(tau).all() and np.isfinite(s).all()):
        raise ValueError("waveform knots must be finite")
    if np.any(np.diff(tau) < 0):
        raise ValueError("tau_knots must be nondecreasing")
    if points < 2:
        raise ValueError("points must be at least 2")

    grid = np.linspace(0.0, 1.0, points)
    regridded = np.interp(grid, tau, s)
    dense = np.linspace(0.0, 1.0, samples)
    error = float(np.abs(np.interp(dense, tau, s) - np.interp(dense, grid, regridded)).max())
    return {"grid": grid, "waveform": regridded, "representation_error": error}


def _rho(a: np.ndarray, b: np.ndarray) -> float | None:
    from scipy.stats import rankdata

    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.size < 3 or np.all(a == a[0]) or np.all(b == b[0]):
        return None
    ra, rb = rankdata(a) - rankdata(a).mean(), rankdata(b) - rankdata(b).mean()
    denominator = float(np.sqrt((ra**2).sum() * (rb**2).sum()))
    return None if denominator == 0 else float(ra @ rb / denominator)


def filter_curve(predicted: Sequence[float], true: Sequence[float], *,
                 keep_fractions: Sequence[float] = (0.125, 0.25, 0.5)) -> dict:
    """What a critic-filtered budget would have reached on one record.

    Keeping the critic's top ``f`` and simulating only those, the realised loss
    is the best *true* loss among them. ``shortfall`` is what that costs against
    simulating everything, in the units the rest of the project reports.
    """
    predicted, true = np.asarray(predicted, dtype=float), np.asarray(true, dtype=float)
    if predicted.shape != true.shape or predicted.ndim != 1 or predicted.size < 2:
        raise ValueError("predicted and true must be 1-D, equal length, at least 2 entries")
    order = np.argsort(predicted, kind="stable")
    full_best = float(true.min())
    rows = {}
    for fraction in keep_fractions:
        if not 0 < fraction <= 1:
            raise ValueError("keep_fractions must lie in (0, 1]")
        keep = max(1, int(round(fraction * predicted.size)))
        kept = order[:keep]
        realised = float(true[kept].min())
        rows[str(fraction)] = {
            "kept": keep, "of": int(predicted.size),
            "realised_loss": realised,
            "shortfall_vs_full_budget": realised - full_best,
            "found_the_true_best": bool(realised == full_best)}
    return {"full_budget_best_loss": full_best, "rank_correlation": _rho(predicted, true),
            "by_keep_fraction": rows}


def aggregate_filter(rows: Sequence[Mapping[str, Any]], *, keep_fractions: Sequence[float],
                     bootstrap_resamples: int = 2000, seed: int = 0) -> dict:
    """Parent-level summary of the budget a critic filter would have saved."""
    from .headroom import _bootstrap, _describe, _parent_means

    rows = list(rows)
    if not rows:
        raise ValueError("aggregate_filter requires a nonempty row set")

    correlations = [r["rank_correlation"] for r in rows if r.get("rank_correlation") is not None]
    by_fraction = {}
    for fraction in keep_fractions:
        key = str(fraction)
        values, parents = _parent_means(
            rows, lambda r, key=key: r["by_keep_fraction"][key]["shortfall_vs_full_budget"])
        block = _describe(values)
        block["parent_bootstrap_ci"] = _bootstrap(values, n_resamples=bootstrap_resamples, seed=seed)
        block["n_parents"] = len(parents)
        block["found_true_best_fraction"] = float(np.mean(
            [r["by_keep_fraction"][key]["found_the_true_best"] for r in rows]))
        block["mean_kept"] = float(np.mean([r["by_keep_fraction"][key]["kept"] for r in rows]))
        block["mean_of"] = float(np.mean([r["by_keep_fraction"][key]["of"] for r in rows]))
        by_fraction[key] = block

    return _safe({
        "schema_version": 1,
        "n_records": len(rows),
        "n_parents": len({str(r["parent_id"]) for r in rows}),
        "mean_rank_correlation": float(np.mean(correlations)) if correlations else None,
        "n_records_with_rank_correlation": len(correlations),
        "by_keep_fraction": by_fraction,
        "scope": ("candidates are search-generated waveforms re-expressed on the critic's "
                  "grid and re-scored by the true simulator, so prediction and label "
                  "describe the same object; the critic never saw these records in training"),
    })
