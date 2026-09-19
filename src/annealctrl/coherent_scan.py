"""The coherent-failure test: does a scalar profile track interference?

The design document reserves this for paths with two or more transition
regions:

> For coherent failure modes, use paths with two or more transition regions and
> scan runtime densely enough to detect interference. A scalar profile might
> work after averaging across noise or runtime uncertainty while failing for a
> sharply specified coherent experiment; report that distinction rather than
> declaring it universally sufficient or useless.

Two questions, in order, because the second is meaningless without the first:

1. **Is there interference to fail at?** A success-versus-runtime curve with
   coherent structure is non-monotone on a fine grid. If the curve is smooth,
   nothing here can distinguish a scalar profile from anything else, and the
   honest report is that the test did not apply rather than that the profile
   passed.
2. **Does the scalar profile track it?** Its advantage over a linear ramp is
   computed at every runtime. A profile that works "after averaging" shows a
   positive mean advantage whose sign flips across the grid; one that works
   coherently keeps its sign.

The teacher's spectral profile does not depend on runtime, so it is computed
once per record and only the density-to-schedule step is repeated. That keeps a
dense scan affordable and removes solver noise between grid points.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .telemetry import _safe


def _turning_points(values: np.ndarray, tolerance: float) -> int:
    """Interior sign changes of the first difference, ignoring flat noise."""
    d = np.diff(values)
    d = np.where(np.abs(d) < tolerance, 0.0, d)
    signs = np.sign(d[d != 0])
    return int(np.sum(signs[1:] != signs[:-1])) if signs.size > 1 else 0


def scan_runtimes(record: Mapping[str, Any], schedules: Mapping[str, Any], *,
                  runtimes: Sequence[float], tolerance: float = 5e-4,
                  max_steps: int = 16384, max_slope: float = 4.0,
                  turning_tolerance: float = 1e-3) -> dict:
    """Every named schedule at every runtime, on one record.

    ``turning_tolerance`` is the smallest change counted as a direction change.
    Without it, integrator noise manufactures turning points and every curve
    looks like interference.
    """
    from .benchmarking import score_schedule
    from .schedules import Schedule

    runtimes = [float(t) for t in runtimes]
    if len(runtimes) < 3 or any(t <= 0 for t in runtimes):
        raise ValueError("need at least three positive runtimes to see structure")
    if not schedules:
        raise ValueError("scan_runtimes needs at least one named schedule")
    if not np.isfinite(turning_tolerance) or turning_tolerance <= 0:
        raise ValueError("turning_tolerance must be positive")

    # score_schedule reads the duration from the record, so the scan varies the
    # record's own runtime field on a shallow copy. Everything else -- the
    # Hamiltonian, the embedding, the observable -- is shared by reference, so
    # nothing but the duration differs between grid points.
    # Feasibility does NOT depend on runtime. score_schedule checks
    # validate_slope(runtime, max_ds_dtau / runtime) against slopes that carry a
    # 1/runtime, so the constraint reduces to ds/dtau <= max_ds_dtau -- the same
    # bound at every runtime. An earlier comment here claimed the opposite and
    # was wrong.
    #
    # Censoring is still needed, because a schedule can exceed the bound at
    # *every* runtime: a construction handed the wrong slope budget produces
    # one, and a d2 teacher built with max_slope=4 instead of 4/runtime reaches
    # ds/dtau = 15 at runtime 4. Those points are recorded as None and counted,
    # never dropped, so a curve is never silently shortened.
    #
    # `schedules` values may be a Schedule or a factory runtime -> Schedule|None.
    # The second form is how a runtime-aware construction such as the spectral
    # teacher is actually deployed: rebuilt at each runtime rather than
    # transplanted from the reference.
    #
    # The two are told apart by type, NOT by callable(): a Schedule defines
    # __call__ to evaluate s(tau), so callable() is true for both and calling a
    # Schedule as if it were a factory silently returns a float.
    curves: dict[str, list[float | None]] = {}
    infeasible: dict[str, int] = {}
    for name, entry in schedules.items():
        losses: list[float | None] = []
        skipped = 0
        for runtime in runtimes:
            schedule = entry if isinstance(entry, Schedule) else entry(runtime)
            if schedule is None:
                losses.append(None)
                skipped += 1
                continue
            scanned = dict(record)
            scanned["runtime"] = float(runtime)
            try:
                # Exactly the check score_schedule performs, so the pre-check
                # cannot pass something the scorer then rejects. An earlier
                # version compared against max_slope directly, which is the
                # ds/dt convention, and let steep schedules through at long
                # runtimes only for the scorer to raise.
                schedule.validate_slope(runtime=float(runtime),
                                        max_slope=max_slope * (1 + 1e-6) / float(runtime))
            except ValueError:
                losses.append(None)
                skipped += 1
                continue
            losses.append(float(score_schedule(scanned, schedule, tolerance=tolerance,
                                               max_steps=max_steps,
                                               max_ds_dtau=max_slope * (1 + 1e-6))["loss"]))
        curves[name] = losses
        infeasible[name] = skipped

    out: dict[str, Any] = {
        "record_id": str(np.asarray(record["record_id"]).item()),
        "parent_id": str(np.asarray(record["parent_id"]).item()),
        "n_physical": int(np.asarray(record["physical_h"]).size),
        "runtimes": runtimes, "curves": curves,
        "turning_tolerance": turning_tolerance,
    }
    out["infeasible_runtimes"] = infeasible
    for name, losses in curves.items():
        finite = np.asarray([v for v in losses if v is not None], dtype=float)
        if finite.size < 3:
            out[f"{name}_turning_points"] = None
            out[f"{name}_range"] = None
            continue
        out[f"{name}_turning_points"] = _turning_points(finite, turning_tolerance)
        out[f"{name}_range"] = float(finite.max() - finite.min())
    if "linear" in curves:
        for name, losses in curves.items():
            if name == "linear":
                continue
            # Only runtimes where both curves exist can be differenced.
            pairs = [(b, v) for b, v in zip(curves["linear"], losses)
                     if b is not None and v is not None]
            if len(pairs) < 3:
                out[f"{name}_advantage_vs_linear"] = {"status": "too_few_shared_runtimes",
                                                      "shared": len(pairs)}
                continue
            advantage = np.array([b - v for b, v in pairs], dtype=float)
            out[f"{name}_advantage_vs_linear"] = {
                "mean": float(advantage.mean()),
                "min": float(advantage.min()), "max": float(advantage.max()),
                "sign_flips": int(np.sum(np.sign(advantage[1:]) != np.sign(advantage[:-1]))),
                "positive_at": int((advantage > 0).sum()), "of": int(advantage.size),
            }
    return _safe(out)


def summarise_scans(rows: Sequence[Mapping[str, Any]], *, method: str) -> dict:
    """Did interference appear, and did the scalar profile keep its sign?"""
    rows = list(rows)
    if not rows:
        raise ValueError("summarise_scans requires a nonempty row set")
    key = f"{method}_advantage_vs_linear"
    missing = [r["record_id"] for r in rows if key not in r]
    if missing:
        raise ValueError(f"{method!r} missing from {len(missing)} scans, e.g. {missing[:3]}")

    usable = [r for r in rows if r["linear_turning_points"] is not None
              and "sign_flips" in r[key]]
    if not usable:
        return _safe({"schema_version": 1, "method": method, "n_records": len(rows),
                      "status": "no record had enough feasible runtimes to compare"})
    linear_turns = [r["linear_turning_points"] for r in usable]
    flips = [r[key]["sign_flips"] for r in usable]
    means = [r[key]["mean"] for r in usable]
    rows = usable
    return _safe({
        "schema_version": 1, "method": method, "n_records": len(rows),
        "records_with_nonmonotone_linear_curve": int(sum(1 for t in linear_turns if t > 0)),
        "mean_linear_turning_points": float(np.mean(linear_turns)),
        "records_where_the_advantage_changes_sign": int(sum(1 for f in flips if f > 0)),
        "mean_advantage": float(np.mean(means)),
        "records_with_positive_mean_advantage": int(sum(1 for m in means if m > 0)),
        "interpretation": (
            "a smooth linear curve means the scan found no interference to fail at, so "
            "the test did not apply; a sign-changing advantage means the profile works "
            "only after averaging over runtime"),
    })
