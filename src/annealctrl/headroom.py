"""G2: control-complexity frontier, and headroom censored against its own numerics.

The question this module exists to answer is the project's go/no-go:

    On this distribution, does instance-specific control choice buy anything
    that exceeds the resolution of the measurement itself?

``benchmark_record_controls`` already runs an equal-budget, exact-waveform search
per control family and reports a per-trial ``loss_ambiguity_indicator``. What it
does not do is compare the effect being claimed against that ambiguity. A
difference between two simulated losses, each uncertain at the same order as the
difference, is not evidence; averaged over a population it is nonetheless a
positive number, and that number would end up in a table. ``record_headroom``
therefore marks such a record ``censored_numerical`` and ``aggregate_frontier``
excludes it from headroom statistics while still counting and reporting it.

Boundaries. ``headroom`` is a **finite-budget lower bound**: a larger budget can
only lower the reference, so it is never a global control gain and never an
optimum. Families are not treated as nested except through the linear incumbent,
which every family search evaluates as its first trial; a one-window waveform is
not exactly representable in the eight-bin duration parameterisation and the two
are not placed on a common nesting ladder. See
``docs/decisions/ADR-0002-censor-headroom-against-numerical-ambiguity.md``.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .search import CONTROL_FAMILIES
from .sweeps import SweepUnit, run_sweep
from .telemetry import _safe

QUANTILES = (10, 25, 50, 75, 90)


def _finite(value, name, *, minimum=None):
    value = float(value)
    if not np.isfinite(value) or (minimum is not None and value < minimum):
        raise ValueError(f"{name} must be finite" + (f" and >= {minimum}" if minimum is not None else ""))
    return value


def _linear_trial(family_result: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    """The linear incumbent every family search evaluates as its first trial.

    Identified by evaluation index rather than candidate id, because
    ``optimize_control_family`` names it ``{family}_0000``. The waveform is
    verified so a reordered or relabelled trajectory cannot pass silently.
    """
    for trial in family_result.get("records", ()):
        if int(trial.get("evaluation_index", -1)) == 0:
            waveform = trial.get("waveform") or {}
            tau, s = waveform.get("tau_knots"), waveform.get("s_knots")
            if tau is not None and s is not None and not (
                    np.allclose(tau, [0.0, 1.0]) and np.allclose(s, [0.0, 1.0])):
                raise ValueError(f"family {name!r} trial 0 is not the linear incumbent waveform")
            return trial
    raise ValueError(f"family {name!r} has no linear incumbent trial at evaluation index 0")


def record_headroom(benchmark: Mapping[str, Any], *, ambiguity_margin: float = 1.0,
                    reference_floor: float = 0.01, record: Mapping[str, Any] | None = None) -> dict:
    """Headroom of one record, censored when it cannot exceed its own numerics.

    ``ambiguity_margin`` multiplies the summed step-doubling ambiguity of the two
    differenced losses; 1.0 is the default and lowering it below 1 must be
    justified, since it is visible in every row. ``reference_floor`` withholds the
    normalised ratio when the linear reference is so small that a ratio would
    explode (protocol §8: do not inflate improvement ratios on low-headroom tasks;
    report their prevalence instead).

    Supplying the stored ``record`` adds shared-bank statistics. Those come from
    the nine-knot resampled bank in the dataset, a *different* control set from
    the exact-waveform families searched here, and are reported under a separate
    key rather than merged.
    """
    ambiguity_margin = _finite(ambiguity_margin, "ambiguity_margin", minimum=0.0)
    reference_floor = _finite(reference_floor, "reference_floor", minimum=0.0)
    families = benchmark.get("families") or {}
    if not families:
        raise ValueError("benchmark must contain at least one evaluated control family")
    unknown = set(families) - set(CONTROL_FAMILIES)
    if unknown:
        raise ValueError(f"unknown control families: {sorted(unknown)}")

    audit_failures: list[str] = []
    linear_trials = {name: _linear_trial(result, name) for name, result in families.items()}
    linear_losses = {name: float(trial["loss"]) for name, trial in linear_trials.items()}

    # Same waveform, same Hamiltonian, same runtime: these must agree to within
    # their own integrator ambiguity. Disagreement means the scoring path is not
    # reproducible, and every downstream difference inherits that.
    hottest = max(linear_losses, key=linear_losses.get)
    coldest = min(linear_losses, key=linear_losses.get)
    spread = linear_losses[hottest] - linear_losses[coldest]
    spread_budget = ambiguity_margin * (float(linear_trials[hottest].get("loss_ambiguity_indicator", 0.0))
                                        + float(linear_trials[coldest].get("loss_ambiguity_indicator", 0.0)))
    consistent = bool(spread <= spread_budget)
    if not consistent:
        audit_failures.append("linear_reference_inconsistent")

    reference_name = "linear" if "linear" in families else coldest
    reference_trial = linear_trials[reference_name]
    linear_loss = linear_losses[reference_name]

    best_losses = {name: float(result["best_loss"]) for name, result in families.items()}
    respected = {}
    for name, result in families.items():
        ok = best_losses[name] <= linear_losses[name] + 1e-12
        respected[name] = bool(ok)
        if not ok:
            # The search evaluated a feasible linear control and then reported a
            # worse best: expressivity was traded for optimisation failure.
            audit_failures.append(f"linear_incumbent_discarded:{name}")

    best_family = min(best_losses, key=best_losses.get)
    best_found_loss = best_losses[best_family]
    best_trial = families[best_family].get("best") or {}
    headroom = linear_loss - best_found_loss

    combined = (float(reference_trial.get("loss_ambiguity_indicator", 0.0))
                + float(best_trial.get("loss_ambiguity_indicator", 0.0)))
    resolved = headroom > ambiguity_margin * combined
    low_reference = linear_loss <= max(reference_floor, ambiguity_margin * combined)

    trial_ambiguities = [float(trial.get("loss_ambiguity_indicator", 0.0))
                         for result in families.values() for trial in result.get("records", ())]

    row = {
        "schema_version": 1,
        "record_id": benchmark.get("record_id"), "parent_id": benchmark.get("parent_id"),
        "family": benchmark.get("family"), "split": benchmark.get("split"),
        "logical_n": benchmark.get("logical_n"), "physical_n": benchmark.get("physical_n"),
        "runtime": benchmark.get("runtime"), "seed": benchmark.get("seed"),
        "evaluated_families": sorted(families),
        "linear_loss": linear_loss, "linear_reference_family": reference_name,
        "linear_reference_spread": float(spread), "linear_reference_consistent": consistent,
        "best_found_loss": best_found_loss, "best_family": best_family,
        "best_candidate_id": best_trial.get("candidate_id"),
        "best_waveform": best_trial.get("waveform"),
        "headroom": float(headroom),
        "relative_headroom": None if low_reference else float(headroom / linear_loss),
        "low_headroom_reference": bool(low_reference),
        "family_best_loss": {name: best_losses[name] for name in sorted(best_losses)},
        "family_restriction_loss": {name: float(best_losses[name] - best_found_loss)
                                    for name in sorted(best_losses)},
        "incumbent_trace": {name: [float(trial.get("incumbent_loss", trial["loss"]))
                                   for trial in families[name].get("records", ())]
                            for name in sorted(families)},
        "combined_loss_ambiguity": float(combined),
        "max_trial_loss_ambiguity": max(trial_ambiguities) if trial_ambiguities else 0.0,
        "ambiguity_margin": ambiguity_margin, "reference_floor": reference_floor,
        "resolution_status": "resolved" if resolved else "censored_numerical",
        "linear_incumbent_respected": respected,
        "audit_failures": audit_failures,
        "objective_calls": {name: int(result["n_evaluations"]) for name, result in sorted(families.items())},
        "total_objective_calls": int(sum(int(result["n_evaluations"]) for result in families.values())),
        "total_integrator_steps": int(sum(int(result.get("total_integrator_steps", 0)) for result in families.values())),
        "search_seconds": float(sum(float(result.get("search_seconds", 0.0)) for result in families.values())),
        "budget_per_tunable_family": benchmark.get("budget_per_tunable_family"),
        "online_adaptation": any(bool(result.get("online_adaptation")) for result in families.values()),
        "reference_status": "best_found_within_evaluated_candidates",
        "headroom_is_global_optimum": False,
        "families_are_nested_only_through_linear": True,
    }
    if record is not None:
        losses = np.asarray(record["candidate_losses"], dtype=float)
        if losses.ndim != 1 or not len(losses) or not np.isfinite(losses).all():
            raise ValueError("stored candidate_losses must be a finite nonempty vector")
        row["bank"] = {"best_loss": float(losses.min()), "worst_loss": float(losses.max()),
                       "spread": float(losses.max() - losses.min()), "mean_loss": float(losses.mean()),
                       "n_candidates": int(losses.size),
                       "control_set": "resampled_nine_knot_shared_bank",
                       "comparable_to_exact_frontier": False}
    return _safe(row)


# --------------------------------------------------------------------------
# Sweep
# --------------------------------------------------------------------------

def _requested_calls(families: Sequence[str], budget: int) -> int:
    return sum(1 if name == "linear" else budget for name in families)


FRONTIER_DEFAULTS: dict[str, Any] = {
    "split": "validation", "families": list(CONTROL_FAMILIES), "budget": 32, "seed": 0,
    "ambiguity_margin": 1.0, "reference_floor": 0.01, "backend": "numpy",
    "tolerance": 5e-4, "initial_steps": 128, "max_steps": 8192, "max_ds_dtau": 4.0,
    "teacher_methods": [], "retain_full_trials": True, "on_error": "raise",
    "strategy": "sobol_local",
}
REPORT_DEFAULTS: dict[str, Any] = {"bootstrap_resamples": 10000, "seed": 0}


def load_frontier_config(config: Mapping[str, Any] | str | Path) -> tuple[dict, dict]:
    """Validate a frontier configuration into (sweep kwargs, report kwargs).

    ``allow_test_adaptation`` is deliberately **not** a configuration key: online
    adaptation on the test split must be a visible command-line act, not
    something a copied config file can turn on silently.
    """
    if isinstance(config, (str, Path)):
        import json as _json
        config = _json.loads(Path(config).read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise ValueError("frontier configuration must be a mapping or a path to one")
    if "allow_test_adaptation" in config:
        raise ValueError("allow_test_adaptation is a command-line flag, not a configuration key; "
                         "test-split search must be an explicit act at the call site")
    # Underscore-prefixed keys are annotations: provenance notes, scope caveats,
    # the reason a config exists. They are ignored by the sweep and kept out of
    # the settings hash. Everything else must be a known key, so a typo like
    # "budgte" is still refused rather than silently taking the default.
    unknown = {key for key in config if not str(key).startswith("_")} \
        - set(FRONTIER_DEFAULTS) - {"report"}
    if unknown:
        raise ValueError(f"unknown frontier configuration keys: {sorted(unknown)}")
    report = dict(config.get("report") or {})
    unknown_report = {key for key in report if not str(key).startswith("_")} - set(REPORT_DEFAULTS)
    if unknown_report:
        raise ValueError(f"unknown frontier report keys: {sorted(unknown_report)}")
    sweep = {**FRONTIER_DEFAULTS,
             **{k: v for k, v in config.items() if k != "report" and not str(k).startswith("_")}}
    sweep["families"] = list(sweep["families"])
    sweep["teacher_methods"] = list(sweep["teacher_methods"])
    return sweep, {**REPORT_DEFAULTS, **report}


def sweep_control_frontier(
    data_dir: str | Path, *, output: str | Path, split: str = "validation",
    families: Sequence[str] = CONTROL_FAMILIES, budget: int = 32, seed: int = 0,
    ambiguity_margin: float = 1.0, reference_floor: float = 0.01,
    allow_test_adaptation: bool = False, backend: str = "numpy",
    tolerance: float = 5e-4, initial_steps: int = 128, max_steps: int = 8192,
    max_ds_dtau: float = 4.0, teacher_methods: Sequence[str] = (),
    strategy: str = "sobol_local",
    resume: bool = False, dry_run: bool = False, on_error: str = "raise",
    retain_full_trials: bool = True, record_ids: Sequence[str] | None = None,
    shard: int = 0, shard_count: int = 1,
) -> dict:
    """Measure the control-complexity frontier over one split of a dataset.

    Test-split family search is online adaptation and must be opted into
    explicitly; the flag is recorded in every row so the resulting numbers cannot
    later be presented as zero-shot.
    """
    from .benchmarking import benchmark_record_controls
    from .experiments import source_hash
    from .pipeline import load_records

    families = tuple(families)
    if not families or len(set(families)) != len(families) or any(f not in CONTROL_FAMILIES for f in families):
        raise ValueError("families must be a nonempty unique subset of CONTROL_FAMILIES")
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        raise ValueError("budget must be a positive integer")
    if split == "test" and not allow_test_adaptation:
        raise ValueError("test-split control search is online adaptation; pass allow_test_adaptation=True "
                         "and report the result as adapted, never zero-shot")

    settings = {"split": split, "families": list(families), "budget": budget, "seed": seed,
                "ambiguity_margin": ambiguity_margin, "reference_floor": reference_floor,
                "allow_test_adaptation": allow_test_adaptation, "backend": backend,
                "tolerance": tolerance, "initial_steps": initial_steps, "max_steps": max_steps,
                "max_ds_dtau": max_ds_dtau, "teacher_methods": list(teacher_methods),
                "strategy": strategy, "retain_full_trials": retain_full_trials}

    if dry_run:
        # The plan reports the real record count, so it needs the real dataset. A
        # plan derived from a guessed count would be fiction, so say what is
        # missing instead of inventing a number.
        if not (Path(data_dir) / "manifest.json").exists():
            raise FileNotFoundError(
                f"no dataset manifest at {Path(data_dir) / 'manifest.json'}; a frontier plan counts "
                "actual records, so generate the dataset first (annealctrl generate)")
        records = load_records(data_dir, split)
        if record_ids is not None:
            records = [r for r in records if str(np.asarray(r["record_id"]).item()) in set(record_ids)]
        per_unit = _requested_calls(families, budget)
        return {"dry_run": True, "planned": len(records), "split": split,
                "requested_objective_calls_per_unit": per_unit,
                "requested_objective_calls_total": per_unit * len(records),
                "settings": settings,
                "note": "requested workload only; no control was evaluated and no output written"}

    if not (Path(data_dir) / "manifest.json").exists():
        raise FileNotFoundError(f"no dataset manifest at {Path(data_dir) / 'manifest.json'}")
    records = load_records(data_dir, split)
    if record_ids is not None:
        wanted = set(record_ids)
        records = [r for r in records if str(np.asarray(r["record_id"]).item()) in wanted]
        missing = wanted - {str(np.asarray(r["record_id"]).item()) for r in records}
        if missing:
            raise ValueError(f"record ids absent from split {split!r}: {sorted(missing)}")
    if not records:
        raise ValueError(f"split {split!r} of {data_dir} contains no records")

    root = Path(output)
    by_id = {str(np.asarray(r["record_id"]).item()): r for r in records}
    units = [SweepUnit(unit_id=key, fingerprint=str(np.asarray(value["fingerprint"]).item()),
                       payload=key) for key, value in by_id.items()]
    if shard_count > 1:
        from .sweeps import select_shard
        units = select_shard(units, shard, shard_count)

    def worker(unit: SweepUnit) -> dict:
        record = by_id[unit.payload]
        result = benchmark_record_controls(
            record, budget_per_family=budget, families=families, seed=seed,
            allow_test_adaptation=allow_test_adaptation, backend=backend, tolerance=tolerance,
            initial_steps=initial_steps, max_steps=max_steps, max_ds_dtau=max_ds_dtau,
            teacher_methods=tuple(teacher_methods), strategy=strategy)
        if retain_full_trials:
            # Reviewers need every evaluated control; rows stay streamable by
            # keeping the full trajectory beside the sweep instead of inside it.
            from .pipeline import write_json
            write_json(root / "benchmarks" / f"{unit.unit_id}.json", result)
        summary = record_headroom(result, ambiguity_margin=ambiguity_margin,
                                  reference_floor=reference_floor, record=record)
        summary["privileged_teachers"] = {name: {"status": value.get("status"),
                                                 "teacher_seconds": value.get("teacher_seconds"),
                                                 "loss": (value.get("outcome") or {}).get("loss"),
                                                 "excluded_from_equal_budget_claim": True}
                                          for name, value in (result.get("privileged_teachers") or {}).items()}
        summary["full_trials_path"] = f"benchmarks/{unit.unit_id}.json" if retain_full_trials else None
        return summary

    return run_sweep(units, worker, output=root, settings=settings, command="control-sweep",
                     source_hash=source_hash(), resume=resume, on_error=on_error,
                     extra_manifest={"data_dir": str(data_dir), "split": split,
                                     "requested_objective_calls_per_unit": _requested_calls(families, budget),
                                     "scope": "best found under a declared budget; not a global control optimum"})


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------

def _parent_means(rows: Sequence[Mapping[str, Any]], key) -> tuple[np.ndarray, list[str]]:
    buckets: dict[str, list[float]] = {}
    for row in rows:
        value = key(row)
        if value is None:
            continue
        buckets.setdefault(str(row["parent_id"]), []).append(float(value))
    parents = sorted(buckets)
    return np.array([float(np.mean(buckets[p])) for p in parents]), parents


def _describe(values: np.ndarray) -> dict:
    if not values.size:
        return {"n_parents": 0, "mean": None, "std": None, "min": None, "max": None,
                "quantiles": {f"p{q}": None for q in QUANTILES}}
    return {"n_parents": int(values.size), "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "min": float(values.min()), "max": float(values.max()),
            "quantiles": {f"p{q}": float(np.percentile(values, q)) for q in QUANTILES}}


def _bootstrap(values: np.ndarray, *, n_resamples: int, seed: int, confidence: float = 0.95) -> dict:
    """Percentile CI of the mean of parent means. Parents are the unit."""
    if values.size < 2:
        return {"low": None, "high": None, "confidence": confidence, "resamples": n_resamples,
                "unit_of_independence": "logical_parent",
                "status": "insufficient_independent_parents",
                "includes_training_seed_uncertainty": False}
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples)
    chunk = max(1, min(512, 1_000_000 // values.size))
    for start in range(0, n_resamples, chunk):
        count = min(chunk, n_resamples - start)
        means[start:start + count] = values[rng.integers(values.size, size=(count, values.size))].mean(axis=1)
    alpha = (1 - confidence) / 2
    low, high = np.quantile(means, [alpha, 1 - alpha])
    return {"low": float(low), "high": float(high), "confidence": confidence,
            "resamples": n_resamples, "unit_of_independence": "logical_parent",
            "status": "ok", "includes_training_seed_uncertainty": False}


def frontier_report(sweep_dir: str | Path, *, output: str | Path | None = None,
                    bootstrap_resamples: int = 2000, seed: int = 0) -> dict:
    """Aggregate a completed ``control-sweep`` and write JSON plus Markdown.

    Failed units stay visible in the summary: a frontier computed only over the
    instances that converged is a conditional statement, and hiding the
    exclusions would make it look unconditional.
    """
    from .pipeline import write_json
    from .sweeps import load_rows

    roots = [Path(part) for part in ([sweep_dir] if isinstance(sweep_dir, (str, Path)) else sweep_dir)]
    root = roots[0]
    rows = [row for part in roots for row in load_rows(part)]
    hashes = {row.get("settings_hash") for row in rows}
    if len(hashes) > 1:
        raise ValueError(f"refusing to aggregate shards computed under different settings: {sorted(hashes)}")
    successful = [row["result"] for row in rows if row.get("status") == "ok"]
    failed = [row for row in rows if row.get("status") == "failed"]
    # A unit that failed and later succeeded on resume is not an exclusion.
    recovered = {row["unit_id"] for row in rows if row.get("status") == "ok"}
    outstanding = [row for row in failed if row["unit_id"] not in recovered]
    if not successful:
        raise ValueError(f"{root} contains no successful rows to aggregate")

    summary = aggregate_frontier(successful, bootstrap_resamples=bootstrap_resamples, seed=seed)
    summary["failed_units"] = len(outstanding)
    summary["failed_unit_ids"] = sorted({row["unit_id"] for row in outstanding})
    summary["failure_reasons"] = dict(sorted(Counter(row.get("error_type", "unknown")
                                                     for row in outstanding).items()))
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        import json as _json
        manifest = _json.loads(manifest_path.read_text())
        summary["sweep"] = {key: manifest.get(key) for key in
                            ("command", "settings", "settings_hash", "source_hash", "status",
                             "data_dir", "requested_objective_calls_per_unit")}

    destination = Path(output) if output is not None else root / "report"
    write_json(destination / "summary.json", summary)
    (destination / "FRONTIER.md").write_text(_frontier_markdown(summary), encoding="utf-8")
    return summary


def _cell(value, digits=6):
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def _frontier_markdown(summary: Mapping[str, Any]) -> str:
    headroom = summary["headroom"]
    quantiles = headroom["quantiles"]
    lines = [
        "# Control-complexity frontier (G2)",
        "",
        "Software output of `annealctrl control-sweep`. Every quantity below is a "
        "**finite-budget best-found** reference, not a global control optimum, and the "
        "search family, seed and budget are part of its definition.",
        "",
        f"- Split: `{summary['split']}`",
        f"- Records: {summary['n_records']} over {summary['n_parents']} independent logical parents",
        f"- Censored by numerical resolution: {summary['n_censored_records']} "
        f"({summary['censored_fraction']:.1%})",
        f"- Low-headroom linear reference: {summary['low_headroom_reference_fraction']:.1%} of records",
        f"- Records with audit failures: {summary['records_with_audit_failures']}",
        f"- Failed units excluded: {summary.get('failed_units', 0)}",
        f"- Verdict: **{summary['verdict']}**",
        "",
        "## Headroom (linear loss − best found), parent-averaged",
        "",
        "| n parents | mean | p10 | p25 | p50 | p75 | p90 | max |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {headroom['n_parents']} | {_cell(headroom['mean'])} | {_cell(quantiles['p10'])} | "
        f"{_cell(quantiles['p25'])} | {_cell(quantiles['p50'])} | {_cell(quantiles['p75'])} | "
        f"{_cell(quantiles['p90'])} | {_cell(headroom['max'])} |",
        "",
    ]
    ci = headroom["parent_bootstrap_ci"]
    lines += [
        f"Parent bootstrap {ci['confidence']:.0%} CI: [{_cell(ci['low'])}, {_cell(ci['high'])}] "
        f"over {ci['resamples']} resamples, unit of independence = logical parent. "
        "This interval does **not** include training-seed uncertainty.",
        "",
        "## Family restriction loss (family best − overall best), parent-averaged",
        "",
        "| family | n parents | mean | p50 | p90 | wins |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    wins = summary.get("best_family_counts", {})
    for name, stats in summary["family_restriction_loss"].items():
        lines.append(f"| `{name}` | {stats['n_parents']} | {_cell(stats['mean'])} | "
                     f"{_cell(stats['quantiles']['p50'])} | {_cell(stats['quantiles']['p90'])} | "
                     f"{wins.get(name, 0)} |")
    strata = summary.get("by_physical_size") or {}
    if len(strata) > 1:
        lines += ["", "## Headroom by physical size", "",
                  "| physical qubits | records | censored | parents | mean | p50 | p90 | bootstrap CI |",
                  "|---:|---:|---:|---:|---:|---:|---:|---|"]
        for key in sorted(strata, key=int):
            block = strata[key]
            head, ci = block["headroom"], block["headroom"]["parent_bootstrap_ci"]
            lines.append(
                f"| {block['physical_n']} | {block['n_records']} | {block['n_censored_records']} | "
                f"{head['n_parents']} | {_cell(head['mean'])} | {_cell(head['quantiles']['p50'])} | "
                f"{_cell(head['quantiles']['p90'])} | "
                f"[{_cell(ci['low'], 4)}, {_cell(ci['high'], 4)}] |")
        lines.append("")
        lines.append("Pooling sizes would hide whether headroom survives as the system grows, "
                     "so the ladder is reported stratified and never averaged across sizes.")

    lines += [
        "",
        "## Reading this table",
        "",
        "- Censored records are excluded from every headroom statistic above and counted "
        "separately. A fully censored population means the distribution offers no control "
        "signal above the integrator's own resolution, which is a result, not a bug.",
        "- Families are nested **only** through the linear incumbent that each search "
        "evaluates first. A one-window waveform is not exactly representable in the "
        "eight-bin duration parameterisation, so the restriction losses above compare "
        "family outcomes at equal budget rather than positions on a nesting ladder.",
        "- Increasing the budget can only lower the reference, so headroom is a lower "
        "bound; it is not a global control optimum.",
        "",
        f"Total objective calls charged: {summary['total_objective_calls']}. "
        f"Total integrator steps: {summary['total_integrator_steps']}.",
        "",
    ]
    return "\n".join(lines)


def aggregate_frontier(rows: Sequence[Mapping[str, Any]], *, bootstrap_resamples: int = 2000,
                       seed: int = 0) -> dict:
    """Parent-level frontier summary with tails, censoring and audit visibility.

    Censored records are excluded from headroom statistics and counted in
    ``censored_fraction``: a population whose headroom is entirely below its own
    numerical resolution yields ``verdict='no_resolved_headroom'``, which is a
    valid and publishable outcome, not a failure of the instrument.
    """
    rows = list(rows)
    if not rows:
        raise ValueError("aggregate_frontier requires a nonempty row set")
    splits = {str(row.get("split")) for row in rows}
    if len(splits) != 1:
        raise ValueError(f"refusing to pool across splits {sorted(splits)}; aggregate each split separately")
    if isinstance(bootstrap_resamples, bool) or not isinstance(bootstrap_resamples, int) or bootstrap_resamples < 1:
        raise ValueError("bootstrap_resamples must be a positive integer")

    resolved = [row for row in rows if row.get("resolution_status") == "resolved"]
    censored = [row for row in rows if row.get("resolution_status") == "censored_numerical"]
    if len(resolved) + len(censored) != len(rows):
        raise ValueError("every row must carry resolution_status resolved or censored_numerical")

    headroom_values, headroom_parents = _parent_means(resolved, lambda row: row.get("headroom"))
    linear_values, _ = _parent_means(resolved, lambda row: row.get("linear_loss"))
    best_values, _ = _parent_means(resolved, lambda row: row.get("best_found_loss"))

    families = sorted({name for row in rows for name in (row.get("family_restriction_loss") or {})})
    restriction = {}
    for name in families:
        values, _ = _parent_means(resolved, lambda row, name=name: (row.get("family_restriction_loss") or {}).get(name))
        restriction[name] = _describe(values)

    winners = Counter(str(row.get("best_family")) for row in resolved)
    reasons = Counter(reason for row in rows for reason in (row.get("audit_failures") or []))
    inconsistent = sum(1 for row in rows if row.get("linear_reference_consistent") is False)
    low_reference = sum(1 for row in rows if row.get("low_headroom_reference"))

    sizes = sorted({int(row["physical_n"]) for row in rows if row.get("physical_n") is not None})
    by_size = {}
    for size in sizes:
        stratum = [row for row in rows if int(row.get("physical_n", -1)) == size]
        kept = [row for row in stratum if row.get("resolution_status") == "resolved"]
        values, parents = _parent_means(kept, lambda row: row.get("headroom"))
        by_size[str(size)] = {
            "physical_n": size, "n_records": len(stratum),
            "n_censored_records": sum(1 for row in stratum
                                      if row.get("resolution_status") == "censored_numerical"),
            "n_parents": len({str(row["parent_id"]) for row in stratum}),
            "headroom": {**_describe(values),
                         "parent_bootstrap_ci": _bootstrap(values, n_resamples=bootstrap_resamples,
                                                           seed=seed)},
            "best_family_counts": dict(sorted(Counter(str(row.get("best_family"))
                                                      for row in kept).items())),
        }

    summary = {
        "schema_version": 1, "split": splits.pop(),
        "n_records": len(rows),
        "n_parents": len({str(row["parent_id"]) for row in rows}),
        "n_resolved_records": len(resolved), "n_censored_records": len(censored),
        "censored_fraction": len(censored) / len(rows),
        "low_headroom_reference_fraction": low_reference / len(rows),
        "headroom": {**_describe(headroom_values),
                     "parent_bootstrap_ci": _bootstrap(headroom_values, n_resamples=bootstrap_resamples, seed=seed),
                     "parents": headroom_parents},
        "linear_loss": _describe(linear_values),
        "best_found_loss": _describe(best_values),
        "family_restriction_loss": restriction,
        # A scale-up ladder is only informative stratified: pooling sizes hides
        # whether headroom survives as the system grows.
        "by_physical_size": by_size,
        "best_family_counts": dict(sorted(winners.items())),
        "records_with_audit_failures": sum(1 for row in rows if row.get("audit_failures")),
        "audit_failure_reasons": dict(sorted(reasons.items())),
        "records_with_inconsistent_linear_reference": inconsistent,
        "total_objective_calls": int(sum(int(row.get("total_objective_calls", 0)) for row in rows)),
        "total_integrator_steps": int(sum(int(row.get("total_integrator_steps", 0)) for row in rows)),
        "verdict": "no_resolved_headroom" if not headroom_values.size else "resolved_headroom_present",
        "scope": ("finite-budget best-found reference; headroom is a lower bound on control gain "
                  "and censored rows are excluded from every headroom statistic above"),
    }
    return _safe(summary)
