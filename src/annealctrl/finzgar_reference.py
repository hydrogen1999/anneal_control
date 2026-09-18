"""Original-system p-spin reference check, separate from embedded-task results.

Equations (3)-(6), Table I, and the Fig. 4(b) setting in Finzgar et al.,
Phys. Rev. Research 6, 023063 (2024). This is an independent implementation
check, not a reproduction of the paper's >=80-run statistical figure.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from math import comb
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.integrate import solve_ivp

from .literature_baselines import REFERENCE, _positive_integer, optimize_finzgar_schedule


@dataclass(frozen=True)
class ReferenceSchedule:
    """Real-space control with endpoint conditions; nonmonotonicity is allowed.

    This type is intentionally separate from the embedded task's Schedule.
    It must not be sent to that task's forward-monotone hardware adapter.
    """
    tau_knots: np.ndarray
    s_knots: np.ndarray

    def __post_init__(self):
        tau, values = np.array(self.tau_knots, float), np.array(self.s_knots, float)
        if (tau.ndim != 1 or len(tau) < 2 or values.shape != tau.shape or
                not np.isfinite(tau).all() or not np.isfinite(values).all() or
                tau[0] != 0 or tau[-1] != 1 or values[0] != 0 or values[-1] != 1 or
                np.any(np.diff(tau) <= 0)):
            raise ValueError("finite equal-length knots with fixed endpoints and increasing time required")
        tau.setflags(write=False)
        values.setflags(write=False)
        object.__setattr__(self, "tau_knots", tau)
        object.__setattr__(self, "s_knots", values)

    def __call__(self, tau):
        return np.interp(tau, self.tau_knots, self.s_knots)

    def to_dict(self):
        return {"tau_knots": self.tau_knots.tolist(), "s_knots": self.s_knots.tolist()}


def reference_real_schedule(parameters, *, zeta=2.):
    x = np.asarray(parameters, float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all() or np.any((x < 0) | (x > 1)):
        raise ValueError("finite nonempty unit-box parameters required")
    if not np.isfinite(zeta) or zeta <= 0:
        raise ValueError("positive finite zeta required")
    segments = len(x) + 1
    values = (np.arange(1, segments) + zeta * (2 * x - 1)) / segments
    return ReferenceSchedule(np.linspace(0., 1., segments + 1), np.r_[0., values, 1.])


def symmetric_pspin(n=15, p=3, gamma=5.):
    """Dicke basis |k down spins>, with Pauli sums (not spin-half sums)."""
    _positive_integer(n, "n")
    _positive_integer(p, "p")
    if p % 2 != 1 or not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("odd p and positive finite gamma required")
    k = np.arange(n + 1)
    target = -n * ((n - 2 * k) / n) ** p
    coupling = -gamma * np.sqrt((np.arange(n) + 1) * (n - np.arange(n)))
    driver = np.diag(coupling, 1) + np.diag(coupling, -1)
    initial = np.sqrt(np.array([comb(n, int(j)) for j in k], float) / float(2 ** n))
    return driver, np.diag(target), initial.astype(complex)


def _evolve(schedule, runtime, driver, target, initial, *, rtol, atol):
    state = initial.copy()
    evaluations = 0
    for left, right in zip(schedule.tau_knots[:-1], schedule.tau_knots[1:]):
        def rhs(tau, vector):
            u = float(schedule(tau))
            return -1j * runtime * ((1. - u) * (driver @ vector) + u * (target @ vector))
        solution = solve_ivp(rhs, (left, right), state, method="DOP853", rtol=rtol, atol=atol)
        evaluations += solution.nfev
        if not solution.success:
            raise ArithmeticError("DOP853 failed in p-spin reference evolution")
        state = solution.y[:, -1]
    return state, evaluations


def score_reference(schedule, *, n=15, p=3, gamma=5., runtime=3., rtol=1e-8,
                    atol=1e-10, state_tolerance=5e-7, return_state=False):
    """Tighten integration tolerance tenfold for every charged query."""
    if any(not np.isfinite(v) or v <= 0 for v in (runtime, rtol, atol, state_tolerance)):
        raise ValueError("runtime and numerical tolerances must be positive and finite")
    driver, target, initial = symmetric_pspin(n, p, gamma)
    started = perf_counter()
    coarse, coarse_calls = _evolve(schedule, runtime, driver, target, initial, rtol=rtol, atol=atol)
    fine, fine_calls = _evolve(schedule, runtime, driver, target, initial, rtol=rtol / 10, atol=atol / 10)
    diagnostic = float(np.linalg.norm(fine - coarse))
    norm_error = float(abs(np.vdot(fine, fine).real - 1.))
    if not np.isfinite(diagnostic + norm_error) or diagnostic > state_tolerance or norm_error > 1e-6:
        raise ArithmeticError("reference integration failed numerical gate")
    fidelity = float(abs(fine[0]) ** 2)
    if not 0 <= fidelity <= 1 + 1e-6:
        raise ArithmeticError("reference fidelity outside numerical tolerance")
    fidelity = float(np.clip(fidelity, 0., 1.))
    result = {"loss": 1. - fidelity, "fidelity": fidelity, "norm_error": norm_error,
              "state_error_diagnostic": diagnostic, "uncertainty_is_certificate": False,
              "rhs_evaluations": coarse_calls + fine_calls, "trajectory_calls": 2,
              "wall_seconds": perf_counter() - started}
    if return_state:
        result["state"] = fine
    return result


def run_reference(*, seeds=(0, 1, 2), budget=60, n=15, p=3, gamma=5., runtime=3.,
                  n_parameters=4, zeta=2., rtol=1e-8, atol=1e-10,
                  state_tolerance=5e-7, bootstrap_resamples=2000, event_callback=None):
    """Paired BO and uniform search with identical first ten parameter draws."""
    _positive_integer(n_parameters, "n_parameters")
    _positive_integer(budget, "budget")
    _positive_integer(bootstrap_resamples, "bootstrap_resamples")
    seeds = tuple(seeds)
    if (not seeds or any(isinstance(s, bool) or not isinstance(s, (int, np.integer)) or s < 0 for s in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("seeds must be unique nonnegative integers")
    symmetric_pspin(n, p, gamma)
    if any(not np.isfinite(x) or x <= 0 for x in (runtime, rtol, atol, state_tolerance)):
        raise ValueError("runtime and numerical tolerances must be positive and finite")
    settings = dict(n=n, p=p, gamma=gamma, runtime=runtime, rtol=rtol, atol=atol,
                    state_tolerance=state_tolerance)
    rows = []
    decoder = lambda x: reference_real_schedule(x, zeta=zeta)
    decoder(np.full(n_parameters, .5))
    for seed in seeds:
        for method in ("gp_ucb", "uniform_random"):
            row = optimize_finzgar_schedule(
                lambda schedule: score_reference(schedule, **settings), budget=budget,
                n_segments=n_parameters + 1, parameterization="real", real_zeta=zeta,
                seed=seed, n_initial=10, search_method=method,
                event_callback=(None if event_callback is None else
                                lambda event: event_callback({"seed": int(seed), "method": method, **event})),
                _schedule_decoder=decoder)
            row.update(method=method, scope="original_pspin_reference_check",
                       physical_settings=settings, n_parameters=n_parameters, zeta=zeta,
                       best_fidelity=None if row["best_loss"] is None else 1. - row["best_loss"])
            # These embedded-task control settings do not apply to the reference.
            row["settings"].pop("max_ds_dtau")
            row["settings"].pop("logit_bound")
            rows.append(row)
    paired_differences = []
    for index in range(0, len(rows), 2):
        bo, random = rows[index:index + 2]
        initial = min(10, budget)
        if ([r["parameters"] for r in bo["history"][:initial]] !=
                [r["parameters"] for r in random["history"][:initial]]):
            raise RuntimeError("original-reference paired initialization diverged")
        paired_differences.append(None if bo["best_fidelity"] is None or random["best_fidelity"] is None
                                  else bo["best_fidelity"] - random["best_fidelity"])
    summary = {"difference_definition": "best fidelity GP-UCB minus uniform random; positive favors GP-UCB",
               "per_seed_difference": dict(zip(map(str, seeds), paired_differences)),
               "mean_difference": None, "ci_low": None, "ci_high": None,
               "scope": "descriptive optimizer-seed bootstrap on one fixed Hamiltonian; not a population generalization claim",
               "all_runs_resolved": all(x is not None for x in paired_differences)}
    if summary["all_runs_resolved"]:
        differences = np.asarray(paired_differences)
        summary["mean_difference"] = float(differences.mean())
        summary["gp_ucb_wins"] = int((differences > 0).sum())
        if len(seeds) >= 2:
            rng = np.random.default_rng(0)
            samples = np.array([rng.choice(differences, len(seeds)).mean() for _ in range(bootstrap_resamples)])
            summary["ci_low"], summary["ci_high"] = map(float, np.quantile(samples, [.025, .975]))
    summary["method_distributions"] = {}
    for method in ("gp_ucb", "uniform_random"):
        values = [r["best_fidelity"] for r in rows if r["method"] == method and r["best_fidelity"] is not None]
        summary["method_distributions"][method] = {
            "n_resolved": len(values), "n_requested": len(seeds),
            "quartile_25_median_quartile_75": None if not values else np.quantile(values, [.25, .5, .75]).tolist(),
            "scope": "resolved runs only; unresolved counts are reported explicitly"}
    return {"reference": REFERENCE,
            "setting": ("Fig4b_N15_T3_real4_Gamma5_p3" if (n, p, gamma, runtime, n_parameters, zeta) == (15, 3, 5., 3., 4, 2.)
                        else "custom_original_model_setting"),
            "figure_reproduction": False, "scope": "bounded_independent_implementation_check",
            "physics": settings, "n_parameters": n_parameters, "zeta": zeta,
            "seeds": [int(s) for s in seeds], "budget_per_method_seed": budget,
            "total_objective_calls": sum(row["n_objective_calls"] for row in rows),
            "total_failed_queries": sum(row["n_failed_calls"] for row in rows), "runs": rows,
            "paired_summary": summary,
            "cost": {"known_trajectory_calls": sum((q.get("metrics") or {}).get("trajectory_calls", 0)
                                                   for row in rows for q in row["history"]),
                     "known_rhs_evaluations": sum((q.get("metrics") or {}).get("rhs_evaluations", 0)
                                                  for row in rows for q in row["history"]),
                     "inner_cost_is_lower_bound": any(row["n_failed_calls"] for row in rows)}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", help="Optional JSON run_reference settings")
    parser.add_argument("--budget", type=int)
    parser.add_argument("--seeds", nargs="+", type=int)
    args = parser.parse_args(argv)
    settings = {} if args.config is None else json.loads(Path(args.config).read_text())
    settings = {key: value for key, value in settings.items() if not key.startswith("_")}
    allowed = {"seeds", "budget", "n", "p", "gamma", "runtime", "n_parameters", "zeta",
               "rtol", "atol", "state_tolerance", "bootstrap_resamples"}
    if set(settings) - allowed:
        raise ValueError("unknown original-reference configuration key")
    if args.budget is not None:
        settings["budget"] = args.budget
    if args.seeds is not None:
        settings["seeds"] = args.seeds
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve output before spending simulator calls; preserve existing results.
    ledger = output.with_suffix(output.suffix + ".queries.jsonl")
    if output.exists() or ledger.exists():
        raise FileExistsError("reference output or query ledger already exists")
    with output.open("x") as stream, ledger.open("x") as events:
        from .pipeline import source_fingerprint
        fingerprint = source_fingerprint()
        events.write(json.dumps({"event": "experiment_started", "settings": settings,
                                 "reference": REFERENCE, "source_fingerprint": fingerprint}, allow_nan=False) + "\n")
        events.flush()
        os.fsync(events.fileno())
        def receipt(event):
            if event["event"] == "query_started" and source_fingerprint() != fingerprint:
                raise RuntimeError("source changed during original-reference run")
            events.write(json.dumps(event, allow_nan=False) + "\n")
            events.flush()
            os.fsync(events.fileno())
        try:
            result = run_reference(event_callback=receipt, **settings)
        except BaseException as error:
            stream.write(json.dumps({"status": "interrupted_or_failed", "settings": settings,
                                     "source_fingerprint_at_start": fingerprint,
                                     "source_fingerprint_at_end": source_fingerprint(),
                                     "error_type": type(error).__name__, "error": str(error),
                                     "query_ledger": ledger.name}, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            raise
        result["source_fingerprint_at_start"] = fingerprint
        result["source_fingerprint_at_end"] = source_fingerprint()
        result["source_frozen"] = result["source_fingerprint_at_end"] == fingerprint
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
        if not result["source_frozen"]:
            raise RuntimeError("source changed during reference run; retained result is invalid for the frozen revision")


if __name__ == "__main__":
    main()
