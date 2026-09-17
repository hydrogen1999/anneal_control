"""Original-system p-spin reference check, separate from embedded-task results.

Equations (3)-(6), Table I, and the Fig. 4(b) setting in Finzgar et al.,
Phys. Rev. Research 6, 023063 (2024). This is an independent implementation
check, not a reproduction of the paper's >=80-run statistical figure.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
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
                  n_parameters=4, zeta=2., rtol=1e-8, atol=1e-10):
    """Paired BO and uniform search with identical first ten parameter draws."""
    _positive_integer(n_parameters, "n_parameters")
    settings = dict(n=n, p=p, gamma=gamma, runtime=runtime, rtol=rtol, atol=atol)
    rows = []
    decoder = lambda x: reference_real_schedule(x, zeta=zeta)
    for seed in seeds:
        for method in ("gp_ucb", "uniform_random"):
            row = optimize_finzgar_schedule(
                lambda schedule: score_reference(schedule, **settings), budget=budget,
                n_segments=n_parameters + 1, parameterization="real", real_zeta=zeta,
                seed=seed, n_initial=10 if method == "gp_ucb" else budget,
                _schedule_decoder=decoder)
            row.update(method=method, scope="original_pspin_reference_check",
                       physical_settings=settings, n_parameters=n_parameters, zeta=zeta,
                       best_fidelity=None if row["best_loss"] is None else 1. - row["best_loss"])
            # These embedded-task control settings do not apply to the reference.
            row["settings"].pop("max_ds_dtau")
            row["settings"].pop("logit_bound")
            rows.append(row)
    return {"reference": REFERENCE, "setting": "Fig4b_N15_T3_real4_Gamma5_p3",
            "figure_reproduction": False, "scope": "bounded_independent_implementation_check",
            "physics": settings, "n_parameters": n_parameters, "zeta": zeta,
            "seeds": [int(s) for s in seeds], "budget_per_method_seed": budget,
            "total_objective_calls": sum(row["n_objective_calls"] for row in rows),
            "total_failed_queries": sum(row["n_failed_calls"] for row in rows), "runs": rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--budget", type=int, default=60)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args(argv)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve output before spending simulator calls; preserve existing results.
    with output.open("x") as stream:
        from .pipeline import source_fingerprint
        fingerprint = source_fingerprint()
        result = run_reference(seeds=args.seeds, budget=args.budget)
        result["source_fingerprint_at_start"] = fingerprint
        result["source_fingerprint_at_end"] = source_fingerprint()
        result["source_frozen"] = result["source_fingerprint_at_end"] == fingerprint
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
        if not result["source_frozen"]:
            raise RuntimeError("source changed during reference run; retained result is invalid for the frozen revision")


if __name__ == "__main__":
    main()
