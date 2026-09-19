"""Does the runtime scan detect interference, and does it refuse to invent it?"""
import numpy as np
import pytest

from annealctrl.coherent_scan import _turning_points, scan_runtimes, summarise_scans


def tiny_record(n=3):
    edges = [[i, i + 1] for i in range(n - 1)]
    return {
        "record_id": "r0", "parent_id": "p0", "runtime": 1.0, "split": "test",
        "logical_h": np.zeros(n), "logical_edges": edges, "logical_J": -np.ones(len(edges)),
        "physical_h": np.zeros(n), "physical_edges": edges, "physical_J": -np.ones(len(edges)),
        "problem_J": -np.ones(len(edges)), "chain_J": np.zeros(0),
        "membership": np.arange(n), "programmed_scale": 1.0, "chain_strength": 1.0,
        "energy_scale": 1.0, "catalyst_strength": 0.0, "logical_n": n, "physical_n": n,
    }


# --- turning points ----------------------------------------------------------

def test_a_monotone_curve_has_no_turning_points():
    assert _turning_points(np.array([0.9, 0.8, 0.7, 0.6]), 1e-3) == 0


def test_an_oscillating_curve_has_them():
    assert _turning_points(np.array([0.9, 0.7, 0.85, 0.65, 0.8]), 1e-3) == 3


def test_noise_below_tolerance_does_not_manufacture_interference():
    """Integrator jitter on a near-flat curve must not read as coherent structure.

    A steep monotone curve is safe at any tolerance because the trend dominates
    the noise; the dangerous case is a curve that has genuinely saturated, where
    every difference is noise and a careless tolerance turns it into dozens of
    turning points.
    """
    rng = np.random.default_rng(0)
    saturated = np.full(40, 0.7) + rng.normal(0, 1e-4, 40)
    assert _turning_points(saturated, 1e-3) == 0
    assert _turning_points(saturated, 1e-12) > 10   # a careless tolerance would


# --- the scan ----------------------------------------------------------------

def test_the_scan_evaluates_every_schedule_at_every_runtime():
    from annealctrl.schedules import Schedule

    out = scan_runtimes(tiny_record(), {"linear": Schedule.linear()},
                        runtimes=[1.0, 2.0, 4.0])
    assert len(out["curves"]["linear"]) == 3
    assert out["linear_range"] >= 0.0


def test_the_advantage_is_reported_against_linear_with_its_sign_flips():
    from annealctrl.schedules import Schedule

    grid = np.linspace(0.0, 1.0, 9)
    out = scan_runtimes(tiny_record(),
                        {"linear": Schedule.linear(),
                         "slow": Schedule(grid, np.sqrt(grid))},
                        runtimes=[1.0, 2.0, 4.0, 8.0])
    block = out["slow_advantage_vs_linear"]
    assert block["of"] == 4
    assert set(block) >= {"mean", "min", "max", "sign_flips", "positive_at"}


def test_the_scan_refuses_too_few_runtimes_to_see_structure():
    from annealctrl.schedules import Schedule

    with pytest.raises(ValueError, match="three positive runtimes"):
        scan_runtimes(tiny_record(), {"linear": Schedule.linear()}, runtimes=[1.0, 2.0])


# --- the summary -------------------------------------------------------------

def rows(turns, flips):
    return [{"record_id": f"r{i}", "linear_turning_points": t,
             "d2_advantage_vs_linear": {"sign_flips": f, "mean": 0.01}}
            for i, (t, f) in enumerate(zip(turns, flips))]


def test_a_smooth_population_is_reported_as_the_test_not_applying():
    out = summarise_scans(rows([0, 0, 0], [0, 0, 0]), method="d2")
    assert out["records_with_nonmonotone_linear_curve"] == 0
    assert "did not apply" in out["interpretation"]


def test_sign_changing_advantage_is_counted():
    out = summarise_scans(rows([2, 3, 0], [1, 0, 0]), method="d2")
    assert out["records_with_nonmonotone_linear_curve"] == 2
    assert out["records_where_the_advantage_changes_sign"] == 1


def test_a_missing_method_is_refused_rather_than_silently_skipped():
    with pytest.raises(ValueError, match="missing from"):
        summarise_scans(rows([0], [0]), method="gap_inverse_square")


def test_a_schedule_is_not_mistaken_for_a_runtime_factory():
    """Schedule defines __call__ to evaluate s(tau), so callable() cannot tell
    the two apart -- and calling one as a factory silently yields a float."""
    from annealctrl.schedules import Schedule

    out = scan_runtimes(tiny_record(), {"linear": Schedule.linear()},
                        runtimes=[1.0, 2.0, 4.0])
    assert all(isinstance(v, float) for v in out["curves"]["linear"])
    assert out["infeasible_runtimes"]["linear"] == 0


def test_a_runtime_factory_is_rebuilt_at_each_runtime():
    from annealctrl.schedules import Schedule

    seen = []

    def factory(runtime):
        seen.append(runtime)
        return Schedule.linear()

    scan_runtimes(tiny_record(), {"linear": Schedule.linear(), "rebuilt": factory},
                  runtimes=[1.0, 2.0, 4.0])
    assert seen == [1.0, 2.0, 4.0]


def test_an_infeasible_schedule_is_censored_at_every_runtime_and_counted():
    """Feasibility is a property of ds/dtau, not of runtime.

    score_schedule checks validate_slope(runtime, max_ds_dtau / runtime) against
    slopes carrying a 1/runtime, so the bound reduces to ds/dtau <= max_ds_dtau
    and does not move with runtime. A schedule that violates it is therefore
    censored everywhere, not just at short runtimes -- and censored rather than
    dropped, so the curve keeps its length.
    """
    from annealctrl.schedules import Schedule

    grid = np.linspace(0.0, 1.0, 9)
    steep = Schedule(grid, np.clip(np.concatenate(([0.0], np.linspace(0.85, 1.0, 8))), 0, 1))
    assert np.abs(np.diff(steep.s_knots) / np.diff(steep.tau_knots)).max() > 4.0
    out = scan_runtimes(tiny_record(), {"linear": Schedule.linear(), "steep": steep},
                        runtimes=[0.3, 1.0, 4.0, 16.0], max_slope=4.0)
    assert out["infeasible_runtimes"]["steep"] == 4, "censoring must not depend on runtime"
    assert out["curves"]["steep"] == [None, None, None, None]
    assert out["infeasible_runtimes"]["linear"] == 0
    # A curve with nothing left cannot be summarised, and says so.
    assert out["steep_turning_points"] is None
    assert out["steep_advantage_vs_linear"]["status"] == "too_few_shared_runtimes"
