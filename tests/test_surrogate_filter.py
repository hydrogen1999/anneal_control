"""Does the critic rank what it did not generate, and does that buy simulator budget?"""
import numpy as np
import pytest

from annealctrl.surrogate_filter import aggregate_filter, filter_curve, regrid


# --- regridding, and the error it must not hide ------------------------------

def test_regridding_a_waveform_already_on_the_grid_is_exact():
    grid = np.linspace(0.0, 1.0, 9)
    out = regrid(grid, grid**2, points=9)
    assert out["representation_error"] == pytest.approx(0.0, abs=1e-12)
    assert out["waveform"] == pytest.approx(grid**2)


def test_regridding_reports_the_error_it_introduces():
    # A knot at tau=0.5 with a sharp corner cannot survive a coarse grid intact.
    out = regrid([0.0, 0.05, 1.0], [0.0, 0.9, 1.0], points=3)
    assert out["representation_error"] > 0.1


def test_regrid_refuses_a_malformed_waveform():
    with pytest.raises(ValueError, match="nondecreasing"):
        regrid([0.0, 0.7, 0.3], [0.0, 0.5, 1.0], points=9)
    with pytest.raises(ValueError, match="equal length"):
        regrid([0.0, 1.0], [0.0], points=9)
    with pytest.raises(ValueError, match="finite"):
        regrid([0.0, np.nan], [0.0, 1.0], points=9)


# --- the budget curve --------------------------------------------------------

def test_a_perfect_critic_finds_the_best_at_every_budget():
    true = np.array([0.5, 0.3, 0.9, 0.4, 0.8, 0.35, 0.7, 0.6])
    out = filter_curve(true, true)      # prediction == truth
    for block in out["by_keep_fraction"].values():
        assert block["found_the_true_best"] is True
        assert block["shortfall_vs_full_budget"] == pytest.approx(0.0)
    assert out["rank_correlation"] == pytest.approx(1.0)


def test_an_inverted_critic_pays_for_it_at_a_small_budget():
    true = np.array([0.5, 0.3, 0.9, 0.4, 0.8, 0.35, 0.7, 0.6])
    out = filter_curve(-true, true)     # ranks worst first
    assert out["rank_correlation"] == pytest.approx(-1.0)
    assert out["by_keep_fraction"]["0.125"]["shortfall_vs_full_budget"] > 0.5
    # At full budget even a useless critic loses nothing: it simulated everything.
    assert filter_curve(-true, true, keep_fractions=[1.0]
                        )["by_keep_fraction"]["1.0"]["shortfall_vs_full_budget"] == pytest.approx(0.0)


def test_the_curve_keeps_at_least_one_candidate():
    true = np.arange(4, dtype=float)
    out = filter_curve(true, true, keep_fractions=[0.01])
    assert out["by_keep_fraction"]["0.01"]["kept"] == 1


def test_filter_curve_refuses_a_fraction_outside_the_unit_interval():
    true = np.arange(4, dtype=float)
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="keep_fractions"):
            filter_curve(true, true, keep_fractions=[bad])


# --- aggregation -------------------------------------------------------------

def rowset():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(6):
        true = rng.random(16)
        noisy = true + 0.05 * rng.standard_normal(16)
        row = filter_curve(noisy, true, keep_fractions=[0.25, 0.5])
        rows.append({**row, "record_id": f"r{i}", "parent_id": f"p{i // 2}"})
    return rows


def test_aggregation_resamples_parents_not_records():
    out = aggregate_filter(rowset(), keep_fractions=[0.25, 0.5])
    assert out["n_records"] == 6 and out["n_parents"] == 3
    for block in out["by_keep_fraction"].values():
        assert block["parent_bootstrap_ci"]["unit_of_independence"] == "logical_parent"
        assert block["n_parents"] == 3


def test_a_larger_budget_never_has_a_worse_shortfall():
    out = aggregate_filter(rowset(), keep_fractions=[0.25, 0.5])
    assert out["by_keep_fraction"]["0.5"]["mean"] <= out["by_keep_fraction"]["0.25"]["mean"]


def test_aggregation_refuses_an_empty_row_set():
    with pytest.raises(ValueError, match="nonempty"):
        aggregate_filter([], keep_fractions=[0.5])
