"""Does searching longer against a noiseless simulator return a worse control?"""
import numpy as np
import pytest

from annealctrl.overoptimisation import incumbent_at_budgets, overoptimisation_curve


# --- the incumbent -----------------------------------------------------------

def test_the_incumbent_is_the_best_seen_so_far():
    losses = [0.9, 0.5, 0.7, 0.3, 0.4]
    assert incumbent_at_budgets(losses, [1, 2, 3, 4, 5]) == [0, 1, 1, 3, 3]


def test_a_budget_longer_than_the_trace_is_refused():
    """Clamping would print a flat curve where the data simply ran out."""
    with pytest.raises(ValueError, match="exceeds"):
        incumbent_at_budgets([0.5, 0.4], [1, 2, 4])


def test_incumbent_refuses_a_nonpositive_budget_and_a_nonfinite_loss():
    with pytest.raises(ValueError, match="positive integers"):
        incumbent_at_budgets([0.5, 0.4], [0])
    with pytest.raises(ValueError, match="finite"):
        incumbent_at_budgets([0.5, np.nan], [1])


# --- the curve ---------------------------------------------------------------

def rows(open_shape):
    """One row per parent; `open_shape` gives the open loss at each budget."""
    budgets = [1, 2, 4, 8]
    out = []
    for i, offsets in enumerate(open_shape):
        out.append({"parent_id": f"p{i}", "record_id": f"r{i}",
                    "closed": {str(b): 0.6 - 0.02 * j for j, b in enumerate(budgets)},
                    "open": {str(b): v for b, v in zip(budgets, offsets)}})
    return out, budgets


def test_a_curve_that_keeps_improving_reports_the_largest_budget():
    r, b = rows([[0.7, 0.65, 0.6, 0.55]] * 4)
    out = overoptimisation_curve(r, budgets=b)
    assert out["budget_minimising_open_loss"] == 8
    assert out["open_loss_is_monotone_in_budget"] is True
    assert out["cost_of_the_largest_budget_against_the_best"]["mean"] == pytest.approx(0.0)


def test_a_curve_that_turns_up_is_detected_and_costed():
    # Best under noise at budget 2; more search makes it worse.
    r, b = rows([[0.70, 0.60, 0.64, 0.68]] * 5)
    out = overoptimisation_curve(r, budgets=b)
    assert out["budget_minimising_open_loss"] == 2
    assert out["open_loss_is_monotone_in_budget"] is False
    cost = out["cost_of_the_largest_budget_against_the_best"]
    assert cost["mean"] == pytest.approx(0.08)
    assert cost["worse_in_parents"] == 5


def test_the_curve_resamples_parents():
    r, b = rows([[0.70, 0.60, 0.64, 0.68]] * 6)
    out = overoptimisation_curve(r, budgets=b)
    assert out["n_parents"] == 6
    for block in out["by_budget"].values():
        assert block["open"]["parent_bootstrap_ci"]["unit_of_independence"] == "logical_parent"


def test_the_curve_refuses_an_empty_row_set():
    with pytest.raises(ValueError, match="nonempty"):
        overoptimisation_curve([], budgets=[1, 2])
