"""A surrogate inside the search loop: does it spend the same budget better?

The offline study showed the critic ranks search-generated waveforms at
rho ~ 0.93. That is a prerequisite, not a saving. The saving only exists if the
filter sits inside the loop, where the budget is simulator calls and the
proposals that were never simulated cost nothing.
"""
import numpy as np
import pytest

from annealctrl.schedules import Schedule
from annealctrl.search import optimize_control_family


def counting(loss_fn):
    calls = []

    def wrapped(schedule):
        calls.append(1)
        return loss_fn(schedule)

    wrapped.calls = calls
    return wrapped


def sup_from_sqrt(schedule: Schedule) -> float:
    """A cheap deterministic objective with a clear optimum away from linear."""
    tau = np.linspace(0.0, 1.0, 65)
    return float(np.abs(schedule(tau) - np.sqrt(tau)).max())


# --- the budget contract -----------------------------------------------------

def test_the_simulator_is_called_exactly_budget_times_whatever_the_oversample():
    for oversample in (1, 4, 16):
        loss = counting(sup_from_sqrt)
        optimize_control_family(loss, "one_window", budget=12, seed=0,
                                surrogate=lambda schedules: np.zeros(len(schedules)),
                                oversample=oversample)
        assert len(loss.calls) == 12, f"oversample={oversample} changed the budget"


def test_oversample_one_reproduces_the_unfiltered_search_exactly():
    """The filter must be a no-op at oversample 1, or the baseline moved."""
    plain = optimize_control_family(sup_from_sqrt, "two_window", budget=16, seed=3)
    filtered = optimize_control_family(sup_from_sqrt, "two_window", budget=16, seed=3,
                                       surrogate=lambda s: np.zeros(len(s)), oversample=1)
    assert [r.candidate.candidate_id for r in plain.records] == \
           [r.candidate.candidate_id for r in filtered.records]
    assert [r.loss for r in plain.records] == pytest.approx([r.loss for r in filtered.records])


# --- does it actually help? --------------------------------------------------

def test_a_perfect_surrogate_beats_the_unfiltered_search_at_equal_budget():
    plain = optimize_control_family(sup_from_sqrt, "eight_bin", budget=16, seed=1)
    oracle = optimize_control_family(
        sup_from_sqrt, "eight_bin", budget=16, seed=1,
        surrogate=lambda schedules: np.array([sup_from_sqrt(s) for s in schedules]),
        oversample=8)
    assert oracle.best.loss < plain.best.loss


def test_a_useless_surrogate_does_not_crash_and_still_spends_the_budget():
    loss = counting(sup_from_sqrt)
    result = optimize_control_family(loss, "one_window", budget=10, seed=2,
                                     surrogate=lambda s: np.ones(len(s)), oversample=6)
    assert len(loss.calls) == 10
    assert result.n_evaluations == 10


def test_the_records_say_how_many_proposals_each_call_was_chosen_from():
    result = optimize_control_family(sup_from_sqrt, "one_window", budget=8, seed=0,
                                     surrogate=lambda s: np.zeros(len(s)), oversample=5)
    considered = [r.candidate.parameters.get("surrogate_considered")
                  for r in result.records[1:]]
    assert all(c == 5 for c in considered), considered
    assert result.records[0].candidate.parameters.get("surrogate_considered") is None


# --- refusals ----------------------------------------------------------------

def test_oversample_above_one_without_a_surrogate_is_refused():
    with pytest.raises(ValueError, match="surrogate"):
        optimize_control_family(sup_from_sqrt, "one_window", budget=8, oversample=4)


def test_a_nonpositive_oversample_is_refused():
    with pytest.raises(ValueError, match="oversample"):
        optimize_control_family(sup_from_sqrt, "one_window", budget=8,
                                surrogate=lambda s: np.zeros(len(s)), oversample=0)


def test_a_surrogate_returning_the_wrong_number_of_scores_is_refused():
    with pytest.raises(ValueError, match="one score per proposal"):
        optimize_control_family(sup_from_sqrt, "one_window", budget=8,
                                surrogate=lambda s: np.zeros(len(s) + 1), oversample=3)


def test_the_surrogate_is_only_wired_into_strategies_that_implement_it():
    with pytest.raises(ValueError, match="surrogate"):
        optimize_control_family(sup_from_sqrt, "one_window", budget=8, strategy="cem",
                                surrogate=lambda s: np.zeros(len(s)), oversample=3)


def test_at_oversample_one_the_sobol_stream_is_the_original_one():
    """A guard the new-against-new comparison above cannot give.

    Both arms of `test_oversample_one_reproduces_the_unfiltered_search_exactly`
    run the same code, so a change to the proposal stream itself would pass it.
    This pins the stream against its definition instead. An earlier version
    advanced a cursor only on exploratory steps, which moved the baseline.
    """
    from scipy.stats import qmc

    budget, dimension, seed = 12, 3, 7
    expected = qmc.Sobol(dimension, scramble=True, seed=seed).random_base2(
        int(np.ceil(np.log2(budget - 1))))[:budget - 1]
    result = optimize_control_family(sup_from_sqrt, "one_window", budget=budget, seed=seed,
                                     surrogate=lambda s: np.zeros(len(s)), oversample=1)
    for record in result.records[1:]:
        if record.candidate.parameters.get("proposal") != "sobol":
            continue
        index = int(record.candidate.candidate_id.rsplit("_", 1)[1])
        assert record.candidate.parameters["unit_parameters"] == \
            pytest.approx(expected[index - 1].tolist())


def test_a_surrogate_may_rank_by_any_rule_including_rejecting_a_subset():
    """The reject-worst arm returns inf for rejected proposals; that must work.

    It is how the study separates 'the critic avoids the bad tail' from 'the
    critic ranks finely', so a surrogate that scores some proposals as
    unacceptable has to be a legal surrogate rather than a special case.
    """
    def reject_all_but_last(schedules):
        scores = np.full(len(schedules), np.inf)
        scores[-1] = 0.0
        return scores

    loss = counting(sup_from_sqrt)
    result = optimize_control_family(loss, "one_window", budget=10, seed=0,
                                     surrogate=reject_all_but_last, oversample=4)
    assert len(loss.calls) == 10
    assert all(r.candidate.parameters.get("surrogate_pick") == 3
               for r in result.records[1:])


def test_a_surrogate_scoring_everything_unacceptable_is_refused():
    """All-inf is not a choice, and silently taking argmin of it would hide that."""
    with pytest.raises(ValueError, match="rejected every proposal"):
        optimize_control_family(sup_from_sqrt, "one_window", budget=8, seed=0,
                                surrogate=lambda s: np.full(len(s), np.inf), oversample=4)


def test_nan_and_negative_infinity_are_still_refused():
    """A rejection is +inf. NaN is undefined and -inf wins without ranking."""
    for bad in (np.nan, -np.inf):
        with pytest.raises(ValueError, match="not rankings"):
            optimize_control_family(sup_from_sqrt, "one_window", budget=8, seed=0,
                                    surrogate=lambda s, b=bad: np.array(
                                        [b] + [1.0] * (len(s) - 1)), oversample=4)
