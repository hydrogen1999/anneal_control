"""The share-of-headroom unit, and the denominator it is measured against.

These tests exist because a published table reported the synthetic set's share
against a 257-call frontier search and Pegasus's against the 64-candidate bank
oracle, in one column headed "headroom a 257-call search finds". The resulting
60% vs 83% gap read as a device-connectivity effect and was an artifact of the
two denominators.

So the contract here is narrow and deliberate: a share is never reported
without naming its reference, the reference's per-instance cost is READ FROM
THE ARTIFACTS rather than asserted, and a frontier reference must cover the
very parents it is the denominator for.
"""
import numpy as np
import pytest

from annealctrl.effect_size import effect_size_report


def evaluation_rows(n_parents=4, *, linear=0.70, selected=0.62, bank_best=0.60,
                    candidate_count=64):
    return [{"parent_id": f"p{i}", "record_id": f"p{i}_r0", "linear_loss": linear,
             "selected_loss": selected, "bank_best_loss": bank_best,
             "candidate_count": candidate_count}
            for i in range(n_parents)]


def frontier_rows(n_parents=4, *, linear=0.70, headroom=0.16, calls=257):
    return [{"parent_id": f"p{i}", "record_id": f"p{i}_r0", "linear_loss": linear,
             "headroom": headroom, "total_objective_calls": calls,
             "evaluated_families": ["linear", "one_window", "two_window", "eight_bin", "pause"]}
            for i in range(n_parents)]


# --- the share, against each reference -------------------------------------

def test_bank_oracle_share_is_gain_over_the_menu_ceiling():
    report = effect_size_report(evaluation_rows(), reference="bank_oracle")
    # gain 0.08, headroom 0.10
    assert report["share_of_headroom"] == pytest.approx(0.8)
    assert report["headroom"]["mean"] == pytest.approx(0.10)


def test_frontier_share_uses_the_searched_headroom_not_the_bank():
    report = effect_size_report(evaluation_rows(), reference="frontier",
                                frontier_rows=frontier_rows())
    # same gain 0.08, but the denominator is now 0.16
    assert report["share_of_headroom"] == pytest.approx(0.5)
    assert report["headroom"]["mean"] == pytest.approx(0.16)


def test_the_two_references_disagree_and_that_is_the_point():
    rows = evaluation_rows()
    bank = effect_size_report(rows, reference="bank_oracle")["share_of_headroom"]
    front = effect_size_report(rows, reference="frontier",
                               frontier_rows=frontier_rows())["share_of_headroom"]
    assert bank > front


# --- the reference must be named, and costed from the artifacts ------------

def test_every_report_carries_its_reference_and_per_instance_cost():
    bank = effect_size_report(evaluation_rows(), reference="bank_oracle")
    assert bank["reference"]["name"] == "bank_oracle"
    assert bank["reference"]["objective_calls_per_instance"] == 64

    front = effect_size_report(evaluation_rows(), reference="frontier",
                               frontier_rows=frontier_rows())
    assert front["reference"]["name"] == "frontier"
    assert front["reference"]["objective_calls_per_instance"] == 257


def test_the_cost_label_comes_from_the_rows_not_from_a_constant():
    """A 97-call sweep must never be able to describe itself as 257-call."""
    report = effect_size_report(evaluation_rows(), reference="frontier",
                                frontier_rows=frontier_rows(calls=97))
    assert report["reference"]["objective_calls_per_instance"] == 97


def test_a_frontier_of_mixed_budgets_refuses_a_single_cost_label():
    mixed = frontier_rows(4)
    mixed[0]["total_objective_calls"] = 97
    with pytest.raises(ValueError, match="single budget|mixed"):
        effect_size_report(evaluation_rows(), reference="frontier", frontier_rows=mixed)


def test_an_unnamed_reference_is_refused():
    for bad in (None, "", "search", "oracle"):
        with pytest.raises(ValueError, match="reference"):
            effect_size_report(evaluation_rows(), reference=bad)


# --- the coverage guard -----------------------------------------------------

def test_frontier_parents_must_cover_the_evaluated_parents():
    """The published error in miniature: validation-parent headroom, test-parent gain."""
    with pytest.raises(ValueError, match="does not cover|missing"):
        effect_size_report(evaluation_rows(n_parents=4), reference="frontier",
                           frontier_rows=frontier_rows(n_parents=2))


def test_extra_frontier_parents_are_dropped_not_averaged_in():
    """A frontier sweep over a superset is usable; the surplus must not shift the mean."""
    surplus = frontier_rows(4) + [{"parent_id": "pX", "record_id": "pX_r0",
                                   "linear_loss": 0.70, "headroom": 999.0,
                                   "total_objective_calls": 257,
                                   "evaluated_families": ["linear"]}]
    report = effect_size_report(evaluation_rows(4), reference="frontier",
                                frontier_rows=surplus)
    assert report["headroom"]["mean"] == pytest.approx(0.16)
    assert report["n_parents"] == 4


def test_bank_oracle_refuses_frontier_rows_it_would_silently_ignore():
    with pytest.raises(ValueError, match="frontier_rows"):
        effect_size_report(evaluation_rows(), reference="bank_oracle",
                           frontier_rows=frontier_rows())


# --- the other two units ----------------------------------------------------

def test_cohens_d_and_win_rate_are_over_parents():
    rows = []
    for i in range(6):
        rows += [{"parent_id": f"p{i}", "record_id": f"p{i}_r0", "linear_loss": 0.70,
                  "selected_loss": 0.62 + 0.001 * i, "bank_best_loss": 0.60,
                  "candidate_count": 64}]
    report = effect_size_report(rows, reference="bank_oracle")
    assert report["n_parents"] == 6
    assert report["gain_vs_linear"]["parents_won"] == 6
    assert report["gain_vs_linear"]["cohens_d"] > 0


def test_a_parent_the_method_loses_is_counted_as_a_loss():
    rows = evaluation_rows(3)
    rows[0]["selected_loss"] = 0.75      # worse than the 0.70 linear ramp
    report = effect_size_report(rows, reference="bank_oracle")
    assert report["gain_vs_linear"]["parents_won"] == 2
    assert report["gain_vs_linear"]["n_parents"] == 3


def test_the_gain_interval_is_a_parent_bootstrap():
    rows = []
    for i in range(8):
        rows.append({"parent_id": f"p{i}", "record_id": f"p{i}_r0", "linear_loss": 0.70,
                     "selected_loss": 0.60 + 0.01 * i, "bank_best_loss": 0.58,
                     "candidate_count": 64})
    report = effect_size_report(rows, reference="bank_oracle", bootstrap_resamples=500)
    ci = report["gain_vs_linear"]["parent_bootstrap_ci"]
    assert ci["unit_of_independence"] == "logical_parent"
    assert ci["low"] < report["gain_vs_linear"]["mean"] < ci["high"]


# --- refusals ---------------------------------------------------------------

def test_an_empty_evaluation_is_refused():
    with pytest.raises(ValueError, match="nonempty|no records"):
        effect_size_report([], reference="bank_oracle")


def test_a_nonpositive_headroom_cannot_produce_a_share():
    """If the reference found nothing, a percentage of it is meaningless."""
    rows = [{"parent_id": "p0", "record_id": "p0_r0", "linear_loss": 0.60,
             "selected_loss": 0.58, "bank_best_loss": 0.60, "candidate_count": 64}]
    report = effect_size_report(rows, reference="bank_oracle")
    assert report["share_of_headroom"] is None
    assert "headroom" in report["share_status"]


# --- pooling across training seeds -----------------------------------------

def test_a_report_exposes_its_per_parent_gains_so_seeds_can_be_pooled():
    report = effect_size_report(evaluation_rows(3), reference="bank_oracle")
    assert set(report["parent_gains"]) == {"p0", "p1", "p2"}
    assert report["parent_gains"]["p0"] == pytest.approx(0.08)


def test_pooling_seeds_can_win_a_parent_that_a_single_seed_loses():
    """Published win rates are pooled; the worst seed can be strictly worse."""
    from annealctrl.effect_size import pooled_across_seeds

    good = effect_size_report(evaluation_rows(2), reference="bank_oracle")
    bad_rows = evaluation_rows(2)
    bad_rows[0]["selected_loss"] = 0.74          # loses p0 by 0.04 in this seed
    bad = effect_size_report(bad_rows, reference="bank_oracle")

    assert good["gain_vs_linear"]["parents_won"] == 2
    assert bad["gain_vs_linear"]["parents_won"] == 1

    pooled = pooled_across_seeds([good, bad])
    # p0 pooled gain = (0.08 + -0.04)/2 = +0.02 > 0, so pooled wins both
    assert pooled["parents_won"] == 2
    assert pooled["worst_seed_parents_won"] == 1
    assert pooled["n_seeds"] == 2


def test_pooling_refuses_reports_built_on_different_references():
    from annealctrl.effect_size import pooled_across_seeds

    bank = effect_size_report(evaluation_rows(), reference="bank_oracle")
    front = effect_size_report(evaluation_rows(), reference="frontier",
                               frontier_rows=frontier_rows())
    with pytest.raises(ValueError, match="reference"):
        pooled_across_seeds([bank, front])


def test_pooling_refuses_reports_over_different_parent_sets():
    from annealctrl.effect_size import pooled_across_seeds

    a = effect_size_report(evaluation_rows(3), reference="bank_oracle")
    b = effect_size_report(evaluation_rows(4), reference="bank_oracle")
    with pytest.raises(ValueError, match="parent"):
        pooled_across_seeds([a, b])


# --- the factorisation ------------------------------------------------------

def test_share_of_findable_factorises_into_critic_and_menu():
    """share_vs_frontier = selector_efficiency x bank_coverage, exactly."""
    from annealctrl.effect_size import decompose_share

    rows = evaluation_rows()                       # gain 0.08, bank headroom 0.10
    bank = effect_size_report(rows, reference="bank_oracle")
    front = effect_size_report(rows, reference="frontier",
                               frontier_rows=frontier_rows())   # headroom 0.16

    d = decompose_share(bank, front)
    assert d["selector_efficiency"] == pytest.approx(0.8)      # 0.08 / 0.10
    assert d["bank_coverage"] == pytest.approx(0.625)          # 0.10 / 0.16
    assert d["share_of_findable"] == pytest.approx(0.5)        # 0.08 / 0.16
    assert d["selector_efficiency"] * d["bank_coverage"] == pytest.approx(d["share_of_findable"])


def test_the_factorisation_names_which_factor_binds():
    from annealctrl.effect_size import decompose_share

    rows = evaluation_rows()
    bank = effect_size_report(rows, reference="bank_oracle")
    # a frontier that finds far more than the bank holds: the menu is the limit
    front = effect_size_report(rows, reference="frontier",
                               frontier_rows=frontier_rows(headroom=0.40))
    assert decompose_share(bank, front)["binding_factor"] == "bank_coverage"

    # a frontier barely better than the bank: the critic is the limit
    tight = effect_size_report(rows, reference="frontier",
                               frontier_rows=frontier_rows(headroom=0.105))
    assert decompose_share(bank, tight)["binding_factor"] == "selector_efficiency"


def test_the_factorisation_refuses_reports_of_different_evaluations():
    from annealctrl.effect_size import decompose_share

    bank = effect_size_report(evaluation_rows(3), reference="bank_oracle")
    front = effect_size_report(evaluation_rows(4), reference="frontier",
                               frontier_rows=frontier_rows(4))
    with pytest.raises(ValueError, match="parent|same evaluation"):
        decompose_share(bank, front)


def test_the_factorisation_refuses_two_reports_of_the_same_reference():
    from annealctrl.effect_size import decompose_share

    a = effect_size_report(evaluation_rows(), reference="bank_oracle")
    with pytest.raises(ValueError, match="bank_oracle.*frontier|one of each"):
        decompose_share(a, a)


def test_the_factorisation_refuses_a_bank_wider_than_the_frontier():
    """A 64-candidate menu cannot out-find a search that also scored those controls."""
    from annealctrl.effect_size import decompose_share

    rows = evaluation_rows()
    bank = effect_size_report(rows, reference="bank_oracle")          # headroom 0.10
    front = effect_size_report(rows, reference="frontier",
                               frontier_rows=frontier_rows(headroom=0.05))
    with pytest.raises(ValueError, match="exceeds|wider"):
        decompose_share(bank, front)


def test_a_pooled_result_can_itself_be_decomposed():
    """Headroom is a property of the records, so pooling seeds must not change it."""
    from annealctrl.effect_size import decompose_share, pooled_across_seeds

    def seeded(selected):
        rows = evaluation_rows()
        for row in rows:
            row["selected_loss"] = selected
        return rows

    bank = [effect_size_report(seeded(s), reference="bank_oracle") for s in (0.60, 0.64)]
    front = [effect_size_report(seeded(s), reference="frontier",
                                frontier_rows=frontier_rows()) for s in (0.60, 0.64)]

    bank_pooled, front_pooled = pooled_across_seeds(bank), pooled_across_seeds(front)
    assert bank_pooled["headroom"]["mean"] == pytest.approx(0.10)
    assert front_pooled["headroom"]["mean"] == pytest.approx(0.16)

    d = decompose_share(bank_pooled, front_pooled)
    # pooled gain = 0.70 - mean(0.60, 0.64) = 0.08
    assert d["selector_efficiency"] == pytest.approx(0.8)
    assert d["bank_coverage"] == pytest.approx(0.625)
    assert d["share_of_findable"] == pytest.approx(0.5)


def test_pooling_refuses_seeds_whose_headroom_disagrees():
    """Headroom cannot depend on the training seed; if it does, the inputs are mismatched."""
    from annealctrl.effect_size import pooled_across_seeds

    a = effect_size_report(evaluation_rows(), reference="bank_oracle")
    b = effect_size_report(evaluation_rows(bank_best=0.50), reference="bank_oracle")
    with pytest.raises(ValueError, match="headroom"):
        pooled_across_seeds([a, b])


# --- budget equivalence -----------------------------------------------------

def budget_curve(points):
    """calls -> headroom, monotone non-decreasing, as a search incumbent must be."""
    return dict(points)


def test_the_equivalent_is_the_smallest_budget_that_reaches_the_gain():
    from annealctrl.effect_size import search_call_equivalent

    curves = {"p0": budget_curve([(5, 0.0), (9, 0.04), (13, 0.05), (17, 0.07)])}
    result = search_call_equivalent(curves, {"p0": 0.055}, censor_at=17)
    assert result["per_parent_calls"]["p0"] == 17        # first budget reaching 0.055
    assert result["per_parent_lower_bound"]["p0"] == 13  # last budget still short
    assert result["n_censored"] == 0


def test_a_gain_the_search_never_reaches_is_censored_not_extrapolated():
    from annealctrl.effect_size import search_call_equivalent

    curves = {"p0": budget_curve([(5, 0.0), (9, 0.04)])}
    result = search_call_equivalent(curves, {"p0": 0.99}, censor_at=9)
    assert result["n_censored"] == 1
    assert result["per_parent_calls"]["p0"] is None
    assert "censored" in result["censoring_note"]


def test_the_median_is_over_parents_with_a_bootstrap():
    from annealctrl.effect_size import search_call_equivalent

    curves = {f"p{i}": budget_curve([(5, 0.0), (9, 0.04), (17, 0.08), (33, 0.12)])
              for i in range(8)}
    gains = {f"p{i}": 0.05 + 0.001 * i for i in range(8)}
    result = search_call_equivalent(curves, gains, censor_at=33, bootstrap_resamples=500)
    assert result["n_parents"] == 8
    assert result["median_calls"] == 17
    ci = result["parent_bootstrap_ci"]
    assert ci["unit_of_independence"] == "logical_parent"


def test_a_non_monotone_curve_is_refused():
    """An incumbent trace cannot get worse; if it does the input is not an incumbent."""
    from annealctrl.effect_size import search_call_equivalent

    curves = {"p0": budget_curve([(5, 0.05), (9, 0.02)])}
    with pytest.raises(ValueError, match="monotone|incumbent"):
        search_call_equivalent(curves, {"p0": 0.01}, censor_at=9)


def test_parents_must_have_both_a_curve_and_a_gain():
    from annealctrl.effect_size import search_call_equivalent

    curves = {"p0": budget_curve([(5, 0.0), (9, 0.04)])}
    with pytest.raises(ValueError, match="p1|same parents"):
        search_call_equivalent(curves, {"p0": 0.01, "p1": 0.02}, censor_at=9)


# --- menu size --------------------------------------------------------------

def test_a_menu_of_every_candidate_reproduces_the_bank_oracle():
    from annealctrl.effect_size import menu_size_curve

    rows = [{"parent_id": "p0", "record_id": "r0", "linear_loss": 1.0,
             "candidate_losses": [0.9, 0.8, 0.7, 0.6]}]
    curve = menu_size_curve(rows, sizes=[4], draws=1, seed=0)
    assert curve["curve"]["4"] == pytest.approx(0.4)      # 1.0 - 0.6


def test_the_menu_is_one_fixed_subset_shared_by_every_record():
    """The bank is shared, so a menu of size k is one subset, not a draw per record."""
    from annealctrl.effect_size import menu_size_curve

    # Candidate 0 is best for r0, candidate 1 for r1. A per-record draw would
    # let each record pick its own favourite and overstate a size-1 menu.
    rows = [{"parent_id": "p0", "record_id": "r0", "linear_loss": 1.0,
             "candidate_losses": [0.0, 1.0]},
            {"parent_id": "p1", "record_id": "r1", "linear_loss": 1.0,
             "candidate_losses": [1.0, 0.0]}]
    curve = menu_size_curve(rows, sizes=[1], draws=200, seed=0)
    # Either fixed choice gives one record headroom 1.0 and the other 0.0.
    assert curve["curve"]["1"] == pytest.approx(0.5, abs=0.02)


def test_headroom_floors_at_zero_when_no_candidate_beats_linear():
    from annealctrl.effect_size import menu_size_curve

    rows = [{"parent_id": "p0", "record_id": "r0", "linear_loss": 0.1,
             "candidate_losses": [0.9, 0.8]}]
    assert menu_size_curve(rows, sizes=[2], draws=1, seed=0)["curve"]["2"] == pytest.approx(0.0)


def test_the_curve_is_non_decreasing_in_menu_size():
    from annealctrl.effect_size import menu_size_curve

    rng = np.random.default_rng(0)
    rows = [{"parent_id": f"p{i}", "record_id": f"r{i}", "linear_loss": 1.0,
             "candidate_losses": list(rng.random(16))} for i in range(12)]
    curve = menu_size_curve(rows, sizes=[1, 2, 4, 8, 16], draws=64, seed=0)
    values = [curve["curve"][str(k)] for k in (1, 2, 4, 8, 16)]
    assert all(b >= a - 1e-9 for a, b in zip(values, values[1:]))


def test_a_size_larger_than_the_bank_is_refused():
    from annealctrl.effect_size import menu_size_curve

    rows = [{"parent_id": "p0", "record_id": "r0", "linear_loss": 1.0,
             "candidate_losses": [0.9, 0.8]}]
    with pytest.raises(ValueError, match="larger than the bank|exceeds"):
        menu_size_curve(rows, sizes=[4], draws=1, seed=0)


def test_records_with_different_bank_sizes_are_refused():
    from annealctrl.effect_size import menu_size_curve

    rows = [{"parent_id": "p0", "record_id": "r0", "linear_loss": 1.0,
             "candidate_losses": [0.9, 0.8]},
            {"parent_id": "p1", "record_id": "r1", "linear_loss": 1.0,
             "candidate_losses": [0.9, 0.8, 0.7]}]
    with pytest.raises(ValueError, match="same bank|differ"):
        menu_size_curve(rows, sizes=[2], draws=1, seed=0)
