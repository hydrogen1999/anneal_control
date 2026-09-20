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
