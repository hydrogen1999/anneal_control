"""G3: paired embedding interventions with exactly one declared factor changed."""
import json

import numpy as np
import pytest

from annealctrl.generation import IsingProblem
from annealctrl.interventions import (
    FACTORS,
    aggregate_interventions,
    build_pairs,
    cross_control_matrix,
)


def problem():
    return IsingProblem(np.array([0.4, -0.3, 0.2]), np.array([[0, 1], [1, 2], [0, 2]]),
                        np.array([0.9, -0.6, 0.5]))


BASE = {"shape": "path", "ports": 1, "field_distribution": "uniform",
        "coupling_distribution": "uniform", "chain_strength": 1.5}


def spec(factor, change):
    return {"factor": factor, "base": dict(BASE), "change": change}


# Chain 0 needs at least three members for "path" and "star" to differ at all:
# on a two-member chain both shapes give the single edge (0,1), and build_pairs
# correctly refuses that as a non-intervention.
LENGTHS = np.array([3, 2, 1])


def pairs(factor, change, **kwargs):
    return build_pairs(problem(), spec(factor, change), lengths=LENGTHS,
                       runtime=2.0, seed=5, **kwargs)


def boundary(arm):
    owner = arm.compiled.embedding.membership
    edges = arm.compiled.embedding.hardware_edges
    return {tuple(edge) for edge in edges.tolist() if owner[edge[0]] != owner[edge[1]]}


def internal(arm):
    owner = arm.compiled.embedding.membership
    edges = arm.compiled.embedding.hardware_edges
    return {tuple(edge) for edge in edges.tolist() if owner[edge[0]] == owner[edge[1]]}


# --- pair construction: exactly one factor -----------------------------------

def test_a_change_touching_two_factors_is_refused():
    with pytest.raises(ValueError, match="exactly one"):
        pairs("geometry", {"shape": "star", "ports": 2})


def test_a_change_key_that_does_not_match_the_declared_factor_is_refused():
    with pytest.raises(ValueError, match="factor 'geometry'"):
        pairs("geometry", {"ports": 2})


def test_an_unknown_factor_is_refused():
    with pytest.raises(ValueError, match="factor"):
        pairs("vibes", {"shape": "star"})


def test_a_change_that_changes_nothing_is_refused():
    with pytest.raises(ValueError, match="differ"):
        pairs("geometry", {"shape": "path"})


def test_every_declared_factor_can_be_built():
    assert set(FACTORS) == {"geometry", "ports", "field_allocation", "chain_strength", "chain_length"}


# --- what is actually held fixed ---------------------------------------------

def test_geometry_changes_intra_chain_edges_and_leaves_boundary_ports_alone():
    a, b = pairs("geometry", {"shape": "star"})[0].arms
    assert boundary(a) == boundary(b), "changing chain shape must not move the ports"
    assert internal(a) != internal(b)
    assert np.array_equal(a.compiled.embedding.membership, b.compiled.embedding.membership)


def test_ports_changes_boundary_edges_and_leaves_chain_shape_alone():
    a, b = pairs("ports", {"ports": 2})[0].arms
    assert internal(a) == internal(b), "changing port count must not reshape the chains"
    assert boundary(a) != boundary(b)


def test_field_allocation_leaves_the_physical_graph_identical():
    a, b = pairs("field_allocation", {"field_distribution": "concentrated"})[0].arms
    assert np.array_equal(a.compiled.embedding.hardware_edges, b.compiled.embedding.hardware_edges)
    assert not np.allclose(a.compiled.physical.h, b.compiled.physical.h)


def test_chain_strength_leaves_the_physical_graph_identical():
    a, b = pairs("chain_strength", {"chain_strength": 3.0})[0].arms
    assert np.array_equal(a.compiled.embedding.hardware_edges, b.compiled.embedding.hardware_edges)
    assert a.compiled.chain_strength == 1.5 and b.compiled.chain_strength == 3.0


def test_the_logical_objective_and_runtime_are_identical_on_both_arms():
    pair = pairs("geometry", {"shape": "star"})[0]
    a, b = pair.arms
    assert np.allclose(a.compiled.logical.h, b.compiled.logical.h)
    assert np.allclose(a.compiled.logical.J, b.compiled.logical.J)
    assert pair.runtime == 2.0
    assert "runtime" in pair.held_fixed and "logical_coefficients" in pair.held_fixed


def test_matched_size_factors_report_physical_size_matched():
    for factor, change in [("geometry", {"shape": "star"}), ("ports", {"ports": 2}),
                           ("field_allocation", {"field_distribution": "concentrated"}),
                           ("chain_strength", {"chain_strength": 3.0})]:
        for pair in pairs(factor, change):
            assert pair.physical_size_matched is True
            assert pair.declared_size_change is False


# --- size-changing factors must be declared ----------------------------------

def test_a_size_changing_factor_is_refused_unless_declared():
    with pytest.raises(ValueError, match="allow_size_change"):
        pairs("chain_length", {"chain_lengths": [3, 2, 2]})


def test_a_declared_size_change_is_built_and_labelled():
    pair = pairs("chain_length", {"chain_lengths": [3, 2, 2]}, allow_size_change=True)[0]
    a, b = pair.arms
    assert a.compiled.physical.n == 6 and b.compiled.physical.n == 7
    assert pair.physical_size_matched is False
    assert pair.declared_size_change is True


# --- the scale-controlled arm (ADR-0003) -------------------------------------

def test_a_chain_strength_intervention_always_emits_both_scale_arms():
    built = pairs("chain_strength", {"chain_strength": 3.0})
    assert [pair.scale_arm for pair in built] == ["total_compiled_effect", "scale_controlled"]


def test_the_scale_controlled_arm_gives_both_sides_one_common_scale():
    total, controlled = pairs("chain_strength", {"chain_strength": 3.0})
    a, b = total.arms
    assert a.compiled.programmed_scale != pytest.approx(b.compiled.programmed_scale), \
        "chain strength is expected to move the natural scale; that is why the second arm exists"
    ca, cb = controlled.arms
    assert ca.compiled.programmed_scale == pytest.approx(cb.compiled.programmed_scale)
    assert ca.compiled.programmed_scale == pytest.approx(min(a.compiled.programmed_scale,
                                                             b.compiled.programmed_scale))


def test_the_scale_controlled_arm_still_satisfies_the_compilation_identity():
    from annealctrl.generation import validate_compilation
    _, controlled = pairs("chain_strength", {"chain_strength": 3.0})
    for arm in controlled.arms:
        assert validate_compilation(arm.compiled)["aligned_energy_max_error"] < 1e-9


def test_a_factor_that_moves_the_scale_also_emits_a_scale_controlled_arm():
    # A logical coupling above the J cap is split across two ports, so the port
    # intervention moves the natural scale and is confounded without a control.
    strong = IsingProblem(np.array([0.1, -0.1, 0.1]), np.array([[0, 1], [1, 2], [0, 2]]),
                          np.array([1.8, -0.4, 0.3]))
    built = build_pairs(strong, spec("ports", {"ports": 2}), lengths=np.array([2, 2, 1]),
                        runtime=2.0, seed=5)
    assert [pair.scale_arm for pair in built] == ["total_compiled_effect", "scale_controlled"]
    assert built[0].scale_moved is True


def test_a_factor_that_does_not_move_the_scale_emits_one_arm_only():
    built = pairs("geometry", {"shape": "star"})
    assert [pair.scale_arm for pair in built] == ["total_compiled_effect"]
    assert built[0].scale_moved is False


def test_the_driver_and_runtime_are_never_rescaled_by_the_scale_control():
    _, controlled = pairs("chain_strength", {"chain_strength": 3.0})
    assert controlled.runtime == 2.0
    assert controlled.metadata["scale_applies_to"] == "H_Z_only"


# --- cross-control matrix ----------------------------------------------------

MATRIX_KWARGS = dict(families=("linear", "one_window"), budget=4, seed=0,
                     tolerance=5e-3, initial_steps=16, max_steps=512)


def test_cross_control_matrix_diagonal_is_each_arm_own_best_found_loss():
    pair = pairs("geometry", {"shape": "star"})[0]
    result = cross_control_matrix(pair, **MATRIX_KWARGS)
    matrix = result["loss_matrix"]
    assert matrix["A_on_A"] == pytest.approx(result["arm_A"]["best_found_loss"])
    assert matrix["B_on_B"] == pytest.approx(result["arm_B"]["best_found_loss"])
    assert matrix["A_on_B"] >= matrix["B_on_B"] - 1e-9


def test_a_negative_transfer_penalty_survives_unclipped():
    # The imported control beat this arm's own best found: the equal-budget search
    # on A was the weaker of the two, and clipping this to zero would hide it.
    from annealctrl.interventions import transfer_penalties
    penalty_a, penalty_b = transfer_penalties({"A_on_A": 0.50, "B_on_A": 0.42,
                                               "B_on_B": 0.30, "A_on_B": 0.35})
    assert penalty_a == pytest.approx(-0.08)
    assert penalty_b == pytest.approx(0.05)


def test_reported_penalties_equal_the_raw_matrix_difference():
    pair = pairs("chain_strength", {"chain_strength": 3.0})[0]
    result = cross_control_matrix(pair, **MATRIX_KWARGS)
    assert result["transfer_penalty_on_A"] == pytest.approx(
        result["loss_matrix"]["B_on_A"] - result["loss_matrix"]["A_on_A"])
    assert result["transfer_penalty_on_B"] == pytest.approx(
        result["loss_matrix"]["A_on_B"] - result["loss_matrix"]["B_on_B"])
    assert result["penalties_are_signed_and_unclipped"] is True


def test_an_identical_selected_waveform_can_never_count_as_a_swap():
    pair = pairs("geometry", {"shape": "star"})[0]
    result = cross_control_matrix(pair, families=("linear",), budget=1, seed=0,
                                  tolerance=5e-3, initial_steps=16, max_steps=512)
    assert result["selected_waveform_identical"] is True
    assert result["preferred_control_swapped"] is False
    assert result["resolution_status"] == "censored_numerical"


def test_a_swap_requires_both_directions_to_exceed_their_ambiguity():
    pair = pairs("chain_strength", {"chain_strength": 3.0})[0]
    result = cross_control_matrix(pair, **MATRIX_KWARGS)
    decisive = result["decisive_on_A"] and result["decisive_on_B"]
    assert result["preferred_control_swapped"] == (decisive and not result["selected_waveform_identical"])
    assert result["resolution_status"] in {"resolved", "one_sided", "censored_numerical"}


def test_cross_control_matrix_charges_every_objective_call():
    pair = pairs("geometry", {"shape": "star"})[0]
    result = cross_control_matrix(pair, **MATRIX_KWARGS)
    # linear is parameter-free (1 call), one_window gets the budget, on each arm,
    # plus the two cross executions.
    assert result["objective_calls"] == 2 * (1 + 4) + 2


def test_cross_control_matrix_records_the_selected_family_on_each_arm():
    pair = pairs("geometry", {"shape": "star"})[0]
    result = cross_control_matrix(pair, **MATRIX_KWARGS)
    assert result["arm_A"]["best_family"] in MATRIX_KWARGS["families"]
    assert result["arm_B"]["best_family"] in MATRIX_KWARGS["families"]
    assert result["factor"] == "geometry" and result["scale_arm"] == "total_compiled_effect"


def test_cross_control_matrix_output_is_json_safe():
    pair = pairs("ports", {"ports": 2})[0]
    json.dumps(cross_control_matrix(pair, **MATRIX_KWARGS), allow_nan=False)


def test_cross_control_matrix_refuses_a_budget_below_one():
    pair = pairs("geometry", {"shape": "star"})[0]
    with pytest.raises(ValueError, match="budget"):
        cross_control_matrix(pair, families=("linear",), budget=0, seed=0)


# --- aggregation -------------------------------------------------------------

def intervention_row(parent, penalty_a, penalty_b, *, factor="geometry",
                     scale_arm="total_compiled_effect", status="resolved", swapped=True):
    return {"pair_id": f"{parent}_{factor}", "parent_id": parent, "factor": factor,
            "scale_arm": scale_arm, "runtime": 2.0,
            "loss_matrix": {"A_on_A": 0.5, "A_on_B": 0.5 + penalty_b,
                            "B_on_A": 0.5 + penalty_a, "B_on_B": 0.5},
            "transfer_penalty_on_A": penalty_a, "transfer_penalty_on_B": penalty_b,
            "mean_transfer_penalty": (penalty_a + penalty_b) / 2,
            "resolution_status": status, "preferred_control_swapped": swapped,
            "selected_waveform_identical": False, "objective_calls": 12,
            "physical_size_matched": True, "scale_moved": False}


def test_aggregate_reports_the_reversal_rate_against_every_pair():
    rows = [intervention_row("p0", 0.10, 0.12),
            intervention_row("p1", 0.08, 0.09),
            intervention_row("p2", 1e-9, 1e-9, status="censored_numerical", swapped=False)]
    summary = aggregate_interventions(rows)
    assert summary["censored_fraction"] == pytest.approx(1 / 3)
    assert summary["decisive_reversals"] == 2
    assert summary["decisive_reversal_rate"] == pytest.approx(2 / 3)
    assert summary["decisive_reversal_denominator"] == 3


def test_aggregate_separates_scale_arms_and_never_pools_them():
    rows = [intervention_row("p0", 0.20, 0.20, factor="chain_strength"),
            intervention_row("p0", 0.05, 0.05, factor="chain_strength", scale_arm="scale_controlled"),
            intervention_row("p1", 0.30, 0.30, factor="chain_strength"),
            intervention_row("p1", 0.06, 0.06, factor="chain_strength", scale_arm="scale_controlled")]
    summary = aggregate_interventions(rows)
    by_arm = summary["by_scale_arm"]
    assert by_arm["total_compiled_effect"]["transfer_penalty"]["mean"] == pytest.approx(0.25)
    assert by_arm["scale_controlled"]["transfer_penalty"]["mean"] == pytest.approx(0.055)
    assert "transfer_penalty" not in summary, "a pooled penalty across scale arms would be meaningless"


def test_aggregate_breaks_down_by_factor():
    rows = [intervention_row("p0", 0.10, 0.10, factor="geometry"),
            intervention_row("p1", 0.20, 0.20, factor="ports")]
    summary = aggregate_interventions(rows)
    assert set(summary["by_factor"]) == {"geometry", "ports"}
    assert summary["by_factor"]["ports"]["transfer_penalty"]["mean"] == pytest.approx(0.2)


def test_aggregate_bootstraps_the_transfer_penalty_over_parents():
    rows = [intervention_row(f"p{i}", v, v) for i, v in enumerate([0.05, 0.1, 0.15, 0.2, 0.25])]
    summary = aggregate_interventions(rows, bootstrap_resamples=200, seed=0)
    ci = summary["by_scale_arm"]["total_compiled_effect"]["transfer_penalty"]["parent_bootstrap_ci"]
    assert ci["unit_of_independence"] == "logical_parent" and ci["resamples"] == 200


def test_aggregate_of_a_fully_censored_population_says_no_swap_measured():
    rows = [intervention_row(f"p{i}", 1e-9, 1e-9, status="censored_numerical", swapped=False)
            for i in range(3)]
    summary = aggregate_interventions(rows)
    assert summary["verdict"] == "no_resolved_preference_change"
    assert summary["decisive_reversal_rate"] == pytest.approx(0.0)


def test_aggregate_refuses_to_pool_size_matched_and_size_changed_pairs():
    rows = [intervention_row("p0", 0.1, 0.1), {**intervention_row("p1", 0.2, 0.2),
                                               "physical_size_matched": False}]
    with pytest.raises(ValueError, match="size"):
        aggregate_interventions(rows)


def test_aggregate_of_an_empty_row_set_is_refused():
    with pytest.raises(ValueError, match="nonempty"):
        aggregate_interventions([])


# --- config-driven planning --------------------------------------------------

def plan_config(**overrides):
    config = {
        "seed": 4242, "parents": 6, "families": ["spin_glass", "weighted_maxcut"],
        "logical_qubits": 3, "chain_lengths": [3, 2, 1], "runtimes": [2.0],
        "splits": ["train", "validation"],
        "base": dict(BASE),
        "interventions": [{"factor": "geometry", "change": {"shape": "star"}},
                          {"factor": "chain_strength", "change": {"chain_strength": 3.0}}],
        "search": {"families": ["linear", "one_window"], "budget": 3, "seed": 0,
                   "tolerance": 0.005, "initial_steps": 16, "max_steps": 512},
        "max_physical_qubits": 8, "allow_size_change": False,
    }
    config.update(overrides)
    return config


def test_plan_builds_one_pair_per_parent_runtime_and_intervention():
    from annealctrl.interventions import plan_intervention_pairs
    built, plan = plan_intervention_pairs(plan_config())
    # geometry gives one pair; chain_strength gives two (total + scale-controlled).
    assert plan["n_parents"] * 1 * 3 == len(built)
    assert {pair.factor for pair in built} == {"geometry", "chain_strength"}
    assert len({pair.pair_id for pair in built}) == len(built)


def test_plan_uses_only_train_and_validation_parents_by_default():
    from annealctrl.interventions import plan_intervention_pairs
    _, plan = plan_intervention_pairs(plan_config())
    assert plan["splits"] == ["train", "validation"]
    assert "test" not in {entry["split"] for entry in plan["parents"]}


def test_plan_refuses_test_parents_unless_explicitly_allowed():
    from annealctrl.interventions import plan_intervention_pairs
    with pytest.raises(ValueError, match="allow_test_parents"):
        plan_intervention_pairs(plan_config(splits=["train", "test"]))


def test_plan_allows_test_parents_when_declared_and_records_it():
    from annealctrl.interventions import plan_intervention_pairs
    _, plan = plan_intervention_pairs(plan_config(splits=["test"]), allow_test_parents=True)
    assert plan["includes_test_parents"] is True


def test_plan_rejects_unknown_config_keys():
    from annealctrl.interventions import plan_intervention_pairs
    with pytest.raises(ValueError, match="unknown"):
        plan_intervention_pairs(plan_config(sneaky=1))


def test_plan_respects_the_physical_qubit_cap():
    from annealctrl.interventions import plan_intervention_pairs
    with pytest.raises(ValueError, match="cap"):
        plan_intervention_pairs(plan_config(chain_lengths=[4, 4, 4], max_physical_qubits=6))


# --- sweep and report --------------------------------------------------------

def test_intervention_sweep_runs_and_resumes(tmp_path):
    from annealctrl.interventions import plan_intervention_pairs, sweep_interventions
    built, _ = plan_intervention_pairs(plan_config(parents=6))
    kwargs = dict(families=("linear", "one_window"), budget=2, seed=0,
                  tolerance=0.005, initial_steps=16, max_steps=512, max_qubits=8)
    first = sweep_interventions(built, output=tmp_path / "sweep", **kwargs)
    assert first["completed"] == len(built)
    again = sweep_interventions(built, output=tmp_path / "sweep", resume=True, **kwargs)
    assert again["completed"] == 0 and again["skipped"] == len(built)


def test_intervention_sweep_dry_run_reports_the_budget(tmp_path):
    from annealctrl.interventions import plan_intervention_pairs, sweep_interventions
    built, _ = plan_intervention_pairs(plan_config(parents=6))
    plan = sweep_interventions(built, output=tmp_path / "sweep",
                               families=("linear", "one_window"), budget=8, dry_run=True)
    assert plan["requested_objective_calls_per_unit"] == 2 * (1 + 8) + 2
    assert not (tmp_path / "sweep").exists()


def test_intervention_report_writes_markdown_with_the_scale_arms_separated(tmp_path):
    from annealctrl.interventions import intervention_report, plan_intervention_pairs, sweep_interventions
    built, _ = plan_intervention_pairs(plan_config(parents=6))
    sweep_interventions(built, output=tmp_path / "sweep", families=("linear", "one_window"),
                        budget=2, seed=0, tolerance=0.005, initial_steps=16, max_steps=512,
                        max_qubits=8)
    summary = intervention_report(tmp_path / "sweep", bootstrap_resamples=100)

    text = (tmp_path / "sweep" / "report" / "INTERVENTIONS.md").read_text()
    assert "scale_controlled" in text and "total_compiled_effect" in text
    assert "closed-system simulator" in text
    assert "1 by construction" in text, "the tautology must be stated where it is reported"
    assert set(summary["by_scale_arm"]) == {"total_compiled_effect", "scale_controlled"}


def test_a_vacuous_intervention_raises_its_own_exception_type():
    from annealctrl.interventions import VacuousIntervention
    # A two-member chain has the same single edge under path and star, so this
    # change is empty rather than confounded.
    with pytest.raises(VacuousIntervention):
        build_pairs(problem(), spec("geometry", {"shape": "star"}),
                    lengths=np.array([2, 1, 1]), runtime=2.0, seed=5)


def test_the_planner_skips_and_counts_vacuous_pairs_instead_of_aborting():
    from annealctrl.interventions import plan_intervention_pairs
    built, plan = plan_intervention_pairs(plan_config(chain_lengths=[2, 1, 1]))
    assert plan["n_vacuous_skipped"] > 0
    assert all(entry["factor"] == "geometry" for entry in plan["vacuous_skipped"])
    assert built and all(pair.factor == "chain_strength" for pair in built)


def test_the_planner_never_skips_a_confounded_pair():
    from annealctrl.interventions import plan_intervention_pairs
    # An unknown base key is a specification error, not a vacuous pair.
    with pytest.raises(ValueError, match="unknown base keys"):
        plan_intervention_pairs(plan_config(base={**BASE, "mystery": 1}))


def test_penalty_inclusion_and_swap_denominators_are_distinct_populations():
    rows = [intervention_row("p0", 0.10, 0.12, status="resolved", swapped=True),
            intervention_row("p1", 0.10, 1e-9, status="one_sided", swapped=False),
            intervention_row("p2", 1e-9, 1e-9, status="censored_numerical", swapped=False)]
    block = aggregate_interventions(rows)["by_scale_arm"]["total_compiled_effect"]["transfer_penalty"]
    # A one-sided pair contributes a measured penalty but can never be a swap.
    assert block["n_pairs"] == 3
    assert block["n_included_pairs"] == 2
    assert block["n_resolved_pairs"] == 1
    assert block["n_one_sided_pairs"] == 1


def test_a_field_allocation_pair_changes_fields_only_even_with_random_couplers():
    built = build_pairs(problem(), {"factor": "field_allocation",
                                    "base": {**BASE, "ports": 2, "coupling_distribution": "random"},
                                    "change": {"field_distribution": "concentrated"}},
                        lengths=LENGTHS, runtime=2.0, seed=5)
    a, b = built[0].arms
    scale_a, scale_b = a.compiled.programmed_scale, b.compiled.programmed_scale
    assert np.allclose(a.compiled.problem_J / scale_a, b.compiled.problem_J / scale_b)
    assert np.allclose(a.compiled.chain_J / scale_a, b.compiled.chain_J / scale_b)
    assert not np.allclose(a.compiled.physical.h / scale_a, b.compiled.physical.h / scale_b)


def test_a_chain_strength_pair_changes_the_penalty_only():
    a, b = pairs("chain_strength", {"chain_strength": 3.0})[0].arms
    scale_a, scale_b = a.compiled.programmed_scale, b.compiled.programmed_scale
    assert np.allclose(a.compiled.problem_J / scale_a, b.compiled.problem_J / scale_b)
    assert np.allclose(a.compiled.physical.h / scale_a, b.compiled.physical.h / scale_b)
    assert not np.allclose(a.compiled.chain_J / scale_a, b.compiled.chain_J / scale_b)


def test_a_coefficient_intervention_that_moved_an_unrelated_coefficient_is_refused():
    from annealctrl import interventions

    built = pairs("chain_strength", {"chain_strength": 3.0})[0]
    a, b = built.arms
    # Simulate a stream-drift confound: same graph, but the couplers moved too.
    b.compiled.problem_J[:] = b.compiled.problem_J * 1.5
    with pytest.raises(ValueError, match="problem coupler"):
        interventions._audit_single_factor("chain_strength", a, b)


def test_two_interventions_on_the_same_factor_get_distinct_pair_ids():
    from annealctrl.interventions import plan_intervention_pairs
    # intervention_research.json declares both star and random_tree geometry
    # changes; a pair id keyed on the factor alone collides between them.
    built, _ = plan_intervention_pairs(plan_config(
        chain_lengths=[5, 2, 1],
        interventions=[{"factor": "geometry", "change": {"shape": "star"}},
                       {"factor": "geometry", "change": {"shape": "random_tree"}}]))
    ids = [pair.pair_id for pair in built]
    assert len(set(ids)) == len(ids), "pair ids must distinguish two changes to one factor"
    assert any("star" in pair_id for pair_id in ids)
    assert any("random_tree" in pair_id for pair_id in ids)


def test_pair_ids_stay_filesystem_safe_for_numeric_and_list_changes():
    built = pairs("chain_strength", {"chain_strength": 3.5})
    lengths = pairs("chain_length", {"chain_lengths": [3, 2, 2]}, allow_size_change=True)
    for pair in [*built, *lengths]:
        assert "/" not in pair.pair_id and " " not in pair.pair_id
        assert all(c.isalnum() or c in "_-." for c in pair.pair_id)


def test_swap_rate_conditional_on_resolved_is_one_by_construction():
    # 'resolved' means BOTH transfer penalties exceed their ambiguity, which can
    # only happen when each arm's own control beats the imported one - that IS a
    # swap. So P(swap | resolved) == 1 identically, and quoting it as a headline
    # would present a tautology as a finding.
    rows = [intervention_row(f"p{i}", 0.1, 0.1, status="resolved", swapped=True) for i in range(4)]
    rows += [intervention_row(f"q{i}", 0.1, 1e-9, status="one_sided", swapped=False) for i in range(3)]
    rows += [intervention_row(f"r{i}", 1e-9, 1e-9, status="censored_numerical", swapped=False)
             for i in range(3)]
    summary = aggregate_interventions(rows)

    assert summary["swap_consistency_check"] == pytest.approx(1.0)
    assert summary["swap_consistency_holds"] is True
    # The reportable quantity is unconditional: 4 decisive reversals out of 10 pairs.
    assert summary["decisive_reversal_rate"] == pytest.approx(0.4)
    assert summary["decisive_reversal_denominator"] == 10


def test_a_resolved_pair_that_is_not_swapped_breaks_the_consistency_check():
    rows = [intervention_row("p0", 0.1, 0.1, status="resolved", swapped=True),
            intervention_row("p1", 0.1, 0.1, status="resolved", swapped=False)]
    summary = aggregate_interventions(rows)
    assert summary["swap_consistency_holds"] is False


def test_verdict_uses_the_unconditional_reversal_rate():
    censored = [intervention_row(f"p{i}", 1e-9, 1e-9, status="censored_numerical", swapped=False)
                for i in range(3)]
    assert aggregate_interventions(censored)["verdict"] == "no_resolved_preference_change"
    assert aggregate_interventions(censored)["decisive_reversal_rate"] == pytest.approx(0.0)


def test_aggregate_counts_pairs_whose_search_was_asymmetric():
    # A negative penalty in some direction means the imported control beat this
    # arm's own best found: the equal-budget search on that arm was the weaker of
    # the two. The count bounds how much of the measured effect is search noise
    # rather than a property of the intervention.
    rows = [intervention_row("p0", 0.10, 0.12),
            intervention_row("p1", -0.03, 0.20),
            intervention_row("p2", 0.05, -0.01)]
    summary = aggregate_interventions(rows)
    assert summary["pairs_with_negative_direction"] == 2
    assert summary["negative_direction_fraction"] == pytest.approx(2 / 3)
