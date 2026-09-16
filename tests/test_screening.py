"""G2b: hardness qualification measured by control gain, fitted on train/validation only."""
import json

import pytest

from annealctrl.screening import apply_threshold, fit_threshold, screen_records


def row(parent, headroom, *, split="validation", status="resolved", calls=25, restriction=None):
    return {"parent_id": parent, "record_id": f"{parent}_r0", "split": split,
            "family": "spin_glass", "logical_n": 3, "physical_n": 6, "runtime": 2.0,
            "headroom": headroom, "linear_loss": 0.6, "best_found_loss": 0.6 - headroom,
            "relative_headroom": headroom / 0.6, "resolution_status": status,
            "combined_loss_ambiguity": 1e-6, "total_objective_calls": calls,
            "family_restriction_loss": restriction or {"linear": headroom, "one_window": 0.0}}


def fit_rows():
    return [row(f"p{i}", value) for i, value in enumerate([0.01, 0.02, 0.05, 0.10, 0.20])]


# --- fitting -----------------------------------------------------------------

def test_threshold_is_the_requested_quantile_of_parent_headroom():
    rule = fit_threshold(fit_rows(), quantile=0.5)
    assert rule["threshold"] == pytest.approx(0.05)
    assert rule["quantity"] == "headroom"
    assert rule["n_fit_parents"] == 5


def test_fitting_on_the_test_split_is_refused():
    rows = [row("p0", 0.1, split="test"), row("p1", 0.2, split="test")]
    with pytest.raises(ValueError, match="test"):
        fit_threshold(rows)


def test_fitting_on_mixed_rows_containing_any_test_row_is_refused():
    with pytest.raises(ValueError, match="test"):
        fit_threshold([*fit_rows(), row("q0", 0.3, split="test")])


def test_train_and_validation_rows_may_be_pooled_for_fitting():
    rows = [row("p0", 0.1, split="train"), row("p1", 0.2, split="validation")]
    rule = fit_threshold(rows, quantile=0.0)
    assert rule["fit_splits"] == ["train", "validation"]
    assert rule["threshold"] == pytest.approx(0.1)


def test_censored_rows_never_contribute_to_the_threshold():
    rows = [*fit_rows(), row("p9", 5.0, status="censored_numerical")]
    rule = fit_threshold(rows, quantile=1.0)
    assert rule["threshold"] == pytest.approx(0.20), "a censored row cannot set the bar"
    assert rule["n_fit_censored_records"] == 1


def test_an_explicit_minimum_headroom_floor_raises_the_threshold():
    rule = fit_threshold(fit_rows(), quantile=0.5, min_headroom=0.12)
    assert rule["threshold"] == pytest.approx(0.12)
    assert rule["min_headroom"] == 0.12


def test_the_screening_quantity_is_restricted_to_measured_control_gain():
    with pytest.raises(ValueError, match="quantity"):
        fit_threshold(fit_rows(), quantity="model_advantage")


def test_family_restriction_loss_is_an_allowed_screening_quantity():
    rule = fit_threshold(fit_rows(), quantity="family_restriction_loss:linear", quantile=0.5)
    assert rule["threshold"] == pytest.approx(0.05)


def test_quantile_must_lie_in_the_unit_interval():
    for bad in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError, match="quantile"):
            fit_threshold(fit_rows(), quantile=bad)


def test_fitting_requires_resolved_rows():
    with pytest.raises(ValueError, match="resolved"):
        fit_threshold([row("p0", 0.1, status="censored_numerical")])


# --- applying ----------------------------------------------------------------

def test_applying_selects_parents_above_the_threshold():
    rule = fit_threshold(fit_rows(), quantile=0.5)
    target = [row("t0", 0.01, split="test"), row("t1", 0.30, split="test")]
    result = apply_threshold(rule, target)
    assert result["selected_parent_ids"] == ["t1"]
    assert result["selected_fraction"] == pytest.approx(0.5)
    assert result["unscreened_parents"] == 2


def test_a_censored_target_row_can_never_be_selected():
    rule = fit_threshold(fit_rows(), quantile=0.0)
    target = [row("t0", 0.99, split="test", status="censored_numerical")]
    result = apply_threshold(rule, target)
    assert result["selected_parent_ids"] == []
    assert result["censored_parents"] == 1


def test_the_selected_subset_is_labelled_conditional_not_a_deployment_frequency():
    rule = fit_threshold(fit_rows(), quantile=0.5)
    result = apply_threshold(rule, [row("t0", 0.30, split="test")])
    assert result["interpretation"].startswith("conditional stress benchmark")
    assert result["is_unbiased_deployment_sample"] is False


def test_applying_to_the_split_the_rule_was_fitted_on_is_flagged_as_in_sample():
    rule = fit_threshold(fit_rows(), quantile=0.5)
    result = apply_threshold(rule, fit_rows())
    assert result["in_sample"] is True
    assert result["parent_overlap"] == 5


def test_applying_to_disjoint_parents_is_out_of_sample():
    rule = fit_threshold(fit_rows(), quantile=0.5)
    result = apply_threshold(rule, [row("t0", 0.30, split="test")])
    assert result["in_sample"] is False and result["parent_overlap"] == 0


def test_screening_cost_counts_objective_calls_on_both_sides():
    rule = fit_threshold(fit_rows(), quantile=0.5)
    result = apply_threshold(rule, [row("t0", 0.30, split="test", calls=257)])
    assert rule["fit_objective_calls"] == 5 * 25
    assert result["apply_objective_calls"] == 257
    assert result["total_screening_objective_calls"] == 5 * 25 + 257


def test_selecting_nothing_is_reported_plainly():
    rule = fit_threshold(fit_rows(), quantile=1.0)
    result = apply_threshold(rule, [row("t0", 0.001, split="test")])
    assert result["selected_parent_ids"] == []
    assert result["selected_fraction"] == 0.0
    assert result["verdict"] == "no_parent_qualifies"


def test_parent_level_aggregation_before_thresholding():
    rule = fit_threshold(fit_rows(), quantile=0.5)
    # One parent, two variants straddling the threshold: the parent mean decides.
    target = [row("t0", 0.01, split="test"), {**row("t0", 0.20, split="test"),
                                              "record_id": "t0_r1"}]
    result = apply_threshold(rule, target)
    assert result["unscreened_parents"] == 1
    assert result["selected_parent_ids"] == ["t0"], "mean 0.105 exceeds the 0.05 threshold"


# --- end to end over sweep directories ---------------------------------------

def write_sweep(tmp_path, name, rows):
    root = tmp_path / name
    root.mkdir(parents=True)
    with (root / "rows.jsonl").open("w") as handle:
        for index, payload in enumerate(rows):
            handle.write(json.dumps({"unit_id": payload["record_id"], "unit_key": f"k{index}",
                                     "fingerprint": f"f{index}", "settings_hash": "s",
                                     "source_hash": "h", "status": "ok", "result": payload}) + "\n")
    return root


def test_screen_records_reads_sweep_directories_and_reports_the_full_rule(tmp_path):
    fit = write_sweep(tmp_path, "val", fit_rows())
    target = write_sweep(tmp_path, "test", [row("t0", 0.01, split="test"), row("t1", 0.30, split="test")])
    result = screen_records([fit], target, quantile=0.5)

    assert result["rule"]["threshold"] == pytest.approx(0.05)
    assert result["selection"]["selected_parent_ids"] == ["t1"]
    assert result["rule"]["fit_sweeps"] == [str(fit)]
    assert result["selection"]["apply_sweep"] == [str(target)]
    json.dumps(result)


def test_screen_records_refuses_a_test_sweep_as_the_fitting_source(tmp_path):
    bad = write_sweep(tmp_path, "test", [row("t0", 0.3, split="test"), row("t1", 0.4, split="test")])
    with pytest.raises(ValueError, match="test"):
        screen_records([bad], bad)


def test_screen_records_applies_across_several_sharded_directories(tmp_path):
    # A sharded sweep is many directories; screening must apply to all of them,
    # exactly as it already fits across several.
    fit_a = write_sweep(tmp_path, "train_a", fit_rows()[:3])
    fit_b = write_sweep(tmp_path, "train_b", fit_rows()[3:])
    apply_a = write_sweep(tmp_path, "val_a", [row("t0", 0.01), row("t1", 0.30)])
    apply_b = write_sweep(tmp_path, "val_b", [row("t2", 0.40)])
    result = screen_records([fit_a, fit_b], [apply_a, apply_b], quantile=0.5)

    assert result["selection"]["unscreened_parents"] == 3
    assert result["selection"]["selected_parent_ids"] == ["t1", "t2"]
    assert result["selection"]["apply_sweep"] == [str(apply_a), str(apply_b)]


def test_a_sweep_directory_without_rows_names_the_missing_file(tmp_path):
    empty = tmp_path / "not_a_sweep"
    empty.mkdir()
    with pytest.raises(ValueError, match="rows.jsonl"):
        screen_records([write_sweep(tmp_path, "fit", fit_rows())], empty)
