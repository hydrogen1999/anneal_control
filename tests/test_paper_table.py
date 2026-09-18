"""The single comparison table: every method on the same records, with its cost class.

A table that puts a linear ramp, an exponentially expensive spectral oracle, an
amortised network and a 257-evaluation per-instance search in one column ordered
by loss is not a comparison, it is a ranking of things that are not comparable.
Each row therefore carries what it had to consume to produce its number, and the
assembler refuses to emit a row whose cost class it does not know.
"""
import json

import numpy as np
import pytest

from annealctrl.paper_table import COST_CLASSES, assemble_comparison


def ml_row(record, parent, method, mode, loss, *, linear=0.60, glob=0.58):
    return {"record_id": record, "parent_id": parent, "method": method, "mode": mode,
            "split": "test", "loss": loss, "linear_loss": linear, "global_loss": glob,
            "runtime": 4.0, "family": "spin_glass", "logical_n": 4, "physical_n": 8}


def frontier_row(record, parent, *, linear=0.60, best=0.50, gap=0.63, d2=0.61,
                 gap_status="sampled_point_audit_passed"):
    return {"record_id": record, "parent_id": parent, "split": "test", "runtime": 4.0,
            "linear_loss": linear, "best_found_loss": best, "online_adaptation": True,
            "total_objective_calls": 257,
            "privileged_teachers": {
                "gap_inverse_square": {"status": gap_status, "loss": gap, "teacher_seconds": 0.02},
                "d2": {"status": "sampled_point_audit_passed", "loss": d2, "teacher_seconds": 0.02}}}


def paired(n=12, **kwargs):
    ml = [ml_row(f"r{i}", f"p{i}", "summary", "bank", 0.545, **kwargs) for i in range(n)]
    ml += [ml_row(f"r{i}", f"p{i}", "summary", "direct", 0.590, **kwargs) for i in range(n)]
    frontier = [frontier_row(f"r{i}", f"p{i}") for i in range(n)]
    return ml, frontier


# --- the join ----------------------------------------------------------------

def test_every_method_is_measured_on_the_same_records():
    ml, frontier = paired()
    table = assemble_comparison(ml, frontier, bootstrap_resamples=500)
    counts = {row["method"]: row["n_records"] for row in table["rows"]}
    assert len(set(counts.values())) == 1, counts
    assert table["n_records"] == 12
    assert table["n_parents"] == 12


def test_records_missing_from_either_source_are_excluded_and_counted():
    ml, frontier = paired(n=10)
    frontier = frontier[:7]
    table = assemble_comparison(ml, frontier, bootstrap_resamples=500)
    assert table["n_records"] == 7
    assert table["n_dropped_no_frontier_row"] == 3
    assert all(row["n_records"] == 7 for row in table["rows"])


def test_refuses_a_join_with_no_overlap():
    ml = [ml_row("a", "pa", "summary", "bank", 0.5)]
    frontier = [frontier_row("b", "pb")]
    with pytest.raises(ValueError, match="no records in common"):
        assemble_comparison(ml, frontier, bootstrap_resamples=100)


def test_refuses_ml_rows_outside_the_test_split():
    ml, frontier = paired()
    ml[0] = {**ml[0], "split": "validation"}
    with pytest.raises(ValueError, match="test"):
        assemble_comparison(ml, frontier, bootstrap_resamples=100)


# --- cost classes ------------------------------------------------------------

def test_each_row_declares_what_it_consumed():
    ml, frontier = paired()
    table = assemble_comparison(ml, frontier, bootstrap_resamples=500)
    by_method = {row["method"]: row for row in table["rows"]}
    assert by_method["linear"]["cost_class"] == "fixed"
    assert by_method["global"]["cost_class"] == "fixed"
    assert by_method["gap_inverse_square"]["cost_class"] == "privileged_spectrum"
    assert by_method["summary/bank"]["cost_class"] == "amortised"
    assert by_method["search_best_found"]["cost_class"] == "online_adaptation"
    for row in table["rows"]:
        assert row["cost_class"] in COST_CLASSES


def test_online_adaptation_rows_report_their_per_instance_budget():
    ml, frontier = paired()
    table = assemble_comparison(ml, frontier, bootstrap_resamples=500)
    search = next(row for row in table["rows"] if row["method"] == "search_best_found")
    assert search["objective_calls_per_instance"] == pytest.approx(257.0)
    assert search["consults_true_outcomes"] is True


def test_amortised_and_fixed_rows_consult_no_outcome():
    ml, frontier = paired()
    table = assemble_comparison(ml, frontier, bootstrap_resamples=500)
    for row in table["rows"]:
        if row["cost_class"] in ("fixed", "amortised", "privileged_spectrum"):
            assert row["consults_true_outcomes"] is False


def test_the_table_refuses_to_rank_across_cost_classes():
    ml, frontier = paired()
    table = assemble_comparison(ml, frontier, bootstrap_resamples=500)
    assert "rank" not in table
    # Ordering is within a class, and the class is part of the key.
    assert set(table["ranked_within_cost_class"]) <= set(COST_CLASSES)
    assert table["scope"].count("cost class") >= 1


# --- the numbers -------------------------------------------------------------

def test_losses_are_parent_level_with_intervals():
    ml, frontier = paired()
    table = assemble_comparison(ml, frontier, bootstrap_resamples=2000)
    bank = next(row for row in table["rows"] if row["method"] == "summary/bank")
    assert bank["mean_loss"] == pytest.approx(0.545)
    assert bank["parent_bootstrap_ci"]["unit_of_independence"] == "logical_parent"
    assert bank["vs_linear"]["mean_difference"] == pytest.approx(-0.055)


def test_an_unresolved_teacher_shrinks_only_its_own_row():
    ml, frontier = paired(n=10)
    frontier[0] = frontier_row("r0", "p0", gap=None, gap_status="unresolved_spectral_points")
    table = assemble_comparison(ml, frontier, bootstrap_resamples=500)
    by_method = {row["method"]: row for row in table["rows"]}
    assert by_method["gap_inverse_square"]["n_records"] == 9
    assert by_method["gap_inverse_square"]["measured_on_full_population"] is False
    assert by_method["summary/bank"]["n_records"] == 10
    assert by_method["summary/bank"]["measured_on_full_population"] is True


def test_failed_interpolation_audit_excludes_even_finite_teacher_loss():
    ml, frontier = paired(n=4)
    frontier[0] = frontier_row("r0", "p0", gap=0.01, gap_status="interpolation_audit_failed")
    table = assemble_comparison(ml, frontier, bootstrap_resamples=100)
    gap = next(row for row in table["rows"] if row["method"] == "gap_inverse_square")
    assert gap["n_records"] == 3
    assert gap["mean_loss"] == pytest.approx(.63)
    assert "gap_inverse_square" not in table["ranked_within_cost_class"].get("privileged_spectrum", [])
    assert table["teacher_populations"]["gap_inverse_square"]["n_excluded_records"] == 1


def test_teacher_contrasts_use_teacher_specific_paired_population():
    ml, frontier = paired(n=3)
    frontier[0] = frontier_row("r0", "p0", gap=.01, gap_status="interpolation_audit_failed")
    # This unusually easy learned record must not enter the gap comparison.
    ml[0]["loss"] = 0.
    table = assemble_comparison(ml, frontier, bootstrap_resamples=100)
    matched = {(row["method"], row["teacher"]): row for row in table["matched_teacher_contrasts"]}
    gap = matched["summary/bank", "gap_inverse_square"]
    assert gap["n_records"] == 2 and gap["record_ids"] == ["r1", "r2"]
    assert gap["mean_difference"] == pytest.approx(.545 - .63)
    assert gap["learned_mean_loss"] == pytest.approx(.545)
    assert gap["method_cost_class"] == "amortised"
    assert gap["teacher_cost_class"] == "privileged_spectrum"
    assert matched["summary/bank", "d2"]["n_records"] == 3


def test_all_failed_teacher_keeps_population_audit_without_a_loss_row():
    ml, frontier = paired(n=3)
    for row in frontier:
        row["privileged_teachers"]["gap_inverse_square"]["status"] = "interpolation_audit_failed"
    table = assemble_comparison(ml, frontier, bootstrap_resamples=100)
    assert table["teacher_populations"]["gap_inverse_square"]["n_eligible_records"] == 0
    assert not any(row["method"] == "gap_inverse_square" for row in table["rows"])
    assert not any(row["teacher"] == "gap_inverse_square" for row in table["matched_teacher_contrasts"])


def test_teacher_pairing_rejects_mismatched_parent_identity():
    ml, frontier = paired(n=3)
    frontier[0]["parent_id"] = "different"
    with pytest.raises(ValueError, match="parent identity"):
        assemble_comparison(ml, frontier, bootstrap_resamples=100)


def test_table_is_json_serialisable():
    ml, frontier = paired()
    table = assemble_comparison(ml, frontier, bootstrap_resamples=200)
    assert json.loads(json.dumps(table))["n_records"] == 12


def test_comparison_table_cli_runs_end_to_end(tmp_path, capsys):
    from annealctrl.workflow_cli import main

    ml, frontier = paired(n=10)
    records = tmp_path / "heldout.json"
    records.write_text(json.dumps({"record_means": ml}))
    sweep = tmp_path / "sweep"
    sweep.mkdir()
    with (sweep / "rows.jsonl").open("w") as handle:
        for index, row in enumerate(frontier):
            handle.write(json.dumps({
                "unit_id": f"u{index}", "unit_key": f"k{index}", "fingerprint": f"f{index}",
                "settings_hash": "s", "source_hash": "h", "status": "ok", "result": row}) + "\n")

    main(["comparison-table", "--records", str(records), "--reference-sweep", str(sweep),
          "--output", str(tmp_path / "table.json"), "--bootstrap-resamples", "500"])
    result = json.loads((tmp_path / "table.json").read_text())
    assert result["n_records"] == 10
    out = capsys.readouterr().out
    assert "online_adaptation" in out and "privileged_spectrum" in out


# --- several search strategies, each its own row -----------------------------

def test_a_second_search_strategy_becomes_its_own_row():
    ml, frontier = paired(n=12)
    bayes = [{**row, "best_found_loss": 0.48} for row in frontier]
    table = assemble_comparison(ml, frontier, searches={"bayesian": bayes},
                                bootstrap_resamples=500)
    by_method = {row["method"]: row for row in table["rows"]}
    assert by_method["search_best_found"]["mean_loss"] == pytest.approx(0.50)
    assert by_method["search_bayesian"]["mean_loss"] == pytest.approx(0.48)
    assert by_method["search_bayesian"]["cost_class"] == "online_adaptation"
    assert by_method["search_bayesian"]["consults_true_outcomes"] is True


def test_extra_searches_are_restricted_to_the_shared_records():
    """A strategy evaluated on more records than the join would not be comparable."""
    ml, frontier = paired(n=10)
    bayes = [{**row, "best_found_loss": 0.48} for row in frontier]
    bayes.append(frontier_row("extra", "p_extra", best=0.01))
    table = assemble_comparison(ml, frontier, searches={"bayesian": bayes},
                                bootstrap_resamples=500)
    by_method = {row["method"]: row for row in table["rows"]}
    assert by_method["search_bayesian"]["n_records"] == 10
    assert by_method["search_bayesian"]["mean_loss"] == pytest.approx(0.48)


def test_a_strategy_missing_records_is_marked_conditional():
    ml, frontier = paired(n=10)
    bayes = [{**row, "best_found_loss": 0.48} for row in frontier[:6]]
    table = assemble_comparison(ml, frontier, searches={"bayesian": bayes},
                                bootstrap_resamples=500)
    row = next(r for r in table["rows"] if r["method"] == "search_bayesian")
    assert row["n_records"] == 6
    assert row["measured_on_full_population"] is False


def test_search_strategies_are_ordered_within_their_class():
    ml, frontier = paired(n=12)
    bayes = [{**row, "best_found_loss": 0.48} for row in frontier]
    table = assemble_comparison(ml, frontier, searches={"bayesian": bayes},
                                bootstrap_resamples=500)
    order = table["ranked_within_cost_class"]["online_adaptation"]
    assert order == ["search_bayesian", "search_best_found"]


def test_comparison_table_cli_accepts_extra_search_directories(tmp_path, capsys):
    from annealctrl.workflow_cli import main

    ml, frontier = paired(n=10)
    (tmp_path / "heldout.json").write_text(json.dumps({"record_means": ml}))

    def write_sweep(name, rows):
        directory = tmp_path / name
        directory.mkdir()
        with (directory / "rows.jsonl").open("w") as handle:
            for index, row in enumerate(rows):
                handle.write(json.dumps({
                    "unit_id": f"u{index}", "unit_key": f"k{index}", "fingerprint": f"f{index}",
                    "settings_hash": "s", "source_hash": "h", "status": "ok",
                    "result": row}) + "\n")
        return directory

    primary = write_sweep("sobol", frontier)
    bayes = write_sweep("bayes", [{**row, "best_found_loss": 0.47} for row in frontier])

    main(["comparison-table", "--records", str(tmp_path / "heldout.json"),
          "--reference-sweep", str(primary), "--search", f"bayesian={bayes}",
          "--output", str(tmp_path / "t.json"), "--bootstrap-resamples", "300"])
    methods = {row["method"] for row in json.loads((tmp_path / "t.json").read_text())["rows"]}
    assert {"search_best_found", "search_bayesian"} <= methods
    assert "search_bayesian" in capsys.readouterr().out


def test_comparison_table_cli_rejects_a_malformed_search_argument(tmp_path):
    from annealctrl.workflow_cli import main

    ml, frontier = paired(n=4)
    (tmp_path / "heldout.json").write_text(json.dumps({"record_means": ml}))
    directory = tmp_path / "sweep"
    directory.mkdir()
    (directory / "rows.jsonl").write_text("")
    with pytest.raises(ValueError, match="NAME=DIR"):
        main(["comparison-table", "--records", str(tmp_path / "heldout.json"),
              "--reference-sweep", str(directory), "--search", "justadir",
              "--output", str(tmp_path / "t.json")])


@pytest.mark.parametrize("field,value,error", [
    ("parent_id", "wrong", "parent_id mismatch"),
    ("split", "validation", "test split"),
    ("total_objective_calls", 256, "budget mismatch"),
    ("total_objective_calls", None, "positive integer"),
    ("runtime", 99., "runtime mismatch"),
    ("linear_loss", .2, "linear reference mismatch"),
    ("best_found_loss", float("nan"), "finite"),
])
def test_extra_search_rejects_mismatched_identity_budget_or_invalid_loss(field, value, error):
    ml, frontier = paired(n=3)
    extra = [{**row} for row in frontier]
    extra[0][field] = value
    with pytest.raises(ValueError, match=error):
        assemble_comparison(ml, frontier, searches={"extra": extra}, bootstrap_resamples=20)


def test_extra_search_duplicate_records_are_not_silently_overwritten():
    ml, frontier = paired(n=3)
    with pytest.raises(ValueError, match="duplicate"):
        assemble_comparison(ml, frontier, searches={"extra": frontier + [frontier[0]]}, bootstrap_resamples=20)


def test_duplicate_learned_rows_fail_before_any_population_averaging():
    ml, frontier = paired(n=3)
    with pytest.raises(ValueError, match="duplicate learned"):
        assemble_comparison(ml + [ml[0]], frontier, bootstrap_resamples=20)


def test_shared_fixed_baseline_cannot_depend_on_method_order():
    ml, frontier = paired(n=3)
    ml[3]["global_loss"] = .01
    with pytest.raises(ValueError, match="global_loss inconsistent"):
        assemble_comparison(ml, frontier, bootstrap_resamples=20)
