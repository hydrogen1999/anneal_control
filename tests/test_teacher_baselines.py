"""The privileged spectral baselines: is the gap schedule the thing to approximate?

`gap_inverse_square` is the local-adiabatic rule, ds/dt proportional to the
square of the instantaneous gap. It is the canonical physics-derived answer to
"how should one anneal", and it needs the exact spectrum at every point, which
no deployed method has. If it wins, learned control selection is a cheap
approximation to it. If it loses, it is not the target at all, and that changes
what the paper is about -- so the instrument has to be able to say either.
"""
import json

import numpy as np
import pytest

from annealctrl.teacher_baselines import aggregate_privileged_teachers


def row(record, parent, *, linear, best, gap, d2, runtime=4.0,
        gap_status="sampled_point_audit_passed", d2_status="sampled_point_audit_passed"):
    return {"record_id": record, "parent_id": parent, "split": "validation", "runtime": runtime,
            "family": "spin_glass", "logical_n": 4, "physical_n": 8,
            "linear_loss": linear, "best_found_loss": best,
            "privileged_teachers": {
                "gap_inverse_square": {"status": gap_status, "loss": gap, "teacher_seconds": 0.02,
                                       "excluded_from_equal_budget_claim": True},
                "d2": {"status": d2_status, "loss": d2, "teacher_seconds": 0.02,
                       "excluded_from_equal_budget_claim": True}}}


# --- resolution accounting ---------------------------------------------------

def test_unresolved_teachers_are_counted_not_dropped_silently():
    rows = [row(f"r{i}", f"p{i}", linear=0.7, best=0.6, gap=0.72, d2=0.71) for i in range(6)]
    rows += [row("r9", "p9", linear=0.7, best=0.6, gap=None, d2=0.71,
                 gap_status="unresolved_spectral_points")]
    result = aggregate_privileged_teachers(rows, bootstrap_resamples=200)
    gap = result["methods"]["gap_inverse_square"]
    assert gap["n_rows"] == 7
    assert gap["n_resolved"] == 6
    assert gap["resolution_rate"] == pytest.approx(6 / 7)
    assert gap["status_counts"]["unresolved_spectral_points"] == 1


def test_audit_failures_are_not_pooled_with_passes():
    rows = [row(f"r{i}", f"p{i}", linear=0.7, best=0.6, gap=0.72, d2=0.71) for i in range(5)]
    rows += [row(f"f{i}", f"q{i}", linear=0.7, best=0.6, gap=0.30, d2=0.71,
                 gap_status="interpolation_audit_failed") for i in range(5)]
    result = aggregate_privileged_teachers(rows, bootstrap_resamples=200)
    gap = result["methods"]["gap_inverse_square"]
    # The headline uses audited waveforms only; the failures are reported beside it.
    assert gap["n_audit_passed"] == 5
    assert gap["n_audit_failed"] == 5
    assert gap["mean_loss"] == pytest.approx(0.72)
    assert gap["mean_loss_including_audit_failures"] == pytest.approx(0.51)
    assert gap["audit_failures_change_the_sign"] is True


def test_selection_bias_is_stated_when_resolution_is_partial():
    resolved = [row(f"r{i}", f"p{i}", linear=0.7, best=0.6, gap=0.72, d2=0.71) for i in range(4)]
    unresolved = [row(f"u{i}", f"q{i}", linear=0.9, best=0.8, gap=None, d2=0.71,
                      gap_status="unresolved_spectral_points") for i in range(6)]
    result = aggregate_privileged_teachers(resolved + unresolved, bootstrap_resamples=200)
    gap = result["methods"]["gap_inverse_square"]
    assert gap["resolution_rate"] < 0.5
    assert gap["comparison_population_is_conditional"] is True
    # The instances it could not schedule are described, not ignored.
    assert gap["unresolved_mean_linear_loss"] == pytest.approx(0.9)
    assert gap["resolved_mean_linear_loss"] == pytest.approx(0.7)


# --- the contrasts -----------------------------------------------------------

def test_teacher_is_compared_against_linear_and_search_on_its_own_records():
    rows = [row(f"r{i}", f"p{i}", linear=0.70, best=0.60, gap=0.75, d2=0.68) for i in range(8)]
    result = aggregate_privileged_teachers(rows, bootstrap_resamples=2000)
    gap = result["methods"]["gap_inverse_square"]
    assert gap["vs_linear"]["mean_difference"] == pytest.approx(0.05)
    assert gap["vs_best_found"]["mean_difference"] == pytest.approx(0.15)
    assert gap["beats_linear_fraction"] == 0.0
    assert gap["verdict"] == "loses_to_linear"


def test_a_winning_teacher_is_reported_as_winning():
    rows = [row(f"r{i}", f"p{i}", linear=0.70, best=0.60, gap=0.55, d2=0.68) for i in range(8)]
    result = aggregate_privileged_teachers(rows, bootstrap_resamples=2000)
    gap = result["methods"]["gap_inverse_square"]
    assert gap["beats_linear_fraction"] == 1.0
    assert gap["verdict"] == "beats_search"


def test_contrast_intervals_are_parent_level():
    rows = [row(f"r{i}", "shared_parent", linear=0.70, best=0.60, gap=0.75, d2=0.68)
            for i in range(20)]
    result = aggregate_privileged_teachers(rows, bootstrap_resamples=500)
    gap = result["methods"]["gap_inverse_square"]
    assert gap["vs_linear"]["n_parents"] == 1
    assert gap["vs_linear"]["parent_bootstrap_ci"]["status"] == "insufficient_independent_parents"


def test_runtime_stratification_shows_the_adiabatic_trend():
    rows = []
    for index in range(6):
        rows.append(row(f"a{index}", f"pa{index}", linear=0.9, best=0.85, gap=0.95, d2=0.94,
                        runtime=1.0))
        rows.append(row(f"c{index}", f"pc{index}", linear=0.52, best=0.41, gap=0.50, d2=0.51,
                        runtime=12.0))
    result = aggregate_privileged_teachers(rows, bootstrap_resamples=500)
    by_runtime = result["methods"]["gap_inverse_square"]["by_runtime"]
    assert by_runtime["1.0"]["beats_linear_fraction"] == 0.0
    assert by_runtime["12.0"]["beats_linear_fraction"] == 1.0


# --- refusals ----------------------------------------------------------------

def test_refuses_rows_without_any_privileged_teacher():
    with pytest.raises(ValueError, match="privileged_teachers"):
        aggregate_privileged_teachers([{"record_id": "r", "parent_id": "p", "linear_loss": 0.7,
                                        "best_found_loss": 0.6, "runtime": 1.0}],
                                      bootstrap_resamples=100)


def test_refuses_an_empty_row_set():
    with pytest.raises(ValueError, match="nonempty"):
        aggregate_privileged_teachers([], bootstrap_resamples=100)


def test_result_is_json_serialisable():
    rows = [row(f"r{i}", f"p{i}", linear=0.70, best=0.60, gap=0.75, d2=0.68) for i in range(6)]
    result = aggregate_privileged_teachers(rows, bootstrap_resamples=200)
    assert json.loads(json.dumps(result))["methods"]["d2"]["n_resolved"] == 6


# --- the CLI -----------------------------------------------------------------

def test_teacher_baseline_report_cli_runs_end_to_end(tmp_path, capsys):
    from annealctrl.workflow_cli import main

    sweep = tmp_path / "sweep"
    sweep.mkdir()
    with (sweep / "rows.jsonl").open("w") as handle:
        for index in range(8):
            handle.write(json.dumps({
                "unit_id": f"u{index}", "unit_key": f"k{index}", "fingerprint": f"f{index}",
                "settings_hash": "s", "source_hash": "h", "status": "ok",
                "result": row(f"r{index}", f"p{index}", linear=0.70, best=0.60,
                              gap=0.75, d2=0.68)}) + "\n")

    main(["teacher-baseline-report", "--sweep", str(sweep),
          "--output", str(tmp_path / "t.json"), "--bootstrap-resamples", "500"])
    result = json.loads((tmp_path / "t.json").read_text())
    assert result["methods"]["gap_inverse_square"]["verdict"] == "loses_to_linear"
    assert "gap_inverse_square" in result["methods_losing_to_linear"]
    assert "vs search" in capsys.readouterr().out
