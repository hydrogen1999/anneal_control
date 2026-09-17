"""Method-against-method contrasts: does the proposed architecture actually win?

The per-method table answers "is this method better than the global baseline?".
It cannot answer "is hierarchy better than summary?", because two overlapping
confidence intervals are not a test of their difference. These contrasts pair on
the logical parent and correct for the whole family of comparisons, so a claim
of architectural advantage has to survive the same scrutiny as any other claim.
"""
import json

import numpy as np
import pytest

from annealctrl.contrasts import (
    contrast_matrix,
    encoder_information_contrast,
    paired_method_contrast,
)


def rows_for(method, mode, losses, *, parents=None, split="test"):
    parents = parents or [f"p{i // 2}" for i in range(len(losses))]
    return [{"method": method, "mode": mode, "split": split, "record_id": f"{method}_{i}",
             "parent_id": parents[i], "loss": float(loss)}
            for i, loss in enumerate(losses)]


# --- the paired contrast -----------------------------------------------------

def test_contrast_is_paired_on_the_parent_not_the_record():
    # Two parents, two records each. B beats A by exactly 0.10 everywhere.
    a = rows_for("a", "bank", [0.60, 0.70, 0.50, 0.40])
    b = rows_for("b", "bank", [0.50, 0.60, 0.40, 0.30])
    result = paired_method_contrast(a + b, "a", "b", mode="bank", bootstrap_resamples=500)
    assert result["n_parents"] == 2
    assert result["mean_difference"] == pytest.approx(-0.10)
    assert result["unit_of_independence"] == "logical_parent"


def test_contrast_sign_convention_is_b_minus_a():
    a = rows_for("a", "bank", [0.60, 0.60], parents=["p0", "p1"])
    b = rows_for("b", "bank", [0.50, 0.50], parents=["p0", "p1"])
    worse = paired_method_contrast(a + b, "b", "a", mode="bank", bootstrap_resamples=500)
    assert worse["mean_difference"] == pytest.approx(+0.10)
    assert worse["better"] == "b"


def test_a_real_difference_separates():
    rng = np.random.default_rng(0)
    base = rng.normal(0.6, 0.02, 40)
    a = rows_for("a", "bank", base, parents=[f"p{i}" for i in range(40)])
    b = rows_for("b", "bank", base - 0.05, parents=[f"p{i}" for i in range(40)])
    result = paired_method_contrast(a + b, "a", "b", mode="bank", bootstrap_resamples=2000)
    assert result["separated"] is True
    assert result["ci_high"] < 0


def test_no_difference_does_not_separate():
    rng = np.random.default_rng(1)
    base = rng.normal(0.6, 0.02, 40)
    a = rows_for("a", "bank", base, parents=[f"p{i}" for i in range(40)])
    b = rows_for("b", "bank", base + rng.normal(0, 1e-4, 40), parents=[f"p{i}" for i in range(40)])
    result = paired_method_contrast(a + b, "a", "b", mode="bank", bootstrap_resamples=2000)
    assert result["separated"] is False
    assert result["ci_low"] < 0 < result["ci_high"]


def test_contrast_refuses_methods_evaluated_on_different_parents():
    a = rows_for("a", "bank", [0.6, 0.6], parents=["p0", "p1"])
    b = rows_for("b", "bank", [0.5, 0.5], parents=["p0", "p2"])
    with pytest.raises(ValueError, match="same parents"):
        paired_method_contrast(a + b, "a", "b", mode="bank", bootstrap_resamples=100)


def test_contrast_refuses_an_absent_method():
    a = rows_for("a", "bank", [0.6, 0.6])
    with pytest.raises(ValueError, match="no rows"):
        paired_method_contrast(a, "a", "ghost", mode="bank", bootstrap_resamples=100)


def test_contrast_refuses_to_mix_modes():
    a = rows_for("a", "bank", [0.6, 0.6])
    b = rows_for("b", "direct", [0.5, 0.5])
    with pytest.raises(ValueError, match="no rows"):
        paired_method_contrast(a + b, "a", "b", mode="bank", bootstrap_resamples=100)


# --- the family --------------------------------------------------------------

def test_matrix_covers_every_unordered_pair_once():
    rng = np.random.default_rng(2)
    rows = []
    for name in ("a", "b", "c"):
        rows += rows_for(name, "bank", rng.normal(0.6, 0.02, 20),
                         parents=[f"p{i}" for i in range(20)])
    matrix = contrast_matrix(rows, mode="bank", bootstrap_resamples=500)
    assert len(matrix["pairs"]) == 3
    assert {tuple(sorted((p["method_a"], p["method_b"]))) for p in matrix["pairs"]} == {
        ("a", "b"), ("a", "c"), ("b", "c")}


def test_matrix_holm_corrects_the_family_of_comparisons():
    rng = np.random.default_rng(3)
    shared = rng.normal(0.6, 0.02, 30)
    rows = rows_for("a", "bank", shared, parents=[f"p{i}" for i in range(30)])
    rows += rows_for("b", "bank", shared + rng.normal(0, 1e-4, 30),
                     parents=[f"p{i}" for i in range(30)])
    rows += rows_for("c", "bank", shared - 0.06, parents=[f"p{i}" for i in range(30)])
    matrix = contrast_matrix(rows, mode="bank", bootstrap_resamples=2000)
    by_pair = {tuple(sorted((p["method_a"], p["method_b"]))): p for p in matrix["pairs"]}
    # Holm adjusts upward, never downward.
    for pair in matrix["pairs"]:
        assert pair["p_value_holm"] >= pair["p_value"] - 1e-12
    assert by_pair[("a", "c")]["separated_after_correction"] is True
    assert by_pair[("a", "b")]["separated_after_correction"] is False
    assert matrix["n_comparisons"] == 3


def test_matrix_reports_which_methods_are_indistinguishable():
    rng = np.random.default_rng(4)
    shared = rng.normal(0.6, 0.02, 30)
    rows = rows_for("a", "bank", shared, parents=[f"p{i}" for i in range(30)])
    rows += rows_for("b", "bank", shared + rng.normal(0, 1e-4, 30),
                     parents=[f"p{i}" for i in range(30)])
    matrix = contrast_matrix(rows, mode="bank", bootstrap_resamples=2000)
    assert ["a", "b"] in matrix["indistinguishable_groups"] or \
           ["b", "a"] in matrix["indistinguishable_groups"]


# --- the question the paper actually asks ------------------------------------

def test_information_contrast_pools_embedding_aware_against_blind():
    rng = np.random.default_rng(5)
    parents = [f"p{i}" for i in range(30)]
    blind = rng.normal(0.62, 0.02, 30)
    rows = rows_for("logical", "bank", blind, parents=parents)
    for name in ("summary", "physical"):
        rows += rows_for(name, "bank", blind - 0.03, parents=parents)
    result = encoder_information_contrast(rows, blind=("logical",), mode="bank",
                                          bootstrap_resamples=2000)
    assert result["n_parents"] == 30
    assert result["mean_difference"] == pytest.approx(-0.03, abs=1e-9)
    assert result["separated"] is True
    assert set(result["aware_methods"]) == {"summary", "physical"}


def test_information_contrast_refuses_an_empty_side():
    rows = rows_for("summary", "bank", [0.6, 0.6])
    with pytest.raises(ValueError, match="blind"):
        encoder_information_contrast(rows, blind=("logical",), mode="bank",
                                     bootstrap_resamples=100)


def test_information_contrast_is_json_serialisable():
    rng = np.random.default_rng(6)
    parents = [f"p{i}" for i in range(20)]
    blind = rng.normal(0.62, 0.02, 20)
    rows = rows_for("logical", "bank", blind, parents=parents)
    rows += rows_for("summary", "bank", blind - 0.02, parents=parents)
    result = encoder_information_contrast(rows, blind=("logical",), mode="bank",
                                          bootstrap_resamples=200)
    assert json.loads(json.dumps(result))["separated"] in (True, False)


# --- the CLI -----------------------------------------------------------------

def test_method_contrast_cli_runs_end_to_end(tmp_path, capsys):
    from annealctrl.workflow_cli import main
    rng = np.random.default_rng(7)
    parents = [f"p{i}" for i in range(25)]
    blind = rng.normal(0.62, 0.02, 25)
    rows = rows_for("logical", "bank", blind, parents=parents)
    rows += rows_for("summary", "bank", blind - 0.03, parents=parents)
    rows += rows_for("hierarchy", "bank", blind - 0.031, parents=parents)
    records = tmp_path / "heldout.json"
    records.write_text(json.dumps({"record_means": rows}))

    main(["method-contrast", "--records", str(records), "--output", str(tmp_path / "c.json"),
          "--mode", "bank", "--bootstrap-resamples", "500"])
    result = json.loads((tmp_path / "c.json").read_text())
    bank = result["modes"]["bank"]
    assert bank["contrast_matrix"]["n_comparisons"] == 3
    assert bank["embedding_information"]["separated"] is True
    assert "SEPARATED" in capsys.readouterr().out


def test_method_contrast_cli_reports_when_no_blind_encoder_is_present(tmp_path):
    from annealctrl.workflow_cli import main
    rng = np.random.default_rng(8)
    parents = [f"p{i}" for i in range(10)]
    base = rng.normal(0.6, 0.02, 10)
    rows = rows_for("summary", "bank", base, parents=parents)
    rows += rows_for("physical", "bank", base - 0.01, parents=parents)
    records = tmp_path / "heldout.json"
    records.write_text(json.dumps({"record_means": rows}))
    main(["method-contrast", "--records", str(records), "--output", str(tmp_path / "c.json"),
          "--mode", "bank", "--bootstrap-resamples", "200"])
    information = json.loads((tmp_path / "c.json").read_text())["modes"]["bank"]["embedding_information"]
    assert information["status"] == "unavailable"
    assert "logical" in information["reason"]
