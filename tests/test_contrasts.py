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
    paired_parent_seed_contrast,
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


def test_nonrejection_is_not_transitive_and_maximal_sets_may_overlap(monkeypatch):
    import annealctrl.contrasts as module
    # a~b and a~c, but b and c differ. The former greedy grouping put all
    # three into one purportedly indistinguishable set.
    def contrast(_rows, a, b, **_kwargs):
        rejected = {a, b} == {"b", "c"}
        return {"method_a": a, "method_b": b, "p_value": 0.001 if rejected else 0.8,
                "separated": rejected}
    monkeypatch.setattr(module, "paired_method_contrast", contrast)
    rows = sum((rows_for(m, "bank", [0.4, 0.5]) for m in ("a", "b", "c")), [])
    matrix = contrast_matrix(rows, methods=["a", "b", "c"])
    assert matrix["nonseparated_maximal_sets"] == [["a", "b"], ["a", "c"]]
    assert matrix["indistinguishable_groups"] == matrix["nonseparated_maximal_sets"]


def test_pooled_information_is_explicitly_outside_the_holm_family():
    rows = rows_for("logical", "bank", [0.6, 0.7], parents=["p0", "p1"])
    rows += rows_for("summary", "bank", [0.5, 0.6], parents=["p0", "p1"])
    result = encoder_information_contrast(rows, bootstrap_resamples=100)
    assert result["correction"] == "none"
    assert result["inference_role"].startswith("exploratory")
    assert "NOT part" in result["scope"]


def test_pooled_information_refuses_silently_dropped_parents():
    rows = rows_for("logical", "bank", [0.6, 0.7, 0.8], parents=["p0", "p1", "p2"])
    rows += rows_for("summary", "bank", [0.5, 0.6], parents=["p0", "p1"])
    with pytest.raises(ValueError, match="same parents"):
        encoder_information_contrast(rows, bootstrap_resamples=100)


def test_centered_bootstrap_has_an_explicit_null_and_never_a_zero_p_value():
    from annealctrl.contrasts import _paired_bootstrap
    result = _paired_bootstrap(np.zeros(20), n_resamples=100, seed=0)
    assert result["p_value"] == 1
    effect = _paired_bootstrap(np.full(20, 0.1), n_resamples=100, seed=0)
    assert effect["p_value"] == pytest.approx(1 / 101)
    assert effect["p_value_method"] == "centered_null_bootstrap_absolute_mean"


def crossed_rows(seed_effects=(-0.1, 0., 0.1)):
    rows = []
    for method in ("a", "b"):
        for parent in range(20):
            for seed, effect in enumerate(seed_effects):
                rows.append({"method": method, "mode": "direct", "parent_id": f"p{parent}",
                             "seed": seed, "record_id": f"r{parent}",
                             "loss": 0.5 + (effect if method == "b" else 0.)})
    return rows


def test_crossed_bootstrap_retains_shared_seed_uncertainty():
    rows = crossed_rows()
    parent_only = paired_method_contrast(rows, "a", "b", mode="direct", bootstrap_resamples=500)
    result = paired_parent_seed_contrast(rows, "a", "b", bootstrap_resamples=1000)
    assert parent_only["ci_high"] - parent_only["ci_low"] < 1e-10
    assert result["ci_high"] - result["ci_low"] > 0.1
    assert result["mean_difference"] == pytest.approx(0., abs=1e-14)
    assert result["n_seeds"] == 3
    assert result["p_value"] is None
    assert result["per_seed_mean_difference"]["0"] == pytest.approx(-0.1)


def test_crossed_bootstrap_requires_a_complete_panel_and_matching_records():
    rows = crossed_rows()
    incomplete = [r for r in rows if not (r["parent_id"] == "p0" and r["seed"] == 0)]
    with pytest.raises(ValueError, match="complete"):
        paired_parent_seed_contrast(incomplete, "a", "b", bootstrap_resamples=100)
    rows[-1]["record_id"] = "different"
    with pytest.raises(ValueError, match="matching record IDs"):
        paired_parent_seed_contrast(rows, "a", "b", bootstrap_resamples=100)


def test_crossed_bootstrap_does_not_report_joint_ci_from_one_seed():
    result = paired_parent_seed_contrast(crossed_rows((-0.1,)), "a", "b", bootstrap_resamples=100)
    assert result["status"] == "insufficient_parents_or_seeds"
    assert result["ci_low"] is None
    assert result["mean_difference"] == pytest.approx(-0.1)


@pytest.mark.parametrize("resamples,confidence", [(0, .95), (True, .95), (100, 1.), (100, 0.)])
def test_bootstrap_refuses_invalid_configuration(resamples, confidence):
    with pytest.raises(ValueError):
        paired_parent_seed_contrast(crossed_rows(), "a", "b",
                                    bootstrap_resamples=resamples, confidence=confidence)
