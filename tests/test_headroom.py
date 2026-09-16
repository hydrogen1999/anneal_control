"""G2: control-complexity frontier and headroom censored against its own numerics."""
import json

import numpy as np
import pytest

from annealctrl.headroom import aggregate_frontier, record_headroom, sweep_control_frontier


def trial(loss, ambiguity=1e-6, candidate_id="c", index=0):
    return {"loss": loss, "success": 1 - loss, "loss_ambiguity_indicator": ambiguity,
            "state_error_diagnostic": ambiguity / 2, "norm_error": 1e-14,
            "accepted_steps": 128, "total_integrator_steps": 384, "propagation_calls": 2,
            "candidate_id": candidate_id, "evaluation_index": index, "incumbent_loss": loss,
            "waveform": {"tau_knots": [0.0, 1.0], "s_knots": [0.0, 1.0]},
            "offline_scoring_seconds": 0.01, "elapsed_seconds": 0.01}


def family(best_loss, *, linear_loss, budget=4, ambiguity=1e-6, parameter_free=False):
    records = [trial(linear_loss, ambiguity, "linear_incumbent", 0)]
    if not parameter_free:
        for index in range(1, budget):
            loss = best_loss if index == budget - 1 else linear_loss
            records.append(trial(loss, ambiguity, f"t{index}", index))
    best_index = int(np.argmin([r["loss"] for r in records]))
    return {"best_loss": min(r["loss"] for r in records), "best_index": best_index,
            "best": records[best_index], "records": records, "n_evaluations": len(records),
            "objective_budget": 1 if parameter_free else budget, "parameter_free": parameter_free,
            "online_adaptation": False, "search_seconds": 0.05,
            "total_integrator_steps": sum(r["total_integrator_steps"] for r in records),
            "reference_status": "best_found_within_evaluated_candidates",
            "exact_switching_waveforms": True}


def benchmark(*, linear_loss=0.60, bests=None, ambiguity=1e-6, budget=4, **identity):
    bests = {"linear": linear_loss, "one_window": 0.50, "eight_bin": 0.55} if bests is None else bests
    families = {name: family(loss, linear_loss=linear_loss, budget=budget, ambiguity=ambiguity,
                             parameter_free=name == "linear")
                for name, loss in bests.items()}
    return {"schema_version": 2, "record_id": identity.get("record_id", "p0_e0_k0_t0"),
            "parent_id": identity.get("parent_id", "p0"), "family": identity.get("family", "spin_glass"),
            "logical_n": 3, "physical_n": 6, "runtime": 2.0,
            "split": identity.get("split", "validation"), "seed": 0,
            "families": families, "privileged_teachers": {},
            "budget_per_tunable_family": budget, "test_adaptation_explicitly_allowed": False,
            "total_objective_calls": sum(f["n_evaluations"] for f in families.values()),
            "wall_seconds": 0.2}


# --- headroom arithmetic -----------------------------------------------------

def test_headroom_is_linear_minus_best_found_across_families():
    result = record_headroom(benchmark(linear_loss=0.60, bests={"linear": 0.60, "one_window": 0.50}))
    assert result["linear_loss"] == pytest.approx(0.60)
    assert result["best_found_loss"] == pytest.approx(0.50)
    assert result["headroom"] == pytest.approx(0.10)
    assert result["best_family"] == "one_window"


def test_family_restriction_loss_is_zero_for_the_winner_and_positive_elsewhere():
    result = record_headroom(benchmark(bests={"linear": 0.60, "one_window": 0.50, "eight_bin": 0.55}))
    restriction = result["family_restriction_loss"]
    assert restriction["one_window"] == pytest.approx(0.0)
    assert restriction["eight_bin"] == pytest.approx(0.05)
    assert restriction["linear"] == pytest.approx(0.10)
    assert all(value >= 0 for value in restriction.values())


def test_headroom_is_labelled_a_finite_budget_lower_bound_not_an_optimum():
    result = record_headroom(benchmark())
    assert result["reference_status"] == "best_found_within_evaluated_candidates"
    assert result["headroom_is_global_optimum"] is False


# --- the censoring rule (ADR-0002) -------------------------------------------

def test_headroom_below_its_own_numerical_ambiguity_is_censored():
    # Two losses each uncertain at 1e-3 cannot support a 1e-4 difference.
    result = record_headroom(benchmark(linear_loss=0.50, bests={"linear": 0.50, "one_window": 0.4999},
                                       ambiguity=1e-3))
    assert result["headroom"] == pytest.approx(1e-4)
    assert result["resolution_status"] == "censored_numerical"
    assert result["combined_loss_ambiguity"] == pytest.approx(2e-3)


def test_headroom_well_above_its_ambiguity_is_resolved():
    result = record_headroom(benchmark(linear_loss=0.50, bests={"linear": 0.50, "one_window": 0.30},
                                       ambiguity=1e-5))
    assert result["resolution_status"] == "resolved"


def test_censoring_margin_is_configurable_and_recorded():
    payload = benchmark(linear_loss=0.50, bests={"linear": 0.50, "one_window": 0.495}, ambiguity=1e-3)
    assert record_headroom(payload, ambiguity_margin=1.0)["resolution_status"] == "resolved"
    strict = record_headroom(payload, ambiguity_margin=5.0)
    assert strict["resolution_status"] == "censored_numerical"
    assert strict["ambiguity_margin"] == 5.0


def test_negative_or_nonfinite_margin_is_refused():
    for bad in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="ambiguity_margin"):
            record_headroom(benchmark(), ambiguity_margin=bad)


# --- internal consistency audits ---------------------------------------------

def test_a_family_that_beats_its_own_linear_incumbent_is_fine_but_one_that_loses_is_flagged():
    payload = benchmark(bests={"linear": 0.60, "one_window": 0.50})
    # Corrupt the family so its reported best is worse than the linear trial it ran.
    payload["families"]["one_window"]["best_loss"] = 0.70
    payload["families"]["one_window"]["best"] = trial(0.70)
    result = record_headroom(payload)
    assert result["linear_incumbent_respected"]["one_window"] is False
    assert result["audit_failures"], "a discarded feasible incumbent must not be silent"


def test_disagreeing_linear_trials_across_families_are_detected():
    payload = benchmark(bests={"linear": 0.60, "one_window": 0.50, "eight_bin": 0.55})
    payload["families"]["eight_bin"]["records"][0]["loss"] = 0.58   # same waveform, different loss
    result = record_headroom(payload)
    assert result["linear_reference_spread"] == pytest.approx(0.02)
    assert result["linear_reference_consistent"] is False
    assert result["audit_failures"]


def test_consistent_linear_trials_pass_the_audit():
    result = record_headroom(benchmark())
    assert result["linear_reference_consistent"] is True
    assert result["audit_failures"] == []


def test_at_least_one_family_is_required():
    payload = benchmark()
    payload["families"] = {}
    with pytest.raises(ValueError, match="famil"):
        record_headroom(payload)


def test_a_benchmark_without_any_linear_trial_is_refused():
    payload = benchmark(bests={"one_window": 0.5})
    payload["families"]["one_window"]["records"][0]["candidate_id"] = "not_the_incumbent"
    payload["families"]["one_window"]["records"][0]["evaluation_index"] = 3
    with pytest.raises(ValueError, match="linear"):
        record_headroom(payload)


# --- low-headroom and ratio guards -------------------------------------------

def test_relative_headroom_is_withheld_when_the_linear_reference_is_near_zero():
    result = record_headroom(benchmark(linear_loss=1e-4, bests={"linear": 1e-4, "one_window": 0.0},
                                       ambiguity=1e-9))
    assert result["relative_headroom"] is None
    assert result["low_headroom_reference"] is True


def test_relative_headroom_is_reported_when_the_reference_is_substantial():
    result = record_headroom(benchmark(linear_loss=0.50, bests={"linear": 0.50, "one_window": 0.25},
                                       ambiguity=1e-9))
    assert result["relative_headroom"] == pytest.approx(0.5)
    assert result["low_headroom_reference"] is False


# --- cost accounting ---------------------------------------------------------

def test_objective_calls_and_integrator_steps_are_totalled_per_family():
    result = record_headroom(benchmark(budget=4, bests={"linear": 0.6, "one_window": 0.5, "eight_bin": 0.55}))
    assert result["objective_calls"]["linear"] == 1
    assert result["objective_calls"]["one_window"] == 4
    assert result["total_objective_calls"] == 9
    assert result["total_integrator_steps"] > 0


def test_output_is_json_safe():
    payload = benchmark()
    payload["families"]["one_window"]["records"][1]["loss_ambiguity_indicator"] = float("inf")
    text = json.dumps(record_headroom(payload), allow_nan=False)
    assert "NaN" not in text and "Infinity" not in text


# --- shared-bank statistics from the stored record ---------------------------

def test_stored_bank_statistics_are_reported_separately_from_the_exact_frontier():
    record = {"candidate_losses": np.array([0.62, 0.58, 0.55]),
              "candidate_ids": np.array(["linear", "one_window_0", "sobol_0000"])}
    result = record_headroom(benchmark(linear_loss=0.60, bests={"linear": 0.60, "one_window": 0.50}),
                             record=record)
    assert result["bank"]["best_loss"] == pytest.approx(0.55)
    assert result["bank"]["spread"] == pytest.approx(0.07)
    assert result["bank"]["control_set"] == "resampled_nine_knot_shared_bank"
    # The exact-waveform frontier stays the headline quantity; they are not merged.
    assert result["best_found_loss"] == pytest.approx(0.50)


# --- sweep -------------------------------------------------------------------

def dataset(tmp_path, *, parents=2, runtimes=(2.0,), candidates=4):
    from annealctrl.pipeline import generate_dataset
    config = {"seed": 77, "parents": parents, "families": ["spin_glass"], "logical_qubits": 3,
              "chain_lengths": [1, 1, 1], "variants": [{"shape": "path", "ports": 1}],
              "chain_strengths": [1.5], "runtimes": list(runtimes), "candidates": candidates,
              "spectral_points": 3, "steps": 16, "max_steps": 1024,
              "label_state_tolerance": 0.005, "max_physical_qubits": 4,
              "teacher": {"mode": "none"}}
    generate_dataset(config, tmp_path / "data")
    return tmp_path / "data"


@pytest.mark.parametrize("split", ["train", "validation"])
def test_sweep_runs_the_frontier_over_a_split_and_writes_rows(tmp_path, split):
    data = dataset(tmp_path, parents=4)
    manifest = sweep_control_frontier(data, output=tmp_path / "sweep", split=split,
                                      families=("linear", "one_window"), budget=3,
                                      tolerance=5e-3, initial_steps=16, max_steps=512)
    assert manifest["completed"] >= 1
    rows = [json.loads(line) for line in (tmp_path / "sweep" / "rows.jsonl").read_text().splitlines()]
    assert rows and all(row["status"] == "ok" for row in rows)
    first = rows[0]["result"]
    assert first["split"] == split
    assert set(first["objective_calls"]) == {"linear", "one_window"}
    assert first["resolution_status"] in {"resolved", "censored_numerical"}


def test_sweep_refuses_the_test_split_without_explicit_adaptation(tmp_path):
    data = dataset(tmp_path, parents=4)
    with pytest.raises(ValueError, match="allow_test_adaptation"):
        sweep_control_frontier(data, output=tmp_path / "sweep", split="test",
                               families=("linear", "one_window"), budget=2)


def test_sweep_resumes_without_recomputing(tmp_path):
    data = dataset(tmp_path, parents=4)
    kwargs = dict(split="train", families=("linear", "one_window"), budget=2,
                  tolerance=5e-3, initial_steps=16, max_steps=512)
    first = sweep_control_frontier(data, output=tmp_path / "sweep", **kwargs)
    second = sweep_control_frontier(data, output=tmp_path / "sweep", resume=True, **kwargs)
    assert second["completed"] == 0 and second["skipped"] == first["completed"]


def test_sweep_dry_run_reports_the_requested_budget_without_running(tmp_path):
    data = dataset(tmp_path, parents=4)
    plan = sweep_control_frontier(data, output=tmp_path / "sweep", split="train",
                                  families=("linear", "one_window", "eight_bin"), budget=8,
                                  dry_run=True)
    assert plan["dry_run"] is True
    assert plan["requested_objective_calls_per_unit"] == 1 + 8 + 8
    assert not (tmp_path / "sweep").exists()


# --- aggregation -------------------------------------------------------------

def rows_for(values, *, parent_prefix="p", statuses=None, split="validation"):
    statuses = statuses or ["resolved"] * len(values)
    return [{"parent_id": f"{parent_prefix}{index}", "record_id": f"{parent_prefix}{index}_r0",
             "split": split, "family": "spin_glass", "logical_n": 3, "physical_n": 6,
             "runtime": 2.0, "headroom": value, "linear_loss": 0.6,
             "best_found_loss": 0.6 - value, "best_family": "one_window",
             "resolution_status": status, "combined_loss_ambiguity": 1e-6,
             "family_restriction_loss": {"linear": value, "one_window": 0.0},
             "total_objective_calls": 9, "total_integrator_steps": 1000,
             "audit_failures": []}
            for index, (value, status) in enumerate(zip(values, statuses))]


def test_aggregate_reports_quantiles_and_tails_not_only_a_mean():
    summary = aggregate_frontier(rows_for([0.01, 0.05, 0.10, 0.20, 0.40]))
    quantiles = summary["headroom"]["quantiles"]
    assert set(quantiles) == {"p10", "p25", "p50", "p75", "p90"}
    assert quantiles["p50"] == pytest.approx(0.10)
    assert summary["headroom"]["max"] == pytest.approx(0.40)
    assert summary["headroom"]["n_parents"] == 5


def test_censored_rows_are_excluded_from_headroom_statistics_but_counted():
    rows = rows_for([0.30, 0.0001, 0.40], statuses=["resolved", "censored_numerical", "resolved"])
    summary = aggregate_frontier(rows)
    assert summary["censored_fraction"] == pytest.approx(1 / 3)
    assert summary["headroom"]["n_parents"] == 2
    assert summary["headroom"]["mean"] == pytest.approx(0.35)
    assert summary["n_records"] == 3


def test_aggregation_averages_variants_within_a_parent_before_pooling():
    rows = rows_for([0.10, 0.30])
    for row in rows:
        row["parent_id"] = "shared"
        row["record_id"] = f"shared_{row['headroom']}"
    summary = aggregate_frontier(rows)
    assert summary["headroom"]["n_parents"] == 1
    assert summary["headroom"]["mean"] == pytest.approx(0.20)


def test_aggregate_bootstraps_the_headroom_over_independent_parents():
    summary = aggregate_frontier(rows_for([0.05, 0.10, 0.15, 0.20, 0.25, 0.30]),
                                 bootstrap_resamples=200, seed=0)
    interval = summary["headroom"]["parent_bootstrap_ci"]
    assert interval["low"] <= summary["headroom"]["mean"] <= interval["high"]
    assert interval["resamples"] == 200
    assert interval["unit_of_independence"] == "logical_parent"


def test_aggregate_reports_the_family_restriction_distribution():
    summary = aggregate_frontier(rows_for([0.1, 0.2, 0.3]))
    assert "one_window" in summary["family_restriction_loss"]
    assert summary["family_restriction_loss"]["one_window"]["mean"] == pytest.approx(0.0)
    assert summary["family_restriction_loss"]["linear"]["mean"] == pytest.approx(0.2)


def test_aggregate_surfaces_audit_failures_and_never_hides_them():
    rows = rows_for([0.1, 0.2])
    rows[1]["audit_failures"] = ["linear_incumbent_discarded:one_window"]
    summary = aggregate_frontier(rows)
    assert summary["records_with_audit_failures"] == 1
    assert "linear_incumbent_discarded:one_window" in summary["audit_failure_reasons"]


def test_aggregate_refuses_to_pool_across_splits():
    rows = rows_for([0.1], split="train") + rows_for([0.2], parent_prefix="q", split="test")
    with pytest.raises(ValueError, match="split"):
        aggregate_frontier(rows)


def test_aggregate_of_an_entirely_censored_population_reports_no_signal():
    summary = aggregate_frontier(rows_for([1e-8, 2e-8], statuses=["censored_numerical"] * 2))
    assert summary["censored_fraction"] == 1.0
    assert summary["headroom"]["n_parents"] == 0
    assert summary["verdict"] == "no_resolved_headroom"


def test_aggregate_of_an_empty_row_set_is_refused():
    with pytest.raises(ValueError, match="nonempty"):
        aggregate_frontier([])


# --- report ------------------------------------------------------------------

def test_frontier_report_writes_summary_json_and_markdown(tmp_path):
    from annealctrl.headroom import frontier_report
    data = dataset(tmp_path, parents=6, runtimes=(2.0, 8.0))
    sweep_control_frontier(data, output=tmp_path / "sweep", split="train",
                           families=("linear", "one_window"), budget=4,
                           tolerance=5e-3, initial_steps=16, max_steps=512)
    summary = frontier_report(tmp_path / "sweep", bootstrap_resamples=200)

    assert (tmp_path / "sweep" / "report" / "summary.json").exists()
    text = (tmp_path / "sweep" / "report" / "FRONTIER.md").read_text()
    assert "censored" in text.lower()
    assert "not a global control optimum" in text
    assert summary["n_records"] >= 1


def test_frontier_report_refuses_a_sweep_with_no_successful_rows(tmp_path):
    from annealctrl.headroom import frontier_report
    from annealctrl.sweeps import run_sweep, SweepUnit
    run_sweep([SweepUnit("a", "fp")], lambda unit: (_ for _ in ()).throw(ArithmeticError("x")),
              output=tmp_path / "sweep", settings={}, command="control-sweep",
              source_hash="src", on_error="record")
    with pytest.raises(ValueError, match="no successful"):
        frontier_report(tmp_path / "sweep")


def test_frontier_report_counts_failed_units_in_the_summary(tmp_path):
    from annealctrl.headroom import frontier_report
    data = dataset(tmp_path, parents=6)
    sweep_control_frontier(data, output=tmp_path / "sweep", split="train",
                           families=("linear", "one_window"), budget=3,
                           tolerance=5e-3, initial_steps=16, max_steps=512)
    # A failed row from a later resumed attempt must remain visible. It carries the
    # run's real settings and source hashes, as a genuine failure row does.
    manifest = json.loads((tmp_path / "sweep" / "manifest.json").read_text())
    with (tmp_path / "sweep" / "rows.jsonl").open("a") as handle:
        handle.write(json.dumps({"unit_id": "zzz", "unit_key": "k", "fingerprint": "f",
                                 "settings_hash": manifest["settings_hash"],
                                 "source_hash": manifest["source_hash"], "status": "failed",
                                 "error_type": "ArithmeticError", "error": "gate"}) + "\n")
    summary = frontier_report(tmp_path / "sweep", bootstrap_resamples=100)
    assert summary["failed_units"] == 1
    assert summary["failed_unit_ids"] == ["zzz"]


# --- configuration -----------------------------------------------------------

def test_frontier_config_rejects_unknown_keys():
    from annealctrl.headroom import load_frontier_config
    with pytest.raises(ValueError, match="unknown"):
        load_frontier_config({"split": "validation", "budgett": 4})


def test_frontier_config_refuses_to_enable_test_adaptation():
    from annealctrl.headroom import load_frontier_config
    with pytest.raises(ValueError, match="allow_test_adaptation"):
        load_frontier_config({"allow_test_adaptation": True})


def test_frontier_config_fills_declared_defaults():
    from annealctrl.headroom import load_frontier_config
    sweep, report = load_frontier_config({"budget": 8})
    assert sweep["budget"] == 8
    assert sweep["split"] == "validation"
    assert sweep["families"] == list(("linear", "one_window", "two_window", "eight_bin", "pause"))
    assert sweep["ambiguity_margin"] == 1.0
    assert report["bootstrap_resamples"] == 10000


def test_sweep_names_the_missing_dataset_manifest(tmp_path):
    for dry in (True, False):
        with pytest.raises(FileNotFoundError, match="manifest.json"):
            sweep_control_frontier(tmp_path / "absent", output=tmp_path / "sweep",
                                   split="train", families=("linear",), budget=1, dry_run=dry)


def test_sharded_sweeps_cover_every_record_and_merge_in_the_report(tmp_path):
    from annealctrl.headroom import frontier_report
    data = dataset(tmp_path, parents=6, runtimes=(2.0, 8.0))
    kwargs = dict(split="train", families=("linear", "one_window"), budget=2,
                  tolerance=5e-3, initial_steps=16, max_steps=512)
    shards = [sweep_control_frontier(data, output=tmp_path / f"shard{i}", shard=i,
                                     shard_count=3, **kwargs) for i in range(3)]
    whole = sweep_control_frontier(data, output=tmp_path / "whole", **kwargs)

    assert sum(shard["completed"] for shard in shards) == whole["completed"]
    merged = frontier_report([tmp_path / f"shard{i}" for i in range(3)],
                             output=tmp_path / "merged", bootstrap_resamples=100)
    single = frontier_report(tmp_path / "whole", bootstrap_resamples=100)
    assert merged["n_records"] == single["n_records"]
    assert merged["headroom"]["mean"] == pytest.approx(single["headroom"]["mean"])


def test_merging_shards_with_different_settings_is_refused(tmp_path):
    from annealctrl.headroom import frontier_report
    data = dataset(tmp_path, parents=6)
    base = dict(split="train", families=("linear", "one_window"),
                tolerance=5e-3, initial_steps=16, max_steps=512)
    sweep_control_frontier(data, output=tmp_path / "a", budget=2, **base)
    sweep_control_frontier(data, output=tmp_path / "b", budget=3, **base)
    with pytest.raises(ValueError, match="different settings"):
        frontier_report([tmp_path / "a", tmp_path / "b"], output=tmp_path / "merged")
