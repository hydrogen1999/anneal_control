"""G5: amortised cost. What does a decision actually cost, and when does learning pay?"""
import json

import pytest

from annealctrl.costs import (
    amortization_curve,
    compare_amortized,
    crossover_deployments,
    experiment_costs,
    search_reference_cost,
)


# --- the curve ---------------------------------------------------------------

def test_curve_is_offline_over_m_plus_online():
    curve = amortization_curve(offline_seconds=1000.0, online_seconds=2.0, deployments=[1, 10, 100])
    assert curve[0] == pytest.approx(1002.0)
    assert curve[1] == pytest.approx(102.0)
    assert curve[2] == pytest.approx(12.0)


def test_curve_never_falls_below_the_online_cost():
    curve = amortization_curve(offline_seconds=1e6, online_seconds=3.0, deployments=[10**9])
    assert curve[0] > 3.0


def test_curve_refuses_nonpositive_deployment_counts():
    for bad in ([0], [-1], [1, 0]):
        with pytest.raises(ValueError, match="deployment"):
            amortization_curve(offline_seconds=1.0, online_seconds=1.0, deployments=bad)


def test_curve_refuses_negative_costs():
    with pytest.raises(ValueError, match="nonnegative"):
        amortization_curve(offline_seconds=-1.0, online_seconds=1.0, deployments=[1])


# --- crossover ---------------------------------------------------------------

def test_crossover_is_the_first_deployment_count_where_learning_is_cheaper():
    # Learning: 1000 s offline, 2 s online. Search: no offline, 50 s online.
    m = crossover_deployments(offline_seconds=1000.0, online_seconds=2.0,
                              reference_online_seconds=50.0)
    assert m == 21          # 1000/21 + 2 = 49.6 < 50; at M=20 it is 52 > 50
    assert crossover_deployments(offline_seconds=0.0, online_seconds=2.0,
                                 reference_online_seconds=50.0) == 1


def test_crossover_is_none_when_online_cost_alone_already_loses():
    assert crossover_deployments(offline_seconds=10.0, online_seconds=60.0,
                                 reference_online_seconds=50.0) is None


def test_crossover_refuses_a_nonpositive_reference():
    with pytest.raises(ValueError, match="reference"):
        crossover_deployments(offline_seconds=1.0, online_seconds=1.0, reference_online_seconds=0.0)


# --- reading a completed experiment -----------------------------------------

def write_experiment(tmp_path, *, methods=("summary", "physical"), seeds=(0, 1)):
    root = tmp_path / "run"
    (root / "evaluations").mkdir(parents=True)
    runs = {}
    for method in methods:
        for seed in seeds:
            runs[f"{method}/seed_{seed}"] = {
                "method": method, "seed": seed, "trained": True, "evaluated": True,
                "observed_training_seconds": 400.0 + 10 * seed,
                "training_seconds": 400.0 + 10 * seed,
                "training_cost_complete": True,
                "evaluation_seconds": 30.0}
            (root / "evaluations" / f"{method}__seed_{seed}.json").write_text(json.dumps({
                "method": method, "training_seed": seed, "n_records": 100, "n_test_parents": 20,
                "mean_loss": 0.5, "mean_linear_loss": 0.6,
                "bank_inference_seconds": 1.0 + seed,
                "costs": {"bank_inference_seconds": 1.0 + seed,
                          "stored_test_bank_label_seconds": 900.0,
                          "online_simulator_search_calls": 0,
                          "direct_inference_seconds": 2.0,
                          "direct_offline_scoring_calls": 100},
                "read_budget_at_deployment": {"candidates_scored_by_critic": 8}}))
    (root / "experiment.json").write_text(json.dumps({
        "status": "complete", "stages": {"generate": {"seconds": 700.0, "record_count": 4320}},
        "runs": runs}))
    return root


def test_experiment_costs_separate_offline_from_online(tmp_path):
    costs = experiment_costs(write_experiment(tmp_path))
    assert costs["data_generation_seconds"] == pytest.approx(700.0)
    summary = costs["methods"]["summary"]
    assert summary["mean_training_seconds"] == pytest.approx(405.0)
    assert summary["n_seeds"] == 2
    # Offline is generation plus this method's own training, not every method's.
    assert summary["offline_seconds"] == pytest.approx(700.0 + 405.0)
    assert summary["online_seconds_per_instance"] > 0


def test_online_cost_is_per_instance_not_per_split(tmp_path):
    costs = experiment_costs(write_experiment(tmp_path))
    summary = costs["methods"]["summary"]
    # 1.0 and 2.0 seconds of inference over 100 records, averaged over two seeds.
    assert summary["online_seconds_per_instance"] == pytest.approx(0.015)


def test_experiment_costs_report_the_critic_budget(tmp_path):
    costs = experiment_costs(write_experiment(tmp_path))
    assert costs["methods"]["summary"]["candidates_scored_by_critic"] == 8
    assert costs["methods"]["summary"]["online_simulator_calls"] == 0


def test_experiment_costs_flag_an_incomplete_training_cost(tmp_path):
    root = write_experiment(tmp_path)
    manifest = json.loads((root / "experiment.json").read_text())
    manifest["runs"]["summary/seed_0"]["training_cost_complete"] = False
    (root / "experiment.json").write_text(json.dumps(manifest))
    costs = experiment_costs(root)
    assert costs["methods"]["summary"]["training_cost_complete"] is False
    assert "summary" in costs["methods_with_incomplete_training_cost"]


def test_experiment_costs_refuse_an_unfinished_run(tmp_path):
    root = write_experiment(tmp_path)
    manifest = json.loads((root / "experiment.json").read_text())
    manifest["runs"]["physical/seed_1"]["evaluated"] = False
    (root / "experiment.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="not evaluated"):
        experiment_costs(root)


# --- the search reference ----------------------------------------------------

def frontier_sweep(tmp_path, name="frontier", *, per_record_seconds=40.0, calls=257):
    root = tmp_path / name
    root.mkdir(parents=True)
    with (root / "rows.jsonl").open("w") as handle:
        for index in range(5):
            handle.write(json.dumps({
                "unit_id": f"r{index}", "unit_key": f"k{index}", "fingerprint": f"f{index}",
                "settings_hash": "s", "source_hash": "h", "status": "ok",
                "result": {"record_id": f"r{index}", "parent_id": f"p{index}", "split": "validation",
                           "search_seconds": per_record_seconds,
                           "total_objective_calls": calls,
                           "total_integrator_steps": 1000}}) + "\n")
    return root


def test_search_reference_cost_is_per_instance(tmp_path):
    reference = search_reference_cost([frontier_sweep(tmp_path)])
    assert reference["n_records"] == 5
    assert reference["online_seconds_per_instance"] == pytest.approx(40.0)
    assert reference["objective_calls_per_instance"] == pytest.approx(257.0)
    assert reference["offline_seconds"] == 0.0


def test_search_reference_has_no_offline_cost_to_amortise(tmp_path):
    reference = search_reference_cost([frontier_sweep(tmp_path)])
    assert reference["amortisable"] is False
    assert "every new instance pays" in reference["note"]


def test_search_reference_refuses_an_empty_sweep(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no successful"):
        search_reference_cost([empty])


# --- the comparison ----------------------------------------------------------

def test_compare_reports_a_curve_and_a_crossover_per_method(tmp_path):
    costs = experiment_costs(write_experiment(tmp_path))
    reference = search_reference_cost([frontier_sweep(tmp_path)])
    result = compare_amortized(costs, reference, deployments=[1, 10, 100, 1000])

    summary = result["methods"]["summary"]
    assert summary["curve"]["1"] == pytest.approx(700.0 + 405.0 + 0.015)
    assert summary["curve"]["1000"] < summary["curve"]["1"]
    assert summary["crossover_deployments"] > 1
    assert result["reference"]["online_seconds_per_instance"] == pytest.approx(40.0)


def test_compare_states_the_unamortised_cost_beside_the_curve(tmp_path):
    result = compare_amortized(experiment_costs(write_experiment(tmp_path)),
                               search_reference_cost([frontier_sweep(tmp_path)]),
                               deployments=[1, 1000])
    summary = result["methods"]["summary"]
    assert summary["unamortised_seconds"] == pytest.approx(summary["curve"]["1"])
    assert result["shows_unamortised_cost"] is True


def test_compare_refuses_a_deployment_list_that_hides_the_unamortised_point(tmp_path):
    with pytest.raises(ValueError, match="M=1"):
        compare_amortized(experiment_costs(write_experiment(tmp_path)),
                          search_reference_cost([frontier_sweep(tmp_path)]),
                          deployments=[100, 1000])


def test_cost_report_cli_runs_end_to_end(tmp_path, capsys):
    from annealctrl.workflow_cli import main
    run = write_experiment(tmp_path)
    sweep = frontier_sweep(tmp_path)
    main(["cost-report", "--run", str(run), "--reference-sweep", str(sweep),
          "--output", str(tmp_path / "cost.json"), "--deployments", "1", "100"])
    result = json.loads((tmp_path / "cost.json").read_text())
    assert result["shows_unamortised_cost"] is True
    assert set(result["methods"]) == {"summary", "physical"}
    assert "crossover" in capsys.readouterr().out


def test_cost_report_cli_refuses_a_deployment_list_without_one(tmp_path):
    from annealctrl.workflow_cli import main
    run = write_experiment(tmp_path)
    sweep = frontier_sweep(tmp_path)
    with pytest.raises(ValueError, match="M=1"):
        main(["cost-report", "--run", str(run), "--reference-sweep", str(sweep),
              "--output", str(tmp_path / "c.json"), "--deployments", "100"])
