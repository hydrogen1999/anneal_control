"""Public workflow integration and statistics barriers."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from annealctrl.experiments import (expand_tuning, load_experiment, output_lock,
    plan_experiment, run_experiment, validate_experiment, tune_validation)
from annealctrl.reporting import aggregate_evaluations, audit_dataset, build_report
from annealctrl.workflow_cli import doctor, main


def small_experiment():
    return {"schema_version": 1,
        "dataset": {"seed": 45, "parents": 6, "families": ["spin_glass"],
            "logical_qubits": 3, "chain_lengths": [1, 1, 1],
            "variants": [{"shape": "path", "ports": 1}], "chain_strengths": [1.5],
            "runtimes": [2.], "candidates": 3, "spectral_points": 3, "steps": 16,
            "max_steps": 512, "label_state_tolerance": .001, "max_physical_qubits": 3},
        "seeds": [0], "methods": [{"name": "hierarchy", "model": {"width": 8}}],
        "training": {"epochs": 2, "patience": 2},
        "execution": {"device": "cpu", "threads": 1},
        "evaluation": {"direct": False, "n_resamples": 20}, "report": {"plots": False, "bootstrap_resamples": 20}}


def test_bundled_configs_validate_without_execution():
    root = Path(__file__).resolve().parents[1]
    for name in ("experiment_smoke.json", "experiment_research.json"):
        cfg = load_experiment(root / "configs" / name)
        assert plan_experiment(cfg)["training_runs"] > 0


@pytest.mark.parametrize("key,value", [("typo", 2), ("seeds", [0, 0]), ("methods", []),
    ("training", {"epochz": 2}), ("execution", {"workers": 0})])
def test_bad_config_rejected(key, value):
    cfg = small_experiment()
    cfg[key] = value
    with pytest.raises(ValueError):
        validate_experiment(cfg)


def test_output_lock_never_stolen(tmp_path):
    with output_lock(tmp_path):
        with pytest.raises(RuntimeError, match="locked"):
            with output_lock(tmp_path):
                pass
    assert not (tmp_path / ".experiment.lock").exists()


def test_full_workflow_resume_report_and_inference(tmp_path):
    pytest.importorskip("torch")
    cfg = small_experiment()
    root = tmp_path / "experiment"
    result = run_experiment(cfg, root)
    assert result["status"] == "complete"
    assert (root / "paper" / "table_results.tex").exists()
    assert json.loads((root / "data_audit_train.json").read_text())["split_scope"] == "train"
    checkpoint = root / "models" / "hierarchy" / "seed_0" / "best.pt"
    before = checkpoint.read_bytes()
    assert run_experiment(cfg, root, resume=True)["status"] == "complete"
    assert before == checkpoint.read_bytes()
    with pytest.raises(FileExistsError):
        run_experiment(cfg, root)
    wrong = deepcopy(cfg)
    wrong["training"]["epochs"] += 1
    with pytest.raises(ValueError, match="config/source"):
        run_experiment(wrong, root, resume=True)
    record = next((root / "data" / "records").glob("*.npz"))
    main(["infer", "--record", str(record), "--checkpoint", str(checkpoint), "--output", str(tmp_path / "wave.json")])
    wave = json.loads((tmp_path / "wave.json").read_text())
    assert wave["true_outcome_observed"] is False and wave["s_knots"][-1] == 1
    # A standalone report cannot silently report an incomplete ablation matrix.
    manifest = json.loads((root / "experiment.json").read_text())
    manifest["config"]["seeds"].append(1)
    (root / "experiment.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="predeclared"):
        build_report(root, plots=False)


def test_tuning_never_runs_test_evaluation(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from annealctrl import benchmarking, reporting
    monkeypatch.setattr(benchmarking, "evaluate_checkpoint", lambda *a, **k: pytest.fail("Tuning accessed test evaluator"))
    original = reporting.audit_dataset
    def audit(*args, **kwargs):
        assert kwargs.get("split") == "train"
        return original(*args, **kwargs)
    monkeypatch.setattr(reporting, "audit_dataset", audit)
    cfg = {"experiment": small_experiment(), "search_space": {"training.learning_rate": [.001, .002]}, "max_trials": 2}
    result = tune_validation(cfg, tmp_path / "tune")
    assert result["test_evaluated"] is False and len(result["trials"]) == 2
    assert not list((tmp_path / "tune").glob("trial_*/evaluations/*.json"))
    assert (tmp_path / "tune" / "selected_experiment.json").exists()


def test_tuning_budget_and_invalid_grid():
    cfg = {"experiment": small_experiment(), "search_space": {"training.learning_rate": [.1, .2]}, "max_trials": 1}
    with pytest.raises(ValueError, match="exceeds"):
        expand_tuning(cfg)


def eval_fixture(seed, losses):
    rows = [{"record_id": str(i), "parent_id": str(i // 2), "selected_loss": loss,
             "linear_loss": .5, "global_loss": .4, "bank_best_loss": .1}
            for i, loss in enumerate(losses)]
    return {"method": "m", "training_seed": seed, "records": rows}


def test_report_keeps_parent_and_seed_uncertainty_separate():
    a, b = eval_fixture(0, [.1, .3, .4, .6]), eval_fixture(1, [.3, .5, .6, .8])
    report = aggregate_evaluations([a, b], bootstrap_resamples=20)
    row = report["summary"][0]
    assert row["parents"] == 2 and row["seeds"] == 2 and row["records"] == 4
    assert row["mean_loss"] == pytest.approx(.45)
    assert row["seed_std_loss"] == pytest.approx(np.std([.35, .55], ddof=1))
    assert row["paired_differences"]["global_loss"]["mean_difference"] == pytest.approx(.05)
    b["records"][0]["global_loss"] = .2
    with pytest.raises(ValueError, match="Reference"):
        aggregate_evaluations([a, b])


def test_doctor_cpu_is_nonmutating():
    assert "capabilities" in doctor()


def test_experiment_config_allows_underscore_annotations_but_still_catches_typos():
    from annealctrl.experiments import validate_experiment
    base = {
        "schema_version": 1,
        "dataset": {"seed": 1, "parents": 3, "families": ["spin_glass"], "logical_qubits": 3,
                    "chain_lengths": [1, 1, 1], "variants": [{"shape": "path", "ports": 1}],
                    "chain_strengths": [1.5], "runtimes": [2.0], "candidates": 2,
                    "spectral_points": 3, "steps": 8, "max_steps": 128,
                    "label_state_tolerance": 0.005, "max_physical_qubits": 3},
        "seeds": [0], "methods": [{"name": "m", "model": {"width": 8}}],
    }
    # Underscore-prefixed keys are free-form annotations; the dataset configs
    # already use that convention and it can never collide with a real key.
    validate_experiment({**base, "_execution_note": "measured on apollo"})
    # A near-miss of a real key must still be rejected.
    with pytest.raises(ValueError, match="Unknown experiment keys"):
        validate_experiment({**base, "excution": {"device": "cpu"}})
