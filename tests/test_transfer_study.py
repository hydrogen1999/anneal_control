"""End-to-end transfer audit and resume checks, not performance claims."""
import copy
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from annealctrl import transfer_study as study
from annealctrl.learning import fit_records
from annealctrl.models import AnnealController
from annealctrl.pipeline import generate_dataset, load_records


def dataset_config():
    return {"seed": 17, "parents": 10, "families": ["spin_glass"], "logical_qubits": 3,
            "chain_lengths": [1, 1, 1], "variants": [{"shape": "path", "ports": 1}],
            "chain_strengths": [1.5], "runtimes": [2.0], "candidates": 4,
            "spectral_points": 3, "steps": 16, "max_steps": 1024,
            "label_state_tolerance": .005, "max_physical_qubits": 3, "teacher": {"mode": "none"}}


@pytest.fixture
def prepared(tmp_path):
    torch.set_num_threads(1)
    data = tmp_path / "data"
    generate_dataset(dataset_config(), data)
    train, validation = load_records(data, "train"), load_records(data, "validation")
    checkpoints = []
    for seed in (0, 1):
        path = tmp_path / f"seed{seed}.pt"
        fit_records(train, validation, model=AnnealController(encoder_variant="summary", width=8),
                    epochs=1, seed=seed, checkpoint=path)
        checkpoints.append({"method": "summary", "seed": seed, "path": str(path)})
    cfg = {"schema_version": 1, "source_data": str(data), "checkpoints": checkpoints,
           "targets": [{"name": "heldout", "data": str(data), "axes": ["heldout_parents"]}],
           "device": "cpu", "threads": 1, "bootstrap_resamples": 20}
    return cfg, tmp_path / "result"


def test_transfer_study_writes_raw_crossed_seed_results_and_resumes(prepared, monkeypatch):
    cfg, output = prepared
    result = study.run_transfer_study(cfg, output)
    assert result["costs"]["online_simulator_calls"] == 0
    assert result["costs"]["attempt_count"] == 2
    assert result["results"][0]["n_seeds"] == 2
    contrast = result["results"][0]["crossed_parent_seed_contrasts"]["source_global"]
    assert contrast["unit_of_resampling"] == "crossed_logical_parent_and_training_seed"
    assert result["transfer_axes"]["heldout"]["in_distribution_reference"]
    raw = list((output / "rows").glob("*.json"))
    assert len(raw) == 2
    assert json.loads(raw[0].read_text())[0]["source_global_loss"] is not None
    monkeypatch.setattr(study, "evaluate_transfer", lambda *a, **k: pytest.fail("completed inference rerun"))
    repeated = study.run_transfer_study(cfg, output, resume=True)
    assert repeated == result
    with pytest.raises(ValueError, match="already exists"):
        study.run_transfer_study(cfg, output)


def test_resume_refuses_modified_checkpoint_and_raw_artifact(prepared):
    cfg, output = prepared
    study.run_transfer_study(cfg, output)
    raw = next((output / "rows").glob("*.json"))
    text = raw.read_text()
    raw.write_text(text + " ")
    with pytest.raises(ValueError, match="artifact changed"):
        study.run_transfer_study(cfg, output, resume=True)
    raw.write_text(text)
    path = cfg["checkpoints"][0]["path"]
    with open(path, "ab") as stream:
        stream.write(b"changed")
    with pytest.raises(RuntimeError, match="input_drift"):
        study.run_transfer_study(cfg, output, resume=True)


def test_failed_attempt_retained_and_counted_on_retry(prepared, monkeypatch):
    cfg, output = prepared
    evaluate = study.evaluate_transfer
    calls = []
    def fail_once(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("deliberate failure")
        return evaluate(*args, **kwargs)
    monkeypatch.setattr(study, "evaluate_transfer", fail_once)
    with pytest.raises(RuntimeError, match="deliberate failure"):
        study.run_transfer_study(cfg, output)
    result = study.run_transfer_study(cfg, output, resume=True)
    assert result["costs"]["attempt_count"] == 3
    assert result["costs"]["completed_or_failed_receipts"] == 3
    assert not result["costs"]["timing_is_lower_bound"]
    assert any(json.loads(p.read_text())["status"] == "failed" for p in (output / "attempts").glob("*.result.json"))


def test_source_drift_cannot_complete_a_stage(prepared, monkeypatch):
    cfg, output = prepared
    evaluate = study.evaluate_transfer
    def changed(*args, **kwargs):
        result = evaluate(*args, **kwargs)
        monkeypatch.setattr(study, "source_hash", lambda: "changed")
        return result
    monkeypatch.setattr(study, "evaluate_transfer", changed)
    with pytest.raises(RuntimeError, match="source_drift"):
        study.run_transfer_study(cfg, output)
    assert not (output / "summary.json").exists()
    assert json.loads((output / "manifest.json").read_text())["status"] == "failed"


def test_checkpoint_seed_and_false_transfer_axis_refused_before_inference(prepared, monkeypatch):
    cfg, output = prepared
    cfg["checkpoints"][0]["seed"] = 999
    monkeypatch.setattr(study, "evaluate_transfer", lambda *a, **k: pytest.fail("inference started before provenance"))
    with pytest.raises(ValueError, match="saved training seed"):
        study.run_transfer_study(cfg, output)
    cfg["checkpoints"][0]["seed"] = 0
    cfg["targets"][0]["axes"] = ["topology"]
    with pytest.raises(ValueError, match="different observed topologies"):
        study.run_transfer_study(cfg, output.parent / "other")


@pytest.mark.parametrize("axis", ["family", "physical_size", "runtime"])
def test_declared_unseen_axes_require_disjoint_observed_values(axis):
    source = {"topology": "synthetic", "family": ["spin_glass"], "physical_size": [3, 4], "runtime": [2.]}
    with pytest.raises(ValueError, match="disjoint observed"):
        study.validate_axes(source, source, [axis])


def test_multiseed_summary_refuses_incomplete_parent_seed_panel():
    rows = []
    for seed, parent in ((0, "a"), (0, "b"), (1, "a")):
        rows.append(dict(target="t", method="m", seed=seed, parent_id=parent, logical_fingerprint=parent,
                         record_id=parent, selected_loss=.5, linear_loss=.6, source_global_loss=.55, bank_best_loss=.4))
    with pytest.raises(ValueError, match="complete parent-by-seed"):
        study.summarize_transfer_study(rows, bootstrap_resamples=20)


def test_resume_refuses_tampered_cost_receipt(prepared):
    cfg, output = prepared
    study.run_transfer_study(cfg, output)
    receipt = next((output / "attempts").glob("*.result.json"))
    value = json.loads(receipt.read_text())
    value["elapsed_seconds"] = 0
    receipt.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="receipt"):
        study.run_transfer_study(cfg, output, resume=True)


@pytest.mark.parametrize("field,value", [("topology", "pegasus"), ("family", ["unseen"]),
                                         ("physical_size", [16]), ("runtime", [100.])])
def test_shifted_target_cannot_be_mislabeled_in_distribution(field, value):
    source = {"topology": "synthetic", "family": ["spin_glass"], "physical_size": [3, 4], "runtime": [2.]}
    target = dict(source, **{field: value})
    with pytest.raises(ValueError, match="observed distribution shift"):
        study.validate_axes(source, target, ["heldout_parents"])
