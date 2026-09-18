"""Small end-to-end checks of the frozen protocol, not performance evidence."""
import copy
import hashlib
import json
import shutil

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from annealctrl import acquisition_study as study
from annealctrl.pipeline import generate_dataset, load_records


def _config():
    return {
        "schema_version": 1,
        "dataset": {"seed": 17, "parents": 10, "families": ["spin_glass"], "logical_qubits": 3,
                    "chain_lengths": [1, 1, 1], "variants": [{"shape": "path", "ports": 1}],
                    "chain_strengths": [1.5], "runtimes": [2.0], "candidates": 4,
                    "spectral_points": 3, "steps": 16, "max_steps": 1024,
                    "label_state_tolerance": 0.005, "max_physical_qubits": 3,
                    "teacher": {"mode": "none"}},
        "seeds": [0, 1, 2],
        "model": {"encoder_variant": "summary", "width": 8, "proposals": 3},
        "training": {"epochs": 1, "response_weight": 0.0},
        "execution": {"device": "cpu", "backend": "numpy", "threads": 1},
        "acquisition": {"n_extra": 3, "random_seed": 41, "logit_std": 1.0},
        "evaluation": {"direct": True, "tolerance": 0.005, "initial_steps": 16,
                       "max_steps": 1024, "n_resamples": 20},
        "latency": {"warmup": 1, "repeats": 1}, "bootstrap_resamples": 20,
    }


def test_mechanism_study_reuses_labels_and_pairs_frozen_initialization(tmp_path, monkeypatch):
    cfg = _config()
    cfg["seeds"] = [0]
    cfg["mechanism"] = {"enabled": True, "epochs": 2, "scopes": ["critic", "policy", "heads"]}
    assert study.plan_study(cfg)["training_runs"] == 11
    data, output = tmp_path / "data", tmp_path / "study"
    generate_dataset(cfg["dataset"], data)
    acquire, calls = study._acquire, []

    def counted(*args, **kwargs):
        calls.append(args[2])
        return acquire(*args, **kwargs)

    monkeypatch.setattr(study, "_acquire", counted)
    manifest = study.run_study(cfg, output, data_dir=data, stage="all")
    assert calls == ["bankext", "decoder_random", "policy"]
    baseline = manifest["runs"]["baseline/seed_0"]
    acquired = manifest["runs"]["policy/seed_0"]
    for scope in cfg["mechanism"]["scopes"]:
        for labels in ("original", "acquired"):
            arm = manifest["runs"][f"mechanism_{scope}_{labels}/seed_0"]
            assert arm["initialization_checkpoint_sha256"] == baseline["checkpoint_sha256"]
            assert arm["best_epoch"] == arm["last_epoch"] == 1
            assert arm["selection_mode"] == "fixed_epochs"
            if labels == "acquired":
                assert arm["acquisition_sha256"] == acquired["acquisition_sha256"]
                assert arm["acquisition_cost"]["objective_calls"] == 0
                assert arm["shared_acquisition_from"] == "policy/seed_0"
            else:
                assert "acquisition" not in arm
    summary = json.loads((output / "summary.json").read_text())
    assert set(summary["mechanism_contrasts"]) == {"critic", "policy", "heads"}
    # Frozen policy is exactly the same in both critic-only fits, so the oracle
    # best proposal loss is identical; only ranking may change.
    a = summary["diagnostics"]["mechanism_critic_original/seed_0"]["proposal_diagnostics"]
    b = summary["diagnostics"]["mechanism_critic_acquired/seed_0"]["proposal_diagnostics"]
    assert [row["best_proposal_loss"] for row in a["rows"]] == [row["best_proposal_loss"] for row in b["rows"]]


def test_bank_extension_uses_independent_candidate_seed(tmp_path):
    from annealctrl.dagger import collect_bank_extension
    from annealctrl.search import shared_candidate_bank

    cfg = _config()
    cfg["dataset"]["candidate_seed"] = 901  # logical parents still use seed 17
    generate_dataset(cfg["dataset"], tmp_path / "data")
    obtained = study._acquire(tmp_path / "data", None, "bankext", cfg, cfg["dataset"], "cpu")
    inferred = collect_bank_extension(tmp_path / "data", bank_size=4, n_extra=3,
                                      tolerance=0.005, initial_steps=16, max_steps=1024)
    expected = shared_candidate_bank(n=7, n_segments=8, seed=901, runtime=2., max_slope=2.)
    waves = np.stack([candidate.schedule(np.linspace(0., 1., 9)) for candidate in expected[4:]])
    assert obtained["provenance"]["bank_seed"] == inferred["provenance"]["bank_seed"] == 901
    for identifier, row in obtained["records"].items():
        np.testing.assert_allclose(row["waveforms"], waves, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(row["waveforms"], inferred["records"][identifier]["waveforms"])


@pytest.fixture(scope="module")
def completed_study(tmp_path_factory):
    from annealctrl import pipeline

    root = tmp_path_factory.mktemp("controlled_study")
    cfg = _config()
    data, output = root / "data", root / "study"
    generate_dataset(cfg["dataset"], data)
    original = pipeline.load_records
    test_openings = []

    def checked_load(path, split=None):
        if split in {None, "test"}:
            frozen = json.loads((output / "study.json").read_text())
            assert frozen.get("training_complete"), "test labels opened before training-complete barrier"
            assert len(frozen["runs"]) == 15
            assert all(state.get("trained") for state in frozen["runs"].values())
            test_openings.append(split)
        return original(path, split)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(pipeline, "load_records", checked_load)
        patch.setattr(study, "load_records", checked_load)
        trained = study.run_study(cfg, output, data_dir=data, stage="train")
        assert trained["training_complete"]
        assert not test_openings
        manifest = study.run_study(cfg, output, data_dir=data, stage="all", resume=True)
    assert test_openings
    return cfg, data, output, manifest


def test_three_seed_study_uses_fixed_validation_and_matched_real_label_budgets(completed_study):
    from annealctrl.dagger import augment_records
    from annealctrl.learning import records_content_digest

    cfg, data, output, manifest = completed_study
    assert manifest["status"] == "complete"
    assert len(manifest["runs"]) == 15
    training, validation = load_records(data, "train"), load_records(data, "validation")
    validation_hash = records_content_digest(validation)
    for seed in cfg["seeds"]:
        for arm in study.ARMS:
            state = manifest["runs"][f"{arm}/seed_{seed}"]
            checkpoint = torch.load(output / state["checkpoint"], map_location="cpu", weights_only=True)
            assert checkpoint["data_provenance"]["validation_content_sha256"] == validation_hash
            if arm == "control":
                assert "acquisition" not in state
                continue
            acquisition = json.loads((output / state["acquisition"]).read_text())
            assert acquisition["split"] == "train"
            assert acquisition["objective_calls"] == len(training) * 3
            assert acquisition["requested_objective_calls"] == acquisition["objective_calls"]
            assert acquisition["cost_counts_complete"]
            augmented = augment_records(training, acquisition)
            assert checkpoint["data_provenance"]["train_content_sha256"] == records_content_digest(augmented)
    summary = json.loads((output / "summary.json").read_text())
    assert set(summary["means"]) == set(study.ARMS)
    assert set(summary["acquisition_contrasts"]) == {
        "policy_minus_control", "policy_minus_bankext", "policy_minus_decoder_random"}
    for contrast in summary["acquisition_contrasts"].values():
        assert contrast["n_seeds"] == 3
        assert contrast["p_value"] is None
    for costs in summary["costs"].values():
        assert costs["deployment_latency"]["online_simulator_calls"] == {"bank": 0, "direct": 0}
        assert costs["offline_evaluation_diagnostics"]["offline_objective_calls"] > 0
        assert costs["shared_frozen_baseline_training_seconds"] > 0
    assert all(np.isfinite(value) for modes in summary["means"].values() for value in modes.values())


def test_same_recipe_control_reproduces_baseline_weights(completed_study):
    cfg, _, output, manifest = completed_study
    for seed in cfg["seeds"]:
        models = [torch.load(output / manifest["runs"][f"{arm}/seed_{seed}"]["checkpoint"],
                             map_location="cpu", weights_only=True)["model_state"]
                  for arm in ("baseline", "control")]
        assert models[0].keys() == models[1].keys()
        assert all(torch.equal(models[0][key], models[1][key]) for key in models[0])


def test_clean_resume_reuses_all_frozen_artifacts(completed_study, monkeypatch):
    from annealctrl import acquisition_evaluation

    cfg, data, output, manifest = completed_study

    def forbidden(*args, **kwargs):
        raise AssertionError("completed study must reuse acquisition, checkpoints and evaluations")

    monkeypatch.setattr(study, "_fit", forbidden)
    monkeypatch.setattr(study, "_acquire", forbidden)
    monkeypatch.setattr(acquisition_evaluation, "evaluate_acquisition_arm", forbidden)
    before = hashlib.sha256((output / "summary.json").read_bytes()).hexdigest()
    resumed = study.run_study(cfg, output, data_dir=data, stage="all", resume=True)
    assert resumed["runs"] == manifest["runs"]
    assert resumed["status"] == "complete"
    assert hashlib.sha256((output / "summary.json").read_bytes()).hexdigest() == before


@pytest.mark.parametrize("section,key,value", [
    ("acquisition", "n_extra", 2), ("acquisition", "unrecognized", True),
    ("evaluation", "direct", False), ("latency", "repeats", 0),
    ("evaluation", "norm_tolerance", 1e-8),
])
def test_invalid_study_design_is_rejected_before_execution(section, key, value):
    cfg = _config()
    cfg[section][key] = value
    with pytest.raises(ValueError):
        study.validate_study(cfg)


def test_resume_rejects_changed_configuration_and_source(completed_study, monkeypatch):
    cfg, data, output, _ = completed_study
    changed = copy.deepcopy(cfg)
    changed["acquisition"]["logit_std"] = 2.0
    with pytest.raises(ValueError, match="config/source/data path changed"):
        study.run_study(changed, output, data_dir=data, resume=True)
    monkeypatch.setattr(study, "source_hash", lambda: "changed-source")
    with pytest.raises(ValueError, match="config/source/data path changed"):
        study.run_study(cfg, output, data_dir=data, resume=True)


def test_resume_rejects_modified_acquisition_artifact(completed_study, tmp_path):
    cfg, data, output, manifest = completed_study
    copied = tmp_path / "copied_study"
    shutil.copytree(output, copied)
    state = manifest["runs"]["policy/seed_0"]
    with (copied / state["acquisition"]).open("a") as handle:
        handle.write(" ")
    with pytest.raises(ValueError, match="changed frozen artifact"):
        study.run_study(cfg, copied, data_dir=data, resume=True)


def test_evaluation_cannot_run_before_all_training_arms_finish(completed_study, tmp_path, monkeypatch):
    from annealctrl import acquisition_evaluation

    cfg, data, _, _ = completed_study

    def forbidden(*args, **kwargs):
        raise AssertionError("evaluation entered before frozen training-complete barrier")

    monkeypatch.setattr(acquisition_evaluation, "evaluate_acquisition_arm", forbidden)
    with pytest.raises(ValueError, match="complete all frozen training arms"):
        study.run_study(cfg, tmp_path / "premature", data_dir=data, stage="evaluate")


def test_external_dataset_must_match_frozen_scientific_config(completed_study, tmp_path):
    cfg, data, _, _ = completed_study
    changed = copy.deepcopy(cfg)
    changed["dataset"]["seed"] += 1
    with pytest.raises(ValueError, match="dataset config differs"):
        study.run_study(changed, tmp_path / "wrong_dataset", data_dir=data, stage="train")


def test_failed_acquisition_costs_survive_resume_in_immutable_hashed_ledger(completed_study, tmp_path, monkeypatch):
    from annealctrl import benchmarking
    from annealctrl.dagger import AcquisitionError

    cfg, data, _, _ = completed_study
    cfg = copy.deepcopy(cfg)
    cfg["seeds"] = [0]
    output = tmp_path / "retry_study"
    original = benchmarking.score_schedule
    completed_scores = []
    failed = False

    def fail_once(record, schedule, **kwargs):
        nonlocal failed
        if len(completed_scores) == 1 and not failed:
            failed = True
            raise ArithmeticError("deliberate second-call convergence failure")
        result = original(record, schedule, **kwargs)
        completed_scores.append(result)
        return result

    monkeypatch.setattr(benchmarking, "score_schedule", fail_once)
    with pytest.raises(AcquisitionError, match="convergence failure"):
        study.run_study(cfg, output, data_dir=data, stage="all")
    failed_manifest = json.loads((output / "study.json").read_text())
    state = failed_manifest["runs"]["bankext/seed_0"]
    assert "acquisition" not in state, "partial labels must not become training labels"
    assert len(state["acquisition_attempts"]) == 1
    attempt = state["acquisition_attempts"][0]
    assert attempt["status"] == "failed"
    first_audit = (output / attempt["audit"]).read_bytes()
    first_request = (output / attempt["request"]).read_bytes()
    assert hashlib.sha256(first_audit).hexdigest() == attempt["audit_sha256"]
    assert failed_manifest["error"]["acquisition_audit"] == attempt["audit"]
    assert state["acquisition_cost"]["objective_calls"] == 2
    assert state["acquisition_cost"]["successful_objective_calls"] == 1
    assert not state["acquisition_cost"]["cost_counts_complete"]

    resumed = study.run_study(cfg, output, data_dir=data, stage="all", resume=True)
    state = resumed["runs"]["bankext/seed_0"]
    assert [row["status"] for row in state["acquisition_attempts"]] == ["failed", "complete"]
    assert (output / attempt["audit"]).read_bytes() == first_audit
    assert (output / attempt["request"]).read_bytes() == first_request
    assert len({row["audit"] for row in state["acquisition_attempts"]}) == 2
    added_labels = len(load_records(data, "train")) * 3
    successful = json.loads((output / state["acquisition"]).read_text())
    assert successful["objective_calls"] == added_labels
    summary = json.loads((output / "summary.json").read_text())
    costs = summary["costs"]["bankext/seed_0"]["acquisition"]
    assert costs["objective_calls"] == added_labels + 2
    assert costs["successful_objective_calls"] == added_labels + 1
    assert costs["failed_attempt_objective_calls"] == 2
    assert costs["failed_objective_calls"] == 1
    assert costs["attempt_count"] == 2
    assert costs["failed_attempts"] == 1
    assert costs["total_integrator_steps"] == (successful["total_integrator_steps"]
                                              + completed_scores[0]["total_integrator_steps"])
    assert costs["propagation_calls"] == successful["propagation_calls"] + completed_scores[0]["propagation_calls"]
    assert costs["objective_calls_complete"]
    assert not costs["cost_counts_complete"]
    assert costs["counts_are_lower_bounds"]
    with (output / attempt["audit"]).open("a") as handle:
        handle.write(" ")
    with pytest.raises(ValueError, match="changed frozen artifact"):
        study.run_study(cfg, output, data_dir=data, resume=True)


def test_interrupted_acquisition_is_retained_as_unknown_work_before_retry(completed_study, tmp_path, monkeypatch):
    cfg, data, _, _ = completed_study
    cfg = copy.deepcopy(cfg)
    cfg["seeds"] = [0]
    output = tmp_path / "interrupted_study"
    original = study._acquire

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt("simulated process interruption")

    monkeypatch.setattr(study, "_acquire", interrupted)
    with pytest.raises(KeyboardInterrupt):
        study.run_study(cfg, output, data_dir=data, stage="train")
    manifest = json.loads((output / "study.json").read_text())
    request = manifest["runs"]["bankext/seed_0"]["acquisition_attempts"][0]
    assert request["status"] == "started"
    request_bytes = (output / request["request"]).read_bytes()
    monkeypatch.setattr(study, "_acquire", original)
    resumed = study.run_study(cfg, output, data_dir=data, stage="train", resume=True)
    state = resumed["runs"]["bankext/seed_0"]
    assert [row["status"] for row in state["acquisition_attempts"]] == ["interrupted", "complete"]
    assert (output / request["request"]).read_bytes() == request_bytes
    assert not state["acquisition_cost"]["objective_calls_complete"]
    assert not state["acquisition_cost"]["cost_counts_complete"]
    assert state["acquisition_cost"]["failed_attempts"] == 1
