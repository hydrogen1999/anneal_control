from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from annealctrl import pipeline, profiling


def config():
    return {"seed": 923, "parents": 3, "families": ["spin_glass"],
            "logical_qubits": 3, "chain_lengths": [1, 1, 1],
            "variants": [{"shape": "path", "ports": 1}], "chain_strengths": [1.5],
            "runtimes": [.5], "candidates": 2, "spectral_points": 3,
            "steps": 8, "max_steps": 128, "label_state_tolerance": .005,
            "max_physical_qubits": 8}


@pytest.fixture(autouse=True)
def fixed_source(monkeypatch):
    # The invariant is tested explicitly below; unrelated concurrent edits
    # should not make the independent real tiny generation test flaky.
    monkeypatch.setattr(profiling, "source_fingerprint", lambda: "profile-test-source")
    monkeypatch.setattr(pipeline, "source_fingerprint", lambda: "profile-test-source")


def records():
    result = []
    for index in range(2):
        metadata = {"teacher": {"mode": "exact", "spectral_seconds_per_path": 2.},
                    "memory": {"endpoint_estimated_mib": .02, "state_estimated_mib": .01},
                    "independent_reference_seconds": .3 if index == 0 else 0.,
                    "independent_reference_errors": [1e-6] if index == 0 else []}
        result.append({
            "record_id": np.array(f"p0_e0_k0_t{index}"), "parent_id": np.array("p0"),
            "split": np.array("train"), "family": np.array("spin_glass"),
            "physical_h": np.array([.1, -.2]), "physical_edges": np.array([[0, 1]]),
            "physical_J": np.array([.3]), "problem_J": np.array([.3]), "chain_J": np.array([0.]),
            "membership": np.array([0, 1]), "logical_h": np.array([.1, -.2]),
            "logical_edges": np.array([[0, 1]]), "logical_J": np.array([.3]),
            "programmed_scale": np.array(1.), "runtime": np.array(1. + index),
            "chain_strength": np.array(1.5), "catalyst_strength": np.array(0.),
            "candidate_schedules": np.array([[0., .5, 1.], [0., .6, 1.]]),
            "candidate_tau": np.array([0., .5, 1.]), "candidate_ids": np.array(["linear", "other"]),
            "candidate_losses": np.array([.6, .5]), "candidate_success": np.array([.4, .5]),
            "candidate_decoded_energy": np.array([-.1, -.2]),
            "candidate_any_chain_break": np.zeros(2), "candidate_chain_break_fraction": np.zeros(2),
            "candidate_state_error": np.array([1e-5, 2e-5]), "candidate_norm_error": np.array([1e-14, 2e-14]),
            "candidate_steps": np.array([32, 64]), "candidate_seconds": np.array([.1, .2]),
            "metadata_json": np.array(json.dumps(metadata)), "fingerprint": np.array("backend-dependent"),
        })
    return result


def fake_backends(monkeypatch, *, gpu_records=None, failure=None):
    seen = []
    def generate(cfg, destination, *, backend, workers):
        seen.append((backend, deepcopy(cfg), workers))
        Path(destination).mkdir()
        if failure is not None and backend == "cupy":
            raise failure
        return {"source_fingerprint": "profile-test-source", "record_count": 2,
                "config_hash": f"{backend}-config", "dataset_fingerprint": f"{backend}-data"}
    monkeypatch.setattr(profiling, "generate_dataset", generate)
    monkeypatch.setattr(profiling, "load_records", lambda path: deepcopy(
        gpu_records if Path(path).name == "cupy" and gpu_records is not None else records()))
    monkeypatch.setattr(profiling, "backend_device_info", lambda backend: {"backend": backend, "device": "test-double"})
    ticks = iter([10., 14., 20., 22.])
    monkeypatch.setattr(profiling, "perf_counter", lambda: next(ticks))
    return seen


def test_real_tiny_cpu_profile_includes_full_generation_and_writes_strict_json(tmp_path):
    result = profiling.profile_generation(config(), tmp_path / "real")
    cpu = result["backends"]["numpy"]
    assert result["status"] == "complete"
    assert cpu["records"] == 3
    assert cpu["accepted_candidate_labels"] == 6
    assert cpu["accepted_labels_per_second"] == pytest.approx(6 / cpu["generation_wall_seconds"])
    assert cpu["max_state_error_diagnostic"] <= .005
    assert cpu["max_norm_error"] <= 1e-9
    assert cpu["cost_components"]["spectral_including_random_audit_worker_seconds"] > 0
    assert cpu["cost_components"]["independent_dynamics_audit_worker_seconds"] > 0
    assert not cpu["memory"]["measured_peak_memory"]
    assert result["cpu_gpu_comparison"] is None
    saved = json.loads((tmp_path / "real" / "profile.json").read_text())
    assert saved == result
    json.dumps(result, allow_nan=False)


def test_identical_workloads_parity_and_observed_ratio_not_invented(monkeypatch, tmp_path):
    gpu = records()
    for record in gpu:
        record["candidate_losses"] += 1e-7
        record["candidate_success"] -= 1e-7
        record["fingerprint"] = np.array("different_gpu_fingerprint")
    calls = fake_backends(monkeypatch, gpu_records=gpu)
    cfg = config()
    result = profiling.profile_generation(cfg, tmp_path / "mocked", backends=("numpy", "cupy"))
    assert calls == [("numpy", cfg, 1), ("cupy", cfg, 1)]
    assert cfg == config()
    comparison = result["cpu_gpu_comparison"]
    assert comparison["same_physics_and_waveforms"]
    assert comparison["record_fingerprints_expected_to_differ"]
    assert comparison["outcome_differences"]["candidate_losses"]["max_absolute"] == pytest.approx(1e-7)
    assert comparison["observed_end_to_end_throughput_ratio_cupy_over_numpy"] == pytest.approx(2.)
    assert not comparison["ratio_is_repeated_benchmark_or_speedup_claim"]
    cpu = result["backends"]["numpy"]
    assert cpu["cost_components"]["spectral_including_random_audit_worker_seconds"] == 2.  # Not 4 for two runtimes.
    assert cpu["cost_components"]["independent_dynamics_audit_worker_seconds"] == .3


@pytest.mark.parametrize("field", ["physical_J", "candidate_schedules", "membership", "runtime"])
def test_parity_rejects_changed_program_or_waveforms(monkeypatch, tmp_path, field):
    gpu = records()
    gpu[0][field] = gpu[0][field] + 1
    fake_backends(monkeypatch, gpu_records=gpu)
    with pytest.raises(ValueError, match="physics or waveform mismatch"):
        profiling.profile_generation(config(), tmp_path / field, backends=("numpy", "cupy"))
    saved = json.loads((tmp_path / field / "profile.json").read_text())
    assert saved["status"] == "failed"
    assert all(value["status"] == "complete" for value in saved["backends"].values())
    assert saved["cpu_gpu_comparison"] is None


def test_missing_backend_is_explicit_failure_with_completed_cpu_intact(monkeypatch, tmp_path):
    fake_backends(monkeypatch)
    def device(backend):
        if backend == "cupy":
            raise ImportError("CuPy intentionally unavailable")
        return {"backend": "numpy", "device": "cpu"}
    monkeypatch.setattr(profiling, "backend_device_info", device)
    with pytest.raises(ImportError, match="unavailable"):
        profiling.profile_generation(config(), tmp_path / "partial", backends=("numpy", "cupy"))
    saved = json.loads((tmp_path / "partial" / "profile.json").read_text())
    assert saved["status"] == "failed"
    assert saved["backends"]["numpy"]["status"] == "complete"
    assert saved["backends"]["cupy"]["status"] == "failed"
    assert "accepted_labels_per_second" not in saved["backends"]["cupy"]
    assert saved["error"]["backend"] == "cupy"


def test_interrupt_preserves_clear_recovery_status(monkeypatch, tmp_path):
    fake_backends(monkeypatch, failure=KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        profiling.profile_generation(config(), tmp_path / "interrupted", backends=("numpy", "cupy"))
    saved = json.loads((tmp_path / "interrupted" / "profile.json").read_text())
    assert saved["status"] == "interrupted"
    assert saved["backends"]["numpy"]["status"] == "complete"
    assert saved["backends"]["cupy"]["status"] == "interrupted"
    assert "NEW output" in saved["recovery"]


def test_failed_numerical_labels_never_count_as_accepted_throughput(monkeypatch, tmp_path):
    invalid = records()
    invalid[0]["candidate_state_error"][0] = 1.
    fake_backends(monkeypatch, gpu_records=invalid)
    with pytest.raises(ArithmeticError, match="numerical gate"):
        profiling.profile_generation(config(), tmp_path / "bad_labels", backends=("cupy",))
    saved = json.loads((tmp_path / "bad_labels" / "profile.json").read_text())
    assert saved["status"] == "failed"
    assert "accepted_labels_per_second" not in saved["backends"]["cupy"]


def test_source_drift_fails_instead_of_misattributing_measurement(monkeypatch, tmp_path):
    fake_backends(monkeypatch)
    hashes = iter(["profile-test-source", "changed-source"])
    monkeypatch.setattr(profiling, "source_fingerprint", lambda: next(hashes))
    with pytest.raises(RuntimeError, match="source changed"):
        profiling.profile_generation(config(), tmp_path / "drift")
    saved = json.loads((tmp_path / "drift" / "profile.json").read_text())
    assert saved["status"] == "failed"


def test_existing_output_is_never_reused_even_if_empty(tmp_path):
    target = tmp_path / "existing"
    target.mkdir()
    with pytest.raises(FileExistsError):
        profiling.profile_generation(config(), target)
    assert not list(target.iterdir())


@pytest.mark.parametrize("kwargs", [
    {"backends": ()}, {"backends": ("numpy", "numpy")}, {"backends": ("pretend",)},
    {"workers": 0}, {"workers": True}, {"workers": 1.5}, {"workers": 2, "backends": ("numpy", "cupy")},
])
def test_invalid_execution_settings_fail_before_output_creation(tmp_path, kwargs):
    target = tmp_path / "invalid"
    with pytest.raises(ValueError):
        profiling.profile_generation(config(), target, **kwargs)
    assert not target.exists()
