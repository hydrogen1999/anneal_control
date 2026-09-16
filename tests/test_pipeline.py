"""Small end-to-end contracts; no learned-model or hardware claims."""
import hashlib
import json

import numpy as np
import pytest

from annealctrl import pipeline
from annealctrl.evaluation import audit_parent_splits
from annealctrl.generation import Embedding, IsingProblem, compile_embedding, validate_compilation
from annealctrl.physics import HamiltonianTerms, dense_reference_propagate
from annealctrl.schedules import Schedule
from annealctrl.search import shared_candidate_bank


def tiny_config():
    return {
        "seed": 17, "parents": 12,
        "families": ["weighted_maxcut", "spin_glass", "planted_loops", "weak_field"],
        "logical_qubits": 3, "chain_lengths": [1, 1, 1],
        "variants": [{"shape": "path", "ports": 1, "field_distribution": "uniform"}],
        "chain_strengths": [1.5], "runtimes": [2.0], "candidates": 2,
        "spectral_points": 3, "steps": 16, "max_steps": 512,
        "label_state_tolerance": 1e-3, "max_physical_qubits": 8,
    }


@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory):
    destination = tmp_path_factory.mktemp("pipeline") / "dataset"
    seen = []
    original = pipeline.propagate
    def audited(terms, schedule, runtime, **kwargs):
        # The actual controls entering propagation are reconstructed on nine
        # time knots, never the source bank's heterogeneous knot representation.
        np.testing.assert_array_equal(schedule.tau_knots, np.linspace(0, 1, 9))
        seen.append(schedule.s_knots.copy())
        return original(terms, schedule, runtime, **kwargs)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(pipeline, "propagate", audited)
        manifest = pipeline.generate_dataset(tiny_config(), destination)
    return destination, manifest, seen


def test_schema_shapes_masks_and_numerical_audits(tiny_dataset):
    root, manifest, seen = tiny_dataset
    assert manifest["schema_version"] == pipeline.SCHEMA_VERSION
    assert manifest["status"] == "complete"
    assert manifest["record_count"] == 12
    records = pipeline.load_records(root)
    assert len(records) == 12 and len(seen) >= 24
    for record in records:
        assert record["physical_h"].shape == (3,)
        assert record["candidate_schedules"].shape == (2, 9)
        assert record["candidate_losses"].shape == (2,)
        assert record["response_moments"].shape == (3, 3)
        assert record["response_mask"].dtype == bool
        assert np.isfinite(record["response_moments"]).all()
        assert np.all(record["response_moments"][~record["response_mask"]] == 0)
        np.testing.assert_allclose(record["candidate_losses"], 1 - record["candidate_success"])
        assert np.min(record["candidate_success"]) >= -1e-10
        assert np.max(record["candidate_success"]) <= 1 + 1e-10
        assert np.max(record["candidate_state_error"]) <= 1e-3
        assert np.max(record["candidate_norm_error"]) <= 1e-9
        assert np.max(record["candidate_steps"]) <= 512
        assert np.max(record["response_residual"]) <= 1e-8
        assert np.all(record["candidate_chain_break_fraction"] == 0)
        meta = json.loads(record["metadata_json"].item())
        assert meta["candidate_rule"] == "common_bank_resampled9_then_simulated"
        assert meta["reference_status"] == "finite_shared_bank_best_not_global"
        assert len(meta["independent_reference_errors"]) == 1


def test_common_bank_is_independent_of_parent_test_spectrum(tiny_dataset):
    root, _, _ = tiny_dataset
    records = pipeline.load_records(root)
    expected_bank = shared_candidate_bank(n=2, n_segments=8, seed=17, runtime=2, max_slope=2)
    expected = np.stack([candidate.schedule(np.linspace(0, 1, 9)) for candidate in expected_bank])
    for record in records:
        # All different Hamiltonians and all splits see exactly the same controls.
        np.testing.assert_array_equal(record["candidate_schedules"], expected)
        np.testing.assert_array_equal(record["candidate_ids"], ["linear", "one_window_0"])
    assert any(not np.allclose(records[0]["response_moments"], record["response_moments"]) for record in records[1:])
    # Source window knots and executed nine-knot control are genuinely distinct.
    assert len(expected_bank[1].schedule.tau_knots) != 9


def test_stored_control_matches_independent_terminal_loss(tiny_dataset):
    root, _, _ = tiny_dataset
    record = pipeline.load_records(root)[0]
    schedule = Schedule(record["candidate_tau"], record["candidate_schedules"][1])
    terms = HamiltonianTerms(3, record["physical_h"], record["physical_edges"], record["physical_J"])
    result = dense_reference_propagate(terms, schedule, float(record["runtime"]))
    # Singleton chains mean physical strings and logical strings coincide.
    from annealctrl.generation import all_spins
    problem = IsingProblem(record["logical_h"], record["logical_edges"], record["logical_J"])
    energy = problem.energy(all_spins(3))
    accepted = np.isclose(energy, energy.min(), atol=1e-9, rtol=0)
    reference = float(np.abs(result.state) ** 2 @ accepted)
    assert record["candidate_success"][1] == pytest.approx(reference, abs=3e-3)


def test_parent_split_integrity(tiny_dataset):
    root, manifest, _ = tiny_dataset
    records = pipeline.load_records(root)
    counts = audit_parent_splits([record["parent_id"].item() for record in records], [record["split"].item() for record in records])
    assert counts == {"train": 4, "validation": 4, "test": 4}
    for record in records:
        assert manifest["splits"][record["parent_id"].item()] == record["split"].item()
    assert len(pipeline.load_records(root, "test")) == 4


def test_resume_keeps_records_and_rejects_overwrite_or_config_change(tiny_dataset):
    root, manifest, _ = tiny_dataset
    files = [root / item["path"] for item in manifest["records"]]
    before = [(path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest()) for path in files]
    with pytest.raises(FileExistsError):
        pipeline.generate_dataset(tiny_config(), root)
    changed = tiny_config()
    changed["seed"] += 1
    with pytest.raises(ValueError, match="mismatch"):
        pipeline.generate_dataset(changed, root, resume=True)
    result = pipeline.generate_dataset(tiny_config(), root, resume=True)
    after = [(path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest()) for path in files]
    assert before == after
    assert result["record_count"] == 12


def test_loader_rejects_incomplete_and_record_split_mismatch(tiny_dataset, tmp_path):
    root, manifest, _ = tiny_dataset
    import shutil
    copy = tmp_path / "copy"
    shutil.copytree(root, copy)
    tampered = dict(manifest, status="in_progress")
    pipeline.write_json(copy / "manifest.json", tampered)
    with pytest.raises(ValueError, match="incomplete"):
        pipeline.load_records(copy)
    tampered = json.loads(json.dumps(manifest))
    tampered["records"][0]["split"] = "test" if tampered["records"][0]["split"] != "test" else "train"
    pipeline.write_json(copy / "manifest.json", tampered)
    with pytest.raises(ValueError, match="split"):
        pipeline.load_records(copy)


def test_weak_chains_are_marked_not_rejected():
    # A frustrated logical triangle can break its doubled first chain when k<1.
    logical = IsingProblem(np.zeros(3), np.array([[0, 1], [0, 2], [1, 2]]), np.ones(3))
    embedding = Embedding(np.array([0, 0, 1, 2]), np.array([[0, 1], [0, 2], [1, 3], [2, 3]]))
    weak = compile_embedding(logical, embedding, 0.1, np.random.default_rng(0))
    strong = compile_embedding(logical, embedding, 2.0, np.random.default_rng(0))
    weak_info, strong_info = validate_compilation(weak), validate_compilation(strong)
    assert not weak_info["has_aligned_physical_ground_state"]
    assert strong_info["has_aligned_physical_ground_state"]
    assert weak_info["aligned_energy_max_error"] < 1e-12
    assert strong_info["aligned_energy_max_error"] < 1e-12
