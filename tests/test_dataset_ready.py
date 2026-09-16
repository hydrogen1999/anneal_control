"""Production data contracts exercised on deliberately tiny numerical problems."""
import json
from pathlib import Path

import numpy as np
import pytest

from annealctrl import pipeline
from annealctrl.generation import (IsingProblem, all_spins, compile_embedding,
    generate_problem, logical_fingerprint, output_observables, sample_logical_support,
    spin_chunks, synthetic_lift, validate_compilation)
from annealctrl.physics import HamiltonianTerms


def config(**updates):
    cfg = {"seed": 923, "parents": 3, "families": ["spin_glass"],
           "logical_qubits": 3, "chain_lengths": [1, 1, 1],
           "variants": [{"shape": "path", "ports": 1}], "chain_strengths": [1.5],
           "runtimes": [.5], "candidates": 2, "spectral_points": 3,
           "steps": 8, "max_steps": 128, "label_state_tolerance": .005,
           "max_physical_qubits": 8}
    cfg.update(updates)
    return cfg


@pytest.fixture(autouse=True)
def stable_source_for_parallel_development(monkeypatch):
    # Other developers may edit unrelated source files while this test runs.
    # Test resume semantics itself against a controlled implementation version.
    monkeypatch.setattr(pipeline, "source_fingerprint", lambda: "test-source-version")


def test_spin_chunks_and_endpoint_match_enumeration():
    np.testing.assert_array_equal(np.concatenate([z for _, z in spin_chunks(7, 11)]), all_spins(7))
    problem = generate_problem(3, "spin_glass", np.random.default_rng(3))
    embedding = synthetic_lift(problem, [2, 1, 2], np.random.default_rng(4))
    compiled = compile_embedding(problem, embedding, .4, np.random.default_rng(5))
    a, b = output_observables(compiled, chunk_size=3), output_observables(compiled, chunk_size=4096)
    for key in a:
        np.testing.assert_allclose(a[key], b[key], atol=1e-12)
    x, y = validate_compilation(compiled, chunk_size=3), validate_compilation(compiled, chunk_size=4096)
    for key in x:
        assert x[key] == pytest.approx(y[key], abs=1e-12)


def test_exact_labelled_fingerprint_ignores_edge_order_and_family():
    a = IsingProblem(np.array([0., 1., 2.]), np.array([[0, 1], [1, 2]]), np.array([.4, .8]), "a")
    b = IsingProblem(np.array([-0., 1., 2.]), np.array([[2, 1], [1, 0]]), np.array([.8, .4]), "b")
    assert logical_fingerprint(a) == logical_fingerprint(b)
    b.h[1] += .01
    assert logical_fingerprint(a) != logical_fingerprint(b)


@pytest.mark.parametrize("kind", ["complete", "path", "cycle", "erdos_renyi", "connected_erdos_renyi"])
def test_support_generation_is_reproducible(kind):
    a = sample_logical_support(6, np.random.default_rng(7), kind=kind)
    b = sample_logical_support(6, np.random.default_rng(7), kind=kind)
    np.testing.assert_array_equal(a, b)
    assert np.all(a[:, 0] < a[:, 1])


def test_multisize_holdout_and_chain_distribution():
    cfg = config(parents=18, families=["spin_glass", "weighted_maxcut"],
        logical_sizes=[3, 4, 5], chain_distribution={"distribution": "powerlaw", "low": 1, "high": 2, "exponent": 1.8},
        split={"holdout_sizes": [5], "test_fraction": 0., "validation_fraction": .2},
        logical_support={"kind": "connected_erdos_renyi", "edge_probability": .4})
    del cfg["logical_qubits"], cfg["chain_lengths"]
    pipeline._validate_config(cfg)
    parents = pipeline._plan_parents(cfg)
    mapping = pipeline._split_parents(parents, cfg)
    for parent in parents:
        assert (mapping[parent["parent_id"]] == "test") == (parent["size"] == 5)
        assert sum(parent["lengths"]) <= 8
        assert parent["length_metadata"]["sampling_condition"].startswith("sum_of_target")
    assert set(mapping.values()) == {"train", "validation", "test"}
    assert len({p["logical_fingerprint"] for p in parents}) == 18


def test_no_duplicate_parent_ids_hidden_by_random_metadata(monkeypatch):
    fixed = generate_problem(3, "spin_glass", np.random.default_rng(0))
    monkeypatch.setattr(pipeline, "generate_problem", lambda *args, **kwargs: fixed)
    with pytest.raises(ValueError, match="unique valid parent"):
        pipeline._plan_parents(config())


def test_exclusive_writer_lock_is_not_automatically_stolen(tmp_path):
    with pipeline.dataset_lock(tmp_path):
        with pytest.raises(FileExistsError, match="locked"):
            with pipeline.dataset_lock(tmp_path):
                pass
    assert not (tmp_path / ".generation.lock").exists()


def test_none_teacher_has_no_invented_targets(tmp_path):
    cfg = config(teacher={"mode": "none"})
    manifest = pipeline.generate_dataset(cfg, tmp_path / "none")
    assert manifest["teacher_mode"] == "none"
    assert manifest["duplicate_audit"]["graph_or_gauge_isomorphism_checked"] is False
    for record in pipeline.load_records(tmp_path / "none"):
        assert not record["response_mask"].any()
        assert record["response_bins"].shape == (3, 0)
        assert np.isfinite(record["candidate_losses"]).all()


def test_fixed_hardware_growth_preserves_truthful_achieved_metadata(tmp_path):
    cfg = config(generation_route="hardware_growth", chain_lengths=[2, 2, 2],
        hardware={"n_qubits": 9, "edges": [[i, i + 1] for i in range(8)]})
    pipeline.generate_dataset(cfg, tmp_path / "hardware")
    for record in pipeline.load_records(tmp_path / "hardware"):
        meta = json.loads(record["metadata_json"].item())
        emb = meta["embedding"]
        assert emb["route"] == "fixed_hardware_growth"
        np.testing.assert_array_equal(np.bincount(record["membership"]), emb["achieved_lengths"])
        assert sum(emb["achieved_lengths"]) + emb["unused_qubits"] == 9
        assert not meta["variant_geometry_applicable"]
        assert meta["logical_support"]["kind"] == "hardware_quotient"


def test_adaptive_teacher_records_audits_separately(tmp_path):
    cfg = config(teacher={"mode": "adaptive", "max_queries": 9, "audit_points": 2})
    pipeline.generate_dataset(cfg, tmp_path / "adaptive")
    for record in pipeline.load_records(tmp_path / "adaptive"):
        meta = json.loads(record["metadata_json"].item())
        assert meta["teacher"]["uniform_certificate"] is False
        assert len(meta["teacher"]["audit"]) == 2
        assert len(record["response_s"]) <= 7
        assert np.all(np.diff(record["response_s"]) > 0)
        assert not set(record["response_s"]).intersection(p["s"] for p in meta["teacher"]["audit"])


def test_partial_resume_uses_spectral_cache_and_completed_records(tmp_path, monkeypatch):
    root = tmp_path / "resume"
    cfg = config()
    manifest = pipeline.generate_dataset(cfg, root)
    record_paths = [root / r["path"] for r in manifest["records"]]
    first_mtime = record_paths[0].stat().st_mtime_ns
    record_paths[-1].unlink()  # remove one disposable test record to simulate interrupted generation
    manifest["status"] = "in_progress"
    pipeline.write_json(root / "manifest.json", manifest)
    def forbidden(*args, **kwargs):
        raise AssertionError("cached spectral path must not be recomputed")
    monkeypatch.setattr(pipeline, "spectral_profile", forbidden)
    rebuilt = pipeline.generate_dataset(cfg, root, resume=True)
    assert rebuilt["record_count"] == 3
    assert record_paths[0].stat().st_mtime_ns == first_mtime
    assert len(pipeline.load_records(root)) == 3
    monkeypatch.setattr(pipeline, "source_fingerprint", lambda: "new-version")
    with pytest.raises(ValueError, match="source fingerprint mismatch"):
        pipeline.generate_dataset(cfg, root, resume=True)


def test_cpu_worker_count_does_not_change_scientific_records(tmp_path):
    cfg = config(teacher={"mode": "none"})
    a = pipeline.generate_dataset(cfg, tmp_path / "serial", workers=1)
    b = pipeline.generate_dataset(cfg, tmp_path / "parallel", workers=2)
    assert a["dataset_fingerprint"] == b["dataset_fingerprint"]
    serial, parallel = pipeline.load_records(tmp_path / "serial"), pipeline.load_records(tmp_path / "parallel")
    for x, y in zip(serial, parallel):
        for key in x:
            if key in {"metadata_json", "candidate_seconds", "payload_fingerprint"}:
                continue
            np.testing.assert_array_equal(x[key], y[key])


def test_explicit_outcome_only_budget_guards():
    cfg = config(logical_qubits=11, chain_lengths=1, max_physical_qubits=11)
    with pytest.raises(ValueError, match="full spectral"):
        pipeline._validate_config(cfg)
    cfg["teacher"] = {"mode": "none"}
    with pytest.raises(ValueError, match="explicit"):
        pipeline._validate_config(cfg)
    cfg.update(endpoint_max_qubits=20, endpoint_memory_mb=16, state_memory_mb=16)
    pipeline._validate_config(cfg)
    terms = HamiltonianTerms(11, np.zeros(11), np.empty((0, 2), int), np.empty(0))
    with pytest.raises(MemoryError, match="state estimate"):
        pipeline._endpoint_budget({**cfg, "state_memory_mb": .01}, terms)


def test_file_integrity_catches_altered_label(tmp_path):
    root = tmp_path / "integrity"
    manifest = pipeline.generate_dataset(config(), root)
    target = root / manifest["records"][0]["path"]
    with np.load(target, allow_pickle=False) as data:
        payload = {k: data[k] for k in data.files}
    payload["candidate_losses"][0] = .12345
    pipeline._atomic_npz(target, payload)
    with pytest.raises(ValueError, match="checksum"):
        pipeline.load_records(root)


def test_batched_candidate_generation_matches_scalar(tmp_path):
    cfg = config(teacher={"mode": "none"}, candidates=3)
    pipeline.generate_dataset(cfg, tmp_path / "scalar")
    pipeline.generate_dataset({**cfg, "candidate_batch_size": 2}, tmp_path / "batch")
    scalar, batch = pipeline.load_records(tmp_path / "scalar"), pipeline.load_records(tmp_path / "batch")
    for a, b in zip(scalar, batch):
        for name in ("candidate_losses", "candidate_state_error", "candidate_steps", "candidate_norm_error"):
            np.testing.assert_allclose(a[name], b[name], atol=1e-11, rtol=1e-9)
        assert json.loads(b["metadata_json"].item())["candidate_batch_size"] == 2


@pytest.mark.parametrize("selected_split", ["train", "validation"])
def test_selected_loader_never_opens_other_split_outcomes(tmp_path, monkeypatch, selected_split):
    root = tmp_path / selected_split
    manifest = pipeline.generate_dataset(config(), root)
    forbidden = {(root / row["path"]).resolve() for row in manifest["records"] if row["split"] != selected_split}
    real_load, real_read = np.load, Path.read_bytes
    def audited_load(file, *args, **kwargs):
        assert Path(file).resolve() not in forbidden
        return real_load(file, *args, **kwargs)
    def audited_read(path, *args, **kwargs):
        assert path.resolve() not in forbidden
        return real_read(path, *args, **kwargs)
    monkeypatch.setattr(np, "load", audited_load)
    monkeypatch.setattr(Path, "read_bytes", audited_read)
    records = pipeline.load_records(root, selected_split)
    assert len(records) == 1 and records[0]["split"].item() == selected_split


def test_manifest_crosssplit_duplicate_audit_does_not_need_outcome_reads(tmp_path, monkeypatch):
    root = tmp_path / "duplicate_manifest"
    manifest = pipeline.generate_dataset(config(), root)
    manifest["records"][1]["logical_fingerprint"] = manifest["records"][0]["logical_fingerprint"]
    pipeline.write_json(root / "manifest.json", manifest)
    monkeypatch.setattr(np, "load", lambda *args, **kwargs: pytest.fail("manifest audit must happen before outcome reads"))
    with pytest.raises(ValueError, match="duplicate exact"):
        pipeline.load_records(root, "train")


@pytest.mark.parametrize("variant", [{"field_distribution": "random"}, {"coupling_distribution": "concentrated"},
                                     {"shape": "banana"}, {"ports": 1.5}, {"unknown": True}])
def test_variant_validation_rejects_typos_before_generation(variant):
    with pytest.raises(ValueError):
        pipeline._validate_config(config(variants=[variant]))
