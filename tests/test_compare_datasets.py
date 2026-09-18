"""Regression tests for dataset audits that previously accepted invalid inputs."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("compare_datasets", Path(__file__).parents[1] / "scripts/compare_datasets.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def copies(tmp_path, left=None, right=None):
    a, b = tmp_path / "old", tmp_path / "new"
    a.mkdir(); b.mkdir()
    np.savez(a / "r.npz", **(left if left is not None else {"loss": np.array([.5])}))
    np.savez(b / "r.npz", **(right if right is not None else {"loss": np.array([.5])}))
    return a, b


def test_missing_and_empty_record_sets_fail_closed(tmp_path):
    a, b = copies(tmp_path)
    np.savez(a / "extra.npz", loss=[.2])
    report = module.compare_datasets(a, b)
    assert not report["passed"] and report["missing_from_new"] == ["extra.npz"]
    (b / "r.npz").unlink()
    with pytest.raises(ValueError, match="at least one"):
        module.compare_datasets(a, b)


@pytest.mark.parametrize("left,right", [
    ({"x": np.array([np.nan])}, {"x": np.array([np.nan])}),
    ({"x": np.array([np.inf])}, {"x": np.array([np.inf])}),
    ({"x": np.array([1+2j])}, {"x": np.array([1+3j])}),
    ({"x": np.array([1.])}, {"x": np.array([[1.]])}),
    ({"x": np.array([1.])}, {"y": np.array([1.])}),
    ({}, {}),
])
def test_invalid_or_changed_science_is_not_equal(tmp_path, left, right):
    assert not module.compare_datasets(*copies(tmp_path, left, right))["passed"]


def test_only_declared_fields_are_excluded_metadata_science_is_checked(tmp_path):
    metadata = lambda runtime, seconds: np.array(json.dumps({"runtime": runtime, "wall_seconds": seconds}))
    a, b = copies(tmp_path, {"loss": np.array([.5]), "metadata_json": metadata(2, 3)},
                  {"loss": np.array([.5]), "metadata_json": metadata(2, 8)})
    report = module.compare_datasets(a, b)
    assert report["passed"] and report["excluded_field_changes"]
    np.savez(b / "r.npz", loss=np.array([.5]), metadata_json=metadata(8, 8))
    assert not module.compare_datasets(a, b)["passed"]


def test_cli_machine_json_exit_and_no_hardcoded_inputs(tmp_path, capsys):
    a, b = copies(tmp_path)
    output = tmp_path / "audit.json"
    assert module.main([str(a), str(b), "--output", str(output)]) == 0
    assert json.loads(capsys.readouterr().out) == json.loads(output.read_text())
    np.savez(b / "r.npz", loss=np.array([.50000001]))
    assert module.main([str(a), str(b)]) == 1
    assert module.main([str(a), str(b), "--atol", "0.001"]) == 0


def test_nonfinite_scientific_metadata_is_rejected(tmp_path):
    value = {"metadata_json": np.array('{"runtime": 1e999}')}
    assert not module.compare_datasets(*copies(tmp_path, value, value))["passed"]


def test_corrupt_records_and_only_timing_do_not_certify_science(tmp_path):
    a, b = copies(tmp_path, {"candidate_seconds": np.array([1.])}, {"candidate_seconds": np.array([2.])})
    assert not module.compare_datasets(a, b)["passed"]
    (a / "r.npz").write_bytes(b"PK damaged archive")
    assert not module.compare_datasets(a, b)["passed"]
