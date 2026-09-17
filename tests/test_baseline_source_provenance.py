import json

import pytest


def test_reference_refuses_source_drift_but_retains_work(tmp_path, monkeypatch):
    from annealctrl import finzgar_reference, pipeline
    hashes = iter(["before", "after"])
    monkeypatch.setattr(pipeline, "source_fingerprint", lambda: next(hashes))
    monkeypatch.setattr(finzgar_reference, "run_reference", lambda **kw: {"total_objective_calls": 60})
    path = tmp_path / "reference.json"
    with pytest.raises(RuntimeError, match="source changed"):
        finzgar_reference.main(["--out", str(path)])
    result = json.loads(path.read_text())
    assert not result["source_frozen"]
    assert result["total_objective_calls"] == 60


def test_embedded_runner_refuses_source_drift_but_retains_work(tmp_path, monkeypatch):
    from annealctrl import literature_baselines, pipeline
    hashes = iter(["before", "after"])
    monkeypatch.setattr(pipeline, "source_fingerprint", lambda: next(hashes))
    monkeypatch.setattr(pipeline, "load_records", lambda *a: [{"record_id": "r"}])
    monkeypatch.setattr(literature_baselines, "benchmark_finzgar_record", lambda *a, **kw: {"n_objective_calls": 60})
    (tmp_path / "manifest.json").write_text("{}")
    path = tmp_path / "baseline.jsonl"
    with pytest.raises(RuntimeError, match="source changed"):
        literature_baselines.main(["--data", str(tmp_path), "--out", str(path)])
    result = json.loads(path.read_text())
    assert not result["source_frozen"]
    assert result["n_objective_calls"] == 60
