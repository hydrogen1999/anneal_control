"""Campaign-level order, freeze and interrupted-work barriers."""
import json

import pytest

from annealctrl import paper_campaign as campaign


def config(tmp_path):
    cfg = {"schema_version": 1, "name": "test", "steps": [
        {"id": "source", "kind": "generate", "config": {"seed": 1}, "output": "source"},
        {"id": "target", "kind": "generate", "config": {"seed": 2}, "depends": ["source"]},
    ]}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    return path, cfg


def test_campaign_freezes_outputs_and_resumes_without_reexecution(tmp_path, monkeypatch):
    path, _ = config(tmp_path)
    calls = []
    def dispatch(step, **kwargs):
        calls.append(step["id"])
        destination = campaign.Path(step["output"])
        destination.mkdir()
        (destination / "result.json").write_text(json.dumps(step["config"]))
    monkeypatch.setattr(campaign, "_dispatch", dispatch)
    root = tmp_path / "run"
    first = campaign.run_campaign(path, root, through="source")
    assert first["status"] == "partial" and calls == ["source"]
    second = campaign.run_campaign(path, root, resume=True)
    assert second["status"] == "complete" and calls == ["source", "target"]
    campaign.run_campaign(path, root, resume=True)
    assert calls == ["source", "target"]
    (root / "source" / "result.json").write_text("changed")
    with pytest.raises(ValueError, match="artifacts changed"):
        campaign.run_campaign(path, root, resume=True)


@pytest.mark.parametrize("mutation", ["cycle", "outside", "overlap", "duplicate", "unknown"])
def test_invalid_campaign_refused_before_work(tmp_path, mutation):
    path, cfg = config(tmp_path)
    if mutation == "cycle":
        cfg["steps"][0]["depends"] = ["target"]
    elif mutation == "outside":
        cfg["steps"][0]["output"] = "../escape"
    elif mutation == "overlap":
        cfg["steps"][1]["output"] = "source/child"
    elif mutation == "duplicate":
        cfg["steps"][1]["id"] = "source"
    else:
        cfg["steps"][0]["kind"] = "shell"
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError):
        campaign.load_campaign(path, tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_failed_attempt_is_retained_and_config_changes_refused(tmp_path, monkeypatch):
    path, cfg = config(tmp_path)
    def fail(*args, **kwargs):
        raise ArithmeticError("real solver failure")
    monkeypatch.setattr(campaign, "_dispatch", fail)
    root = tmp_path / "run"
    with pytest.raises(ArithmeticError):
        campaign.run_campaign(path, root)
    manifest = json.loads((root / "campaign.json").read_text())
    assert manifest["steps"]["source"]["attempts"][0]["status"] == "failed"
    cfg["steps"][0]["config"]["seed"] = 3
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="config/source changed"):
        campaign.run_campaign(path, root, resume=True)


def test_plan_resolves_config_without_claiming_a_result(tmp_path):
    path, _ = config(tmp_path)
    result = campaign.plan_campaign(path, tmp_path / "run")
    assert result["scientific_verdict"] == "unmeasured"
    assert not (tmp_path / "run").exists()


def test_referenced_study_paths_resolve_against_its_file(tmp_path):
    path, cfg = config(tmp_path)
    folder = tmp_path / "nested"
    folder.mkdir()
    (folder / "noise.json").write_text(json.dumps({"data": "../target", "source_data": "../source",
                                                 "checkpoint": "../model.pt"}))
    cfg["steps"] = [{"id": "noise", "kind": "noise", "config": "nested/noise.json"}]
    path.write_text(json.dumps(cfg))
    resolved = campaign.load_campaign(path, tmp_path / "run")["steps"][0]["config"]
    assert resolved["data"] == str(tmp_path / "target")
    assert resolved["checkpoint"] == str(tmp_path / "model.pt")
