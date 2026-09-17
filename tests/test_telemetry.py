import json

import pytest

from annealctrl.telemetry import RunLog, peak_rss_bytes


def read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_every_line_is_json_with_the_same_run_id(tmp_path):
    log = RunLog(tmp_path / "sweep.jsonl", command="control-sweep", settings={"budget": 8})
    log.event("unit_start", unit="rec_a")
    log.event("unit_end", unit="rec_a", objective_calls=8)
    log.close(units=1)

    lines = read(tmp_path / "sweep.jsonl")
    assert [line["event"] for line in lines] == ["run_start", "unit_start", "unit_end", "run_end"]
    assert len({line["run_id"] for line in lines}) == 1
    assert [line["seq"] for line in lines] == [0, 1, 2, 3]


def test_run_start_records_command_settings_and_environment(tmp_path):
    log = RunLog(tmp_path / "sweep.jsonl", command="control-sweep", settings={"budget": 8, "seed": 3})
    log.close(units=0)

    start = read(tmp_path / "sweep.jsonl")[0]
    assert start["command"] == "control-sweep"
    assert start["settings"] == {"budget": 8, "seed": 3}
    assert start["environment"]["python"]
    assert start["telemetry_is_operational_not_scientific_evidence"] is True


def test_run_end_totals_wall_time_and_peak_memory(tmp_path):
    log = RunLog(tmp_path / "sweep.jsonl", command="x", settings={})
    log.close(units=4, objective_calls=32)

    end = read(tmp_path / "sweep.jsonl")[-1]
    assert end["units"] == 4 and end["objective_calls"] == 32
    assert end["wall_seconds"] >= 0
    assert end["peak_rss_bytes"] > 0
    assert end["peak_rss_is_estimate"] is True


def test_reopening_an_existing_log_appends_and_starts_a_new_run_id(tmp_path):
    path = tmp_path / "sweep.jsonl"
    first = RunLog(path, command="x", settings={})
    first.close(units=1)
    second = RunLog(path, command="x", settings={})
    second.close(units=1)

    lines = read(path)
    assert len(lines) == 4
    assert lines[0]["run_id"] != lines[2]["run_id"]
    # Resume boundary must stay legible: sequence restarts with the new run id.
    assert lines[2]["seq"] == 0


def test_credential_shaped_field_names_are_refused(tmp_path):
    log = RunLog(tmp_path / "sweep.jsonl", command="x", settings={})
    for name in ("api_key", "token", "password", "SECRET", "access_token"):
        with pytest.raises(ValueError, match="credential"):
            log.event("unit_end", **{name: "value"})
    log.close(units=0)


def test_credential_shaped_settings_keys_are_refused(tmp_path):
    with pytest.raises(ValueError, match="credential"):
        RunLog(tmp_path / "sweep.jsonl", command="x", settings={"auth_token": "abc"})


def test_nonfinite_values_are_written_as_null_not_nan(tmp_path):
    log = RunLog(tmp_path / "sweep.jsonl", command="x", settings={})
    log.event("unit_end", loss=float("nan"), ratio=float("inf"))
    log.close(units=1)

    line = read(tmp_path / "sweep.jsonl")[1]
    assert line["loss"] is None and line["ratio"] is None


def test_reserved_field_names_cannot_be_overwritten(tmp_path):
    log = RunLog(tmp_path / "sweep.jsonl", command="x", settings={})
    for name in ("run_id", "seq", "t", "event"):
        with pytest.raises(ValueError, match="reserved"):
            log.event("unit_end", **{name: 1})
    log.close(units=0)


def test_event_name_must_be_a_nonempty_identifier(tmp_path):
    log = RunLog(tmp_path / "sweep.jsonl", command="x", settings={})
    for bad in ("", "  ", "has space", 3):
        with pytest.raises(ValueError):
            log.event(bad)
    log.close(units=0)


def test_closing_twice_does_not_write_a_second_run_end(tmp_path):
    log = RunLog(tmp_path / "sweep.jsonl", command="x", settings={})
    log.close(units=1)
    log.close(units=1)
    assert sum(line["event"] == "run_end" for line in read(tmp_path / "sweep.jsonl")) == 1


def test_events_after_close_are_refused(tmp_path):
    log = RunLog(tmp_path / "sweep.jsonl", command="x", settings={})
    log.close(units=0)
    with pytest.raises(RuntimeError, match="closed"):
        log.event("unit_start")


def test_context_manager_closes_and_records_a_failure(tmp_path):
    with pytest.raises(ZeroDivisionError):
        with RunLog(tmp_path / "sweep.jsonl", command="x", settings={}) as log:
            log.event("unit_start", unit="a")
            raise ZeroDivisionError("boom")

    end = read(tmp_path / "sweep.jsonl")[-1]
    assert end["event"] == "run_end"
    assert end["status"] == "failed"
    assert end["error_type"] == "ZeroDivisionError"


def test_peak_rss_bytes_is_positive_and_plausible():
    value = peak_rss_bytes()
    assert isinstance(value, int)
    # Any live CPython interpreter holds more than 1 MiB and less than 1 TiB.
    assert 2**20 < value < 2**40


# --- the shared config allowlist ---------------------------------------------

def test_unknown_config_keys_permits_annotations_and_rejects_typos():
    """Written out three separate times before; one implementation now, three callers."""
    from annealctrl.telemetry import unknown_config_keys

    allowed = {"budget", "split"}
    assert unknown_config_keys({"budget": 1, "_why": "a note"}, allowed) == []
    assert unknown_config_keys({"budget": 1, "_scope_note": "x", "_provenance": "y"}, allowed) == []
    assert unknown_config_keys({"budgte": 1}, allowed) == ["budgte"]
    assert unknown_config_keys({"budget": 1, "zzz": 2, "aaa": 3}, allowed) == ["aaa", "zzz"]


def test_unknown_config_keys_is_used_by_every_config_validator():
    """A fourth copy of this rule is how the same failure comes back."""
    from annealctrl.experiments import validate_experiment
    from annealctrl.headroom import load_frontier_config
    from annealctrl.interventions import plan_intervention_pairs

    for call in (lambda: load_frontier_config({"split": "test", "_note": "x"}),
                 lambda: plan_intervention_pairs({"parents": 3, "logical_qubits": 3,
                                                  "chain_lengths": [2, 2, 2], "_note": "x"}),
                 lambda: validate_experiment({"dataset": "d.json", "seeds": [0],
                                              "methods": [{"name": "m", "model": {}}],
                                              "_note": "x"})):
        try:
            call()
        except ValueError as error:
            assert "_note" not in str(error), error
