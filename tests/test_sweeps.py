import json

import pytest

from annealctrl.sweeps import SweepUnit, completed_rows, run_sweep, settings_hash


def units(names=("a", "b", "c")):
    return [SweepUnit(unit_id=name, fingerprint=f"fp-{name}", payload={"n": index})
            for index, name in enumerate(names)]


def rows(path):
    return [json.loads(line) for line in (path / "rows.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


def test_settings_hash_is_order_independent_but_value_sensitive():
    assert settings_hash({"a": 1, "b": 2}) == settings_hash({"b": 2, "a": 1})
    assert settings_hash({"a": 1}) != settings_hash({"a": 2})
    assert settings_hash({"a": 1}) != settings_hash({"a": 1, "b": None})


def test_settings_hash_refuses_unserialisable_settings():
    with pytest.raises(ValueError):
        settings_hash({"a": {1, 2}})


def test_sweep_runs_every_unit_once_and_records_rows(tmp_path):
    seen = []

    def worker(unit):
        seen.append(unit.unit_id)
        return {"value": unit.payload["n"] * 2}

    manifest = run_sweep(units(), worker, output=tmp_path / "out",
                         settings={"budget": 4}, command="test-sweep", source_hash="src")

    assert seen == ["a", "b", "c"]
    assert manifest["completed"] == 3 and manifest["failed"] == 0 and manifest["skipped"] == 0
    written = rows(tmp_path / "out")
    assert [row["unit_id"] for row in written] == ["a", "b", "c"]
    assert [row["result"]["value"] for row in written] == [0, 2, 4]
    assert {row["settings_hash"] for row in written} == {settings_hash({"budget": 4})}
    assert all(row["status"] == "ok" for row in written)


def test_existing_output_without_resume_is_refused(tmp_path):
    run_sweep(units(("a",)), lambda unit: {}, output=tmp_path / "out",
              settings={}, command="c", source_hash="src")
    with pytest.raises(FileExistsError):
        run_sweep(units(("a",)), lambda unit: {}, output=tmp_path / "out",
                  settings={}, command="c", source_hash="src")


def test_resume_skips_completed_units_and_runs_only_the_new_ones(tmp_path):
    run_sweep(units(("a", "b")), lambda unit: {"value": unit.payload["n"]},
              output=tmp_path / "out", settings={"budget": 4}, command="c", source_hash="src")

    seen = []

    def worker(unit):
        seen.append(unit.unit_id)
        return {"value": unit.payload["n"]}

    manifest = run_sweep(units(("a", "b", "c")), worker, output=tmp_path / "out",
                         settings={"budget": 4}, command="c", source_hash="src", resume=True)

    assert seen == ["c"]
    assert manifest["skipped"] == 2 and manifest["completed"] == 1
    assert [row["unit_id"] for row in rows(tmp_path / "out")] == ["a", "b", "c"]


def test_resume_refuses_a_changed_settings_hash(tmp_path):
    run_sweep(units(("a",)), lambda unit: {}, output=tmp_path / "out",
              settings={"budget": 4}, command="c", source_hash="src")
    with pytest.raises(ValueError, match="settings"):
        run_sweep(units(("a",)), lambda unit: {}, output=tmp_path / "out",
                  settings={"budget": 8}, command="c", source_hash="src", resume=True)


def test_resume_refuses_a_changed_source_hash(tmp_path):
    run_sweep(units(("a",)), lambda unit: {}, output=tmp_path / "out",
              settings={}, command="c", source_hash="src-1")
    with pytest.raises(ValueError, match="source"):
        run_sweep(units(("a",)), lambda unit: {}, output=tmp_path / "out",
                  settings={}, command="c", source_hash="src-2", resume=True)


def test_a_changed_record_fingerprint_forces_recomputation(tmp_path):
    run_sweep(units(("a",)), lambda unit: {"v": 1}, output=tmp_path / "out",
              settings={}, command="c", source_hash="src")
    changed = [SweepUnit(unit_id="a", fingerprint="fp-a-modified", payload={"n": 0})]

    seen = []
    run_sweep(changed, lambda unit: seen.append(unit.unit_id) or {"v": 2},
              output=tmp_path / "out", settings={}, command="c", source_hash="src", resume=True)

    assert seen == ["a"]
    assert [row["result"]["v"] for row in rows(tmp_path / "out")] == [1, 2]


def test_a_truncated_final_line_is_repaired_on_resume(tmp_path):
    out = tmp_path / "out"
    run_sweep(units(("a", "b")), lambda unit: {"v": unit.payload["n"]},
              output=out, settings={}, command="c", source_hash="src")
    path = out / "rows.jsonl"
    path.write_text(path.read_text(encoding="utf-8")[:-12], encoding="utf-8")

    seen = []
    manifest = run_sweep(units(("a", "b")), lambda unit: seen.append(unit.unit_id) or {"v": 9},
                         output=out, settings={}, command="c", source_hash="src", resume=True)

    assert seen == ["b"], "the unit whose row was lost must be recomputed"
    assert manifest["repaired_truncated_rows"] == 1
    assert [row["unit_id"] for row in rows(out)] == ["a", "b"]


def test_corruption_that_is_not_the_final_line_is_refused(tmp_path):
    out = tmp_path / "out"
    run_sweep(units(("a", "b")), lambda unit: {"v": 0}, output=out,
              settings={}, command="c", source_hash="src")
    path = out / "rows.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(["{ broken", *lines]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="corrupt"):
        run_sweep(units(("a", "b")), lambda unit: {"v": 0}, output=out,
                  settings={}, command="c", source_hash="src", resume=True)


def test_a_failing_unit_aborts_by_default_and_records_the_failure(tmp_path):
    def worker(unit):
        if unit.unit_id == "b":
            raise ArithmeticError("convergence gate")
        return {"v": 1}

    with pytest.raises(ArithmeticError):
        run_sweep(units(), worker, output=tmp_path / "out", settings={},
                  command="c", source_hash="src")

    written = rows(tmp_path / "out")
    assert [(row["unit_id"], row["status"]) for row in written] == [("a", "ok"), ("b", "failed")]
    assert written[1]["error_type"] == "ArithmeticError"


def test_continue_on_error_records_failures_and_finishes(tmp_path):
    def worker(unit):
        if unit.unit_id == "b":
            raise ArithmeticError("convergence gate")
        return {"v": 1}

    manifest = run_sweep(units(), worker, output=tmp_path / "out", settings={},
                         command="c", source_hash="src", on_error="record")

    assert manifest["completed"] == 2 and manifest["failed"] == 1
    assert manifest["failed_unit_ids"] == ["b"]


def test_resume_retries_a_previously_failed_unit(tmp_path):
    out = tmp_path / "out"
    run_sweep(units(("a", "b")), lambda unit: (_ for _ in ()).throw(ArithmeticError("x"))
              if unit.unit_id == "b" else {"v": 1},
              output=out, settings={}, command="c", source_hash="src", on_error="record")

    seen = []
    run_sweep(units(("a", "b")), lambda unit: seen.append(unit.unit_id) or {"v": 2},
              output=out, settings={}, command="c", source_hash="src", resume=True)

    assert seen == ["b"], "a failed unit is not complete and must be retried"
    assert completed_rows(out)["b"]["result"]["v"] == 2


def test_completed_rows_returns_the_last_row_per_unit(tmp_path):
    out = tmp_path / "out"
    run_sweep(units(("a",)), lambda unit: {"v": 1}, output=out, settings={},
              command="c", source_hash="src")
    changed = [SweepUnit(unit_id="a", fingerprint="fp-a-2", payload={"n": 0})]
    run_sweep(changed, lambda unit: {"v": 2}, output=out, settings={},
              command="c", source_hash="src", resume=True)

    assert completed_rows(out)["a"]["result"]["v"] == 2


def test_dry_run_plans_without_executing(tmp_path):
    seen = []
    manifest = run_sweep(units(), lambda unit: seen.append(unit) or {}, output=tmp_path / "out",
                         settings={"budget": 4}, command="c", source_hash="src", dry_run=True)

    assert seen == []
    assert manifest["planned"] == 3 and manifest["dry_run"] is True
    assert not (tmp_path / "out").exists(), "a plan must not create an output directory"


def test_duplicate_unit_ids_are_refused(tmp_path):
    duplicated = [SweepUnit("a", "fp1", {}), SweepUnit("a", "fp2", {})]
    with pytest.raises(ValueError, match="duplicate"):
        run_sweep(duplicated, lambda unit: {}, output=tmp_path / "out",
                  settings={}, command="c", source_hash="src")


def test_manifest_and_telemetry_are_written(tmp_path):
    out = tmp_path / "out"
    run_sweep(units(("a",)), lambda unit: {"v": 1}, output=out, settings={"budget": 2},
              command="control-sweep", source_hash="src")

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["command"] == "control-sweep"
    assert manifest["settings"] == {"budget": 2}
    assert manifest["settings_hash"] == settings_hash({"budget": 2})
    assert manifest["source_hash"] == "src"
    assert manifest["environment"]["python"]

    events = [json.loads(line) for line in (out / "telemetry.jsonl").read_text().splitlines()]
    assert [event["event"] for event in events] == ["run_start", "unit_start", "unit_end", "run_end"]


def test_worker_must_return_a_mapping(tmp_path):
    with pytest.raises(TypeError, match="mapping"):
        run_sweep(units(("a",)), lambda unit: [1, 2], output=tmp_path / "out",
                  settings={}, command="c", source_hash="src")


def test_aborted_sweep_still_writes_a_manifest_marked_aborted(tmp_path):
    def worker(unit):
        if unit.unit_id == "b":
            raise ArithmeticError("convergence gate")
        return {"v": 1}

    with pytest.raises(ArithmeticError):
        run_sweep(units(), worker, output=tmp_path / "out", settings={},
                  command="c", source_hash="src")

    manifest = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert manifest["status"] == "aborted"
    assert manifest["completed"] == 1 and manifest["failed"] == 1
    events = [json.loads(line) for line in (tmp_path / "out" / "telemetry.jsonl").read_text().splitlines()]
    assert events[-1]["event"] == "run_end" and events[-1]["status"] == "aborted"
