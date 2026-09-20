"""Guards on the file that becomes a committed audit input.

rebuild_evidence.py recomputes EVIDENCE.md's contrasts from these rows, so a
malformed export does not fail loudly at export time -- it fails as a wrong
number in a table three commits later. Everything the audit will later refuse
is refused here instead.
"""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "export_heldout_records", Path(__file__).parents[1] / "scripts/export_heldout_records.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def row(method="summary", mode="bank", record="r0", parent="p0", **over):
    base = {"method": method, "mode": mode, "record_id": record, "parent_id": parent,
            "split": "test", "loss": 0.5, "linear_loss": 0.6, "global_loss": 0.55,
            "bank_best_loss": 0.45, "bank_regret": 0.05, "family": "spin_glass",
            "logical_n": 5, "physical_n": 10, "runtime": 4.0, "inference_seconds": 0.001}
    base.update(over)
    return base


def test_projection_keeps_exactly_the_audit_fields():
    kept = module.project(row(metrics={"junk": 1}, selected_index=3, candidate_count=64))
    assert set(kept) == set(module.AUDIT_FIELDS)
    assert "metrics" not in kept and "selected_index" not in kept


def test_a_row_missing_an_audit_field_is_refused():
    incomplete = row()
    del incomplete["global_loss"]
    with pytest.raises(ValueError, match="global_loss"):
        module.project(incomplete)


def test_balance_accepts_matched_panels():
    rows = [row(method=m, mode=d, record=f"r{i}", parent=f"p{i}")
            for m in ("summary", "logical") for d in ("bank", "direct") for i in range(3)]
    panels, records, parents = module.check_balanced(rows)
    assert (panels, records, parents) == (4, 3, 3)


def test_a_non_test_row_is_refused():
    with pytest.raises(ValueError, match="held-out test"):
        module.check_balanced([row(split="validation")])


def test_a_duplicate_record_within_a_panel_is_refused():
    with pytest.raises(ValueError, match="duplicate"):
        module.check_balanced([row(), row()])


def test_panels_over_different_populations_are_refused():
    """Unpaired methods would silently produce an unpaired 'paired' contrast."""
    rows = [row(method="summary", record="r0", parent="p0"),
            row(method="logical", record="r1", parent="p1")]
    with pytest.raises(ValueError, match="different record-parent population"):
        module.check_balanced(rows)


def test_a_record_reassigned_to_another_parent_is_refused():
    rows = [row(method="summary", record="r0", parent="p0"),
            row(method="logical", record="r0", parent="p9")]
    with pytest.raises(ValueError, match="different record-parent population"):
        module.check_balanced(rows)


def test_export_end_to_end(tmp_path):
    rows = [row(method=m, mode=d, record=f"r{i}", parent=f"p{i}", metrics={"junk": 1})
            for m in ("summary", "logical") for d in ("bank", "direct") for i in range(2)]
    results = tmp_path / "results.json"
    results.write_text(json.dumps({"record_means": rows, "summary": [{"method": "summary"}]}))
    out = tmp_path / "heldout_records.json"
    module.main(["--results", str(results), "--experiment", "demo", "--output", str(out)])
    payload = json.loads(out.read_text())
    assert payload["experiment"] == "demo" and payload["n_rows"] == 8
    assert all(set(r) == set(module.AUDIT_FIELDS) for r in payload["record_means"])


def test_a_results_file_without_record_means_is_rejected(tmp_path):
    results = tmp_path / "results.json"
    results.write_text(json.dumps({"summary": []}))
    with pytest.raises(SystemExit):
        module.main(["--results", str(results), "--experiment", "x",
                     "--output", str(tmp_path / "o.json")])
