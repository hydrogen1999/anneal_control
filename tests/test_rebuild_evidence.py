"""The paper-facing artifact must be reproducible and drift detectable."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("rebuild_evidence", Path(__file__).parents[1] / "scripts/rebuild_evidence.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_archived_outcomes_rebuild_and_check_fail_on_drift(tmp_path):
    args = ["--output", str(tmp_path), "--bootstrap-resamples", "100"]
    module.main(args)
    table = json.loads((tmp_path / "comparison_table.json").read_text())
    assert table["teacher_populations"]["d2"]["n_eligible_records"] == 711
    assert table["teacher_populations"]["gap_inverse_square"]["n_eligible_records"] == 339
    searches = {row["method"]: row for row in table["rows"] if row["cost_class"] == "online_adaptation"}
    assert set(searches) == {"search_best_found", "search_bayesian", "search_policy_gradient"}
    assert all(row["n_records"] == 864 and row["objective_calls_per_instance"] == 257 for row in searches.values())
    module.main(args + ["--check"])
    (tmp_path / "RESULTS.md").write_text("untracked manual claim")
    with pytest.raises(SystemExit, match="differs"):
        module.main(args + ["--check"])


def test_reanalysis_requires_identical_record_populations_not_just_parent_sets():
    base = {"split": "test", "mode": "bank", "parent_id": "same_parent"}
    rows = [{**base, "record_id": "r1", "method": "a"}, {**base, "record_id": "r2", "method": "b"}]
    with pytest.raises(ValueError, match="same record-parent"):
        module._balanced(rows)
