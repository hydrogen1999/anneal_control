"""The budget axis behind "one forward pass is worth N calls".

The headline number depends entirely on how a per-family search budget is
converted into a total call count. Each tunable family is searched under its
own equal budget and the linear reference is one free evaluation, so the
incumbent after m evaluations per family sits at 1 + (#tunable) * m total
calls. If that mapping is wrong the whole unit is wrong, so it is pinned here.
"""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "search_equivalence", Path(__file__).parents[1] / "scripts/search_equivalence.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def benchmark(tmp_path, record_id, parent_id, *, linear, traces, sub="shard_0"):
    folder = tmp_path / sub / "benchmarks"
    folder.mkdir(parents=True, exist_ok=True)
    payload = {"record_id": record_id, "parent_id": parent_id,
               "families": {"linear": {"best_loss": linear, "records": [{"loss": linear}]}}}
    for name, losses in traces.items():
        payload["families"][name] = {"best_loss": min(losses),
                                     "records": [{"loss": v} for v in losses]}
    (folder / f"{record_id}.json").write_text(json.dumps(payload))
    return tmp_path


def test_the_budget_axis_is_one_plus_families_times_depth(tmp_path):
    benchmark(tmp_path, "r0", "p0", linear=1.0,
              traces={"a": [0.9, 0.8, 0.7], "b": [0.95, 0.85, 0.75]})
    curves = module.parent_curves([tmp_path])
    assert sorted(curves["p0"]) == [3, 5, 7]        # 1 + 2*1, 1 + 2*2, 1 + 2*3


def test_headroom_is_linear_minus_the_running_best(tmp_path):
    benchmark(tmp_path, "r0", "p0", linear=1.0,
              traces={"a": [0.9, 0.8, 0.7], "b": [0.95, 0.85, 0.75]})
    curve = module.parent_curves([tmp_path])["p0"]
    assert curve[3] == pytest.approx(0.10)   # best of 0.9, 0.95
    assert curve[5] == pytest.approx(0.20)   # best of 0.8, 0.85
    assert curve[7] == pytest.approx(0.30)


def test_the_curve_is_an_incumbent_so_a_worse_trial_does_not_lower_it(tmp_path):
    benchmark(tmp_path, "r0", "p0", linear=1.0, traces={"a": [0.5, 0.9, 0.95]})
    curve = module.parent_curves([tmp_path])["p0"]
    assert curve[2] == pytest.approx(0.5)
    assert curve[3] == pytest.approx(0.5)    # 0.9 is worse; the incumbent holds
    assert curve[4] == pytest.approx(0.5)


def test_a_search_never_beating_linear_has_zero_headroom_not_negative(tmp_path):
    benchmark(tmp_path, "r0", "p0", linear=0.5, traces={"a": [0.9, 0.8]})
    curve = module.parent_curves([tmp_path])["p0"]
    assert curve[2] == pytest.approx(0.0)
    assert curve[3] == pytest.approx(0.0)


def test_records_are_averaged_within_their_parent(tmp_path):
    benchmark(tmp_path, "r0", "p0", linear=1.0, traces={"a": [0.8]})
    benchmark(tmp_path, "r1", "p0", linear=1.0, traces={"a": [0.6]})
    curve = module.parent_curves([tmp_path])["p0"]
    assert curve[2] == pytest.approx(0.30)    # mean of 0.2 and 0.4


def test_a_record_with_no_retained_trials_is_refused(tmp_path):
    folder = tmp_path / "shard_0" / "benchmarks"
    folder.mkdir(parents=True)
    (folder / "r0.json").write_text(json.dumps(
        {"record_id": "r0", "parent_id": "p0",
         "families": {"linear": {"best_loss": 1.0, "records": [{"loss": 1.0}]}}}))
    with pytest.raises(ValueError, match="retain_full_trials|no trials"):
        module.parent_curves([tmp_path])


def test_a_record_missing_its_linear_reference_is_refused(tmp_path):
    folder = tmp_path / "shard_0" / "benchmarks"
    folder.mkdir(parents=True)
    (folder / "r0.json").write_text(json.dumps(
        {"record_id": "r0", "parent_id": "p0",
         "families": {"a": {"best_loss": 0.8, "records": [{"loss": 0.8}]}}}))
    with pytest.raises(ValueError, match="linear"):
        module.parent_curves([tmp_path])


def test_records_searching_different_families_are_refused(tmp_path):
    """A shared budget axis is meaningless if the records did not share families."""
    benchmark(tmp_path, "r0", "p0", linear=1.0, traces={"a": [0.8], "b": [0.7]})
    benchmark(tmp_path, "r1", "p1", linear=1.0, traces={"a": [0.8]})
    with pytest.raises(ValueError, match="families|budget axis"):
        module.parent_curves([tmp_path])


def test_uneven_trace_depths_truncate_to_the_shortest(tmp_path):
    """The budget axis must stay aligned across families."""
    benchmark(tmp_path, "r0", "p0", linear=1.0, traces={"a": [0.9, 0.8, 0.7], "b": [0.95]})
    curves = module.parent_curves([tmp_path])
    assert sorted(curves["p0"]) == [3]


# --- the direct-policy join -------------------------------------------------

def evaluation_payload(direct_losses, *, linear=1.0):
    records = [{"record_id": f"r{i}", "parent_id": f"p{i}", "linear_loss": linear,
                "selected_loss": 0.5, "bank_best_loss": 0.4, "candidate_count": 64}
               for i in range(len(direct_losses))]
    direct = [{"record_id": f"r{i}", "parent_id": f"p{i}", "loss": v}
              for i, v in enumerate(direct_losses)]
    return {"records": records, "direct_policy": {"records": direct}}


def test_direct_losses_are_joined_to_the_linear_reference_by_record():
    rows = module.direct_rows(evaluation_payload([0.8, 0.6]))
    assert [r["selected_loss"] for r in rows] == [0.8, 0.6]
    assert all(r["linear_loss"] == 1.0 for r in rows)


def test_a_direct_record_with_no_linear_counterpart_is_refused():
    """Dropping it silently would change the population the gain is measured over."""
    payload = evaluation_payload([0.8, 0.6])
    payload["direct_policy"]["records"].append(
        {"record_id": "rX", "parent_id": "pX", "loss": 0.5})
    with pytest.raises(ValueError, match="no linear reference|different populations"):
        module.direct_rows(payload)


def test_bank_ceiling_is_undefined_for_the_direct_policy(tmp_path):
    """The direct policy generates a control; it has no candidate bank to top out on."""
    (tmp_path / "summary__seed_0.json").write_text(json.dumps(evaluation_payload([0.8])))
    with pytest.raises(ValueError, match="no candidate bank|undefined"):
        module.parent_gains(tmp_path, "summary", target="bank_ceiling", mode="direct")
