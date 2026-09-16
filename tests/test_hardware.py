import copy
import json

import numpy as np
import pytest

from annealctrl.hardware import DeviceConstraints, export_program, ingest_samples
from annealctrl.schedules import Schedule


def constraints(**updates):
    data = {"device_id": "fake-device-no-qpu", "hardware_nodes": [10, 20, 30],
            "hardware_edges": [[10, 20], [20, 30]], "h_range": [-2., 2.], "J_range": [-2., 2.],
            "runtime_range": [1., 100.], "time_unit": "us", "max_schedule_points": 5,
            "max_slope": 1., "coefficient_unit": "dimensionless_ising"}
    data.update(updates)
    return DeviceConstraints.from_mapping(data)


def record():
    # Logical energy z0+z1; chain 0 is physical 0,1; chain 1 is physical 2.
    return {"physical_h": np.array([.5, .5, 1.]), "physical_edges": np.array([[0, 1], [1, 2]]),
            "physical_J": np.array([-1., 0.]), "membership": np.array([0, 0, 1]),
            "logical_h": np.array([1., 1.]), "logical_edges": np.empty((0, 2), dtype=int),
            "logical_J": np.array([]), "record_id": np.array("fixture"),
            "fingerprint": np.array("fixture-only-fingerprint"), "runtime": np.array(2.),
            "catalyst_strength": np.array(0.)}


def program(**updates):
    params = {"physical_ids": [10, 20, 30], "runtime": 10., "time_unit": "us",
              "tie_policy": "plus", "gauge": "identity"}
    params.update(updates)
    return export_program(record(), Schedule.linear(), constraints(), **params)


def samples(plan, rows=None, **updates):
    result = {"program_hash": plan["program_hash"], "vartype": "SPIN", "variable_order": [30, 10, 20],
              "energy_convention": "programmed_ising",
              "rows": rows or [{"values": [-1, -1, -1], "count": 3, "energy": -3.},
                               {"values": [-1, -1, -1], "count": 2, "energy": -3.},
                               {"values": [-1, 1, -1], "count": 5, "energy": 0.}]}
    result.update(updates)
    return result


def test_export_is_json_safe_explicit_and_deterministic():
    first = program()
    assert json.loads(json.dumps(first)) == first
    assert first["program_hash"] == program()["program_hash"]
    assert first["autoscale"] is False
    assert first["logical_ground_energy"] == -2.
    assert first["schedule"] == [[0., 0.], [10., 1.]]
    assert "not_submitted" in first["status"]


def test_ingest_reorders_and_aggregates_duplicates_without_losing_counts():
    plan = program()
    result = ingest_samples(plan, samples(plan))
    assert result["shots"] == 10
    assert result["unique_physical_samples"] == 2
    assert result["input_rows"] == 3
    assert result["successes"] == 5
    assert result["success_probability"] == .5
    assert result["mean_decoded_energy_accepted"] == -1.
    assert result["any_chain_break_probability"] == .5
    assert result["mean_chain_break_fraction"] == .25
    low, high = result["success_confidence_interval"]
    assert 0 < low < .5 < high < 1


@pytest.mark.parametrize("policy,successes,accepted", [("plus", 5, 10), ("minus", 10, 10), ("reject", 5, 5)])
def test_explicit_tie_policy_changes_decoder_not_raw_counts(policy, successes, accepted):
    plan = program(tie_policy=policy)
    result = ingest_samples(plan, samples(plan))
    assert result["successes"] == successes
    assert result["accepted_shots"] == accepted
    assert result["shots"] == 10


def test_gauge_undo_and_binary_convention_are_explicit():
    plan = program(gauge=[-1, 1, -1])
    # Original --- maps to programmed +-+; input order is qubits 30,10,20.
    supplied = samples(plan, rows=[{"values": [0, 0, 1], "count": 10, "energy": -3.}],
                       vartype="BINARY", binary_zero_spin=1)
    result = ingest_samples(plan, supplied)
    assert result["success_probability"] == 1.
    assert result["decoded_samples"][0]["original_spins"] == [-1, -1, -1]
    assert result["success_confidence_interval"][1] == 1.
    supplied.pop("binary_zero_spin")
    with pytest.raises(ValueError, match="binary_zero_spin"):
        ingest_samples(plan, supplied)


@pytest.mark.parametrize("updates", [
    {"physical_ids": [10, 10, 30]}, {"physical_ids": [10, 20, 99]},
    {"time_unit": "dimensionless"}, {"runtime": .1}, {"runtime": np.nan},
    {"coefficient_scale": 4.}, {"coefficient_scale": -1.}, {"gauge": None},
    {"gauge": [0, 1, 1]}, {"tie_policy": "random"}, {"logical_ground_energy": -100.},
])
def test_export_rejects_ambiguous_or_invalid_settings(updates):
    with pytest.raises(ValueError):
        program(**updates)


def test_topology_schedule_quantization_and_catalyst_rejections():
    kwargs = {"physical_ids": [10, 20, 30], "runtime": 10., "time_unit": "us", "tie_policy": "plus", "gauge": "identity"}
    with pytest.raises(ValueError, match="topology"):
        export_program(record(), Schedule.linear(), constraints(hardware_edges=[[10, 20]]), **kwargs)
    with pytest.raises(ValueError, match="max_schedule_points"):
        export_program(record(), Schedule([0., .3, 1.], [0., .5, 1.]), constraints(max_schedule_points=2), **kwargs)
    with pytest.raises(ValueError, match="quantization"):
        export_program(record(), Schedule([0., .33, 1.], [0., .5, 1.]), constraints(time_resolution=1.), **kwargs)
    with pytest.raises(ValueError, match="maximum"):
        export_program(record(), Schedule([0., .01, 1.], [0., .5, 1.]), constraints(), **kwargs)
    changed = record()
    changed["catalyst_strength"] = .1
    with pytest.raises(ValueError, match="catalyst"):
        export_program(changed, Schedule.linear(), constraints(), **kwargs)


@pytest.mark.parametrize("changes", [
    {"program_hash": "wrong"}, {"energy_convention": "logical"}, {"vartype": "unknown"},
    {"variable_order": [10, 20, 99]}, {"variable_order": [10, 20, 20]},
    {"rows": []}, {"rows": [{"values": [-1, -1, -1], "count": 0, "energy": -3.}]},
    {"rows": [{"values": [-1, -1, -1], "count": 1.5, "energy": -3.}]},
    {"rows": [{"values": [-1, -1, -1], "count": True, "energy": -3.}]},
    {"rows": [{"values": [-1, -1, -1], "count": 1, "energy": 9.}]},
    {"rows": [{"values": [0, -1, -1], "count": 1, "energy": -3.}]},
])
def test_invalid_sample_provenance_values_counts_and_energies(changes):
    plan = program()
    supplied = samples(plan)
    supplied.update(changes)
    with pytest.raises(ValueError):
        ingest_samples(plan, supplied)


def test_tampering_with_exported_program_is_detected():
    plan = program()
    supplied = samples(plan)
    plan["h"][0] += .1
    with pytest.raises(ValueError, match="hash"):
        ingest_samples(plan, supplied)


@pytest.mark.parametrize("updates", [
    {"hardware_nodes": [10, 20, 20]}, {"hardware_edges": [[10, 10]]},
    {"hardware_edges": [[10, 20], [20, 10]]}, {"h_range": [1., -1.]},
    {"time_unit": "ns"}, {"max_schedule_points": 1}, {"max_slope": 0.},
    {"coefficient_unit": "GHz"}, {"time_resolution": -1.},
])
def test_device_constraint_validation(updates):
    with pytest.raises(ValueError):
        constraints(**updates)
