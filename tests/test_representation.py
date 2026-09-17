"""G3 model side: does physical information change a learned model's decision?

The structural fact this rests on: a `logical` encoder receives bitwise identical
input on both arms of a single-factor pair, because both arms share the logical
graph, the logical coefficients and the runtime, and the logical variant sees no
physical or chain feature and no programmed scale. It therefore *cannot* propose
different controls. On a pair whose preferred control genuinely swaps, it must be
wrong on at least one arm, and that excess loss is the measurable cost of
embedding blindness.
"""
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from annealctrl.generation import IsingProblem  # noqa: E402
from annealctrl.interventions import build_pairs, cross_control_matrix  # noqa: E402
from annealctrl.representation import (  # noqa: E402
    aggregate_model_interventions,
    arm_record,
    model_intervention_response,
    sweep_model_interventions,
)

BASE = {"shape": "path", "ports": 1, "field_distribution": "uniform",
        "coupling_distribution": "uniform", "chain_strength": 1.5}
LENGTHS = np.array([3, 2, 1])


def problem():
    return IsingProblem(np.array([0.4, -0.3, 0.2]), np.array([[0, 1], [1, 2], [0, 2]]),
                        np.array([0.9, -0.6, 0.5]))


def pair(factor="chain_strength", change=None, **kwargs):
    change = change or {"chain_strength": 3.0}
    return build_pairs(problem(), {"factor": factor, "base": dict(BASE), "change": change},
                       lengths=LENGTHS, runtime=2.0, seed=5, **kwargs)[0]


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory):
    """Two tiny real checkpoints: logical-only and physical-aware."""
    from annealctrl.learning import fit_records
    from annealctrl.pipeline import generate_dataset, load_records

    root = tmp_path_factory.mktemp("repr")
    config = {"seed": 91, "parents": 6, "families": ["spin_glass"], "logical_qubits": 3,
              "chain_lengths": [2, 1, 1], "variants": [{"shape": "path", "ports": 1}],
              "chain_strengths": [1.5], "runtimes": [2.0], "candidates": 4,
              "spectral_points": 3, "steps": 16, "max_steps": 1024,
              "label_state_tolerance": 0.005, "max_physical_qubits": 4,
              "teacher": {"mode": "none"}}
    generate_dataset(config, root / "data")
    train = load_records(root / "data", "train")
    validation = load_records(root / "data", "validation")
    made = {}
    for variant in ("logical", "physical"):
        path = root / f"{variant}.pt"
        fit_records(train, validation, model_config={"encoder_variant": variant, "width": 8},
                    epochs=1, seed=3, checkpoint=path, response_weight=0.0)
        made[variant] = path
    return made


# --- arm records -------------------------------------------------------------

def test_arm_record_carries_everything_graph_from_record_needs():
    from annealctrl.models import graph_from_record
    built = pair()
    for arm in built.arms:
        record = arm_record(arm, built.runtime)
        graph_from_record(record)          # must not raise
        assert float(record["runtime"]) == 2.0
        assert len(record["physical_h"]) == arm.compiled.physical.n


def test_the_logical_view_of_both_arms_is_identical():
    """The premise of the whole experiment, asserted rather than assumed."""
    built = pair()
    a, b = (arm_record(arm, built.runtime) for arm in built.arms)
    for key in ("logical_h", "logical_J", "logical_edges"):
        np.testing.assert_array_equal(np.asarray(a[key]), np.asarray(b[key]))
    assert float(a["runtime"]) == float(b["runtime"])


def test_the_physical_view_of_the_two_arms_differs():
    built = pair()
    a, b = (arm_record(arm, built.runtime) for arm in built.arms)
    assert not np.allclose(np.asarray(a["physical_J"]), np.asarray(b["physical_J"]))


# --- model response ----------------------------------------------------------

def test_a_logical_model_proposes_the_same_control_on_both_arms(checkpoints):
    built = pair()
    result = model_intervention_response(built, checkpoints["logical"],
                                         tolerance=5e-3, initial_steps=16, max_steps=512)
    assert result["identical_choice"] is True
    assert result["waveform_distance"] == pytest.approx(0.0, abs=1e-12)
    assert result["encoder_variant"] == "logical"
    assert result["embedding_blind_by_construction"] is True


def test_a_physical_model_is_not_blind_by_construction(checkpoints):
    built = pair()
    result = model_intervention_response(built, checkpoints["physical"],
                                         tolerance=5e-3, initial_steps=16, max_steps=512)
    assert result["embedding_blind_by_construction"] is False
    assert result["waveform_distance"] >= 0.0


def test_response_reports_true_simulator_losses_on_each_arm(checkpoints):
    built = pair()
    result = model_intervention_response(built, checkpoints["physical"],
                                         tolerance=5e-3, initial_steps=16, max_steps=512)
    for arm in ("A", "B"):
        assert 0.0 <= result["model_loss"][arm] <= 1.0
    assert result["true_outcome_observed_after_selection"] is True
    assert result["objective_calls"] == 2


def test_excess_loss_is_measured_against_the_pair_best_found(checkpoints):
    built = pair()
    matrix = cross_control_matrix(built, families=("linear", "one_window"), budget=3,
                                  seed=0, tolerance=5e-3, initial_steps=16, max_steps=512)
    result = model_intervention_response(built, checkpoints["physical"], matrix=matrix,
                                         tolerance=5e-3, initial_steps=16, max_steps=512)
    assert result["excess_loss"]["A"] == pytest.approx(
        result["model_loss"]["A"] - matrix["loss_matrix"]["A_on_A"])
    assert result["mean_excess_loss"] == pytest.approx(
        (result["excess_loss"]["A"] + result["excess_loss"]["B"]) / 2)
    assert result["preferred_control_swapped"] == matrix["preferred_control_swapped"]


def test_excess_loss_is_absent_without_a_reference_matrix(checkpoints):
    result = model_intervention_response(pair(), checkpoints["physical"],
                                         tolerance=5e-3, initial_steps=16, max_steps=512)
    assert result["excess_loss"] is None and result["mean_excess_loss"] is None


def test_output_is_json_safe(checkpoints):
    json.dumps(model_intervention_response(pair(), checkpoints["logical"],
                                           tolerance=5e-3, initial_steps=16, max_steps=512),
               allow_nan=False)


# --- sweep -------------------------------------------------------------------

def test_sweep_runs_every_pair_against_every_checkpoint(tmp_path, checkpoints):
    pairs = [pair(), pair("geometry", {"shape": "star"})]
    manifest = sweep_model_interventions(pairs, checkpoints, output=tmp_path / "sweep",
                                         tolerance=5e-3, initial_steps=16, max_steps=512)
    assert manifest["completed"] == len(pairs) * len(checkpoints)
    from annealctrl.sweeps import load_rows
    methods = {row["result"]["method"] for row in load_rows(tmp_path / "sweep")}
    assert methods == {"logical", "physical"}


def test_sweep_resumes(tmp_path, checkpoints):
    kwargs = dict(tolerance=5e-3, initial_steps=16, max_steps=512)
    pairs = [pair()]
    first = sweep_model_interventions(pairs, checkpoints, output=tmp_path / "s", **kwargs)
    again = sweep_model_interventions(pairs, checkpoints, output=tmp_path / "s",
                                      resume=True, **kwargs)
    assert again["completed"] == 0 and again["skipped"] == first["completed"]


# --- aggregation -------------------------------------------------------------

def row(method, parent, excess_a, excess_b, *, swapped=True, blind=False,
        identical=None, status="resolved"):
    return {"pair_id": f"{parent}_k", "parent_id": parent, "method": method,
            "factor": "chain_strength", "scale_arm": "total_compiled_effect",
            "encoder_variant": method, "embedding_blind_by_construction": blind,
            "identical_choice": blind if identical is None else identical,
            "waveform_distance": 0.0 if blind else 0.2,
            "model_loss": {"A": 0.5, "B": 0.5},
            "excess_loss": {"A": excess_a, "B": excess_b},
            "mean_excess_loss": (excess_a + excess_b) / 2,
            "preferred_control_swapped": swapped, "resolution_status": status,
            "objective_calls": 2}


def test_aggregate_contrasts_methods_on_the_same_pairs():
    rows = []
    for index in range(4):
        rows.append(row("logical", f"p{index}", 0.10, 0.12, blind=True))
        rows.append(row("physical", f"p{index}", 0.04, 0.05))
    summary = aggregate_model_interventions(rows)
    assert summary["by_method"]["logical"]["mean_excess_loss"]["mean"] == pytest.approx(0.11)
    assert summary["by_method"]["physical"]["mean_excess_loss"]["mean"] == pytest.approx(0.045)
    assert summary["n_parents"] == 4


def test_aggregate_reports_the_paired_decision_value_against_a_declared_baseline():
    rows = []
    for index in range(5):
        rows.append(row("logical", f"p{index}", 0.10, 0.10, blind=True))
        rows.append(row("physical", f"p{index}", 0.06, 0.06))
    summary = aggregate_model_interventions(rows, baseline="logical", bootstrap_resamples=200)
    contrast = summary["decision_value"]["physical"]
    assert contrast["mean_difference"] == pytest.approx(-0.04)
    assert contrast["baseline"] == "logical"
    assert contrast["favours_method"] is True
    assert contrast["parent_bootstrap_ci"]["unit_of_independence"] == "logical_parent"


def test_aggregate_restricts_the_contrast_to_swap_pairs_when_asked():
    rows = []
    for index in range(4):
        swapped = index < 2
        rows.append(row("logical", f"p{index}", 0.10, 0.10, blind=True, swapped=swapped,
                        status="resolved" if swapped else "censored_numerical"))
        rows.append(row("physical", f"p{index}", 0.02, 0.02, swapped=swapped,
                        status="resolved" if swapped else "censored_numerical"))
    everything = aggregate_model_interventions(rows, baseline="logical")
    swaps_only = aggregate_model_interventions(rows, baseline="logical", swap_pairs_only=True)
    assert everything["n_pairs_per_method"] == 4
    assert swaps_only["n_pairs_per_method"] == 2
    assert swaps_only["restricted_to_swap_pairs"] is True


def test_aggregate_verifies_the_blindness_claim_rather_than_trusting_it():
    rows = [row("logical", "p0", 0.1, 0.1, blind=True, identical=False)]
    summary = aggregate_model_interventions(rows)
    assert summary["blindness_violations"] == 1
    assert "logical" in summary["blindness_violation_methods"]


def test_aggregate_refuses_methods_evaluated_on_different_pair_sets():
    rows = [row("logical", "p0", 0.1, 0.1, blind=True), row("physical", "p1", 0.1, 0.1)]
    with pytest.raises(ValueError, match="same pairs"):
        aggregate_model_interventions(rows, baseline="logical")


def test_aggregate_refuses_an_unknown_baseline():
    rows = [row("logical", "p0", 0.1, 0.1, blind=True), row("physical", "p0", 0.1, 0.1)]
    with pytest.raises(ValueError, match="baseline"):
        aggregate_model_interventions(rows, baseline="nonexistent")


def test_aggregate_of_an_empty_row_set_is_refused():
    with pytest.raises(ValueError, match="nonempty"):
        aggregate_model_interventions([])


# --- leakage between intervention pairs and training data --------------------

def test_disjoint_logical_instances_is_verified_not_assumed(tmp_path):
    from annealctrl.representation import logical_overlap

    built = [pair(), pair("geometry", {"shape": "star"})]
    # A record set built from the pairs themselves must overlap completely.
    records = [{"logical_h": p.arms[0].compiled.logical.h,
                "logical_edges": p.arms[0].compiled.logical.edges,
                "logical_J": p.arms[0].compiled.logical.J} for p in built]
    same = logical_overlap(built, records)
    assert same["n_pair_instances"] == 1        # both pairs share one parent
    assert same["n_overlapping"] == 1
    assert same["disjoint"] is False


def test_unrelated_records_are_reported_disjoint():
    from annealctrl.representation import logical_overlap
    import numpy as np

    built = [pair()]
    other = [{"logical_h": np.array([0.9, -0.8, 0.7]),
              "logical_edges": np.array([[0, 1], [1, 2], [0, 2]]),
              "logical_J": np.array([0.11, -0.22, 0.33])}]
    result = logical_overlap(built, other)
    assert result["n_overlapping"] == 0
    assert result["disjoint"] is True


def test_logical_overlap_refuses_an_empty_pair_set():
    from annealctrl.representation import logical_overlap
    with pytest.raises(ValueError, match="nonempty"):
        logical_overlap([], [])
