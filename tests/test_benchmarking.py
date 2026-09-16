"""Selection-before-scoring, physical path fidelity and budget regression tests."""
import copy
import json

import numpy as np
import pytest

from annealctrl.benchmarking import (
    _checkpoint_provenance, _paired, benchmark_record_controls, probability_tts,
    record_physics, score_schedule, validate_candidate_banks,
)
from annealctrl.generation import Embedding, IsingProblem, compile_embedding
from annealctrl.physics import dense_reference_propagate
from annealctrl.schedules import Schedule, pause_schedule
from annealctrl.search import CONTROL_FAMILIES, optimize_control_family


@pytest.fixture
def record():
    logical = IsingProblem(np.array([.3, -.2]), np.array([[0, 1]]), np.array([-.6]))
    embedding = Embedding(np.array([0, 1]), np.array([[0, 1]]))
    compiled = compile_embedding(logical, embedding, 1., np.random.default_rng(0))
    tau = np.linspace(0, 1, 9)
    return dict(record_id=np.array("r0"), parent_id=np.array("p0"), split=np.array("validation"),
                family=np.array("test"), fingerprint=np.array("fingerprint0"),
                logical_h=logical.h, logical_edges=logical.edges, logical_J=logical.J,
                physical_h=compiled.physical.h, physical_edges=compiled.physical.edges,
                physical_J=compiled.physical.J, problem_J=compiled.problem_J,
                chain_J=compiled.chain_J, membership=embedding.membership,
                runtime=np.array(2.), programmed_scale=np.array(compiled.programmed_scale),
                chain_strength=np.array(1.), catalyst_strength=np.array(0.),
                candidate_tau=tau, candidate_schedules=np.stack([tau, tau**2]),
                candidate_ids=np.array(["linear", "quadratic"]), candidate_losses=np.array([.7, .6]))


@pytest.mark.parametrize("family", CONTROL_FAMILIES)
def test_family_budgets_incumbents_and_reproducibility(family):
    seen = []
    def loss(schedule):
        schedule.validate_slope(runtime=2., max_slope=2.)
        seen.append(schedule)
        return float((schedule(.5) - .25)**2)
    result = optimize_control_family(loss, family, budget=9, runtime=2., max_slope=2., seed=4)
    assert result.n_evaluations == (1 if family == "linear" else 9)
    assert len(seen) == result.n_evaluations
    assert result.best.loss <= result.records[0].loss
    repeat = optimize_control_family(loss, family, budget=9, runtime=2., max_slope=2., seed=4)
    assert [x.loss for x in result.records] == [x.loss for x in repeat.records]
    if family == "pause":
        assert np.any(np.diff(result.records[1].candidate.schedule.s_knots) == 0)


@pytest.mark.parametrize("family", CONTROL_FAMILIES)
def test_no_implicit_test_search(family):
    with pytest.raises(ValueError, match="test"):
        optimize_control_family(lambda s: 0., family, split="test")


@pytest.mark.parametrize("bad", [0, -1, True, 2.5])
def test_invalid_objective_budgets(bad):
    with pytest.raises(ValueError):
        optimize_control_family(lambda s: 0., "eight_bin", budget=bad)


def test_bank_identity_uses_actual_waveform_and_ids(record):
    reference = validate_candidate_banks([record, copy.deepcopy(record)])
    assert reference["linear_index"] == 0
    changed = copy.deepcopy(record)
    changed["candidate_schedules"][1, 4] += .01
    with pytest.raises(ValueError, match="banks differ"):
        validate_candidate_banks([record, changed])
    changed = copy.deepcopy(record)
    changed["candidate_ids"] = changed["candidate_ids"][::-1]
    with pytest.raises(ValueError, match="banks differ"):
        validate_candidate_banks([record, changed])


def test_linear_not_assumed_index_zero(record):
    swapped = copy.deepcopy(record)
    swapped["candidate_schedules"] = swapped["candidate_schedules"][::-1]
    swapped["candidate_ids"] = swapped["candidate_ids"][::-1]
    assert validate_candidate_banks([swapped])["linear_index"] == 1


def test_exact_switch_scoring_matches_dense_reference(record):
    schedule = pause_schedule(.371, .173, runtime=2., max_slope=2.)
    score = score_schedule(record, schedule, initial_steps=16, max_steps=8192, tolerance=1e-5)
    terms, path, observables = record_physics(record)
    reference = dense_reference_propagate(terms, schedule, 2., path=path)
    loss = 1 - (np.abs(reference.state)**2 @ observables["success"])
    assert abs(score["loss"] - loss) < 3e-5
    assert score["switching_knots_aligned"]
    assert score["total_integrator_steps"] >= score["accepted_steps"]
    assert score["norm_error"] < 1e-10
    assert score["any_chain_break"] == 0
    json.dumps(score, allow_nan=False)


def test_whole_path_scale_and_catalyst_preserved(record):
    record.update(energy_scale=np.array(1.7), catalyst_strength=np.array(.4),
                  xx_edges=np.array([[0, 1]]), xx_weights=np.array([.7]))
    terms, path, observables = record_physics(record)
    assert path.energy_scale == 1.7 and path.catalyst_strength == .4
    score = score_schedule(record, Schedule.linear(), initial_steps=32, max_steps=8192, tolerance=1e-5)
    ref = dense_reference_propagate(terms, Schedule.linear(), 2., path=path)
    assert abs(score["success"] - np.abs(ref.state)**2 @ observables["success"]) < 3e-5
    del record["xx_weights"]
    with pytest.raises(ValueError, match="nonzero catalyst"):
        record_physics(record)


@pytest.mark.parametrize("bad", [0., -1., float("nan"), float("inf")])
def test_invalid_tolerance_before_scoring(record, bad):
    with pytest.raises(ValueError, match="tolerance"):
        score_schedule(record, Schedule.linear(), tolerance=bad)


def test_failed_convergence_is_not_silently_accepted(record):
    with pytest.raises(ArithmeticError, match="convergence"):
        score_schedule(record, Schedule.linear(), initial_steps=1, max_steps=2, tolerance=1e-15)


def test_single_parent_ci_is_explicitly_unavailable():
    result = _paired([.2, .3], [.5, .5], ["p", "p"], 0, 100)
    assert result["ci_low"] is None and result["n_parents"] == 1
    assert result["mean_difference"] == pytest.approx(-.25)


def test_parent_means_not_variant_pseudoreplication():
    result = _paired([0., 0., 0., 1.], [0., 0., 0., 0.], ["a", "a", "a", "b"], 0, 500)
    assert result["mean_difference"] == .5
    assert result["n_parents"] == 2


def test_zero_success_is_null_censored_not_smoothed():
    result = probability_tts(0., 2.)
    assert result["censored"] and result["nominal_time"] is None
    assert probability_tts(1., 2.)["nominal_time"] == 2.
    json.dumps(result, allow_nan=False)


def test_provenance_rejects_exposure_and_changed_data(record):
    train, validation, test = [copy.deepcopy(record) for _ in range(3)]
    for r, parent, split, fingerprint in zip((train, validation, test), ("a", "b", "c"),
                                            ("train", "validation", "test"), ("f1", "f2", "f3")):
        r.update(parent_id=np.array(parent), split=np.array(split), fingerprint=np.array(fingerprint))
    checkpoint = {"train_parent_ids": ["a"], "validation_parent_ids": ["b"],
                  "data_fingerprints": ["f1", "f2"]}
    assert _checkpoint_provenance(checkpoint, [train], [validation], [test])["record_fingerprints_verified"]
    checkpoint["data_fingerprints"] = ["wrong"]
    with pytest.raises(ValueError, match="fingerprint"):
        _checkpoint_provenance(checkpoint, [train], [validation], [test])
    checkpoint["train_parent_ids"] = ["c"]
    with pytest.raises(ValueError, match="test parent"):
        _checkpoint_provenance(checkpoint, [train], [validation], [test])


def test_end_to_end_family_benchmark(record):
    result = benchmark_record_controls(record, budget_per_family=2, initial_steps=16, max_steps=2048)
    assert result["total_objective_calls"] == 9
    assert all(x["exact_switching_waveforms"] for x in result["families"].values())
    for family, output in result["families"].items():
        assert output["n_evaluations"] == (1 if family == "linear" else 2)
        assert output["best_loss"] <= output["records"][0]["loss"]
    json.dumps(result, allow_nan=False)
    record["split"] = np.array("test")
    with pytest.raises(ValueError, match="explicit"):
        benchmark_record_controls(record)


def test_checkpoint_evaluation_selection_independent_of_test_labels(record, tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    from annealctrl.benchmarking import evaluate_checkpoint
    from annealctrl.learning import fit_records
    from annealctrl.models import AnnealController
    import annealctrl.pipeline as pipeline

    split_records = {}
    for split in ("train", "validation", "test"):
        item = copy.deepcopy(record)
        item.update(record_id=np.array(f"{split}_r"), parent_id=np.array(f"{split}_p"),
                    split=np.array(split), fingerprint=np.array(f"{split}_fingerprint"))
        split_records[split] = [item]
    checkpoint = tmp_path / "model.pt"
    fit_records(split_records["train"], split_records["validation"],
                model=AnnealController(width=8), epochs=1, checkpoint=checkpoint)
    monkeypatch.setattr(pipeline, "load_records", lambda root, split: split_records[split])
    result = evaluate_checkpoint("unused", checkpoint, initial_steps=16, max_steps=2048, n_resamples=20)
    assert result["provenance"]["training_validation_content_verified"]
    assert result["n_test_parents"] == 1
    assert result["paired_loss_difference_vs_linear"]["ci_low"] is None
    assert result["records"][0]["record_id"] == "test_r"
    assert result["direct_policy"]["records"][0]["record_id"] == "test_r"
    json.dumps(result, allow_nan=False)
    split_records["test"][0]["candidate_losses"] = np.array([.99, .01])
    changed = evaluate_checkpoint("unused", checkpoint, initial_steps=16, max_steps=2048, n_resamples=20)
    assert changed["records"][0]["selected_index"] == result["records"][0]["selected_index"]
    assert changed["direct_policy"]["records"][0]["waveform"] == result["direct_policy"]["records"][0]["waveform"]
    assert changed["direct_policy"]["mean_loss"] == result["direct_policy"]["mean_loss"]
    # Updating a train label while leaving nominal fingerprints untouched fails.
    split_records["train"][0]["candidate_losses"][0] += .01
    with pytest.raises(ValueError, match="content mismatch"):
        evaluate_checkpoint("unused", checkpoint, direct=False)
