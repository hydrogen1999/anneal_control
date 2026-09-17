"""Acquisition evaluation keeps deployment banks and checkpoint provenance honest."""
import copy
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from annealctrl.acquisition_evaluation import evaluate_acquisition_arm
from annealctrl.benchmarking import evaluate_checkpoint, validate_training_augmentation


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    from annealctrl.learning import fit_records
    from annealctrl.pipeline import _payload_fingerprint, generate_dataset, load_records

    torch.set_num_threads(1)
    root = tmp_path_factory.mktemp("acquisition_evaluation")
    config = {"seed": 17, "parents": 6, "families": ["spin_glass"], "logical_qubits": 3,
              "chain_lengths": [1, 1, 1], "variants": [{"shape": "path", "ports": 1}],
              "chain_strengths": [1.5], "runtimes": [.5], "candidates": 4,
              "spectral_points": 3, "steps": 8, "max_steps": 256,
              "label_state_tolerance": .005, "max_physical_qubits": 3,
              "teacher": {"mode": "none"}}
    generate_dataset(config, root / "data")
    original, validation = (load_records(root / "data", split) for split in ("train", "validation"))
    augmented = copy.deepcopy(original)
    for record in augmented:
        for key, value in list(record.items()):
            if key.startswith("candidate_") and key != "candidate_tau":
                record[key] = np.concatenate((value, value[1:2]), axis=0)
        record["candidate_ids"] = np.asarray([*record["candidate_ids"][:-1], "extra_control"])
        record["payload_fingerprint"] = np.array(_payload_fingerprint(record))
    options = dict(model_config={"encoder_variant": "summary", "width": 8, "proposals": 2},
                   epochs=1, seed=0, response_weight=0.)
    fit_records(original, validation, checkpoint=root / "frozen.pt", **options)
    fit_records(augmented, validation, checkpoint=root / "updated.pt", **options)
    return root, original, augmented


def test_augmented_provenance_uses_actual_data_but_original_deployment_bank(study):
    root, original, augmented = study
    report = evaluate_checkpoint(root / "data", root / "updated.pt", direct=False,
                                 training_records=augmented, n_resamples=20)
    assert report["common_bank"]["candidate_count"] == len(original[0]["candidate_losses"])
    assert report["provenance"]["explicit_training_records_verified"]
    assert report["provenance"]["training_validation_content_verified"]
    with pytest.raises(ValueError, match="content mismatch"):
        evaluate_checkpoint(root / "data", root / "updated.pt", direct=False)


@pytest.mark.parametrize("field", ["physical_h", "candidate_losses", "candidate_schedules", "parent_id", "candidate_tau"])
def test_augmentation_cannot_rewrite_original_problem_or_labels(study, field):
    _, original, augmented = study
    changed = copy.deepcopy(augmented)
    # Remove payload digests so the field-specific gate itself is exercised.
    old = copy.deepcopy(original)
    for records in (old, changed):
        for record in records:
            record.pop("payload_fingerprint")
    if field == "parent_id":
        changed[0][field] = np.array("other_parent")
    else:
        changed[0][field].flat[0] += .001
    with pytest.raises(ValueError, match="changed original"):
        validate_training_augmentation(old, changed)


def test_augmentation_rejects_missing_records_and_misaligned_diagnostics(study):
    _, original, augmented = study
    with pytest.raises(ValueError, match="record IDs"):
        validate_training_augmentation(original, augmented[:-1])
    changed = copy.deepcopy(augmented)
    for record in changed:
        record.pop("payload_fingerprint")
    old = copy.deepcopy(original)
    for record in old:
        record.pop("payload_fingerprint")
    changed[0]["candidate_state_error"] = changed[0]["candidate_state_error"][:-1]
    with pytest.raises(ValueError, match="misaligned"):
        validate_training_augmentation(old, changed)


def test_frozen_proposals_are_scored_after_both_critics_select(study):
    root, _, augmented = study
    report = evaluate_acquisition_arm(
        root / "data", root / "updated.pt", training_records=augmented,
        frozen_checkpoint=root / "frozen.pt", seed=0, initial_steps=8, max_steps=256,
        tolerance=.005, n_resamples=20, latency_warmup=1, latency_repeats=2)
    rows = report["proposal_diagnostics"]["rows"]
    fixed = report["frozen_proposal_diagnostics"]["rows"]
    assert report["costs"]["offline_objective_calls"] == 4 * len(rows)
    assert report["costs"]["offline_propagation_calls"] >= 2 * report["costs"]["offline_objective_calls"]
    for row, fixed_row in zip(rows, fixed):
        selected = int(np.argmin(row["predicted_losses"]))
        assert selected == row["selected_proposal"]
        assert row["selected_proposal_loss"] == row["outcomes"][selected]["loss"]
        assert row["ranking_regret"] + row["generation_gap_vs_bank"] == pytest.approx(
            row["selected_proposal_loss"] - row["bank_loss"])
        assert fixed_row["new_critic"]["best_proposal_loss"] == fixed_row["old_critic"]["best_proposal_loss"]
        assert fixed_row["new_selected_proposal"] == int(np.argmin(fixed_row["new_predicted_losses"]))
    assert report["deployment_latency"]["online_simulator_calls"] == {"bank": 0, "direct": 0}
    assert report["deployment_latency"]["preprocessing_included"]
    assert report["evaluation"]["direct_policy"]["records"]
    json.dumps(report, allow_nan=False)


def test_frozen_diagnostic_rejects_checkpoint_trained_on_augmented_labels(study):
    root, _, augmented = study
    with pytest.raises(ValueError, match="content mismatch"):
        evaluate_acquisition_arm(root / "data", root / "updated.pt", training_records=augmented,
                                 frozen_checkpoint=root / "updated.pt", n_resamples=20)


def test_failed_proposal_scoring_does_not_silently_drop_test_records(study, monkeypatch):
    import annealctrl.acquisition_evaluation as module

    root, _, augmented = study
    def reject(*args, **kwargs):
        raise ValueError("invalid numerical configuration")
    monkeypatch.setattr(module, "score_schedule", reject)
    with pytest.raises(ValueError, match="invalid numerical configuration"):
        evaluate_acquisition_arm(root / "data", root / "updated.pt", training_records=augmented,
                                 frozen_checkpoint=root / "frozen.pt", n_resamples=20)
