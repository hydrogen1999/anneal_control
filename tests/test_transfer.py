"""Does a selector trained on one distribution work on another?

Every learned number in this project is in-distribution: trained and evaluated on
splits of the same generator. That is the weakest kind of ML evidence, and the
project had no cross-distribution measurement at all. This one evaluates a
checkpoint on records it was never trained on, and it inverts the usual
provenance guard: instead of requiring the checkpoint's parents to MATCH the
dataset, it requires them to be DISJOINT from it. A "transfer" result measured on
parents the model trained on is leakage wearing a different name.
"""
import numpy as np
import pytest

from annealctrl.transfer import evaluate_transfer, transfer_report


def fake_checkpoint(tmp_path, train_parents, validation_parents, *, variant="summary"):
    import torch

    from annealctrl.models import AnnealController
    model = AnnealController(encoder_variant=variant, width=64)
    path = tmp_path / "ckpt.pt"
    torch.save({"model_state": model.state_dict(), "config": model.config,
                "normalizer": None,
                "train_parent_ids": list(train_parents),
                "validation_parent_ids": list(validation_parents)}, path)
    return path


# --- the guard that makes this a transfer test -------------------------------

def test_refuses_when_target_parents_appear_in_checkpoint_training(tmp_path):
    path = fake_checkpoint(tmp_path, ["p0", "p1"], ["p2"])
    with pytest.raises(ValueError, match="disjoint"):
        evaluate_transfer(path, records=[{"parent_id": "p0", "record_id": "r0"}])


def test_refuses_when_target_parents_appear_in_checkpoint_validation(tmp_path):
    path = fake_checkpoint(tmp_path, ["p0"], ["p9"])
    with pytest.raises(ValueError, match="disjoint"):
        evaluate_transfer(path, records=[{"parent_id": "p9", "record_id": "r0"}])


def test_refuses_a_checkpoint_without_parent_provenance(tmp_path):
    import torch

    from annealctrl.models import AnnealController
    model = AnnealController(encoder_variant="summary", width=64)
    path = tmp_path / "bare.pt"
    torch.save({"model_state": model.state_dict(), "config": model.config}, path)
    with pytest.raises(ValueError, match="provenance"):
        evaluate_transfer(path, records=[{"parent_id": "zz", "record_id": "r0"}])


# --- the aggregate -----------------------------------------------------------

def rows(n=6, *, selected=0.55, linear=0.60, best=0.50):
    return [{"record_id": f"r{i}", "parent_id": f"p{i}", "selected_loss": selected,
             "linear_loss": linear, "bank_best_loss": best, "selected_index": 3,
             "physical_n": 12, "runtime": 4.0} for i in range(n)]


def test_report_pairs_against_linear_on_the_parent():
    result = transfer_report(rows(), bootstrap_resamples=500)
    assert result["n_parents"] == 6
    assert result["mean_selected_loss"] == pytest.approx(0.55)
    assert result["vs_linear"]["mean_difference"] == pytest.approx(-0.05)
    assert result["unit_of_independence"] == "logical_parent"


def test_report_states_the_bank_regret_it_leaves_on_the_table():
    result = transfer_report(rows(selected=0.55, best=0.50), bootstrap_resamples=500)
    assert result["mean_bank_regret"] == pytest.approx(0.05)


def test_report_flags_a_selector_that_loses_to_linear():
    result = transfer_report(rows(selected=0.62, linear=0.60), bootstrap_resamples=2000)
    assert result["beats_linear_fraction"] == 0.0
    assert result["verdict"] == "worse_than_linear"


def test_report_flags_a_selector_that_transfers():
    result = transfer_report(rows(selected=0.52, linear=0.60), bootstrap_resamples=2000)
    assert result["beats_linear_fraction"] == 1.0
    assert result["verdict"] == "beats_linear"


def test_report_refuses_an_empty_row_set():
    with pytest.raises(ValueError, match="nonempty"):
        transfer_report([], bootstrap_resamples=100)


# --- names collide across datasets; content does not -------------------------

def test_parent_id_collision_across_datasets_does_not_block_a_real_transfer(tmp_path):
    """Two independent datasets both number parents from parent_0000.

    The first run of this evaluation refused every checkpoint on seven such
    collisions while the content fingerprints overlapped by zero. Names are
    dataset-local; fingerprints are not.
    """
    import torch

    from annealctrl.models import AnnealController
    model = AnnealController(encoder_variant="summary", width=64)
    path = tmp_path / "ckpt.pt"
    torch.save({"model_state": model.state_dict(), "config": model.config, "normalizer": None,
                "train_parent_ids": ["parent_0000"], "validation_parent_ids": ["parent_0001"],
                "data_fingerprints": ["aaa", "bbb"]}, path)
    # Same parent name, different content: a genuinely different problem.
    records = [{"parent_id": "parent_0000", "record_id": "r0", "fingerprint": "zzz"}]
    # The guard must not be what stops this. It gets past disjointness and fails
    # later on the fake checkpoint's missing fields, which is the point.
    with pytest.raises(Exception) as caught:
        evaluate_transfer(path, records=records)
    assert "disjoint" not in str(caught.value), caught.value


def test_fingerprint_overlap_still_blocks(tmp_path):
    import torch

    from annealctrl.models import AnnealController
    model = AnnealController(encoder_variant="summary", width=64)
    path = tmp_path / "ckpt.pt"
    torch.save({"model_state": model.state_dict(), "config": model.config, "normalizer": None,
                "train_parent_ids": ["p0"], "validation_parent_ids": ["p1"],
                "data_fingerprints": ["shared", "other"]}, path)
    with pytest.raises(ValueError, match="disjoint"):
        evaluate_transfer(path, records=[{"parent_id": "zz", "record_id": "r0",
                                          "fingerprint": "shared"}])


def test_the_basis_of_the_disjointness_claim_is_recorded(tmp_path):
    """A claim resting on names is weaker than one resting on content; say which."""
    from annealctrl.transfer import _seen_identifiers

    seen, basis = _seen_identifiers({"data_fingerprints": ["a", "b"]})
    assert basis == "fingerprint" and seen == {"a", "b"}
    seen, basis = _seen_identifiers({"train_parent_ids": ["p0"], "validation_parent_ids": ["p1"]})
    assert basis == "parent_id" and seen == {"p0", "p1"}
