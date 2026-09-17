"""One DAgger round: label the policy's own proposals and retrain on them.

The critic is trained on a candidate bank and deployed on waveforms its policy
emits. Those manifolds do not coincide - 92% of policy waveforms sit further from
the bank than a typical bank waveform sits from its own nearest neighbour - and
random extra candidates from the same family were measured not to close it. What
closes it is labelling the model's ACTUAL proposals and adding those.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from annealctrl.dagger import augment_records, collect_labelled_proposals


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    from annealctrl.learning import fit_records
    from annealctrl.pipeline import generate_dataset, load_records

    root = tmp_path_factory.mktemp("dagger")
    config = {"seed": 17, "parents": 6, "families": ["spin_glass"], "logical_qubits": 3,
              "chain_lengths": [1, 1, 1], "variants": [{"shape": "path", "ports": 1}],
              "chain_strengths": [1.5], "runtimes": [2.0], "candidates": 4,
              "spectral_points": 3, "steps": 16, "max_steps": 1024,
              "label_state_tolerance": 0.005, "max_physical_qubits": 3,
              "teacher": {"mode": "none"}}
    generate_dataset(config, root / "data")
    fit_records(load_records(root / "data", "train"), load_records(root / "data", "validation"),
                model_config={"encoder_variant": "summary", "width": 8, "proposals": 3},
                epochs=1, seed=0, checkpoint=root / "m.pt", response_weight=0.0)
    return root


def test_collected_proposals_carry_true_losses_and_their_waveforms(trained):
    result = collect_labelled_proposals(trained / "data", trained / "m.pt", split="train",
                                        tolerance=5e-3, initial_steps=16, max_steps=512)
    assert result["n_records"] >= 1
    entry = result["records"][next(iter(result["records"]))]
    assert entry["waveforms"].shape[1] == 9
    assert entry["losses"].shape[0] == entry["waveforms"].shape[0]
    assert np.all((entry["losses"] >= 0) & (entry["losses"] <= 1))
    assert result["propagations"] == sum(
        len(v["losses"]) for v in result["records"].values())


def test_augmented_records_keep_every_original_candidate(trained):
    from annealctrl.pipeline import load_records

    records = load_records(trained / "data", "train")
    collected = collect_labelled_proposals(trained / "data", trained / "m.pt", split="train",
                                           tolerance=5e-3, initial_steps=16, max_steps=512)
    augmented = augment_records(records, collected)
    for before, after in zip(records, augmented):
        n = len(before["candidate_losses"])
        np.testing.assert_allclose(after["candidate_losses"][:n], before["candidate_losses"])
        np.testing.assert_allclose(after["candidate_schedules"][:n], before["candidate_schedules"])
        assert len(after["candidate_losses"]) > n


def test_augmented_banks_stay_index_aligned(trained):
    from annealctrl.pipeline import load_records

    records = load_records(trained / "data", "train")
    collected = collect_labelled_proposals(trained / "data", trained / "m.pt", split="train",
                                           tolerance=5e-3, initial_steps=16, max_steps=512)
    augmented = augment_records(records, collected)
    sizes = {len(r["candidate_losses"]) for r in augmented}
    assert len(sizes) == 1, "every record must end with the same bank size or training cannot batch"
    for record in augmented:
        assert len(record["candidate_ids"]) == len(record["candidate_losses"])
        assert record["candidate_schedules"].shape[0] == len(record["candidate_losses"])


def test_added_candidates_are_marked_so_provenance_survives(trained):
    from annealctrl.pipeline import load_records

    records = load_records(trained / "data", "train")
    collected = collect_labelled_proposals(trained / "data", trained / "m.pt", split="train",
                                           tolerance=5e-3, initial_steps=16, max_steps=512)
    augmented = augment_records(records, collected)
    ids = [str(x) for x in augmented[0]["candidate_ids"]]
    assert any(i.startswith("dagger_") for i in ids)
    assert sum(i.startswith("dagger_") for i in ids) == collected["proposals_per_record"]


def test_a_record_without_collected_proposals_is_refused(trained):
    from annealctrl.pipeline import load_records

    records = load_records(trained / "data", "train")
    collected = collect_labelled_proposals(trained / "data", trained / "m.pt", split="train",
                                           tolerance=5e-3, initial_steps=16, max_steps=512)
    collected["records"].pop(next(iter(collected["records"])))
    with pytest.raises(ValueError, match="no collected proposals"):
        augment_records(records, collected)


def test_augmentation_closes_the_manifold_gap_it_was_built_for(trained):
    """The measurement that justifies the whole round."""
    from annealctrl.pipeline import load_records

    records = load_records(trained / "data", "train")
    collected = collect_labelled_proposals(trained / "data", trained / "m.pt", split="train",
                                           tolerance=5e-3, initial_steps=16, max_steps=512)
    augmented = augment_records(records, collected)

    record_id = str(np.asarray(records[0]["record_id"]).item())
    policy = collected["records"][record_id]["waveforms"]
    before = np.abs(policy[:, None, :] - records[0]["candidate_schedules"][None, :, :]
                    ).max(axis=2).min(axis=1)
    after = np.abs(policy[:, None, :] - augmented[0]["candidate_schedules"][None, :, :]
                   ).max(axis=2).min(axis=1)
    assert after.max() < 1e-12, "the policy's own proposals are now IN the bank"
    assert before.min() > 0, "and they were not there before"
