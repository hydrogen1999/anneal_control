"""Acquisition integrity tests; predictive benefit requires held-out experiments."""
import copy
import json
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from annealctrl.dagger import (AcquisitionError, augment_records, collect_bank_extension,
                              collect_decoder_random, collect_labelled_proposals)


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
    assert len(sizes) == 1, "the acquisition experiment requires equal added label budgets"
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


def test_augmentation_preserves_exact_acquired_waveforms(trained):
    """Bank inclusion is an integrity property, not evidence of generalization."""
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


# --- the control arm ---------------------------------------------------------

def test_bank_extension_refuses_the_test_split():
    from annealctrl.dagger import collect_bank_extension

    with pytest.raises(ValueError, match="leakage"):
        collect_bank_extension("unused", split="test", bank_seed=0, bank_size=64)


def test_bank_extension_refuses_a_nonpositive_count():
    from annealctrl.dagger import collect_bank_extension

    with pytest.raises(ValueError, match="at least one"):
        collect_bank_extension("unused", split="train", n_extra=0, bank_seed=0, bank_size=64)


def test_the_extension_continues_the_same_sobol_sequence():
    """The control is only matched if the first entries are byte-identical."""
    import numpy as np

    from annealctrl.search import shared_candidate_bank

    tau = np.linspace(0.0, 1.0, 9)
    base = shared_candidate_bank(n=64, n_segments=8, seed=20260916, runtime=1.0, max_slope=4.0)
    extended = shared_candidate_bank(n=67, n_segments=8, seed=20260916, runtime=1.0, max_slope=4.0)
    a = np.stack([c.schedule(tau) for c in base])
    b = np.stack([c.schedule(tau) for c in extended])
    assert b.shape[0] == 67
    assert np.abs(a - b[:64]).max() < 1e-12
    # The three new entries are genuinely new waveforms, not repeats.
    for extra in b[64:]:
        assert np.abs(a - extra).max(axis=1).min() > 1e-6


@pytest.mark.parametrize("split", ["validation", "test", "unknown", None])
@pytest.mark.parametrize("collector", ["policy", "bank", "random"])
def test_every_acquisition_arm_is_strictly_train_only(split, collector):
    with pytest.raises(ValueError, match="train-only"):
        if collector == "policy":
            collect_labelled_proposals("unused", "unused.pt", split=split)
        elif collector == "bank":
            collect_bank_extension("unused", split=split, bank_seed=0, bank_size=4)
        else:
            collect_decoder_random("unused", split=split, seed=0)


def test_real_diagnostics_and_uncertainty_survive_json_augmentation(trained):
    from annealctrl.pipeline import _payload_fingerprint, jsonable, load_records

    records = load_records(trained / "data", "train")
    collected = collect_labelled_proposals(trained / "data", trained / "m.pt",
                                          tolerance=5e-3, initial_steps=16, max_steps=512)
    restored = json.loads(json.dumps(collected, default=jsonable))
    augmented = augment_records(records, restored)
    mapping = {"candidate_decoded_energy": "decoded_energy", "candidate_any_chain_break": "any_chain_break",
               "candidate_chain_break_fraction": "chain_break_fraction", "candidate_norm_error": "norm_error",
               "candidate_state_error": "state_error_diagnostic", "candidate_steps": "accepted_steps",
               "candidate_seconds": "offline_scoring_seconds", "candidate_loss_uncertainty": "loss_ambiguity_indicator"}
    for before, after in zip(records, augmented):
        n = len(before["candidate_losses"])
        outcomes = collected["records"][str(before["record_id"].item())]["outcomes"]
        for column, metric in mapping.items():
            np.testing.assert_array_equal(after[column][n:], [row[metric] for row in outcomes])
        delta = before["candidate_state_error"]
        np.testing.assert_array_equal(after["candidate_loss_uncertainty"][:n], 2 * delta + delta**2)
        assert after["payload_fingerprint"].item() == _payload_fingerprint(after)
    outcomes = [o for entry in collected["records"].values() for o in entry["outcomes"]]
    assert collected["objective_calls"] == collected["requested_objective_calls"] == len(outcomes)
    assert collected["propagation_calls"] == sum(o["propagation_calls"] for o in outcomes)
    assert collected["total_integrator_steps"] == sum(o["total_integrator_steps"] for o in outcomes)
    assert collected["cost_counts_complete"]
    assert not collected["uncertainty_is_certificate"]


def test_decoder_random_reproduces_the_policy_decoder_and_shared_design(trained):
    from annealctrl.models import monotone_samples

    settings = dict(n_extra=3, seed=29, logit_std=0.8, tolerance=5e-3, initial_steps=16, max_steps=512)
    first = collect_decoder_random(trained / "data", **settings)
    second = collect_decoder_random(trained / "data", **settings)
    expected_logits = (np.random.default_rng(29).standard_normal((3, 8)) * 0.8).astype(np.float32)
    expected = monotone_samples(torch.from_numpy(expected_logits)).double().numpy()
    for identifier, entry in first["records"].items():
        np.testing.assert_array_equal(entry["waveforms"], expected)
        np.testing.assert_array_equal(entry["waveforms"], second["records"][identifier]["waveforms"])
        assert entry["candidate_ids"] == second["records"][identifier]["candidate_ids"]
    assert first["acquisition_id"] == second["acquisition_id"]
    assert first["provenance"]["shared_across_records"]
    third = collect_decoder_random(trained / "data", round_id="round2", **settings)
    assert first["acquisition_id"] != third["acquisition_id"]


def test_augmentation_rejects_modified_base_and_modified_acquisition(trained):
    from annealctrl.pipeline import load_records

    records = load_records(trained / "data", "train")
    collected = collect_decoder_random(trained / "data", seed=4, tolerance=5e-3, initial_steps=16, max_steps=512)
    wrong_records = copy.deepcopy(records)
    wrong_records[0]["physical_h"][0] += 0.01
    with pytest.raises(ValueError, match="base record digest"):
        augment_records(wrong_records, collected)
    bad = copy.deepcopy(collected)
    identifier = next(iter(bad["records"]))
    bad["records"][identifier]["losses"][0] += 0.01
    with pytest.raises(ValueError, match="artifact digest"):
        augment_records(records, bad)
    with pytest.raises(ValueError, match="base record digest"):
        augment_records(augment_records(records, collected), collected)
    wrong_records[0]["split"] = np.array("validation")
    with pytest.raises(ValueError, match="train-only"):
        augment_records(wrong_records, collected)


def test_numerical_failure_fails_matched_arm_without_cherry_picking(trained, monkeypatch):
    from annealctrl import benchmarking

    original = benchmarking.score_schedule
    completed = []

    def score(record, schedule, **kwargs):
        if completed:
            raise ArithmeticError("deliberate convergence failure")
        outcome = original(record, schedule, **kwargs)
        completed.append(outcome)
        return outcome

    monkeypatch.setattr(benchmarking, "score_schedule", score)
    with pytest.raises(AcquisitionError, match="candidate 1") as error:
        collect_decoder_random(trained / "data", n_extra=3, seed=9,
                               tolerance=5e-3, initial_steps=16, max_steps=512)
    audit = error.value.audit
    assert audit["status"] == "failed"
    assert audit["objective_calls"] == 2
    assert audit["successful_objective_calls"] == 1
    assert audit["propagation_calls"] == completed[0]["propagation_calls"]
    assert not audit["cost_counts_complete"]
    assert len(audit["failures"]) == 1


def test_configuration_error_is_not_mislabelled_as_infeasible_proposal(trained, monkeypatch):
    from annealctrl import benchmarking

    def invalid(*args, **kwargs):
        raise ValueError("invalid simulator configuration")

    monkeypatch.setattr(benchmarking, "score_schedule", invalid)
    with pytest.raises(ValueError, match="invalid simulator configuration"):
        collect_decoder_random(trained / "data", seed=0,
                               tolerance=5e-3, initial_steps=16, max_steps=512)


def test_bank_extension_enforces_complete_base_size(trained):
    with pytest.raises(ValueError, match="complete original bank"):
        collect_bank_extension(trained / "data", bank_seed=17, bank_size=3,
                               tolerance=5e-3, initial_steps=16, max_steps=512)
