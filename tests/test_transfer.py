"""Transfer requires invariant logical identity and frozen source selection."""
import copy
import hashlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from annealctrl.transfer import (evaluate_transfer, transfer_report, validate_transfer_provenance,
                                 record_logical_fingerprint, source_global_schedule, _seen_identifiers)


def example(index=0, split="train"):
    from annealctrl.generation import Embedding, IsingProblem, compile_embedding
    logical = IsingProblem(np.array([.1 + .07 * index, -.2]), np.array([[0, 1]]), np.array([-.6]))
    embedding = Embedding(np.array([0, 1]), np.array([[0, 1]]))
    compiled = compile_embedding(logical, embedding, 1., np.random.default_rng(0))
    tau = np.linspace(0, 1, 9)
    record = dict(record_id=np.array(f"r{index}"), parent_id=np.array(f"p{index}"), split=np.array(split),
                  family=np.array("spin_glass"), fingerprint=np.array(hashlib.sha256(str(index).encode()).hexdigest()),
                  logical_h=logical.h, logical_edges=logical.edges, logical_J=logical.J,
                  physical_h=compiled.physical.h, physical_edges=compiled.physical.edges,
                  physical_J=compiled.physical.J, problem_J=compiled.problem_J,
                  chain_J=compiled.chain_J, membership=embedding.membership,
                  runtime=np.array(2.), programmed_scale=np.array(compiled.programmed_scale),
                  chain_strength=np.array(1.), catalyst_strength=np.array(0.),
                  candidate_tau=tau, candidate_schedules=np.stack([tau, tau**2]),
                  candidate_ids=np.array(["linear", "quadratic"]), candidate_losses=np.array([.7, .6]))
    record["logical_fingerprint"] = np.array(record_logical_fingerprint(record))
    return record


def checkpoint(tmp_path, legacy=False):
    from annealctrl.learning import _data_provenance, FeatureNormalizer
    from annealctrl.models import AnnealController, graph_from_record
    train, validation = [example(0)], [example(1, "validation")]
    provenance = _data_provenance(train, validation, None)
    if legacy:
        provenance = {k: v for k, v in provenance.items() if "logical" not in k}
    model = AnnealController(encoder_variant="summary", width=8)
    payload = {"checkpoint_version": 2, "model_state": model.state_dict(), "model_config": model.config,
               "normalizer": FeatureNormalizer.fit([graph_from_record(train[0])]).statistics,
               "seed": 0, "data_provenance": provenance,
               **{k: v for k, v in provenance.items() if k.endswith("_ids") or k == "data_fingerprints"}}
    path = tmp_path / "ckpt.pt"
    torch.save(payload, path)
    return path, payload, train + validation


@pytest.mark.parametrize("split_index", [0, 1])
@pytest.mark.parametrize("change", ["runtime", "source", "config", "parent_id"])
def test_same_logical_parent_still_refused_after_record_changes(tmp_path, split_index, change):
    path, payload, source = checkpoint(tmp_path)
    target = copy.deepcopy(source[split_index])
    target.update(record_id=np.array("new_record"), fingerprint=np.array("f" * 64), split=np.array("test"))
    if change == "runtime":
        target["runtime"] = np.array(8.)
    elif change == "parent_id":
        target["parent_id"] = np.array("renamed")
    else:
        target[change + "_fingerprint"] = np.array("changed")
    with pytest.raises(ValueError, match="disjoint"):
        evaluate_transfer(path, [target])


def test_name_collision_is_allowed_only_when_logical_coefficients_are_distinct(tmp_path):
    path, _, _ = checkpoint(tmp_path)
    target = example(2, "test")
    target["parent_id"] = np.array("p0")
    result = evaluate_transfer(path, [target])
    assert result[0]["disjointness_basis"] == "logical_fingerprint"


@pytest.mark.parametrize("provenance", [{}, {"data_fingerprints": ["a" * 64, "b" * 64]},
                                        {"train_parent_ids": ["p0"], "validation_parent_ids": ["p1"]}])
def test_record_hashes_or_names_are_never_direct_disjointness_proof(provenance):
    with pytest.raises(ValueError, match="provenance"):
        validate_transfer_provenance(provenance, [example(2, "test")])


def test_legacy_checkpoint_migrates_only_after_source_content_verification(tmp_path):
    path, payload, source = checkpoint(tmp_path, legacy=True)
    with pytest.raises(ValueError, match="source records"):
        evaluate_transfer(path, [example(2, "test")])
    result = evaluate_transfer(path, [example(2, "test")], source_records=source)
    assert result[0]["provenance_migration_basis"] == "source_content_digests"
    changed = copy.deepcopy(source)
    changed[0]["candidate_losses"][0] += .01
    with pytest.raises(ValueError, match="content digest"):
        evaluate_transfer(path, [example(2, "test")], source_records=changed)


def test_legacy_matching_compiled_hashes_recover_logical_parent_but_never_hide_overlap(tmp_path):
    _, payload, source = checkpoint(tmp_path, legacy=True)
    payload["data_provenance"] = {}
    target = copy.deepcopy(source[0])
    target.update(fingerprint=np.array("c" * 64), runtime=np.array(8.))
    with pytest.raises(ValueError, match="disjoint"):
        validate_transfer_provenance(payload, [target], source_records=source)
    result = validate_transfer_provenance(payload, [example(2, "test")], source_records=source)
    assert result["migration_basis"] == "complete_source_record_fingerprints"
    payload.pop("data_fingerprints")
    with pytest.raises(ValueError, match="never IDs alone"):
        validate_transfer_provenance(payload, [example(2, "test")], source_records=source)


def test_incomplete_modern_provenance_rejected_not_silently_migrated(tmp_path):
    _, payload, source = checkpoint(tmp_path)
    del payload["data_provenance"]["validation_logical_fingerprints"]
    with pytest.raises(ValueError, match="incomplete"):
        validate_transfer_provenance(payload, [example(2, "test")], source_records=source)


def test_forged_target_fingerprint_rejected(tmp_path):
    path, _, _ = checkpoint(tmp_path)
    target = example(2, "test")
    target["logical_fingerprint"] = np.array("0" * 64)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        evaluate_transfer(path, [target])


def test_global_baseline_selected_only_on_source_validation_and_equal_parents(tmp_path):
    path, _, source = checkpoint(tmp_path)
    source[0]["candidate_losses"] = np.array([.01, .99])  # train prefers opposite
    wave, details = source_global_schedule(source)
    assert np.allclose(wave, source[1]["candidate_schedules"][1])
    target = example(2, "test")
    target["candidate_losses"] = np.array([.01, .99])  # target prefers opposite
    result = evaluate_transfer(path, [target], source_global_schedule=wave)
    assert result[0]["source_global_loss"] == .99
    assert details["selected_index"] == 1


def test_frozen_normalizer_and_outcome_independent_selection(tmp_path, monkeypatch):
    from annealctrl.learning import FeatureNormalizer
    path, _, _ = checkpoint(tmp_path)
    monkeypatch.setattr(FeatureNormalizer, "fit", lambda *a, **k: pytest.fail("target normalization refit"))
    target = example(2, "test")
    first = evaluate_transfer(path, [target])[0]
    target["candidate_losses"] = target["candidate_losses"][::-1]
    second = evaluate_transfer(path, [target])[0]
    assert first["selected_index"] == second["selected_index"]


@pytest.mark.parametrize("bad", ["no_linear", "nan", "dimension", "global_absent", "normalizer", "version"])
def test_invalid_transfer_inputs_fail_closed(tmp_path, bad):
    path, payload, _ = checkpoint(tmp_path)
    target = example(2, "test")
    kwargs = {}
    if bad == "no_linear":
        target["candidate_schedules"][0] = np.linspace(0, 1, 9)**1.1
    elif bad == "nan":
        target["candidate_losses"][0] = np.nan
    elif bad == "dimension":
        target["candidate_losses"] = np.array([.4])
    elif bad == "global_absent":
        kwargs["source_global_schedule"] = np.linspace(0, 1, 9)**1.5
    elif bad == "normalizer":
        payload["normalizer"] = None
        torch.save(payload, path)
    else:
        payload["checkpoint_version"] = 999
        torch.save(payload, path)
    with pytest.raises(ValueError):
        evaluate_transfer(path, [target], **kwargs)


def rows(n=6, *, selected=.55, linear=.60, best=.50):
    return [{"record_id": f"r{i}", "parent_id": f"p{i}", "selected_loss": selected,
             "linear_loss": linear, "bank_best_loss": best} for i in range(n)]


def test_report_pairs_all_metrics_on_equal_parent_population():
    data = rows(2)
    data[0].update(selected_loss=.9, linear_loss=.95, bank_best_loss=.8)
    data.extend([dict(data[1], record_id=f"extra{i}") for i in range(20)])
    result = transfer_report(data, bootstrap_resamples=100)
    assert result["mean_selected_loss"] == pytest.approx((.9 + .55) / 2)
    assert result["mean_linear_loss"] == pytest.approx((.95 + .60) / 2)
    assert result["mean_bank_best_loss"] == pytest.approx((.8 + .50) / 2)
    assert result["mean_bank_regret"] == pytest.approx(.075)


@pytest.mark.parametrize("selected,verdict", [(.52, "beats_linear"), (.62, "worse_than_linear")])
def test_report_direction_requires_parent_interval_separation(selected, verdict):
    result = transfer_report(rows(selected=selected), bootstrap_resamples=200)
    assert result["verdict"] == verdict


def test_mean_win_does_not_produce_positive_verdict_when_ci_contains_zero():
    data = rows(3, selected=.55)
    data[-1]["selected_loss"] = .69  # 2/3 wins, negative mean, wide CI
    result = transfer_report(data, bootstrap_resamples=1000)
    assert result["vs_linear"]["mean_difference"] < 0
    assert result["vs_linear"]["parent_bootstrap_ci"]["high"] > 0
    assert result["verdict"] == "inconclusive"


def test_single_parent_and_empty_data_do_not_establish_transfer():
    assert transfer_report(rows(1), bootstrap_resamples=20)["verdict"] == "inconclusive"
    with pytest.raises(ValueError, match="nonempty"):
        transfer_report([], bootstrap_resamples=20)


def test_modern_augmented_checkpoint_can_use_original_source_validation(tmp_path):
    from annealctrl.learning import _data_provenance
    from annealctrl.transfer import validate_source_reference
    _, payload, source = checkpoint(tmp_path)
    augmented = copy.deepcopy(source[0])
    augmented["candidate_schedules"] = np.vstack([augmented["candidate_schedules"], np.linspace(0, 1, 9)**1.5])
    augmented["candidate_losses"] = np.r_[augmented["candidate_losses"], .4]
    payload["data_provenance"] = _data_provenance([augmented], [source[1]], None)
    result = validate_source_reference(payload, source)
    assert not result["training_content_verified"]
    assert not result["training_labels_reconstructed"]
    assert result["validation_content_verified"]
    assert validate_transfer_provenance(payload, [example(2, "test")])["basis"] == "logical_fingerprint"
    with pytest.raises(ValueError, match="disjoint"):
        validate_transfer_provenance(payload, [source[0]])
    changed = copy.deepcopy(source)
    changed[1]["candidate_losses"][0] += .01
    with pytest.raises(ValueError, match="validation content"):
        validate_source_reference(payload, changed)
    changed = copy.deepcopy(source)
    changed[0]["fingerprint"] = np.array("0" * 64)
    with pytest.raises(ValueError, match="compiled-record"):
        validate_source_reference(payload, changed)


def test_unknown_logical_provenance_version_cannot_be_silently_migrated(tmp_path):
    _, payload, source = checkpoint(tmp_path)
    payload["data_provenance"]["logical_provenance_version"] = 999
    with pytest.raises(ValueError, match="unsupported logical"):
        validate_transfer_provenance(payload, [example(2, "test")], source_records=source)
