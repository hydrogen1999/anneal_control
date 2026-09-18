"""Training, provenance, ablation isolation, and exact-resume regressions."""
from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="Training tests require the ml extra")

from annealctrl.learning import FeatureNormalizer, fit_records, load_checkpoint, seed_everything
from annealctrl.models import AnnealController, graph_from_record


@pytest.fixture(scope="module", autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def example(parent, split="train", offset=0.0):
    tau = np.linspace(0, 1, 9)
    # Distinct IDs alone do not make independent logical problems.
    parent_bias = sum((i + 1) * ord(char) for i, char in enumerate(str(parent))) / 1000
    return dict(record_id=f"record-{parent}", parent_id=parent, split=split, fingerprint="dataset-v1",
                physical_h=np.array([.1, .1, -.1]), physical_edges=np.array([[0, 1], [1, 2]]),
                physical_J=np.array([-1., .5]), membership=np.array([0, 0, 1]),
                logical_h=np.array([.2 + parent_bias, -.1]), logical_edges=np.array([[0, 1]]), logical_J=np.array([.5]),
                runtime=2., programmed_scale=.5, response_s=np.array([.2, .5, .8]),
                candidate_schedules=np.stack((tau, tau**2, 1 - (1 - tau)**2)),
                candidate_losses=np.array([.4, .2 + offset, .3]),
                response_moments=np.ones((3, 3)), response_mask=np.ones((3, 3), dtype=bool))


def config(variant="hierarchical"):
    return dict(width=16, physical_layers=1, logical_layers=1, encoder_variant=variant)


def data():
    return [example("a"), example("b", offset=.04), example("c", offset=.09)], [example("v", "validation")]


@pytest.mark.parametrize("scope", ["critic", "policy", "heads"])
def test_mechanism_freezes_shared_modules_and_changes_only_designated_head(scope, tmp_path):
    train, validation = data()
    initial_path = tmp_path / "initial.pt"
    initial = fit_records(train, validation, model_config=config("summary"), epochs=1,
                          seed=7, checkpoint=initial_path)
    updated_path = tmp_path / "updated.pt"
    fitted = fit_records(train, validation, initialize_from=initial_path, trainable_scope=scope,
                         selection_mode="fixed_epochs", epochs=3, patience=1,
                         seed=7, checkpoint=updated_path)
    before, after = initial.model.state_dict(), fitted.model.state_dict()
    changed = {name for name in before if not torch.equal(before[name], after[name])}
    enabled = {name for name, parameter in fitted.model.named_parameters() if parameter.requires_grad}
    assert changed and changed <= enabled
    assert not any(name.startswith(("attention.", "baseline_summary_encoder.", "response_head.")) for name in enabled)
    assert fitted.best_epoch == fitted.last_epoch == 2
    assert not fitted.stopped_early
    for name in initial.normalizer.statistics:
        for old, new in zip(initial.normalizer.statistics[name], fitted.normalizer.statistics[name]):
            assert torch.equal(old, new)
    graph = fitted.normalizer.transform(graph_from_record(validation[0]))
    bank = torch.tensor(validation[0]["candidate_schedules"], dtype=torch.float32)
    a, b = initial.model(graph, bank), fitted.model(graph, bank)
    if scope == "critic":
        assert torch.equal(a["proposal_schedules"], b["proposal_schedules"])
        assert torch.equal(a["proposal_logits"], b["proposal_logits"])
        assert not torch.equal(a["predicted_losses"], b["predicted_losses"])
    if scope == "policy":
        assert torch.equal(a["predicted_losses"], b["predicted_losses"])
        assert not torch.equal(a["proposal_schedules"], b["proposal_schedules"])
    saved = torch.load(updated_path, weights_only=True)
    assert saved["selection_mode"] == "fixed_epochs"
    assert set(saved["trainable_parameter_names"]) == enabled
    assert len(saved["frozen_parameter_sha256"]) == 64


def test_mechanism_exact_resume_and_unchanged_validation_requirement(tmp_path):
    train, validation = data()
    initial_path = tmp_path / "initial.pt"
    fit_records(train, validation, model_config=config("summary"), epochs=1, checkpoint=initial_path)
    settings = dict(initialize_from=initial_path, trainable_scope="policy", selection_mode="fixed_epochs", seed=9)
    whole = fit_records(train, validation, epochs=3, **settings)
    latest = tmp_path / "latest.pt"
    fit_records(train, validation, epochs=1, latest_checkpoint=latest, **settings)
    resumed = fit_records(train, validation, epochs=3, resume_from=latest, **settings)
    assert whole.history == resumed.history
    for name, value in whole.model.state_dict().items():
        assert torch.equal(value, resumed.model.state_dict()[name])
    changed_validation = [dict(validation[0], candidate_losses=np.array([.9, .8, .7]))]
    with pytest.raises(ValueError, match="unchanged validation_content"):
        fit_records(train, changed_validation, epochs=1, **settings)
    with pytest.raises(ValueError, match="fixed_epochs"):
        fit_records(train, validation, initialize_from=initial_path, trainable_scope="policy", epochs=1)


@pytest.mark.parametrize("variant", ["hierarchical", "physical", "logical", "summary"])
def test_all_encoder_variants_train_and_round_trip(variant, tmp_path):
    train, val = data()
    fitted = fit_records(train, val, model_config=config(variant), epochs=1, seed=13,
                         checkpoint=tmp_path / f"{variant}.pt")
    loaded, normalizer = load_checkpoint(tmp_path / f"{variant}.pt")
    graph = normalizer.transform(graph_from_record(val[0]))
    bank = torch.as_tensor(val[0]["candidate_schedules"], dtype=torch.float32)
    torch.testing.assert_close(loaded(graph, bank)["predicted_losses"],
                               fitted.model(graph, bank)["predicted_losses"], rtol=0, atol=0)
    assert loaded.encoder_variant == variant
    assert fitted.last_epoch == 0


def test_logical_baseline_has_no_physical_or_scale_channel():
    seed_everything(12)
    model = AnnealController(**config("logical")).eval()
    first = graph_from_record(example("a"))
    normalizer = FeatureNormalizer.fit([first])
    second = replace(first, node_features=first.node_features * 100 + 50,
                     edge_features=first.edge_features * -30,
                     membership=torch.tensor([0, 1, 1]), context=torch.tensor([2., .0001]))
    bank = torch.as_tensor(example("a")["candidate_schedules"], dtype=torch.float32)
    a, b = model(normalizer.transform(first), bank), model(normalizer.transform(second), bank)
    for name in a:
        torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)
    assert not hasattr(model, "node_encoder")
    assert not hasattr(model, "chain_encoder")


@pytest.mark.parametrize("variant", ["physical", "logical", "summary"])
def test_alternative_encoders_are_permutation_invariant(variant):
    seed_everything(1)
    model = AnnealController(**config(variant)).eval()
    graph = graph_from_record(example("a"))
    order = torch.tensor([2, 0, 1])
    inverse = torch.argsort(order)
    permuted = replace(graph, node_features=graph.node_features[order],
                       edge_index=inverse[graph.edge_index], membership=graph.membership[order])
    torch.testing.assert_close(model.encode(graph)[1], model.encode(permuted)[1], rtol=1e-5, atol=1e-6)


def test_resume_matches_uninterrupted_optimizer_rng_and_parameters(tmp_path):
    train, val = data()
    kwargs = dict(model_config=config(), seed=31, batch_size=2, accumulation_steps=1, patience=10)
    whole = fit_records(train, val, epochs=3, checkpoint=tmp_path / "whole-best.pt",
                        latest_checkpoint=tmp_path / "whole-latest.pt", **kwargs)
    fit_records(train, val, epochs=1, checkpoint=tmp_path / "part-best.pt",
                latest_checkpoint=tmp_path / "part-latest.pt", **kwargs)
    resumed = fit_records(train, val, epochs=3, resume_from=tmp_path / "part-latest.pt",
                          checkpoint=tmp_path / "resume-best.pt", latest_checkpoint=tmp_path / "resume-latest.pt", **kwargs)
    assert whole.history == resumed.history
    assert whole.best_epoch == resumed.best_epoch
    for key, value in whole.model.state_dict().items():
        torch.testing.assert_close(value, resumed.model.state_dict()[key], rtol=0, atol=0)
    whole_payload = torch.load(tmp_path / "whole-latest.pt", weights_only=True)
    resumed_payload = torch.load(tmp_path / "resume-latest.pt", weights_only=True)
    for key, value in whole_payload["model_state"].items():
        torch.testing.assert_close(value, resumed_payload["model_state"][key], rtol=0, atol=0)
    assert whole_payload["resume_state"]["rng_state"]["order"] == resumed_payload["resume_state"]["rng_state"]["order"]
    for key, state in whole_payload["optimizer_state"]["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(value, resumed_payload["optimizer_state"]["state"][key][name], rtol=0, atol=0)


def test_gradient_accumulation_has_equal_effective_batch_semantics():
    train, val = data()
    a = fit_records(train, val, model_config=config(), epochs=1, seed=4, batch_size=2, accumulation_steps=1)
    b = fit_records(train, val, model_config=config(), epochs=1, seed=4, batch_size=1, accumulation_steps=2)
    assert a.history == b.history
    assert a.history[0]["optimizer_updates"] == 2  # One group of two, one of one.
    for key, value in a.model.state_dict().items():
        torch.testing.assert_close(value, b.model.state_dict()[key], rtol=0, atol=0)


@pytest.mark.parametrize("mutation", ["labels", "order", "fingerprint", "physical"])
def test_resume_rejects_mutated_records_even_with_same_parent_ids(mutation, tmp_path):
    train, val = data()
    fit_records(train, val, model_config=config(), epochs=1, checkpoint=tmp_path / "fit.pt")
    if mutation == "labels":
        train[0]["candidate_losses"][0] += .01
    elif mutation == "order":
        train.reverse()
    elif mutation == "fingerprint":
        train[0]["fingerprint"] = "dataset-v2"
    else:
        train[0]["physical_J"][0] += .1
    with pytest.raises(ValueError, match="provenance/content"):
        fit_records(train, val, epochs=2, resume_from=tmp_path / "fit.pt")


def test_resume_rejects_training_and_model_configuration_changes(tmp_path):
    train, val = data()
    fit_records(train, val, model_config=config(), epochs=1, checkpoint=tmp_path / "fit.pt")
    with pytest.raises(ValueError, match="training configuration"):
        fit_records(train, val, epochs=2, resume_from=tmp_path / "fit.pt", learning_rate=.01)
    with pytest.raises(ValueError, match="model configuration"):
        fit_records(train, val, epochs=2, resume_from=tmp_path / "fit.pt", model_config=config("logical"))


def test_best_latest_provenance_and_noop_resume_materialization(tmp_path):
    train, val = data()
    fitted = fit_records(train, val, model_config=config(), epochs=2, checkpoint=tmp_path / "best.pt",
                         latest_checkpoint=tmp_path / "latest.pt", dataset_fingerprint="external-manifest")
    best = torch.load(tmp_path / "best.pt", weights_only=True)
    latest = torch.load(tmp_path / "latest.pt", weights_only=True)
    assert best["checkpoint_role"] == "best" and latest["checkpoint_role"] == "latest"
    assert best["train_record_ids"] == ["record-a", "record-b", "record-c"]
    assert best["validation_parent_ids"] == ["v"]
    assert best["data_fingerprints"] == ["dataset-v1"]
    assert best["data_provenance"]["dataset_fingerprint"] == "external-manifest"
    assert latest["resume_state"]["epoch"] == 1
    again = fit_records(train, val, epochs=2, resume_from=tmp_path / "latest.pt",
                        checkpoint=tmp_path / "copied-best.pt", dataset_fingerprint="external-manifest")
    assert (tmp_path / "copied-best.pt").exists()
    assert again.history == fitted.history


def test_legacy_inference_checkpoint_is_not_claimed_resumable(tmp_path):
    model = AnnealController(**config())
    legacy = dict(model_config={k: v for k, v in model.config.items() if k != "encoder_variant"},
                  model_state=model.state_dict(), normalizer=FeatureNormalizer.fit([graph_from_record(example("a"))]).statistics)
    torch.save(legacy, tmp_path / "legacy.pt")
    loaded, _ = load_checkpoint(tmp_path / "legacy.pt")
    assert loaded.encoder_variant == "hierarchical"
    train, val = data()
    with pytest.raises(ValueError, match="Legacy checkpoint"):
        fit_records(train, val, resume_from=tmp_path / "legacy.pt")


def test_knot_mismatch_and_bad_batch_fail_before_training():
    train, val = data()
    with pytest.raises(ValueError, match="knots"):
        fit_records(train, val, model_config={**config(), "schedule_points": 5}, epochs=1)
    with pytest.raises(ValueError, match="Positive integer"):
        fit_records(train, val, batch_size=0)
    with pytest.raises(ValueError, match="nonnegative"):
        fit_records(train, val, response_weight=-1)


def test_cuda_is_not_silently_replaced_with_cpu():
    if torch.cuda.is_available():
        pytest.skip("Unavailable-device behavior is a CPU-host test")
    train, val = data()
    with pytest.raises(RuntimeError, match="CUDA training was requested"):
        fit_records(train, val, device="cuda", epochs=1)


def test_model_proposal_count_knots_and_bound_are_configurable():
    model = AnnealController(**config(), proposals=5, schedule_points=17, max_ds_dtau=1.5)
    proposals = model(graph_from_record(example("a")))["proposal_schedules"]
    assert proposals.shape == (5, 17)
    assert float((torch.diff(proposals, dim=-1) * 16).max().detach()) <= 1.5 + 1e-5


def test_training_never_accepts_test_records():
    train, val = data()
    with pytest.raises(ValueError, match="explicitly"):
        fit_records(train, [example("test", "test")], epochs=1)


def test_graph_adapter_never_silently_drops_global_path_energy_scale():
    record = example("a")
    record["energy_scale"] = 2.0
    with pytest.raises(ValueError, match="energy_scale"):
        graph_from_record(record)
