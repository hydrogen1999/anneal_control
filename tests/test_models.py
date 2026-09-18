"""Optional Torch tests: a missing dependency is reported as SKIPPED, not passed."""
from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="Install the ml extra to validate neural components")

from annealctrl.models import AnnealController, graph_from_record, monotone_samples
from annealctrl.learning import (FeatureNormalizer, candidate_loss, evaluate_records,
                                fit_records, load_checkpoint, masked_response_loss,
                                proposal_bank_logits, soft_targets, validate_splits)


@pytest.fixture(scope="module", autouse=True)
def small_cpu_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def record(parent="parent-a", split="train", frustrated=False):
    tau = np.linspace(0, 1, 9)
    return dict(parent_id=parent, split=split, physical_h=np.zeros(4),
                physical_edges=np.array([[0, 1], [1, 2], [2, 3], [3, 0]]),
                physical_J=np.array([-1.5, 0.5, -1.0, -0.5 if frustrated else 0.5]),
                membership=np.array([0, 0, 1, 2]), logical_h=np.zeros(3),
                logical_edges=np.array([[0, 1], [1, 2], [0, 2]]),
                logical_J=np.array([0.5, -1.0, -0.5 if frustrated else 0.5]),
                runtime=2.0, programmed_scale=0.8, response_s=np.array([0.1, 0.5, 0.9]),
                candidate_schedules=np.stack((tau, tau**2, np.sqrt(tau))),
                candidate_losses=np.array([0.4, 0.1, 0.35]),
                response_moments=np.ones((3, 3)), response_mask=np.ones((3, 3), bool))


def small_model():
    return AnnealController(width=16, physical_layers=1, logical_layers=1)


def test_physical_and_logical_permutations_leave_predictions_unchanged():
    torch.manual_seed(7)
    graph, model = graph_from_record(record()), small_model().eval()
    physical_order = torch.tensor([2, 0, 3, 1])
    logical_order = torch.tensor([2, 0, 1])
    physical_inverse, logical_inverse = torch.argsort(physical_order), torch.argsort(logical_order)
    permuted = replace(graph, node_features=graph.node_features[physical_order],
                       edge_index=physical_inverse[graph.edge_index],
                       membership=logical_inverse[graph.membership[physical_order]],
                       logical_node_features=graph.logical_node_features[logical_order],
                       logical_edge_index=logical_inverse[graph.logical_edge_index])
    schedules = torch.tensor(record()["candidate_schedules"], dtype=torch.float32)
    a, b = model(graph, schedules), model(permuted, schedules)
    for name in a:
        torch.testing.assert_close(a[name], b[name], rtol=1e-5, atol=2e-6)


def test_signed_zero_field_systems_do_not_collapse():
    torch.manual_seed(11)
    model = small_model().eval()
    a, b = graph_from_record(record()), graph_from_record(record(frustrated=True))
    assert torch.count_nonzero(a.node_features[:, 0]) == 0
    assert torch.count_nonzero(b.node_features[:, 0]) == 0
    torch.testing.assert_close(a.edge_features[:, 1], b.edge_features[:, 1])
    assert not torch.allclose(model.encode(a)[1], model.encode(b)[1], atol=1e-6)


@pytest.mark.parametrize("bound", [1.0, 1.1, 2.0, 4.0, 10.0])
def test_monotone_decoder_endpoints_slope_and_extreme_logits(bound):
    logits = torch.tensor([[1000., -1000., 20., 0., 0., 0., 0., 0.],
                           [0., 1., -2., 0., 1., -1., 2., -3.]], requires_grad=True)
    samples = monotone_samples(logits, max_ds_dtau=bound)
    assert torch.equal(samples[:, 0], torch.zeros(2))
    assert torch.equal(samples[:, -1], torch.ones(2))
    slope = torch.diff(samples, dim=-1) * 8
    assert slope.min() >= -1e-6
    assert slope.max() <= bound + 2e-6
    samples[:, 1:-1].square().sum().backward()
    assert torch.isfinite(logits.grad).all()


def test_all_heads_receive_finite_gradients():
    torch.manual_seed(0)
    r, model = record(), small_model()
    schedules = torch.tensor(r["candidate_schedules"], dtype=torch.float32)
    output = model(graph_from_record(r), schedules)
    tasks = candidate_loss(output, schedules, torch.tensor(r["candidate_losses"], dtype=torch.float32),
                           response_labels=torch.tensor(r["response_moments"], dtype=torch.float32))
    tasks["total"].backward()
    for head in (model.node_encoder, model.policy_head, model.response_head, model.critic_head):
        gradients = [p.grad for p in head.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in gradients)
        assert any(g.abs().sum() > 0 for g in gradients)


def test_soft_targets_prefer_lower_loss_and_keep_equal_modes():
    p = soft_targets(torch.tensor([0.0, 0.5, 0.0]), temperature=0.1)
    assert p[0] == p[2] and p[0] > p[1]
    with pytest.raises(ValueError):
        soft_targets(torch.tensor([0.0, float("nan")]))


def test_mixture_distillation_can_represent_two_separated_modes():
    tau = torch.linspace(0, 1, 9)
    left, right = tau**3, 1 - (1 - tau)**3
    bank = torch.stack((left, (left + right) / 2, right))
    logits = proposal_bank_logits(torch.stack((left, right)), torch.zeros(2), bank, bandwidth=0.03)
    p = torch.softmax(logits, 0)
    assert p[0] > 0.49 and p[2] > 0.49 and p[1] < 0.01


def test_unresolved_response_labels_are_masked_without_nan_gradients():
    prediction = torch.ones(2, 3, requires_grad=True)
    labels = torch.tensor([[float("nan"), float("inf"), 2.0], [1.0, -1.0, 4.0]])
    loss = masked_response_loss(prediction, labels, torch.tensor([False, True]))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(prediction.grad).all()
    assert torch.equal(prediction.grad[0], torch.zeros(3))


def test_numerically_unresolved_candidate_ranks_are_not_trained_as_certain():
    r, model = record(), small_model()
    schedules = torch.tensor(r["candidate_schedules"], dtype=torch.float32)
    output = model(graph_from_record(r), schedules)
    loss = candidate_loss(output, schedules, torch.tensor([0.400, 0.401, 0.399]),
                          loss_uncertainty=torch.full((3,), 0.01))
    assert loss["ranking"] == 0


def test_normalizer_uses_training_graphs_and_does_not_change_membership():
    graph = graph_from_record(record())
    normalizer = FeatureNormalizer.fit([graph])
    original = {key: (mean.clone(), std.clone()) for key, (mean, std) in normalizer.statistics.items()}
    unusual = replace(graph, context=graph.context * 1000)
    transformed = normalizer.transform(unusual)
    assert torch.equal(transformed.membership, graph.membership)
    for key in original:
        torch.testing.assert_close(original[key][0], normalizer.statistics[key][0])
        torch.testing.assert_close(original[key][1], normalizer.statistics[key][1])
    assert transformed.context[0] > 1000


def test_parent_leakage_and_wrong_split_are_rejected():
    with pytest.raises(ValueError, match="leakage"):
        validate_splits([record("a", "train")], [record("a", "validation")])
    with pytest.raises(ValueError, match="explicitly"):
        validate_splits([record("a", "test")], [record("b", "validation")])


def test_catalyst_is_not_silently_ignored():
    r = record()
    r["catalyst_strength"] = 0.5
    with pytest.raises(ValueError, match="driver-graph"):
        graph_from_record(r)


def test_no_oracle_labels_or_identifiers_enter_model_inputs():
    a = record()
    b = {**a, "parent_id": "unseen", "split": "test", "candidate_losses": np.array([100., 300., 200.]),
         "response_moments": np.full((3, 3), 987.)}
    ga, gb = graph_from_record(a), graph_from_record(b)
    for field in ga.__dataclass_fields__:
        torch.testing.assert_close(getattr(ga, field), getattr(gb, field))


def test_tiny_training_checkpoint_round_trip(tmp_path):
    validation = record("c", "validation")
    validation["logical_h"] = np.array([0.2, 0., 0.])
    validation["physical_h"] = np.array([0.1, 0.1, 0., 0.])
    fit = fit_records([record("a", "train"), record("b", "train", True)],
                       [validation], model=small_model(), epochs=2,
                       checkpoint=tmp_path / "model.pt", seed=9)
    model, normalizer = load_checkpoint(tmp_path / "model.pt")
    before = evaluate_records(fit.model, [record("d", "test")], fit.normalizer)
    after = evaluate_records(model, [record("d", "test")], normalizer)
    assert before == after
    assert before["selection_mode"] == "finite_bank_critic"
    assert len(fit.history) == 2
