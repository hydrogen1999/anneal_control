"""The design document's mandatory G -> D2 -> rho chain, as a model variant.

The factorial table asks to isolate "loss caused by scalar compression, not
merely model size". These tests pin the two properties that make that isolation
valid: everything really does pass through the profile, and the constricted
model is not the smaller one.
"""
import numpy as np
import pytest
import torch

from annealctrl.models import AnnealController


def graph(seed=0, n=6, logical=3):
    from annealctrl.models import GraphInput

    rng = np.random.default_rng(seed)
    edges = np.array([[i, i + 1] for i in range(n - 1)] + [[i + 1, i] for i in range(n - 1)]).T
    logical_edges = np.array([[i, i + 1] for i in range(logical - 1)]
                             + [[i + 1, i] for i in range(logical - 1)]).T
    membership = np.repeat(np.arange(logical), n // logical)
    t = lambda a, d: torch.tensor(np.asarray(a, dtype=np.float32).reshape(-1, d))
    return GraphInput(
        node_features=t(rng.normal(size=(n, 7)), 7),
        edge_index=torch.tensor(edges, dtype=torch.long),
        edge_features=t(rng.normal(size=(edges.shape[1], 3)), 3),
        logical_node_features=t(rng.normal(size=(logical, 2)), 2),
        logical_edge_index=torch.tensor(logical_edges, dtype=torch.long),
        logical_edge_features=t(rng.normal(size=(logical_edges.shape[1], 2)), 2),
        membership=torch.tensor(membership, dtype=torch.long),
        context=torch.tensor([1.0, 1.0]),
        path_queries=t(rng.normal(size=(4, 7)), 7))


# --- the defining property ---------------------------------------------------

def test_a_constant_profile_makes_every_graph_give_the_same_control():
    """If anything bypassed the bottleneck, two graphs would still differ."""
    model = AnnealController(bottleneck_dim=2)
    model.eval()
    with torch.no_grad():
        model.bottleneck_in.weight.zero_()      # profile no longer depends on the graph
        model.bottleneck_in.bias.fill_(0.5)
        a = model(graph(0))["proposal_schedules"]
        b = model(graph(1, n=9, logical=3))["proposal_schedules"]
    assert torch.allclose(a, b, atol=0, rtol=0), "information reached the heads around the bottleneck"


def test_without_the_bottleneck_the_same_two_graphs_do_differ():
    """Otherwise the test above would pass for a model that ignores its input."""
    model = AnnealController()
    model.eval()
    with torch.no_grad():
        a = model(graph(0))["proposal_schedules"]
        b = model(graph(1, n=9, logical=3))["proposal_schedules"]
    assert not torch.allclose(a, b, atol=1e-6)


def test_the_token_bank_collapses_so_attention_cannot_carry_extra_information():
    model = AnnealController(bottleneck_dim=4)
    model.eval()
    with torch.no_grad():
        tokens, summary = model.encode(graph(0, n=9, logical=3))
    assert tokens.shape[0] == 1, "per-node tokens survived the constriction"
    assert summary.shape[-1] == model.width


def test_the_profile_is_nonnegative_as_a_difficulty_density_must_be():
    model = AnnealController(bottleneck_dim=5)
    model.eval()
    with torch.no_grad():
        model.encode(graph(3))
    assert (model._last_profile >= 0).all()


# --- the isolation the factorial table requires ------------------------------

def test_the_constricted_model_is_not_the_smaller_one():
    """A loss must be attributable to compression, not to capacity."""
    plain = sum(p.numel() for p in AnnealController().parameters())
    for k in (1, 8, 33):
        narrow = sum(p.numel() for p in AnnealController(bottleneck_dim=k).parameters())
        assert narrow > plain, f"bottleneck_dim={k} removed parameters"


def test_the_encoder_is_unchanged_by_the_constriction():
    plain, narrow = AnnealController(), AnnealController(bottleneck_dim=1)
    shared = {k: v.shape for k, v in plain.state_dict().items()}
    for key, shape in shared.items():
        assert key in narrow.state_dict(), f"{key} disappeared"
        assert narrow.state_dict()[key].shape == shape, f"{key} changed shape"


# --- refusals and round trip -------------------------------------------------

def test_an_invalid_bottleneck_dimension_is_refused():
    for bad in (0, -3, 1.5, True):
        with pytest.raises(ValueError, match="bottleneck_dim"):
            AnnealController(bottleneck_dim=bad)


def test_the_variant_survives_a_checkpoint_round_trip(tmp_path):
    """bottleneck_dim must ride in model_config, or a trained variant reloads plain."""
    from annealctrl.learning import load_checkpoint

    model = AnnealController(bottleneck_dim=3)
    model.eval()
    with torch.no_grad():
        before = model(graph(7))["proposal_schedules"]
    torch.save({"model_config": model.config, "model_state": model.state_dict(),
                "normalizer": {}}, tmp_path / "m.pt")
    loaded, _ = load_checkpoint(tmp_path / "m.pt", device="cpu")
    loaded.eval()
    assert loaded.bottleneck_dim == 3
    with torch.no_grad():
        after = loaded(graph(7))["proposal_schedules"]
    assert torch.allclose(before, after, atol=0, rtol=0)
