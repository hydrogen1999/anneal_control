"""Factor 2 rung 3: response bins as the auxiliary learning target.

The design document's spectral-target ladder is first gap, all resolved gaps,
**response bins**, then response plus intervention labels. The first two rungs
were answered by comparing privileged teachers. The third asks something
different -- not which spectral summary predicts better, but which one is
worth *supervising a network with* -- and it had never been run because the
training loop hard-coded `response_moments`.

The datasets already carry `response_bins`, eight frequency bins per s-point
against the moments' three. Selecting between them is the whole of rung 3.
"""
import numpy as np
import pytest
import torch

from annealctrl.learning import fit_records


def example(parent, split="train", offset=0.0, bins=True):
    tau = np.linspace(0, 1, 9)
    parent_bias = sum((i + 1) * ord(c) for i, c in enumerate(str(parent))) / 1000
    record = dict(record_id=f"record-{parent}", parent_id=parent, split=split,
                  fingerprint="dataset-v1",
                  physical_h=np.array([.1, .1, -.1]), physical_edges=np.array([[0, 1], [1, 2]]),
                  physical_J=np.array([-1., .5]), membership=np.array([0, 0, 1]),
                  logical_h=np.array([.2 + parent_bias, -.1]),
                  logical_edges=np.array([[0, 1]]), logical_J=np.array([.5]),
                  runtime=2., programmed_scale=.5, response_s=np.array([.2, .5, .8]),
                  candidate_schedules=np.stack((tau, tau**2, 1 - (1 - tau)**2)),
                  candidate_losses=np.array([.4, .2 + offset, .3]),
                  response_moments=np.ones((3, 3)),
                  response_mask=np.ones((3, 3), dtype=bool))
    if bins:
        record["response_bins"] = np.abs(np.linspace(0, 1, 24)).reshape(3, 8)
    return record


def data(bins=True):
    return ([example("a", bins=bins), example("b", offset=.04, bins=bins),
             example("c", offset=.09, bins=bins)],
            [example("v", "validation", bins=bins)])


def config(response_dim=3):
    return dict(width=16, physical_layers=1, logical_layers=1,
                encoder_variant="hierarchical", response_dim=response_dim)


def saved_config(tmp_path, name, **kwargs):
    train, validation = data(kwargs.pop("bins", True))
    path = tmp_path / f"{name}.pt"
    fit_records(train, validation, epochs=1, seed=0, checkpoint=path, **kwargs)
    return torch.load(path, map_location="cpu", weights_only=True)["training_config"]


def test_the_default_target_is_moments_and_is_recorded(tmp_path):
    assert saved_config(tmp_path, "d", model_config=config(3))["response_target"] == "moments"


def test_bins_can_be_selected_as_the_target(tmp_path):
    saved = saved_config(tmp_path, "b", model_config=config(8), response_target="bins")
    assert saved["response_target"] == "bins"


def test_the_bins_are_actually_the_supervision_signal():
    """Changing the bin values must change the trained model, or the rung is untested."""
    train, validation = data()
    a = fit_records(train, validation, model_config=config(8), epochs=2, seed=3,
                    response_target="bins", response_weight=0.5)
    for record in train + validation:
        record["response_bins"] = record["response_bins"][:, ::-1].copy()
    b = fit_records(train, validation, model_config=config(8), epochs=2, seed=3,
                    response_target="bins", response_weight=0.5)
    sa, sb = a.model.state_dict(), b.model.state_dict()
    assert any(not torch.equal(sa[k], sb[k]) for k in sa)


def test_a_response_dim_that_does_not_match_the_target_is_refused():
    """Eight bins cannot be supervised by a three-wide head."""
    train, validation = data()
    with pytest.raises(ValueError, match="response_dim|width"):
        fit_records(train, validation, model_config=config(3), epochs=1, seed=0,
                    response_target="bins")


def test_an_unknown_target_is_refused():
    train, validation = data()
    with pytest.raises(ValueError, match="response_target"):
        fit_records(train, validation, model_config=config(3), epochs=1, seed=0,
                    response_target="gaps")


def test_records_without_bins_train_without_a_response_term(tmp_path):
    """Pegasus datasets carry no resolved response; that must not crash."""
    saved = saved_config(tmp_path, "nb", model_config=config(8),
                         response_target="bins", bins=False)
    assert saved["response_target"] == "bins"


def test_the_bins_mask_comes_from_the_per_s_point_column():
    """response_mask is (n_s, n_moments) with identical columns; bins are wider."""
    train, validation = data()
    for record in train + validation:
        record["response_mask"] = np.zeros((3, 3), dtype=bool)   # nothing valid
    masked = fit_records(train, validation, model_config=config(8), epochs=2, seed=5,
                         response_target="bins", response_weight=0.5)
    for record in train + validation:
        record["response_mask"] = np.ones((3, 3), dtype=bool)
    unmasked = fit_records(train, validation, model_config=config(8), epochs=2, seed=5,
                           response_target="bins", response_weight=0.5)
    sa, sb = masked.model.state_dict(), unmasked.model.state_dict()
    assert any(not torch.equal(sa[k], sb[k]) for k in sa)
