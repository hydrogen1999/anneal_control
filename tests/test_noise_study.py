"""Learned-noise checks are about frozen decisions and real numerical evidence."""
import json

import numpy as np
import pytest

from annealctrl.noise_study import _summarize, run_noise_study


def test_noise_summary_weights_parents_equally_and_pairs_rates():
    rows = []
    for parent, count, difference in (("a", 1, -.2), ("b", 3, .1)):
        for record in range(count):
            for rate in ([0., 0.], [.1, 0.]):
                for method in ("linear", "bank"):
                    rows.append({"parent_id": parent, "logical_fingerprint": parent, "record_id": f"{parent}{record}",
                                 "method": method, "rates": rate,
                                 "loss": .5 + (difference if method == "bank" else 0.) + rate[0]})
    result = _summarize(rows, 100)
    bank = [r for r in result if r["method"] == "bank" and r["rates"] == [.1, 0.]][0]
    assert bank["vs_linear"]["mean"] == pytest.approx(-.05)
    assert bank["vs_noiseless"]["mean"] == pytest.approx(.1)


def test_noise_runner_freezes_decisions_and_keeps_real_refinement_receipts(tmp_path):
    pytest.importorskip("torch")
    from annealctrl.learning import fit_records
    from annealctrl.pipeline import generate_dataset, load_records
    cfg = {"seed": 510, "parents": 6, "families": ["spin_glass"],
           "logical_qubits": 2, "chain_lengths": [1, 1], "variants": [{"shape": "path", "ports": 1}],
           "chain_strengths": [1.5], "runtimes": [.3], "candidates": 3, "spectral_points": 3,
           "teacher": {"mode": "none"}, "steps": 16, "max_steps": 1024,
           "label_state_tolerance": .001, "max_physical_qubits": 2}
    data = tmp_path / "data"
    generate_dataset(cfg, data)
    checkpoint = tmp_path / "model.pt"
    fit_records(load_records(data, "train"), load_records(data, "validation"),
                checkpoint=checkpoint, epochs=1, patience=1,
                model_config={"encoder_variant": "summary", "width": 8, "proposals": 2})
    study = {"schema_version": 1, "data": str(data), "source_data": str(data),
             "checkpoint": str(checkpoint), "rates": [[0., 0.], [.01, .01]],
             "max_qubits": 2, "tolerance": 1e-4, "bootstrap_resamples": 30}
    output = tmp_path / "noise"
    result = run_noise_study(study, output)
    assert result["status"] == "complete"
    assert {r["method"] for r in result["rows"]} == {"linear", "source_global", "bank", "direct"}
    assert all(r["refinement_difference"] < 1e-4 for r in result["rows"])
    assert result["costs"]["solver_calls_started"] == 2 * len(result["rows"])
    assert result["costs"]["complete"]
    before = json.loads((output / "decisions.json").read_text())
    again = run_noise_study(study, output, resume=True)
    assert again == result and json.loads((output / "decisions.json").read_text()) == before


@pytest.mark.parametrize("rates", [[[.1, 0]], [[0, 0], [0, 0]], [[0, 0], [-.1, 0]], [[0, 0], [np.nan, 0]]])
def test_noise_grid_fails_before_loading_or_scoring(rates, tmp_path):
    with pytest.raises(ValueError, match="rates"):
        run_noise_study({"rates": rates}, tmp_path / "no_output")
    assert not (tmp_path / "no_output").exists()
