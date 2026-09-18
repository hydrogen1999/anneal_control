"""Decompose a failing direct policy into generation quality and ranking quality.

Reporting only the critic-selected proposal conflates two different failures: a
policy that cannot generate a good control, and a critic that cannot rank the
controls its own policy generated. They have different fixes, so they have to be
measured apart.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from annealctrl.policy_diagnostics import decompose_proposals, aggregate_proposal_decomposition


def test_decomposition_separates_generation_from_ranking():
    # Three proposals; the critic ranks the worst one first.
    result = decompose_proposals(true_losses=[0.30, 0.50, 0.60],
                                 predicted_losses=[0.10, 0.20, 0.30],
                                 selected_index=0, bank_loss=0.40, linear_loss=0.60)
    assert result["best_proposal_loss"] == pytest.approx(0.30)
    assert result["selected_proposal_loss"] == pytest.approx(0.30)
    assert result["ranking_regret"] == pytest.approx(0.0)


def test_ranking_regret_is_what_the_critic_threw_away():
    result = decompose_proposals(true_losses=[0.50, 0.30, 0.60],
                                 predicted_losses=[0.10, 0.20, 0.30],
                                 selected_index=0, bank_loss=0.40, linear_loss=0.60)
    assert result["best_proposal_loss"] == pytest.approx(0.30)
    assert result["selected_proposal_loss"] == pytest.approx(0.50)
    assert result["ranking_regret"] == pytest.approx(0.20)


def test_generation_gap_is_measured_against_the_bank():
    result = decompose_proposals(true_losses=[0.50, 0.55], predicted_losses=[0.1, 0.2],
                                 selected_index=0, bank_loss=0.40, linear_loss=0.60)
    # Even a perfect ranker could only reach 0.50 here, which is worse than the bank.
    assert result["generation_gap_vs_bank"] == pytest.approx(0.10)
    assert result["oracle_beats_bank"] is False


def test_a_policy_that_can_beat_the_bank_is_recognised():
    result = decompose_proposals(true_losses=[0.35, 0.55], predicted_losses=[0.2, 0.1],
                                 selected_index=1, bank_loss=0.40, linear_loss=0.60)
    assert result["oracle_beats_bank"] is True
    assert result["generation_gap_vs_bank"] == pytest.approx(-0.05)
    assert result["ranking_regret"] == pytest.approx(0.20)


def test_critic_rank_correlation_is_reported():
    perfect = decompose_proposals(true_losses=[0.3, 0.4, 0.5], predicted_losses=[0.1, 0.2, 0.3],
                                  selected_index=0, bank_loss=0.4, linear_loss=0.6)
    inverted = decompose_proposals(true_losses=[0.3, 0.4, 0.5], predicted_losses=[0.3, 0.2, 0.1],
                                   selected_index=2, bank_loss=0.4, linear_loss=0.6)
    assert perfect["critic_rank_correlation"] == pytest.approx(1.0)
    assert inverted["critic_rank_correlation"] == pytest.approx(-1.0)


def test_a_single_proposal_has_no_ranking_to_get_wrong():
    result = decompose_proposals(true_losses=[0.5], predicted_losses=[0.1],
                                 selected_index=0, bank_loss=0.4, linear_loss=0.6)
    assert result["ranking_regret"] == pytest.approx(0.0)
    assert result["critic_rank_correlation"] is None


def test_rank_correlation_uses_average_ranks_for_ties():
    # Average ranks are [1.5, 1.5, 3] and [1, 2.5, 2.5], yielding rho=.5.
    result = decompose_proposals(true_losses=[0.1, 0.1, 0.3],
                                 predicted_losses=[0.1, 0.2, 0.2],
                                 selected_index=0, bank_loss=0.4, linear_loss=0.6)
    assert result["critic_rank_correlation"] == pytest.approx(0.5)
    permuted = decompose_proposals(true_losses=[0.1, 0.3, 0.1],
                                   predicted_losses=[0.2, 0.2, 0.1],
                                   selected_index=2, bank_loss=0.4, linear_loss=0.6)
    assert permuted["critic_rank_correlation"] == pytest.approx(0.5)


def test_constant_losses_have_undefined_rank_correlation():
    result = decompose_proposals(true_losses=[0.3, 0.3, 0.3],
                                 predicted_losses=[0.1, 0.2, 0.2],
                                 selected_index=0, bank_loss=0.4, linear_loss=0.6)
    assert result["critic_rank_correlation"] is None


def test_mismatched_lengths_are_refused():
    with pytest.raises(ValueError, match="same length"):
        decompose_proposals(true_losses=[0.3, 0.4], predicted_losses=[0.1],
                            selected_index=0, bank_loss=0.4, linear_loss=0.6)


def test_an_out_of_range_selection_is_refused():
    with pytest.raises(ValueError, match="selected_index"):
        decompose_proposals(true_losses=[0.3], predicted_losses=[0.1],
                            selected_index=5, bank_loss=0.4, linear_loss=0.6)


# --- aggregation -------------------------------------------------------------

def rows(n=6, *, ranking=0.10, generation=0.05):
    return [{"record_id": f"r{i}", "parent_id": f"p{i}",
             "selected_proposal_loss": 0.55, "best_proposal_loss": 0.55 - ranking,
             "bank_loss": 0.55 - ranking - generation, "linear_loss": 0.60,
             "ranking_regret": ranking, "generation_gap_vs_bank": generation,
             "oracle_beats_bank": generation < 0, "critic_rank_correlation": 0.2,
             "n_proposals": 3} for i in range(n)]


def test_aggregate_attributes_the_shortfall_to_its_two_causes():
    summary = aggregate_proposal_decomposition(rows(), bootstrap_resamples=200)
    assert summary["ranking_regret"]["mean"] == pytest.approx(0.10)
    assert summary["generation_gap_vs_bank"]["mean"] == pytest.approx(0.05)
    assert summary["dominant_cause"] == "critic_ranking"


def test_aggregate_names_generation_when_that_dominates():
    summary = aggregate_proposal_decomposition(rows(ranking=0.01, generation=0.09))
    assert summary["dominant_cause"] == "policy_generation"


def test_aggregate_reports_how_often_the_policy_could_beat_the_bank():
    mixed = rows(4, ranking=0.10, generation=0.05) + rows(4, ranking=0.10, generation=-0.05)
    for index, row in enumerate(mixed):
        row["parent_id"] = f"q{index}"
    summary = aggregate_proposal_decomposition(mixed)
    assert summary["oracle_beats_bank_fraction"] == pytest.approx(0.5)


def test_aggregate_refuses_an_empty_row_set():
    with pytest.raises(ValueError, match="nonempty"):
        aggregate_proposal_decomposition([])


def test_diagnose_checkpoint_scores_every_proposal_on_a_real_dataset(tmp_path, monkeypatch):
    from annealctrl.learning import fit_records
    from annealctrl.pipeline import generate_dataset, load_records
    from annealctrl.policy_diagnostics import diagnose_checkpoint

    config = {"seed": 55, "parents": 6, "families": ["spin_glass"], "logical_qubits": 3,
              "chain_lengths": [1, 1, 1], "variants": [{"shape": "path", "ports": 1}],
              "chain_strengths": [1.5], "runtimes": [2.0], "candidates": 4,
              "spectral_points": 3, "steps": 16, "max_steps": 1024,
              "label_state_tolerance": 0.005, "max_physical_qubits": 3,
              "teacher": {"mode": "none"}}
    generate_dataset(config, tmp_path / "data")
    fit_records(load_records(tmp_path / "data", "train"),
                load_records(tmp_path / "data", "validation"),
                model_config={"encoder_variant": "summary", "width": 8, "proposals": 3},
                epochs=1, seed=0, checkpoint=tmp_path / "m.pt", response_weight=0.0)

    result = diagnose_checkpoint(tmp_path / "data", tmp_path / "m.pt", split="test",
                                 tolerance=5e-3, initial_steps=16, max_steps=512)
    assert result["n_records"] >= 1
    row = result["rows"][0]
    assert row["n_proposals"] == 3
    assert row["ranking_regret"] >= 0.0
    assert row["best_proposal_loss"] <= row["selected_proposal_loss"] + 1e-12
    summary = aggregate_proposal_decomposition(result["rows"], bootstrap_resamples=100)
    assert summary["dominant_cause"] in {"critic_ranking", "policy_generation"}

    import annealctrl.benchmarking as benchmarking
    def broken_physics(*args, **kwargs):
        raise ValueError("invalid physics provenance")
    monkeypatch.setattr(benchmarking, "score_schedule", broken_physics)
    with pytest.raises(ValueError, match="physics provenance"):
        diagnose_checkpoint(tmp_path / "data", tmp_path / "m.pt", split="test",
                            tolerance=5e-3, initial_steps=16, max_steps=512)


def test_diagnostics_reject_bad_backend_before_loading_data(tmp_path):
    from annealctrl.policy_diagnostics import diagnose_checkpoint
    with pytest.raises(ValueError, match="backend"):
        diagnose_checkpoint(tmp_path / "missing", tmp_path / "missing.pt", backend="bad")


def test_decomposition_refuses_nonfinite_reference_losses():
    with pytest.raises(ValueError, match="reference losses"):
        decompose_proposals(true_losses=[0.3], predicted_losses=[0.3],
                            selected_index=0, bank_loss=float("nan"), linear_loss=0.6)
