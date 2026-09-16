"""Figure rendering must be camera-ready: TrueType, never Type 3 (ADR-0006)."""
import sys

import pytest

matplotlib = pytest.importorskip("matplotlib")

from annealctrl.figures import figure_frontier, figure_interventions, load_plot_utils  # noqa: E402


def frontier_rows():
    rows = []
    for index, headroom in enumerate([0.02, 0.05, 0.09, 0.14, 0.21]):
        rows.append({
            "record_id": f"p{index}_r0", "parent_id": f"p{index}", "split": "validation",
            "family": "spin_glass", "runtime": 2.0, "logical_n": 3, "physical_n": 6,
            "linear_loss": 0.6, "best_found_loss": 0.6 - headroom, "headroom": headroom,
            "best_family": "two_window", "resolution_status": "resolved",
            "combined_loss_ambiguity": 1e-5, "ambiguity_margin": 1.0,
            "family_restriction_loss": {"linear": headroom, "one_window": headroom / 2,
                                        "two_window": 0.0, "eight_bin": headroom / 3},
            "incumbent_trace": {"linear": [0.6], "one_window": [0.6, 0.6, 0.6 - headroom / 2],
                                "two_window": [0.6, 0.6 - headroom / 3, 0.6 - headroom],
                                "eight_bin": [0.6, 0.6, 0.6 - headroom / 3]},
            "total_objective_calls": 10, "total_integrator_steps": 900, "audit_failures": []})
    rows.append({**rows[0], "record_id": "p9_r0", "parent_id": "p9", "headroom": 1e-9,
                 "best_found_loss": 0.6 - 1e-9, "resolution_status": "censored_numerical",
                 "combined_loss_ambiguity": 1e-3})
    return rows


def intervention_rows():
    rows = []
    for index, penalty in enumerate([0.01, 0.03, 0.06, 0.10]):
        for factor in ("geometry", "chain_strength"):
            rows.append({
                "pair_id": f"p{index}_{factor}", "parent_id": f"p{index}", "factor": factor,
                "scale_arm": "total_compiled_effect", "runtime": 2.0,
                "loss_matrix": {"A_on_A": 0.5, "B_on_A": 0.5 + penalty,
                                "B_on_B": 0.45, "A_on_B": 0.45 + penalty / 2},
                "transfer_penalty_on_A": penalty, "transfer_penalty_on_B": penalty / 2,
                "mean_transfer_penalty": 0.75 * penalty,
                "resolution_status": "resolved", "preferred_control_swapped": True,
                "selected_waveform_identical": False, "objective_calls": 12,
                "physical_size_matched": True, "scale_moved": False})
    rows.append({**rows[0], "pair_id": "p8_geometry", "parent_id": "p8",
                 "resolution_status": "censored_numerical", "preferred_control_swapped": False,
                 "mean_transfer_penalty": 1e-9, "transfer_penalty_on_A": 1e-9,
                 "transfer_penalty_on_B": 1e-9,
                 "loss_matrix": {"A_on_A": 0.5, "B_on_A": 0.5, "B_on_B": 0.45, "A_on_B": 0.45}})
    rows.append({**rows[1], "pair_id": "p9_chain_strength", "parent_id": "p9",
                 "scale_arm": "scale_controlled", "mean_transfer_penalty": 0.008,
                 "transfer_penalty_on_A": 0.008, "transfer_penalty_on_B": 0.008})
    return rows


def no_type3(path):
    plot_utils = load_plot_utils()
    ok, report = plot_utils.check_fonts(path)
    assert ok, report


def test_plot_utils_is_loadable_and_sets_truetype():
    plot_utils = load_plot_utils()
    plot_utils.use_venue("neurips", "single")
    assert matplotlib.rcParams["pdf.fonttype"] == 42


def test_a_missing_plot_utils_fails_loudly_and_names_the_path(tmp_path, monkeypatch):
    monkeypatch.setenv("ANNEALCTRL_PLOT_UTILS", str(tmp_path / "absent"))
    monkeypatch.delitem(sys.modules, "plot_utils", raising=False)
    with pytest.raises(RuntimeError, match="absent"):
        load_plot_utils()


def test_frontier_figure_writes_pdf_and_png_without_type3(tmp_path):
    result = figure_frontier(frontier_rows(), tmp_path / "figure3", venue="neurips")
    pdf = tmp_path / "figure3.pdf"
    assert pdf.exists() and (tmp_path / "figure3.png").exists()
    no_type3(pdf)
    assert result["venue"] == "neurips"


def test_frontier_figure_shows_censored_records_rather_than_dropping_them(tmp_path):
    result = figure_frontier(frontier_rows(), tmp_path / "figure3")
    assert result["n_censored"] == 1
    assert result["censored_shown"] is True
    assert result["n_records"] == 6


def test_frontier_figure_draws_a_budget_sensitivity_panel(tmp_path):
    result = figure_frontier(frontier_rows(), tmp_path / "figure3")
    assert result["panels"] == ["family_restriction", "budget_sensitivity"]
    assert set(result["families"]) == {"linear", "one_window", "two_window", "eight_bin"}


def test_frontier_figure_refuses_an_empty_row_set(tmp_path):
    with pytest.raises(ValueError, match="nonempty"):
        figure_frontier([], tmp_path / "figure3")


def test_intervention_figure_writes_pdf_and_png_without_type3(tmp_path):
    figure_interventions(intervention_rows(), tmp_path / "figure4", venue="icml")
    pdf = tmp_path / "figure4.pdf"
    assert pdf.exists() and (tmp_path / "figure4.png").exists()
    no_type3(pdf)


def test_intervention_figure_separates_scale_arms(tmp_path):
    result = figure_interventions(intervention_rows(), tmp_path / "figure4")
    assert set(result["scale_arms"]) == {"total_compiled_effect", "scale_controlled"}
    assert result["scale_arms_pooled"] is False


def test_intervention_figure_shows_censored_pairs(tmp_path):
    result = figure_interventions(intervention_rows(), tmp_path / "figure4")
    assert result["n_censored"] == 1 and result["censored_shown"] is True


def test_intervention_figure_refuses_an_empty_row_set(tmp_path):
    with pytest.raises(ValueError, match="nonempty"):
        figure_interventions([], tmp_path / "figure4")


def test_an_unknown_venue_is_refused(tmp_path):
    with pytest.raises(ValueError, match="venue"):
        figure_frontier(frontier_rows(), tmp_path / "figure3", venue="neurlps")
