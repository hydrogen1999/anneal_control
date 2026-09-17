"""Figure rendering must be camera-ready: TrueType, never Type 3 (ADR-0006)."""
import sys
from pathlib import Path

import numpy as np

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


def representation_rows():
    rows = []
    for index in range(6):
        for method, excess in (("logical", 0.10), ("summary", 0.12),
                               ("physical", 0.05), ("hierarchy_physics", 0.04)):
            jitter = 0.01 * ((index % 3) - 1)
            rows.append({
                "pair_id": f"p{index}_k", "parent_id": f"p{index}", "method": method,
                "factor": "chain_strength", "scale_arm": "total_compiled_effect",
                "encoder_variant": method,
                "embedding_blind_by_construction": method == "logical",
                "identical_choice": method == "logical",
                "waveform_distance": 0.0 if method == "logical" else 0.2,
                "model_loss": {"A": 0.5, "B": 0.5},
                "excess_loss": {"A": excess + jitter, "B": excess - jitter},
                "mean_excess_loss": excess + jitter / 2,
                "preferred_control_swapped": index < 4,
                "resolution_status": "resolved" if index < 4 else "censored_numerical",
                "objective_calls": 2})
    return rows


def test_representation_figure_writes_pdf_and_png_without_type3(tmp_path):
    from annealctrl.figures import figure_representation
    result = figure_representation(representation_rows(), tmp_path / "figure5",
                                   baseline="logical", venue="neurips")
    pdf = tmp_path / "figure5.pdf"
    assert pdf.exists() and (tmp_path / "figure5.png").exists()
    no_type3(pdf)
    assert result["baseline"] == "logical"
    assert set(result["methods"]) == {"logical", "summary", "physical", "hierarchy_physics"}


def test_representation_figure_marks_which_methods_are_embedding_blind(tmp_path):
    from annealctrl.figures import figure_representation
    result = figure_representation(representation_rows(), tmp_path / "figure5", baseline="logical")
    assert set(result["blind_methods"]) == {"logical"}
    assert result["blindness_shown"] is True


def test_representation_figure_can_restrict_to_swap_pairs(tmp_path):
    from annealctrl.figures import figure_representation
    everything = figure_representation(representation_rows(), tmp_path / "a", baseline="logical")
    swaps = figure_representation(representation_rows(), tmp_path / "b", baseline="logical",
                                  swap_pairs_only=True)
    assert everything["n_pairs_per_method"] == 6
    assert swaps["n_pairs_per_method"] == 4
    assert swaps["restricted_to_swap_pairs"] is True


def test_representation_figure_refuses_an_empty_row_set(tmp_path):
    from annealctrl.figures import figure_representation
    with pytest.raises(ValueError, match="nonempty"):
        figure_representation([], tmp_path / "figure5")


# --- reproducibility on a clean machine --------------------------------------

def test_falls_back_to_the_vendored_copy_when_upstream_is_absent(tmp_path, monkeypatch):
    """A clean checkout with no research-os must still render camera-ready figures."""
    from annealctrl import figures

    monkeypatch.delenv("ANNEALCTRL_PLOT_UTILS", raising=False)
    monkeypatch.setattr(figures, "DEFAULT_PLOT_UTILS", tmp_path / "no-research-os")
    monkeypatch.delitem(sys.modules, "plot_utils", raising=False)

    assert figures.plot_utils_source() == figures.VENDORED_PLOT_UTILS
    module = figures.load_plot_utils()
    assert module.__annealctrl_vendored__ is True
    # The whole reason the module exists: TrueType, never Type 3.
    module.use_venue("neurips", "single")
    assert matplotlib.rcParams["pdf.fonttype"] == 42
    assert matplotlib.rcParams["ps.fonttype"] == 42


def test_an_explicit_pointer_is_never_silently_replaced_by_the_vendored_copy(tmp_path, monkeypatch):
    """Falling back on an explicit override would render figures with the wrong module."""
    from annealctrl import figures

    monkeypatch.setenv("ANNEALCTRL_PLOT_UTILS", str(tmp_path / "absent"))
    monkeypatch.delitem(sys.modules, "plot_utils", raising=False)
    with pytest.raises(RuntimeError, match="absent"):
        figures.plot_utils_source()


def test_upstream_wins_over_the_vendored_copy_when_both_exist(tmp_path, monkeypatch):
    from annealctrl import figures

    upstream = tmp_path / "upstream"
    upstream.mkdir()
    (upstream / "plot_utils.py").write_text("")
    monkeypatch.delenv("ANNEALCTRL_PLOT_UTILS", raising=False)
    monkeypatch.setattr(figures, "DEFAULT_PLOT_UTILS", upstream)
    assert figures.plot_utils_source() == upstream


def test_the_vendored_copy_records_its_upstream_provenance():
    """A vendored file with no recorded origin cannot be audited or re-synced."""
    from annealctrl import figures

    header = (figures.VENDORED_PLOT_UTILS / "plot_utils.py").read_text()[:1200]
    assert "Upstream:" in header
    assert "sha256" in header
    assert "Do not edit this copy" in header


# --- the teacher baseline figure ---------------------------------------------

def teacher_summary():
    def block(name, mean, rate, per_runtime):
        return {"method": name, "mean_loss": mean, "mean_linear_loss": 0.70,
                "mean_best_found_loss": 0.61, "resolution_rate": rate, "n_rows": 3447,
                "n_resolved": int(3447 * rate),
                "by_runtime": {
                    runtime: {"vs_linear": {
                        "mean_difference": difference,
                        "parent_bootstrap_ci": {"low": difference - 0.01, "high": difference + 0.01}}}
                    for runtime, difference in per_runtime.items()}}
    return {"methods": {
        "gap_inverse_square": block("gap_inverse_square", 0.7159, 0.447,
                                    {"1.0": 0.0063, "4.0": 0.0625, "12.0": -0.0001}),
        "d2": block("d2", 0.5569, 0.981, {"1.0": 0.0057, "4.0": 0.0232, "12.0": -0.0341})}}


def test_teacher_baseline_figure_renders_truetype(tmp_path):
    from annealctrl.figures import figure_teacher_baselines

    result = figure_teacher_baselines(teacher_summary(), tmp_path / "figure_teachers")
    assert result["methods"] == ["d2", "gap_inverse_square"]
    assert any(str(path).endswith(".pdf") for path in result["files"])
    for path in result["files"]:
        assert Path(path).exists()


def test_teacher_baseline_figure_refuses_when_nothing_resolved():
    from annealctrl.figures import figure_teacher_baselines

    empty = {"methods": {"gap_inverse_square": {"mean_loss": None, "status": "no_audited_waveforms"}}}
    with pytest.raises(ValueError, match="at least one resolved"):
        figure_teacher_baselines(empty, "unused")


def test_teacher_figure_never_compares_across_populations(tmp_path, monkeypatch):
    """Each baseline must be drawn against its own linear/search, not a shared one."""
    from annealctrl import figures

    summary = teacher_summary()
    # Give the two baselines deliberately different references; if the figure
    # shared one, the drawn bars could not reproduce both.
    summary["methods"]["d2"]["mean_linear_loss"] = 0.5667
    summary["methods"]["d2"]["mean_best_found_loss"] = 0.4641
    summary["methods"]["d2"]["n_audit_passed"] = 2946
    summary["methods"]["gap_inverse_square"]["mean_linear_loss"] = 0.7013
    summary["methods"]["gap_inverse_square"]["mean_best_found_loss"] = 0.6116
    summary["methods"]["gap_inverse_square"]["n_audit_passed"] = 1381

    drawn = []
    real_bar = None

    def record_bar(self, x, height, **kwargs):
        drawn.extend(float(v) for v in np.atleast_1d(height))
        return real_bar(self, x, height, **kwargs)

    import matplotlib.axes
    real_bar = matplotlib.axes.Axes.bar
    monkeypatch.setattr(matplotlib.axes.Axes, "bar", record_bar)
    figures.figure_teacher_baselines(summary, tmp_path / "fig")

    for expected in (0.5667, 0.4641, 0.7013, 0.6116):
        assert any(abs(value - expected) < 1e-9 for value in drawn), (expected, drawn)
