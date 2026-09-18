"""Does the control advantage survive decoherence, or is it an artefact of a closed system?

Every loss in this project comes from unitary evolution. A reviewer's first
question about that is whether the ranking of controls is an artefact of
simulating no environment at all. This asks the question with the independent
Lindblad solver already in `adapters`, on the same records and the same
waveforms, and reports the answer whichever way it comes out.
"""
import numpy as np
import pytest

from annealctrl.open_system import open_loss, robustness_sweep


def tiny_record(n=3):
    """A small ferromagnetic chain with a one-to-one embedding."""
    edges = [[i, i + 1] for i in range(n - 1)]
    return {
        "record_id": "r0", "parent_id": "p0", "runtime": 1.0, "split": "test",
        "logical_h": np.zeros(n), "logical_edges": edges, "logical_J": -np.ones(len(edges)),
        "physical_h": np.zeros(n), "physical_edges": edges, "physical_J": -np.ones(len(edges)),
        "problem_J": -np.ones(len(edges)), "chain_J": np.zeros(0),
        "membership": np.arange(n), "programmed_scale": 1.0, "chain_strength": 1.0,
        "energy_scale": 1.0, "catalyst_strength": 0.0, "logical_n": n, "physical_n": n,
    }


# --- the loss ----------------------------------------------------------------

def test_open_loss_at_zero_noise_is_a_closed_system_loss():
    from annealctrl.benchmarking import score_schedule
    from annealctrl.schedules import Schedule

    record, schedule = tiny_record(), Schedule.linear()
    closed = score_schedule(record, schedule, tolerance=1e-6, max_steps=65536)["loss"]
    opened = open_loss(record, schedule, dephasing_rate=0.0)
    assert opened == pytest.approx(closed, abs=2e-3)


def test_dephasing_never_improves_the_loss_of_a_good_control():
    from annealctrl.schedules import Schedule

    record = tiny_record()
    clean = open_loss(record, Schedule.linear(), dephasing_rate=0.0)
    noisy = open_loss(record, Schedule.linear(), dephasing_rate=0.5)
    assert noisy >= clean - 1e-9


def test_open_loss_refuses_a_system_too_large_to_hold()  :
    from annealctrl.schedules import Schedule

    with pytest.raises(ValueError, match="density matrix"):
        open_loss(tiny_record(3), Schedule.linear(), dephasing_rate=0.0, max_qubits=2)


def test_open_loss_refuses_a_negative_rate():
    from annealctrl.schedules import Schedule

    with pytest.raises(ValueError, match="nonnegative"):
        open_loss(tiny_record(), Schedule.linear(), dephasing_rate=-0.1)


# --- the sweep ---------------------------------------------------------------

def test_sweep_compares_the_same_waveforms_at_every_rate():
    from annealctrl.schedules import Schedule

    record = tiny_record()
    waveforms = {"linear": Schedule.linear(),
                 "pause": Schedule(np.array([0.0, 0.4, 0.6, 1.0]), np.array([0.0, 0.45, 0.45, 1.0]))}
    result = robustness_sweep([record], waveforms, rates=[0.0, 0.2], max_qubits=6)
    assert result["n_records"] == 1
    assert sorted(result["rates"]) == [0.0, 0.2]
    for rate in ("0.0", "0.2"):
        assert set(result["by_rate"][rate]["mean_loss"]) == {"linear", "pause"}


def test_sweep_reports_whether_the_ordering_survives():
    from annealctrl.schedules import Schedule

    record = tiny_record()
    waveforms = {"linear": Schedule.linear(),
                 "pause": Schedule(np.array([0.0, 0.4, 0.6, 1.0]), np.array([0.0, 0.45, 0.45, 1.0]))}
    result = robustness_sweep([record], waveforms, rates=[0.0, 0.2], max_qubits=6)
    assert isinstance(result["ordering_preserved"], bool)
    assert result["best_at_each_rate"]["0.0"] in waveforms


def test_sweep_refuses_an_empty_waveform_set():
    with pytest.raises(ValueError, match="at least two"):
        robustness_sweep([tiny_record()], {}, rates=[0.0], max_qubits=6)


def test_sweep_refuses_rates_without_zero():
    """Without the noiseless point there is nothing to measure degradation against."""
    from annealctrl.schedules import Schedule

    waveforms = {"linear": Schedule.linear(),
                 "pause": Schedule(np.array([0.0, 0.4, 0.6, 1.0]), np.array([0.0, 0.45, 0.45, 1.0]))}
    with pytest.raises(ValueError, match="zero"):
        robustness_sweep([tiny_record()], waveforms, rates=[0.2], max_qubits=6)


# --- does a selection rule trained without an environment still choose well? --

def banked_record(record_id="r0", parent_id="p0", n=3):
    """A record carrying a small bank, so a selection rule has something to pick from."""
    record = tiny_record(n)
    record["record_id"], record["parent_id"] = record_id, parent_id
    grid = np.linspace(0.0, 1.0, 9)
    record["candidate_schedules"] = np.stack([
        grid,                       # linear
        grid**2,                    # slow start
        np.sqrt(grid),              # fast start
        np.clip(1.6 * grid - 0.3, 0, 1)])  # a pause-like ramp
    return record


def test_a_selector_that_picks_the_noisy_best_has_no_regret():
    from annealctrl.open_system import selection_robustness

    records = [banked_record()]
    # An oracle that sees the noise: the sweep must score it as zero regret.
    result = selection_robustness(records, select=lambda r: None, rates=[0.0, 0.05],
                                  oracle_selects=True)
    for rate in ("0.0", "0.05"):
        assert result["by_rate"][rate]["selection_regret"]["mean"] == pytest.approx(0.0, abs=1e-12)


def test_a_deliberately_bad_selector_shows_positive_regret():
    from annealctrl.open_system import selection_robustness

    records = [banked_record()]
    worst = selection_robustness(records, select=lambda r: 3, rates=[0.0])
    best = selection_robustness(records, select=lambda r: 0, rates=[0.0])
    assert worst["by_rate"]["0.0"]["selection_regret"]["mean"] >= 0.0
    assert (worst["by_rate"]["0.0"]["selection_regret"]["mean"]
            >= best["by_rate"]["0.0"]["selection_regret"]["mean"])


def test_the_linear_reference_is_the_candidate_closest_to_the_identity_ramp():
    from annealctrl.open_system import selection_robustness

    result = selection_robustness([banked_record()], select=lambda r: 1, rates=[0.0])
    assert result["rows"][0]["linear_index"] == 0


def test_advantage_against_linear_is_reported_at_every_rate_with_a_parent_ci():
    from annealctrl.open_system import selection_robustness

    records = [banked_record(f"r{i}", f"p{i}") for i in range(3)]
    result = selection_robustness(records, select=lambda r: 1, rates=[0.0, 0.05, 0.1])
    assert sorted(result["by_rate"]) == ["0.0", "0.05", "0.1"]
    for block in result["by_rate"].values():
        assert block["advantage_vs_linear"]["parent_bootstrap_ci"]["unit_of_independence"] \
            == "logical_parent"
        assert block["n_parents"] == 3


def test_selection_robustness_refuses_rates_without_zero():
    from annealctrl.open_system import selection_robustness

    with pytest.raises(ValueError, match="zero"):
        selection_robustness([banked_record()], select=lambda r: 0, rates=[0.05])


def test_selection_robustness_refuses_a_record_without_a_bank():
    from annealctrl.open_system import selection_robustness

    with pytest.raises(ValueError, match="candidate_schedules"):
        selection_robustness([tiny_record()], select=lambda r: 0, rates=[0.0])


def test_selection_robustness_refuses_an_out_of_range_choice():
    from annealctrl.open_system import selection_robustness

    with pytest.raises(ValueError, match="index"):
        selection_robustness([banked_record()], select=lambda r: 99, rates=[0.0])


def test_a_precomputed_table_gives_exactly_the_same_numbers():
    """The saving is only legitimate if it changes nothing it measures."""
    from annealctrl.open_system import bank_loss_table, selection_robustness

    records = [banked_record(f"r{i}", f"p{i}") for i in range(2)]
    rates = [0.0, 0.05]
    direct = selection_robustness(records, select=lambda r: 2, rates=rates)
    table = bank_loss_table(records, rates=rates)
    reused = selection_robustness(records, select=lambda r: 2, rates=rates, table=table)

    assert reused["reused_precomputed_table"] is True
    for rate in ("0.0", "0.05"):
        for name in ("advantage_vs_linear", "selection_regret", "selected_loss", "linear_loss"):
            assert reused["by_rate"][rate][name]["mean"] == \
                pytest.approx(direct["by_rate"][rate][name]["mean"], abs=0.0, rel=0.0)


def test_the_table_is_shared_across_selectors_without_resolving():
    from annealctrl.open_system import bank_loss_table, selection_robustness

    records = [banked_record()]
    table = bank_loss_table(records, rates=[0.0])
    a = selection_robustness(records, select=lambda r: 1, rates=[0.0], table=table)
    b = selection_robustness(records, select=lambda r: 3, rates=[0.0], table=table)
    # Different picks, identical underlying bank losses.
    assert a["rows"][0]["linear_loss"] == b["rows"][0]["linear_loss"]
    assert a["rows"][0]["best_loss_at_rate"] == b["rows"][0]["best_loss_at_rate"]
    assert a["rows"][0]["selected_loss"] != b["rows"][0]["selected_loss"]


def test_a_table_built_for_a_different_bank_is_refused():
    from annealctrl.open_system import bank_loss_table, selection_robustness

    records = [banked_record()]
    table = bank_loss_table(records, rates=[0.0])
    table["losses"]["r0"]["0.0"] = table["losses"]["r0"]["0.0"][:2]
    with pytest.raises(ValueError, match="bank has"):
        selection_robustness(records, select=lambda r: 1, rates=[0.0], table=table)


def test_the_table_refuses_rates_without_zero():
    from annealctrl.open_system import bank_loss_table

    with pytest.raises(ValueError, match="zero"):
        bank_loss_table([banked_record()], rates=[0.05])
