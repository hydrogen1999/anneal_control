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
