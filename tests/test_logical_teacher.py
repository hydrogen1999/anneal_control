"""Does a schedule built from the logical spectrum control the embedded system?"""
import numpy as np
import pytest

from annealctrl.logical_teacher import compare_spectra, logical_spectrum_schedule, logical_terms


def chain_record(n_logical=3, chain=2):
    """A logical chain embedded with `chain` physical qubits per logical one."""
    logical_edges = [[i, i + 1] for i in range(n_logical - 1)]
    membership = np.repeat(np.arange(n_logical), chain)
    n_phys = n_logical * chain
    physical_edges, physical_J = [], []
    for i in range(n_logical):                      # intra-chain ferromagnetic links
        for k in range(chain - 1):
            physical_edges.append([i * chain + k, i * chain + k + 1])
            physical_J.append(-2.0)
    for a, b in logical_edges:                      # one inter-chain link per logical edge
        physical_edges.append([a * chain, b * chain])
        physical_J.append(0.7)
    return {
        "record_id": "r0", "parent_id": "p0", "runtime": 1.0, "split": "test",
        "logical_h": np.full(n_logical, 0.3), "logical_edges": logical_edges,
        "logical_J": np.full(len(logical_edges), 0.7),
        "physical_h": np.full(n_phys, 0.15), "physical_edges": physical_edges,
        "physical_J": np.asarray(physical_J, dtype=float),
        "problem_J": np.full(len(logical_edges), 0.7),
        "chain_J": np.full(n_logical * (chain - 1), -2.0),
        "membership": membership, "programmed_scale": 1.0, "chain_strength": 2.0,
        "energy_scale": 1.0, "catalyst_strength": 0.0,
        "logical_n": n_logical, "physical_n": n_phys,
    }


def identity_record(n=3):
    """A one-to-one embedding: the logical and physical Hamiltonians coincide."""
    edges = [[i, i + 1] for i in range(n - 1)]
    J = np.full(len(edges), 0.7)
    h = np.full(n, 0.3)
    return {
        "record_id": "r1", "parent_id": "p1", "runtime": 1.0, "split": "test",
        "logical_h": h, "logical_edges": edges, "logical_J": J,
        "physical_h": h, "physical_edges": edges, "physical_J": J,
        "problem_J": J, "chain_J": np.zeros(0), "membership": np.arange(n),
        "programmed_scale": 1.0, "chain_strength": 1.0, "energy_scale": 1.0,
        "catalyst_strength": 0.0, "logical_n": n, "physical_n": n,
    }


# --- the terms ---------------------------------------------------------------

def test_logical_terms_carry_no_chain_and_no_physical_qubits():
    record = chain_record()
    terms = logical_terms(record)
    assert terms.n_qubits == 3                      # not the six physical ones
    assert terms.zz_weights.shape[0] == 2           # the two logical edges only


def test_logical_terms_refuse_mismatched_edges_and_couplings():
    record = chain_record()
    record["logical_J"] = np.full(5, 0.7)
    with pytest.raises(ValueError, match="agree in length"):
        logical_terms(record)


# --- the decisive correctness check ------------------------------------------

def test_a_one_to_one_embedding_makes_both_spectra_give_the_same_schedule():
    """With no chains the two Hamiltonians coincide, so the schedules must too.

    This is what separates a real logical-spectrum path from a bug that quietly
    rebuilds the physical one.
    """
    from annealctrl.benchmarking import record_physics
    from annealctrl.physics_baselines import exact_teacher_baseline

    record = identity_record()
    terms, path, _ = record_physics(record)
    physical = exact_teacher_baseline(terms, "d2", runtime=1.0, max_slope=4.0, path=path)
    logical = logical_spectrum_schedule(record, method="d2", max_slope=4.0)
    assert physical.schedule is not None and logical.schedule is not None
    assert logical.schedule.s_knots == pytest.approx(physical.schedule.s_knots, abs=1e-9)
    assert logical.schedule.tau_knots == pytest.approx(physical.schedule.tau_knots, abs=1e-9)


def test_chains_make_the_two_spectra_disagree():
    """If they agreed here the embedding would carry no spectral information."""
    from annealctrl.benchmarking import record_physics
    from annealctrl.physics_baselines import exact_teacher_baseline

    record = chain_record()
    terms, path, _ = record_physics(record)
    physical = exact_teacher_baseline(terms, "d2", runtime=1.0, max_slope=4.0, path=path)
    logical = logical_spectrum_schedule(record, method="d2", max_slope=4.0)
    assert physical.schedule is not None and logical.schedule is not None
    grid = np.linspace(0.0, 1.0, 65)
    assert np.abs(logical.schedule(grid) - physical.schedule(grid)).max() > 1e-6


# --- the comparison ----------------------------------------------------------

def test_comparison_scores_both_schedules_on_the_same_embedded_record():
    result = compare_spectra(chain_record(), method="d2")
    assert result["n_logical"] == 3 and result["n_physical"] == 6
    for key in ("physical_spectrum_loss", "logical_spectrum_loss", "linear_loss"):
        assert result[key] is not None and 0.0 <= result[key] <= 1.0
    assert result["logical_minus_physical"] == pytest.approx(
        result["logical_spectrum_loss"] - result["physical_spectrum_loss"])


def test_an_unresolved_teacher_is_reported_not_replaced():
    result = compare_spectra(identity_record(), method="gap_inverse_square")
    # Whatever the status, a missing schedule must surface as None rather than a fallback.
    if result["logical_spectrum_loss"] is None:
        assert result["logical_minus_physical"] is None
        assert result["logical_status"] != "sampled_point_audit_passed"


def scaled_identity_record(scale=0.35):
    """A one-to-one embedding whose physical coefficients carry a programmed scale.

    This is what the real dataset looks like, and the earlier synthetic
    identity_record (scale 1.0) could not have caught the confound.
    """
    record = identity_record()
    record["physical_h"] = np.asarray(record["logical_h"], dtype=float) * scale
    record["physical_J"] = np.asarray(record["logical_J"], dtype=float) * scale
    record["problem_J"] = np.asarray(record["logical_J"], dtype=float) * scale
    record["programmed_scale"] = scale
    return record


def test_the_programmed_scale_is_applied_so_a_one_to_one_embedding_still_matches():
    """Without this the comparison confounds re-embedding with common rescaling.

    On the real dataset every one-to-one record with programmed_scale == 1
    reproduced the physical schedule exactly and every one with a smaller scale
    did not, by up to 0.034 in loss. The scale, not the embedding, was moving.
    """
    from annealctrl.benchmarking import record_physics
    from annealctrl.physics_baselines import exact_teacher_baseline

    record = scaled_identity_record()
    terms, path, _ = record_physics(record)
    physical = exact_teacher_baseline(terms, "d2", runtime=1.0, max_slope=4.0, path=path)
    logical = logical_spectrum_schedule(record, method="d2", max_slope=4.0)
    assert physical.schedule is not None and logical.schedule is not None
    assert logical.schedule.s_knots == pytest.approx(physical.schedule.s_knots, abs=1e-9)
    assert logical.schedule.tau_knots == pytest.approx(physical.schedule.tau_knots, abs=1e-9)


def test_the_raw_scale_option_reproduces_the_confound_it_is_named_for():
    """'raw' is kept because it answers a different question; it must differ."""
    from annealctrl.benchmarking import record_physics
    from annealctrl.physics_baselines import exact_teacher_baseline

    record = scaled_identity_record()
    terms, path, _ = record_physics(record)
    physical = exact_teacher_baseline(terms, "d2", runtime=1.0, max_slope=4.0, path=path)
    raw = logical_spectrum_schedule(record, method="d2", max_slope=4.0, scale="raw")
    grid = np.linspace(0.0, 1.0, 65)
    assert np.abs(raw.schedule(grid) - physical.schedule(grid)).max() > 1e-6


def test_an_unknown_scale_is_refused():
    with pytest.raises(ValueError, match="programmed.*raw"):
        logical_terms(chain_record(), scale="physical")
