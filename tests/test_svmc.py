"""A semiclassical surrogate, and the only reason to trust one: it is checked where truth is known.

Exact state-vector dynamics caps around 30-40 qubits anywhere; a deployed
annealer has thousands. Spin-vector Monte Carlo (Shin, Smith, Smolin and
Vazirani, 2014) replaces each qubit with a classical O(2) rotor and costs
O(N x sweeps), so it reaches the sizes that matter. It is not quantum, and a
number produced by it is a claim about the surrogate until the surrogate is
shown to agree with exact dynamics somewhere.

That overlap is the point of this module. The tests pin the physics of the rotor
energy and the agreement protocol; the campaign then validates on the 10-14 qubit
records where exact outcomes already exist, and only afterwards runs at sizes
where they cannot.
"""
import numpy as np
import pytest

from annealctrl.svmc import rotor_energy, svmc_success


def ferromagnet(n=4):
    from annealctrl.physics import HamiltonianTerms
    edges = [[i, i + 1] for i in range(n - 1)]
    return HamiltonianTerms(n, np.zeros(n), edges, -np.ones(len(edges)))


# --- the rotor energy --------------------------------------------------------

def test_energy_at_theta_zero_is_the_classical_ising_energy():
    """cos(0) = 1, so every rotor is the +1 spin and only the Z terms survive."""
    from annealctrl.physics import AnnealPath

    terms = ferromagnet(4)
    theta = np.zeros(4)
    a, b, c = AnnealPath().coefficients(1.0)
    energy = rotor_energy(theta, terms, a, b, c)
    # All spins +1 on a ferromagnet with J = -1: three bonds at -1 each.
    assert energy == pytest.approx(b * -3.0)


def test_transverse_term_is_lowest_when_every_rotor_lies_in_plane():
    from annealctrl.physics import AnnealPath

    terms = ferromagnet(4)
    a, b, c = AnnealPath().coefficients(0.0)          # pure transverse
    flat = rotor_energy(np.full(4, np.pi / 2), terms, a, b, c)
    upright = rotor_energy(np.zeros(4), terms, a, b, c)
    assert flat < upright
    assert flat == pytest.approx(-a * 4.0)


def test_energy_is_invariant_to_a_global_spin_flip_without_fields():
    from annealctrl.physics import AnnealPath

    terms = ferromagnet(4)
    a, b, c = AnnealPath().coefficients(1.0)
    theta = np.array([0.3, 1.1, 2.2, 0.7])
    assert rotor_energy(theta, terms, a, b, c) == pytest.approx(
        rotor_energy(np.pi - theta, terms, a, b, c))


# --- the sampler -------------------------------------------------------------

def frustrated(n, seed):
    """A fully connected +-1 glass. Frustration is what makes the schedule matter."""
    from annealctrl.physics import HamiltonianTerms

    rng = np.random.default_rng(seed)
    edges = [[i, j] for i in range(n) for j in range(i + 1, n)]
    return HamiltonianTerms(n, rng.normal(0, 0.3, n), edges,
                            rng.choice([-1.0, 1.0], size=len(edges)))


def exact_ground_states(terms):
    import itertools

    best, states = None, []
    for bits in itertools.product([1.0, -1.0], repeat=terms.n_qubits):
        z = np.array(bits)
        energy = float(np.asarray(terms.h) @ z)
        for (i, j), w in zip(terms.zz_edges, terms.zz_weights):
            energy += w * z[i] * z[j]
        if best is None or energy < best - 1e-9:
            best, states = energy, [z]
        elif abs(energy - best) < 1e-9:
            states.append(z)
    return np.array(states)


def test_a_slow_schedule_beats_an_instant_quench_when_the_problem_is_frustrated():
    """If the surrogate cannot tell a good schedule from a bad one it is useless here."""
    from annealctrl.physics import AnnealPath
    from annealctrl.schedules import Schedule

    quench = Schedule(np.array([0.0, 0.02, 1.0]), np.array([0.0, 1.0, 1.0]))
    wins = 0
    for n, seed in ((8, 0), (8, 1), (10, 0)):
        terms = frustrated(n, seed)
        ground = exact_ground_states(terms)
        kwargs = dict(ground_states=ground, steps=60, sweeps=2, restarts=300,
                      temperature=0.2, seed=0)
        slow = svmc_success(terms, Schedule.linear(), AnnealPath(), **kwargs)["success"]
        fast = svmc_success(terms, quench, AnnealPath(), **kwargs)["success"]
        wins += slow > fast
    assert wins == 3, f"schedule sensitivity on only {wins} of 3 frustrated instances"


def test_an_unfrustrated_ferromagnet_cannot_discriminate_schedules():
    """Recorded so the validation is never run on an instance that cannot answer.

    With no frustration there is no small-gap bottleneck to traverse: quenching
    straight to s=1 and relaxing reaches the aligned ground state, and the slow
    schedule has nothing to buy. The first version of the test above used a
    ferromagnet and failed for this reason, which is a property of the instance
    rather than of the surrogate.
    """
    from annealctrl.physics import AnnealPath
    from annealctrl.schedules import Schedule

    terms = ferromagnet(6)
    ground = np.array([[1.0] * 6, [-1.0] * 6])
    kwargs = dict(ground_states=ground, steps=200, sweeps=6, restarts=200,
                  temperature=0.05, seed=0)
    slow = svmc_success(terms, Schedule.linear(), AnnealPath(), **kwargs)["success"]
    fast = svmc_success(terms, Schedule(np.array([0.0, 0.02, 1.0]),
                                        np.array([0.0, 1.0, 1.0])), AnnealPath(), **kwargs)["success"]
    assert slow == pytest.approx(1.0) and fast == pytest.approx(1.0)


def test_success_is_a_probability_and_restarts_are_reported():
    from annealctrl.physics import AnnealPath
    from annealctrl.schedules import Schedule

    terms = ferromagnet(4)
    out = svmc_success(terms, Schedule.linear(), AnnealPath(),
                       ground_states=np.array([[1.0] * 4, [-1.0] * 4]),
                       steps=50, sweeps=4, restarts=64, temperature=0.05, seed=1)
    assert 0.0 <= out["success"] <= 1.0
    assert out["restarts"] == 64
    assert out["n_qubits"] == 4


def test_the_sampler_is_deterministic_given_a_seed():
    from annealctrl.physics import AnnealPath
    from annealctrl.schedules import Schedule

    terms = ferromagnet(5)
    kwargs = dict(ground_states=np.array([[1.0] * 5, [-1.0] * 5]), steps=40, sweeps=4,
                  restarts=32, temperature=0.05, seed=7)
    a = svmc_success(terms, Schedule.linear(), AnnealPath(), **kwargs)
    b = svmc_success(terms, Schedule.linear(), AnnealPath(), **kwargs)
    assert a["success"] == b["success"]


def test_cost_is_linear_in_qubits_not_exponential():
    """The whole reason this exists: O(N) per sweep, so thousands of spins are reachable."""
    from annealctrl.svmc import svmc_cost_estimate

    small = svmc_cost_estimate(n_qubits=20, steps=100, sweeps=10, restarts=100)
    large = svmc_cost_estimate(n_qubits=2000, steps=100, sweeps=10, restarts=100)
    assert large["spin_updates"] == 100 * small["spin_updates"]
    assert large["state_bytes"] < 2e6          # 2000 spins x 100 restarts x 8 bytes


def test_sampler_refuses_a_nonpositive_temperature():
    from annealctrl.physics import AnnealPath
    from annealctrl.schedules import Schedule

    with pytest.raises(ValueError, match="temperature"):
        svmc_success(ferromagnet(4), Schedule.linear(), AnnealPath(),
                     ground_states=np.array([[1.0] * 4]), steps=10, sweeps=2,
                     restarts=4, temperature=0.0, seed=0)
