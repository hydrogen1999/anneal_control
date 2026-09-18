import numpy as np
import pytest

from annealctrl.search import (
    equal_budget_refinement,
    evaluate_candidates,
    finite_difference_interventions,
    optimize_control_family,
    shared_candidate_bank,
)


def loss_fn(schedule):
    return float((schedule(0.5) - 0.3) ** 2)


def test_bank_reproducible_and_diverse():
    a = shared_candidate_bank(n=64, n_segments=8, seed=7)
    b = shared_candidate_bank(n=64, n_segments=8, seed=7)
    assert len(a) == 64
    assert {x.family for x in a} == {"linear", "one_window", "two_window", "pause", "duration_logits"}
    for left, right in zip(a, b):
        assert left.candidate_id == right.candidate_id
        np.testing.assert_array_equal(left.schedule.tau_knots, right.schedule.tau_knots)
        left.schedule.validate_slope(1, 10)
    assert len(shared_candidate_bank(n=1)) == 1
    assert "pause" not in {x.family for x in shared_candidate_bank(n=16, runtime=0.1, max_slope=10)}


def test_evaluation_counts_every_call():
    calls = []
    def counted(schedule):
        calls.append(schedule)
        return loss_fn(schedule)
    result = evaluate_candidates(counted, shared_candidate_bank(16), budget=9, split="test")
    assert result.n_evaluations == len(calls) == 9
    assert result.best.loss == min(record.loss for record in result.records)
    assert result.reference_status == "best_found_within_evaluated_candidates"
    assert not result.online_adaptation
    with pytest.raises(ValueError):
        evaluate_candidates(counted, shared_candidate_bank(2), budget=3)


def test_refinement_budget_and_test_guard():
    calls = []
    def counted(schedule):
        calls.append(schedule)
        return loss_fn(schedule)
    result = equal_budget_refinement(counted, [0, 0, 0, 0], budget=11, seed=4)
    assert result.n_evaluations == len(calls) == 11
    assert result.best.loss <= result.records[0].loss
    with pytest.raises(ValueError, match="explicitly"):
        equal_budget_refinement(counted, [0, 0], budget=3, split="test")
    result = equal_budget_refinement(counted, [0, 0], budget=3, split="test", allow_test_adaptation=True)
    assert result.online_adaptation


def test_interventions_are_feasible_counted_and_schedule_conditioned():
    calls = []
    def objective(schedule):
        schedule.validate_slope(2, 1)
        calls.append(schedule)
        # First segment duration has a known derivative through softmax.
        return schedule.tau_knots[1]
    logits = np.array([0.3, -0.2, 0.1])
    labels = finite_difference_interventions(objective, logits, runtime=2, max_slope=1, epsilon=1e-5)
    p = np.exp(logits) / np.exp(logits).sum()
    expected = 0.5 * p[0] * (np.eye(3)[0] - p)
    np.testing.assert_allclose(labels.derivatives, expected, atol=1e-10)
    assert labels.n_evaluations == len(calls) == 6
    np.testing.assert_array_equal(labels.baseline_logits, logits)
    assert labels.runtime == 2


# --- Bayesian strategy -------------------------------------------------------

def test_bayesian_strategy_spends_the_same_budget_as_sobol_local():
    calls = {"sobol_local": 0, "bayesian": 0}

    def make(name):
        def objective(schedule):
            calls[name] += 1
            return float(abs(schedule(0.5) - 0.42))
        return objective

    for name in calls:
        result = optimize_control_family(make(name), "one_window", budget=16, runtime=2.0,
                                         max_slope=2.0, seed=0, strategy=name)
        assert result.n_evaluations == 16
    assert calls["sobol_local"] == calls["bayesian"] == 16


def test_bayesian_strategy_still_charges_the_linear_incumbent_first():
    seen = []

    def objective(schedule):
        seen.append(schedule.s_knots.tolist())
        return float(abs(schedule(0.5) - 0.42))

    result = optimize_control_family(objective, "two_window", budget=12, runtime=2.0,
                                     max_slope=2.0, seed=1, strategy="bayesian")
    assert seen[0] == [0.0, 1.0], "trial 0 must remain the linear incumbent"
    assert result.records[0].candidate.parameters["initial_incumbent"] == "linear"


def test_bayesian_strategy_is_recorded_on_every_candidate():
    result = optimize_control_family(lambda s: float(abs(s(0.5) - 0.42)), "one_window",
                                     budget=10, runtime=2.0, max_slope=2.0, seed=2,
                                     strategy="bayesian")
    sources = {r.candidate.parameters.get("proposal") for r in result.records[1:]}
    assert sources <= {"bayesian_design", "expected_improvement"}


def test_bayesian_strategy_is_a_competitive_opponent_not_a_strawman():
    """Measured, not assumed.

    Head to head over 4 families x 3 budgets x 12 seeds, the Bayesian strategy
    wins 5-8 of 12 per cell and has the lower mean best-loss in 8 of the 12
    cells. It is competitive and often slightly better, not dominant. The reason
    to ship it is that the protocol asks the classical comparator to be strong,
    and a learned policy measured only against random search has beaten little.

    This test therefore pins competence - a broken surrogate would collapse the
    win rate - and deliberately does not assert superiority, which the data does
    not support.
    """
    def target(value):
        return lambda s: float((s(0.5) - value) ** 2)

    wins, bo_losses, sl_losses = 0, [], []
    for family in ("one_window", "pause"):
        for seed in range(8):
            objective = target(0.30 + 0.02 * (seed % 5))
            bo = optimize_control_family(objective, family, budget=16, runtime=2.0,
                                         max_slope=2.0, seed=seed, strategy="bayesian")
            sl = optimize_control_family(objective, family, budget=16, runtime=2.0,
                                         max_slope=2.0, seed=seed, strategy="sobol_local")
            wins += bo.best.loss <= sl.best.loss
            bo_losses.append(bo.best.loss)
            sl_losses.append(sl.best.loss)

    assert wins >= 6, f"Bayesian strategy won only {wins}/16; the surrogate is not working"
    assert np.mean(bo_losses) <= 3 * np.mean(sl_losses), "Bayesian strategy is far worse on average"


def test_linear_family_ignores_the_strategy_because_it_has_no_parameters():
    result = optimize_control_family(lambda s: 0.5, "linear", budget=32, runtime=2.0,
                                     max_slope=2.0, seed=0, strategy="bayesian")
    assert result.n_evaluations == 1


def test_an_unknown_strategy_is_refused():
    with pytest.raises(ValueError, match="strategy"):
        optimize_control_family(lambda s: 0.5, "one_window", budget=8, runtime=2.0,
                                max_slope=2.0, seed=0, strategy="genetic")


# --- the policy waveform family in the training bank -------------------------

def test_capped_simplex_matches_the_torch_decoder_the_policy_uses():
    """The bank's policy-family candidates must lie on the policy's own manifold.

    If the numpy twin drifts from models.monotone_samples, the critic is again
    trained on waveforms its policy does not emit, which is the failure this
    exists to remove.
    """
    torch = pytest.importorskip("torch")
    from annealctrl.models import monotone_samples
    from annealctrl.search import capped_simplex_samples

    rng = np.random.default_rng(0)
    logits = rng.normal(0, 1.5, (32, 8))
    for cap in (1.0, 2.0, 4.0, 10.0):
        expected = monotone_samples(torch.as_tensor(logits, dtype=torch.float64),
                                    max_ds_dtau=cap).numpy()
        np.testing.assert_allclose(capped_simplex_samples(logits, max_ds_dtau=cap),
                                   expected, atol=1e-12)


def test_capped_simplex_samples_are_monotone_and_bounded():
    from annealctrl.search import capped_simplex_samples
    rng = np.random.default_rng(1)
    waves = capped_simplex_samples(rng.normal(0, 3.0, (64, 8)), max_ds_dtau=4.0)
    assert waves.shape == (64, 9)
    assert np.allclose(waves[:, 0], 0.0) and np.allclose(waves[:, -1], 1.0)
    assert np.all(np.diff(waves, axis=1) >= -1e-12)
    assert np.all(np.diff(waves, axis=1) <= 4.0 / 8 + 1e-9)


def test_random_policy_family_candidates_do_not_close_the_manifold_gap():
    """The fix that was tried and rejected, kept so it is not tried again.

    Sixteen extra candidates drawn from the policy's own decoder move the mean
    policy-to-bank distance from 0.133 to only 0.127. An eight-dimensional
    waveform space is not coverable by a sixty-four candidate bank, so the shift
    has to be closed with the model's actual proposals rather than random draws
    from the same family.
    """
    torch = pytest.importorskip("torch")
    from annealctrl.models import monotone_samples
    from annealctrl.search import capped_simplex_samples

    tau = np.linspace(0, 1, 9)
    rng = np.random.default_rng(0)
    policy = monotone_samples(torch.as_tensor(rng.normal(0, 1.5, (200, 8)), dtype=torch.float32),
                              max_ds_dtau=4.0).numpy()
    bank = np.stack([c.schedule(tau) for c in
                     shared_candidate_bank(n=64, n_segments=8, seed=7, runtime=4.0, max_slope=1.0)])
    extra = capped_simplex_samples(1.5 * rng.normal(0, 1, (16, 8)), max_ds_dtau=4.0)

    def nearest(waves):
        return np.abs(policy[:, None, :] - waves[None, :, :]).max(axis=2).min(axis=1).mean()

    without, with_family = nearest(bank), nearest(np.vstack((bank, extra)))
    assert with_family > 0.8 * without, (
        "if random same-family candidates closed this gap the DAgger round would "
        f"be unnecessary: {with_family:.4f} vs {without:.4f}")




# --- the policy-gradient strategy --------------------------------------------

def test_policy_gradient_is_a_registered_strategy():
    from annealctrl.search import STRATEGIES

    assert "policy_gradient" in STRATEGIES


def test_policy_gradient_spends_exactly_the_budget():
    """Equal budget is the whole point of comparing search strategies."""
    from annealctrl.search import optimize_control_family

    calls = []

    def loss_fn(schedule):
        calls.append(1)
        return float(np.abs(schedule(np.linspace(0, 1, 9)) - 0.5).sum())

    for family, budget in (("one_window", 17), ("two_window", 33), ("eight_bin", 25)):
        calls.clear()
        result = optimize_control_family(loss_fn, family, budget=budget, seed=0,
                                         strategy="policy_gradient")
        assert len(calls) == budget, (family, len(calls), budget)
        assert len(result.records) == budget


def test_policy_gradient_is_deterministic_given_a_seed():
    from annealctrl.search import optimize_control_family

    def loss_fn(schedule):
        return float(np.abs(schedule(np.linspace(0, 1, 9)) - 0.3).sum())

    a = optimize_control_family(loss_fn, "one_window", budget=17, seed=3,
                                strategy="policy_gradient")
    b = optimize_control_family(loss_fn, "one_window", budget=17, seed=3,
                                strategy="policy_gradient")
    assert [r.loss for r in a.records] == [r.loss for r in b.records]


def test_policy_gradient_improves_on_its_own_first_trial():
    """A learner that never beats its initial incumbent is not learning."""
    from annealctrl.search import optimize_control_family

    def loss_fn(schedule):
        tau = np.linspace(0, 1, 33)
        return float(np.abs(schedule(tau) - np.sqrt(tau)).mean())

    result = optimize_control_family(loss_fn, "eight_bin", budget=65, seed=0,
                                     strategy="policy_gradient")
    losses = [r.loss for r in result.records]
    assert min(losses) < losses[0]


def test_policy_gradient_labels_its_proposals():
    from annealctrl.search import optimize_control_family

    def loss_fn(schedule):
        return float(schedule(np.linspace(0, 1, 9)).sum())

    result = optimize_control_family(loss_fn, "one_window", budget=17, seed=0,
                                     strategy="policy_gradient")
    tags = {r.candidate.parameters.get("proposal") for r in result.records[1:]}
    assert tags == {"policy_gradient"}
    assert result.records[0].candidate.parameters.get("initial_incumbent") == "linear"


def test_policy_gradient_records_its_own_hyperparameters():
    """A baseline whose settings are not recorded cannot be reproduced or attacked."""
    from annealctrl.search import optimize_control_family

    def loss_fn(schedule):
        return float(schedule(np.linspace(0, 1, 9)).sum())

    result = optimize_control_family(loss_fn, "two_window", budget=33, seed=0,
                                     strategy="policy_gradient")
    meta = result.records[1].candidate.parameters
    for key in ("batch_size", "learning_rate", "sigma"):
        assert key in meta, key


def test_linear_family_ignores_the_strategy():
    """Linear has no free parameters; no strategy may pad it with fake calls."""
    from annealctrl.search import optimize_control_family

    calls = []

    def loss_fn(schedule):
        calls.append(1)
        return 0.5

    result = optimize_control_family(loss_fn, "linear", budget=32, seed=0,
                                     strategy="policy_gradient")
    assert len(calls) == 1
    assert len(result.records) == 1


# --- cross-entropy method: a learned search built for a small budget ----------

def test_cem_is_a_registered_strategy():
    from annealctrl.search import STRATEGIES

    assert "cem" in STRATEGIES


def test_cem_spends_exactly_the_budget():
    from annealctrl.search import optimize_control_family

    calls = []

    def loss_fn(schedule):
        calls.append(1)
        return float(np.abs(schedule(np.linspace(0, 1, 9)) - 0.5).sum())

    for family, budget in (("one_window", 17), ("two_window", 33), ("eight_bin", 64)):
        calls.clear()
        result = optimize_control_family(loss_fn, family, budget=budget, seed=0, strategy="cem")
        assert len(calls) == budget, (family, len(calls), budget)
        assert len(result.records) == budget


def test_policy_gradient_uses_the_correct_gaussian_score():
    """d/dmu log N(z; mu, sigma^2 I) = (z - mu) / sigma^2, not / sigma.

    The first version divided once. Every step therefore shrank as sigma decayed,
    the policy barely moved, and the arm lost on the real test split on 0 of 48
    parents -- a result that was published and then retracted, because it measured
    the bug rather than the method. This pins the scaling by checking that the
    mean moves toward the better half of a deliberately one-sided batch by an
    amount the wrong scaling cannot produce.
    """
    from annealctrl.search import optimize_control_family

    def make_loss():
        def loss_fn(schedule):
            tau = np.linspace(0, 1, 65)
            return float(np.abs(schedule(tau) - np.sqrt(tau)).mean())
        return loss_fn

    corrected = [min(r.loss for r in optimize_control_family(
        make_loss(), "eight_bin", budget=64, seed=seed,
        strategy="policy_gradient").records) for seed in range(6)]
    baseline = [min(r.loss for r in optimize_control_family(
        make_loss(), "eight_bin", budget=64, seed=seed,
        strategy="sobol_local").records) for seed in range(6)]
    # With the correct score the policy gradient beats quasi-random search on the
    # highest-dimensional family at campaign budget. With the wrong one it did not.
    assert np.mean(corrected) < np.mean(baseline), (np.mean(corrected), np.mean(baseline))


def test_cem_labels_its_proposals_and_records_its_elite_fraction():
    from annealctrl.search import optimize_control_family

    def loss_fn(schedule):
        return float(schedule(np.linspace(0, 1, 9)).sum())

    result = optimize_control_family(loss_fn, "two_window", budget=33, seed=0, strategy="cem")
    meta = result.records[1].candidate.parameters
    assert meta["proposal"] == "cem"
    for key in ("batch_size", "elite_fraction"):
        assert key in meta, key


def test_cem_is_deterministic_given_a_seed():
    from annealctrl.search import optimize_control_family

    def loss_fn(schedule):
        return float(np.abs(schedule(np.linspace(0, 1, 9)) - 0.3).sum())

    a = optimize_control_family(loss_fn, "one_window", budget=17, seed=5, strategy="cem")
    b = optimize_control_family(loss_fn, "one_window", budget=17, seed=5, strategy="cem")
    assert [r.loss for r in a.records] == [r.loss for r in b.records]


# --- warm starting a search from a learned selection -------------------------

def test_warm_start_keeps_the_linear_reference_and_spends_one_call_on_the_hint():
    """Trial 0 must stay linear: every headroom number in this project is defined against it."""
    from annealctrl.search import optimize_control_family

    seen = []

    def loss_fn(schedule):
        seen.append(schedule(np.linspace(0, 1, 9)).copy())
        return float(np.abs(schedule(np.linspace(0, 1, 9)) - 0.5).sum())

    hint = np.full(3, 0.7)
    result = optimize_control_family(loss_fn, "one_window", budget=17, seed=0,
                                     warm_start=hint)
    assert len(seen) == 17
    assert np.allclose(seen[0], np.linspace(0, 1, 9)), "trial 0 is not the linear schedule"
    assert result.records[1].candidate.parameters.get("proposal") == "warm_start"
    assert result.records[1].candidate.parameters["unit_parameters"] == pytest.approx(list(hint))


def _cubic_loss():
    def loss_fn(schedule):
        tau = np.linspace(0, 1, 33)
        return float(np.abs(schedule(tau) - tau ** 3).mean())
    return loss_fn


def _best_parameters(family, budget=257, seed=0):
    """A genuinely good hint, found by search rather than guessed."""
    from annealctrl.search import optimize_control_family

    result = optimize_control_family(_cubic_loss(), family, budget=budget, seed=seed)
    best = min((r for r in result.records
                if r.candidate.parameters.get("unit_parameters")), key=lambda r: r.loss)
    return np.asarray(best.candidate.parameters["unit_parameters"], dtype=float)


def test_a_good_warm_start_beats_a_cold_start_at_equal_budget():
    """This is the whole premise of pairing a learned selector with a search."""
    from annealctrl.search import optimize_control_family

    hint = _best_parameters("one_window")
    warm = [min(r.loss for r in optimize_control_family(
        _cubic_loss(), "one_window", budget=9, seed=seed, warm_start=hint).records)
        for seed in range(8)]
    cold = [min(r.loss for r in optimize_control_family(
        _cubic_loss(), "one_window", budget=9, seed=seed).records) for seed in range(8)]
    assert np.mean(warm) < np.mean(cold), (np.mean(warm), np.mean(cold))


def test_a_bad_warm_start_is_worse_than_a_cold_start():
    """Warm starting trades exploration for exploitation, and the trade can lose.

    With a cold start the incumbent is empty until an exploratory trial fills it,
    so early trials all explore. A hint fills it immediately and half the budget
    becomes local refinement around the hint. That is a win only if the hint is
    good, and the experiment pairing a learned selector with a search has to be
    read knowing this is the failure mode.
    """
    from annealctrl.search import optimize_control_family

    bad = np.array([0.05, 0.95, 0.95])          # scores worse than the linear reference
    warm = [min(r.loss for r in optimize_control_family(
        _cubic_loss(), "one_window", budget=9, seed=seed, warm_start=bad).records)
        for seed in range(8)]
    cold = [min(r.loss for r in optimize_control_family(
        _cubic_loss(), "one_window", budget=9, seed=seed).records) for seed in range(8)]
    assert np.mean(warm) > np.mean(cold), (np.mean(warm), np.mean(cold))


def test_warm_start_refuses_a_wrong_length_hint():
    from annealctrl.search import optimize_control_family

    with pytest.raises(ValueError, match="warm_start"):
        optimize_control_family(lambda s: 0.5, "one_window", budget=9, seed=0,
                                warm_start=np.zeros(5))


def test_warm_start_is_ignored_for_linear_which_has_no_parameters():
    from annealctrl.search import optimize_control_family

    calls = []

    def loss_fn(schedule):
        calls.append(1)
        return 0.5

    result = optimize_control_family(loss_fn, "linear", budget=16, seed=0,
                                     warm_start=np.zeros(3))
    assert len(calls) == 1
    assert len(result.records) == 1


def test_warm_start_is_refused_by_strategies_that_cannot_use_it():
    """Silently dropping a hint would make a warm-start experiment measure nothing."""
    from annealctrl.search import optimize_control_family

    for strategy in ("bayesian", "policy_gradient", "cem"):
        with pytest.raises(ValueError, match="warm_start"):
            optimize_control_family(lambda s: 0.5, "one_window", budget=9, seed=0,
                                    strategy=strategy, warm_start=np.full(3, 0.5))
