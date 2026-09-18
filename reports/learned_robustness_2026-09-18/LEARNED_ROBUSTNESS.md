# The learned selector under decoherence: it decays, and it decays more slowly
than the thing it was trained to imitate

## What was missing

The archived open-system check took two fixed waveforms and asked whether
dephasing reordered them. That is not the question a learned method has to
answer. A critic fit to closed-system labels can be hurt twice by an
environment: the control it likes may degrade, and the control it *should* have
liked may change. Only the second is a failure of the learning, and a single
total would credit or blame the wrong component.

The evidence ledger named this gap explicitly: *"no learned-policy robustness
evaluation."* This closes it.

## Design

558 test records at ≤ 6 physical qubits, 31 logical parents, `research_v1`
checkpoints. Every bank candidate of every record is re-solved under a Lindblad
master equation at dephasing rates 0, 0.02, 0.05, 0.1 — **one shared loss
table**, so the selectors below are compared on identical numbers rather than on
two runs of a stochastic solver. A test pins the shared-table path against the
direct path at `rel=0, abs=0`.

Three rules are scored on that table:

| rule | sees the noise? | role |
|---|---|---|
| `summary/seed_{0,1,2}` | no | the deployed critic, selecting from the bank |
| closed-system oracle | no | best candidate by the stored noiseless label — the **ceiling on any noise-blind rule** |
| noise-aware oracle | yes | best candidate at that rate — the ceiling overall |

The critic is called once, before any noise is simulated, and never sees a
rate. That asymmetry is the deployment condition, not an oversight.

## Result 1: the advantage survives every rate tested

Advantage against the linear candidate; negative is better; parent bootstrap.

| rule | rate 0 | rate 0.02 | rate 0.05 | rate 0.1 |
|---|---|---|---|---|
| noise-aware oracle | −0.06084 | −0.05951 | −0.05810 | −0.05021 |
| closed-system oracle | −0.06084 | −0.05373 | −0.04622 | −0.03740 |
| summary/seed_0 | −0.05278 | −0.05168 | −0.04728 | −0.03934 |
| summary/seed_1 | −0.05025 | −0.04817 | −0.04369 | −0.03661 |
| summary/seed_2 | −0.05125 | −0.04860 | −0.04347 | −0.03564 |

Every interval excludes zero at every rate. At the strongest noise the learned
selection still beats linear by −0.03934 [−0.04840, −0.03067] (seed 0). The
advantage is not an artefact of simulating no environment.

It does **decay**. The method is not immune to noise; it is merely still ahead.

## Result 2: the learned rule loses less than the exact noiseless argmax

Retention = advantage(rate) ÷ advantage(0), parent-level, bootstrapped:

| rule | rate 0.05 | rate 0.1 |
|---|---|---|
| noise-aware oracle | 95.5% [82.9, 109.4] | 82.5% [69.1, 98.5] |
| summary/seed_0 | 89.6% [76.9, 104.5] | **74.5%** [60.3, 90.4] |
| summary/seed_1 | 86.9% [77.5, 97.6] | **72.9%** [61.6, 86.4] |
| summary/seed_2 | 84.8% [73.3, 96.9] | **69.5%** [57.5, 83.0] |
| closed-system oracle | 76.0% [65.7, 86.8] | **61.5%** [50.7, 73.6] |

The learned selector sits **between** the noiseless argmax and the noise-aware
oracle, having never seen noise.

The obvious objection is that the learned rule simply had less to lose, since it
starts at −0.051 against the oracle's −0.061. Both tests answer it:

*Difference-in-differences* (absolute, paired per record, parent bootstrap) —
all nine intervals exclude zero, same direction:

| seed | rate 0.02 | rate 0.05 | rate 0.1 |
|---|---|---|---|
| 0 | −0.00600 [−0.00981, −0.00273] | −0.00912 [−0.01506, −0.00398] | −0.00999 [−0.01672, −0.00426] |
| 1 | −0.00502 [−0.00876, −0.00093] | −0.00806 [−0.01348, −0.00224] | −0.00980 [−0.01530, −0.00415] |
| 2 | −0.00445 [−0.00738, −0.00149] | −0.00684 [−0.01125, −0.00240] | −0.00783 [−0.01257, −0.00306] |

*Paired retention contrast* (proportional, the form the objection asks for) —
learned minus closed-oracle retention, all six intervals exclude zero:

| seed | rate 0.05 | rate 0.1 |
|---|---|---|
| 0 | +13.6% [+5.1, +23.7] | +13.1% [+4.3, +25.5] |
| 1 | +11.0% [+1.7, +20.3] | +11.4% [+2.8, +20.8] |
| 2 | +8.8% [+1.3, +16.7] | +8.1% [+0.6, +16.1] |

Seed 2 at rate 0.1 is +8.1% [+0.6, +16.1] — it excludes zero by a margin small
enough that it should be read as the weakest of the six, not as a separate
confirmation.

## Result 3: the price of not seeing the noise

Closed-system oracle minus noise-aware oracle, the intrinsic cost of
noise-blindness for a *perfect* noiseless picker:

    rate 0.02   +0.00577 [+0.00357, +0.00829]
    rate 0.05   +0.01188 [+0.00826, +0.01592]
    rate 0.1    +0.01281 [+0.00924, +0.01698]

So roughly a fifth of the achievable advantage at rate 0.1 is lost purely by
being unable to see the environment — and the learned critic recovers part of
that gap without being shown it.

## The mechanism was tested and is NOT established

The natural story is the one in `open_system.py`'s docstring: a control that
lingers buys adiabaticity in a closed system and buys the bath more time in an
open one, so the exact noiseless argmax over-commits to lingering and a smoothed
critic does not. That is a plausible story and **the measurement does not
support it.**

Rank correlation between a candidate's dwell time (1 ÷ minimum local slope) and
its degradation loss(0.1) − loss(0), over 120 records × 8 candidates:

    per-record mean   +0.0655
    per-record median +0.0737
    positive in       72/120 records = 60.0%
    pooled            +0.0097

That is no relationship worth the name. Either the mechanism is wrong, or dwell
measured *anywhere* is the wrong proxy for lingering *near the gap* — this test
cannot distinguish them. **The effect in Results 1–3 is measured; its
explanation is not.** It is written here rather than omitted because a
plausible mechanism that fails its own test is exactly the thing a paper should
not quietly drop.

## Limits

- One noise axis: local dephasing at a declared rate. No thermal bath, no
  measured T1 or T2, no per-qubit calibration, no working-graph exclusions.
  This is **not** a calibrated device model and says nothing about what a real
  annealer would do.
- ≤ 6 physical qubits, because a density matrix costs O(4^N). 31 parents.
- Bank selection only. The direct-generation policy is not evaluated here.
- Intervals resample logical parents and condition on the archived checkpoints;
  they do not include training-seed uncertainty, which is instead shown by
  reporting three seeds separately.
