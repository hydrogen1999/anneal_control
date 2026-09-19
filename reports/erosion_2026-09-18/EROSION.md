# Noiseless optimisation is what the environment punishes

## The claim

Within a problem instance, **the better a control is in a noiseless simulator,
the more it loses when an environment is added.** The ordering largely survives
— the noiseless favourite is usually still the favourite — but it is
systematically the candidate that gives up the most.

This is the mechanism behind the robustness result: it is why a smoothed
learned preference retains 69.5–74.5% of its advantage under dephasing while
the exact noiseless argmax retains only 61.5%.

## The headline number, after both controls

| analysis | mean ρ | negative in |
|---|---:|---:|
| raw | −0.9081 | 200/200 |
| ÷ headroom (removes the [0,1] ceiling) | −0.7925 | 198/200 |
| candidate-demeaned (removes waveform identity) | −0.9320 | 200/200 |
| **both controls** | **−0.6312** | **192/200** |

**Quote −0.63.** The raw −0.91 is real but flattered by two artefacts, and the
number that survives removing both is the one the claim rests on.

## Control 1: the [0, 1] ceiling

A loss lives in [0, 1], so a candidate with a low noiseless loss has more room
to rise and a negative level–change correlation can appear with no mechanism.
Dividing the degradation by the available headroom (1 − loss) removes exactly
that, and the correlation moves from −0.9081 to −0.7925 while staying negative
in 198 of 200 records. The ceiling explains a part, not the effect.

## Control 2: the bank is shared, and that is dangerous

All 558 test records at ≤ 6 qubits draw from **one** bank of eight waveforms,
and the noiseless winner is concentrated: candidate 2 wins 47.5% of records and
candidate 3 wins 32.4%. If those two happened to be noise-fragile, the whole
correlation would follow with no principle behind it.

Subtracting each candidate's own mean across records — a candidate fixed effect
— leaves only instance-specific variation: *for this problem*, is candidate j
unusually good and unusually fragile? The correlation does not weaken. It
strengthens, to −0.9320 (negative in 200/200 raw; −0.6312 and 192/200 once the
ceiling is also removed).

The per-candidate table shows why there was little fixed effect to remove:

| candidate | mean noiseless loss | mean degradation | times it won |
|---:|---:|---:|---:|
| 0 | 0.64123 | 0.14266 | 10 |
| 1 | **0.70850** | **0.09497** | 0 |
| 2 | 0.63540 | 0.14062 | 99 |
| 3 | 0.60760 | 0.14727 | 76 |
| 4 | 0.64208 | 0.14062 | 1 |
| 5 | 0.65045 | 0.12503 | 3 |
| 6 | 0.64489 | 0.13977 | 2 |
| 7 | 0.62372 | 0.14828 | 9 |

Degradation is nearly uniform at 0.14 ± 0.01 across the six waveforms that ever
win. The one genuine outlier, candidate 1, is the **worst** noiseless control
(0.7085), never wins, and degrades **least** (0.0950) — which is the same story
one level up.

## Generality: a second noise channel

If the effect were specific to dephasing it would be a curiosity about one
perturbation. It is not, but the qualification is real.

| channel | rate | ρ raw | ρ ÷ headroom | negative in | rank of the noiseless-best |
|---|---:|---:|---:|---:|---:|
| dephasing | 0.02 | −0.8024 | −0.6824 | 96/100 | 6.83 |
| dephasing | 0.05 | −0.8452 | −0.7248 | 99/100 | 7.01 |
| dephasing | 0.10 | −0.9000 | −0.7826 | 99/100 | 7.27 |
| relaxation | 0.02 | −0.5424 | −0.4064 | 76/100 | 6.37 |
| relaxation | 0.05 | −0.5255 | −0.3898 | 75/100 | 6.37 |
| relaxation | 0.10 | −0.5136 | −0.3781 | 78/100 | 6.40 |

(Rank of the noiseless-best in headroom-normalised degradation, out of 8;
chance is 4.5, higher means it degrades more.)

Two things to say honestly:

- **The effect is not dephasing-specific.** Under amplitude relaxation it is
  still there at ρ ≈ −0.39, negative in three records out of four, with the
  noiseless-best sitting at rank 6.4 of 8 against a chance value of 4.5.
- **It is about half as strong under relaxation, and it does not grow with
  rate.** Dephasing strengthens monotonically (−0.68 → −0.72 → −0.78) while
  relaxation is flat (−0.41 → −0.39 → −0.38). Whatever drives the effect
  couples more tightly to phase noise than to amplitude damping, and this
  measurement does not say why.

## Ranking quality does not fall with system size

A separate diagnostic on the **full** 864-record test split, all eight
candidates, `summary/seed_0`:

| physical qubits | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ρ | +0.83 | +0.81 | +0.86 | +0.86 | +0.88 | +0.76 | +0.72 | +0.83 |
| n | 54 | 162 | 198 | 144 | 144 | 90 | 54 | 18 |

No trend, a mild dip at 8–9. `hierarchy_physics/seed_0` gives the same picture
(+0.78 to +0.88). This matters twice: it is why the Pegasus ranking drop is not
attributed to larger records, and it is why the erosion measurement's ≤ 6 qubit
cap is a cost limit rather than a regime boundary.

Reproduce with `scripts/critic_ranking_diagnostics.py`; artifact in
`reports/diagnostics_2026-09-18/ranking.json`.

## Is it a small-system artefact? Not at the sizes reachable

The main measurement caps at 6 physical qubits because of the solver, not
because of the physics, so the effect was re-measured at the next size the
solver allows.

| pool | records | raw | ÷ headroom | candidate-demeaned ÷ headroom | negative in |
|---|---:|---:|---:|---:|---:|
| ≤ 6 qubits | 200 | −0.9081 | −0.7925 | **−0.6312** | 192/200 |
| 7 qubits | 40 | −0.9167 | −0.8214 | **−0.5952** | 38/40 |

The doubly-controlled figure is the same to within the noise of 40 records. The
per-candidate pattern repeats too: at 7 qubits candidate 1 is again the worst
noiseless control (0.6515), degrades least (0.0839) and never wins — the same
shape as at ≤ 6 qubits.

An 8-qubit pool is the last size the density solver accepts and costs 30.4 s
per solve; that run is smaller still (12 records) and is reported when it
lands rather than promised here.

## Erosion, not inversion

Two numbers keep the claim the right size:

    ρ(loss at rate 0, loss at rate 0.1)      +0.8340
    noiseless-best still best at rate 0.1    157/200 = 78.5%

The ranking largely survives. The noiseless favourite usually wins anyway — it
simply gives up more of its margin than anything else does.

## What it means for practice

**Past some point, optimising harder against a noiseless simulator is
self-defeating.** The controls a closed-system search rewards are the ones an
environment erodes fastest, so the marginal return on a longer noiseless search
is worse than its closed-system curve suggests. That is an argument for
learning a smoothed preference rather than for buying more noiseless
optimisation, and it applies to any pipeline that fits quantum control in a
closed-system simulator — not only to this one.

## Limits

- One bank of eight waveforms, one dataset, ≤ 6 physical qubits for the main
  measurement. The wall is measured, not guessed: one Lindblad solve costs
  **1.16 s at 6 qubits, 5.12 s at 7, 30.4 s at 8**, and the density solver
  refuses 9 outright (`adapters.simulate_lindblad`, O(4^N) memory). A full
  200-record study is therefore 1 hour at 6 qubits, 4.5 at 7 and 27 at 8.
- **Not replicated on Pegasus, and it cannot be.** The smallest Pegasus records
  here are 10 physical qubits, past the solver's hard cap. That is a genuine
  gap, not an omission: the Pegasus bank is different (64 candidates, two
  distinct banks rather than one) and would have been a much stronger
  independent check than anything available at ≤ 8 qubits.
- Two channels, both uniform across qubits at a declared rate. No thermal bath,
  no measured T1/T2, no per-qubit calibration, no working-graph exclusions.
  **This is not a calibrated device model.**
- The mechanism is measured, not explained. An earlier proxy — dwell time,
  1 ÷ minimum local slope — finds nothing (pooled ρ = +0.0097), so whatever
  structural property makes a noiseless-optimal control fragile, it is not
  simply "it lingers".
