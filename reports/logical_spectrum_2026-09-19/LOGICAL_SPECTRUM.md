# Does the logical spectrum suffice? The question cannot be answered here,
and that is the finding

## Why this was run

The design document names Tx-NQDT (Lu et al. 2026) as "the strongest overlap in
spectrum-guided hardware scheduling" and requires "an implementation faithful
to the closest spectral-learning method". It also records the one structural
difference: *"The reported spectrum is logical, rather than the complete
physical embedded system."*

Its construction — transition matrix elements, a normalised time density,
inverse cumulative reconstruction — is what `physics_baselines` already builds.
So the faithful comparison is not a second implementation of that machinery but
the same machinery fed the **logical** Hamiltonian and asked to control the
**physical embedded** one. That doubles as the sharpest available test of this
project's central thesis.

## Result: a pooled null that is a mixture of two opposite populations

48 held-out parents, one record each, ≤10 physical qubits, `d2` teacher, both
schedules scored by the same simulator on the same embedded record. 47 usable,
1 unresolved teacher, 0 failures, 0 one-to-one violations.

Pooled, it is a null on both scale conventions:

| logical coefficients | logical − physical | logical worse in |
|---|---|---:|
| at the programmed scale | +0.00268 [−0.00148, +0.00668] | 25/47 |
| raw | −0.00115 [−0.00616, +0.00376] | 24/47 |

**That pooled number is a composition artefact**, but not in the way an earlier
version of this report claimed. Half of these instances have a *monotone* gap
— [the crossing census](../crossings_2026-09-19/CROSSINGS.md) finds no interior
gap minimum in 24 of 48 parents — and a time density derived from D₂ has
nothing to localise there.

### Runtime coverage, and a claim withdrawn

The first version of this study covered **runtime 1 only**, silently: taking
one record per parent sorted by record_id always selects `_t0`. All three
runtimes have now been run.

| runtime | logical − physical (pooled) | logical | physical | linear |
|---:|---|---:|---:|---:|
| 1 | +0.00268 [−0.00148, +0.00668] | 0.77566 | 0.77298 | 0.77436 |
| 4 | −0.00390 [−0.01853, +0.00914] | 0.57142 | 0.57532 | **0.56436** |
| 12 | +0.00218 [−0.01287, +0.01663] | 0.39900 | **0.39682** | 0.40749 |

The teacher itself is worse than linear at runtime 4 and better at runtime 12,
so its value depends on runtime before any question of which spectrum feeds it.

**A claim from the first version is withdrawn.** It said the teacher "works
where it has a bottleneck to work with", on this split at runtime 1:

| runtime | 0 interior minima | ≥1 interior minimum |
|---:|---|---|
| 1 | +0.01065, 6/23 | **−0.01291** [−0.01815, −0.00768], 21/24 |
| 4 | −0.03788, 12/23 | **+0.05777** [+0.01517, +0.10218], 7/24 |
| 12 | **−0.04834** [−0.08882, −0.01220], 15/23 | +0.02544, 9/24 |

The stratification **reverses** between runtime 1 and runtime 4, and pooled
across all three — averaging within parent, then resampling parents — the
teacher helps the *monotone* group (−0.02519 [−0.04990, −0.00060]) rather than
the other (+0.02343 [−0.00587, +0.05259]). A runtime-1 result was reported as a
general one.

### What survives, and is stronger for the pooling

The contrast this study exists to measure keeps its sign at every runtime and
clears zero when pooled properly:

| subgroup | logical − physical, pooled over 3 runtimes | positive in |
|---|---|---:|
| all | +0.00032 [−0.00893, +0.00925] | 21/47 |
| 0 interior minima | −0.01122 [−0.02630, +0.00331] | 6/23 |
| **≥1 interior minimum** | **+0.01137** [+0.00320, +0.01975] | **15/24** |

Per runtime the estimate is +0.0087, +0.0135, +0.0120 — same sign three times,
individually significant only at runtime 1, and significant pooled.

**On instances whose gap has an interior minimum, a schedule built from the
physical embedded spectrum controls the embedded system better than one built
from the logical spectrum.** That is direct evidence for this project's central
thesis, through the comparison the design document asked for, and it is the one
claim here that survived both the runtime extension and the slope-convention
fix.

Two qualifications stay attached. It is a **subgroup analysis on 24 parents**,
one stratification chosen from a stated mechanism on a variable that is a
property of the instance and not of the outcome. And the monotone-half sign is
negative throughout, which is the difference between two constructions that
both have nothing to localise, not evidence that the logical spectrum is better
there.

## What it settles about the related work

The document frames Tx-NQDT's logical spectrum as its limitation, and on the
instances where the method it belongs to actually functions, **that framing is
correct and now measured**: +0.00871 [+0.00450, +0.01304], logical worse in
19 of 24 parents.

The complete statement needs both halves, because the pooled figure is a null
and a reader who sees only the subgroup would be misled:

*Spectrum-guided schedule construction beats a linear ramp by 0.013 on the half
of instances whose gap has an interior minimum and not at all on the monotone
half, so pooled it sits at linear. On the half where it works, the physical
embedded spectrum beats the logical one by 0.009. Equal-budget search beats the
construction by 0.110 regardless, and learned selection beats the
physical-spectrum teacher by 0.050 on its audited population.*

## A confound that was found and removed

The first run of this study compared **raw** logical coefficients against
**programmed** physical ones, so "logical versus physical spectrum" silently
also meant "unscaled versus scaled Hamiltonian" — and the document lists
re-embedding and common rescaling as *separate* falsification axes.

The synthetic test asserting that a one-to-one embedding must give identical
schedules passed, because it used `programmed_scale = 1.0`. Real records do not:

| programmed_scale | 1.0000 | 0.9228 | 0.8432 | 0.8369 | 0.4748 | 0.2965 |
|---|---:|---:|---:|---:|---:|---:|
| delta | **0.00000** | −0.0034 | −0.0065 | −0.0075 | −0.0013 | **−0.0338** |

The scale was moving, not the embedding. `logical_terms` now applies
`programmed_scale` by default so a one-to-one embedding yields the identical
Hamiltonian; `scale="raw"` is kept because it answers the different question of
what a solver of the stated logical problem would see, and both are reported
above. The driver checks the identity property on real records and warns if any
one-to-one record disagrees — it does not, in the corrected run.

## Limits

- `d2` only. `gap_inverse_square` was not run through this comparison.
- 47 parents, ≤10 physical qubits, one dataset, closed system.
- Labelled **inspired by** as the document requires, and the label is earned:
  no neural quantum state, no variational reconstruction, and the logical
  spectrum here is exact rather than approximated. A method whose spectrum is
  *approximate* could behave differently, in either direction.
