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

**That pooled number is a composition artefact.** Half of these instances have
a *monotone* gap — [the crossing census](../crossings_2026-09-19/CROSSINGS.md)
finds no interior gap minimum in 24 of 48 parents — and a time density derived
from D₂ has nothing to localise there. The construction does nothing on that
half, so its input cannot matter on that half either.

Splitting on that mechanism, which is a property of the instance and not of the
outcome:

| subgroup | n | teacher − linear | beats linear |
|---|---:|---|---:|
| 0 interior gap minima | 23 | **+0.01065** [−0.00093, +0.02183] | 6/23 |
| ≥1 interior gap minimum | 24 | **−0.01291** [−0.01815, −0.00768] | **21/24** |

The spectral teacher works, clearly, exactly where it has a bottleneck to work
with, and not otherwise. Rank correlation between the minimum gap and the
teacher's advantage over linear is **−0.399**: the smaller the gap, the more it
helps, which is what adiabatic theory predicts.

### Where it works, the physical spectrum beats the logical one

| logical coefficients | subgroup | logical − physical | logical worse in |
|---|---|---|---:|
| programmed | ≥1 minimum (n=24) | **+0.00871** [+0.00450, +0.01304] | **19/24** |
| programmed | 0 minima (n=23) | −0.00360 [−0.00966, +0.00241] | 6/23 |
| raw | ≥1 minimum (n=24) | **+0.00771** [+0.00309, +0.01238] | **18/24** |
| raw | 0 minima (n=23) | −0.01040 [−0.01754, −0.00344] | 6/23 |

**On the half where spectrum-guided scheduling does anything at all, the
physical embedded spectrum beats the logical one, with the interval clearing
zero under both scale conventions.** That is direct evidence for this project's
central thesis, obtained through the comparison the design document asked for.

Two honest qualifications. This is a **subgroup analysis on 24 parents**: one
stratification, chosen from a stated mechanism before the split was examined,
using a variable that is a property of the instance rather than of the outcome
— but it needs replication, not treatment as a confirmatory result. And the
sign on the monotone-gap half is *negative* under both conventions, which
should not be read as "the logical spectrum is better there": it is the sign of
a difference between two constructions that both do nothing.

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
