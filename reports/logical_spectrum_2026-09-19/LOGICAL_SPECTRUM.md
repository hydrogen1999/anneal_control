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

## Result: a clean null, on both scale conventions

48 held-out parents, one record each, ≤10 physical qubits, `d2` teacher,
both schedules scored by the same simulator on the same embedded record.
47 usable, 1 unresolved teacher, 0 failures.

| logical coefficients | logical − physical | logical worse in |
|---|---|---:|
| at the programmed scale (isolates the embedding) | **+0.00268** [−0.00148, +0.00668] | 25/47 |
| raw (also carries the common rescaling) | **−0.00115** [−0.00616, +0.00376] | 24/47 |

Both intervals straddle zero and both win-rates are coin flips.

## The context that decides how to read it

| | mean loss |
|---|---:|
| physical-spectrum teacher | 0.77298 |
| **linear at matched duration** | **0.77436** |
| logical-spectrum teacher (programmed) | 0.77566 |

**Neither teacher meaningfully beats linear on this population** — the
physical-spectrum one by 0.00138, the logical-spectrum one not at all. Asking
which spectrum feeds the construction better is asking which fuel a stalled
engine prefers.

So this is **not** evidence against the embedding thesis. It is evidence that
the `d2` construction is too weak a probe to test it. The direct evidence for
the thesis is elsewhere and is not weak: re-embedding the same logical
objective produces a decisive reversal of the preferred control in **53.9 %** of
pairs at 14 qubits, measured on best-found controls rather than through a
teacher.

It is consistent with the rest of the ledger, which already records that
spectral oracles lose to equal-budget search by +0.110.

*Population note:* the ledger's `summary/bank − d2 = −0.050399` uses the audited
711-record population; this run is 47 records at ≤10 physical qubits, one per
parent. The teacher is not equally weak everywhere, and these are not the same
population.

## What it settles about the related work

Usefully, and in our favour — though not the way the document anticipated. The
document frames Tx-NQDT's logical spectrum as its limitation. The measurement
says the spectrum choice is **not** what separates methods here: the whole
spectrum-guided construction sits at linear on this task, from either spectrum.

The honest related-work position is therefore stronger than arguing about whose
spectrum is better: *spectrum-guided schedule construction, given an exact
spectrum of either the logical or the physical system, does not beat a linear
ramp on this population, while equal-budget search beats it by 0.110 and
learned selection beats the physical-spectrum teacher by 0.050 on its audited
population.*

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
