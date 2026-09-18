# Headroom on Pegasus connectivity through 14 physical qubits

`configs/data_pegasus.json` builds instances on **genuine D-Wave Pegasus P16
connectivity** taken from `dwave-networkx` (5640 qubits, 40484 couplers), not on
a toy graph. Every qubit and coupler used was checked against the real topology:
0 non-real qubits, 0 non-real couplers.

This matters for the scale question. The main campaign runs 3–10 physical qubits;
this one reaches **14**, on connectivity a deployed annealer actually has.

## Validation split, complete

24 records over 12 independent logical parents, **0 censored**, 0 audit failures,
verdict `resolved_headroom_present`. Budget is 97 objective calls per record
(`frontier_scaleup.json`, budget 32 per tunable family), so these numbers are
**not** comparable to the 257-call main campaign — a smaller budget gives a
weaker best-found reference and therefore *understates* headroom.

Headroom (linear loss − best found), parent-averaged: **0.1441**, 95% parent
bootstrap CI **[0.1047, 0.1891]**.

### Stratified by physical size

| physical qubits | records | parents | mean headroom | bootstrap CI |
|---:|---:|---:|---:|---|
| 10 | 8 | 4 | 0.1362 | [0.0977, 0.1765] |
| 12 | 8 | 4 | 0.1649 | [0.0760, 0.2538] |
| 14 | 8 | 4 | 0.1312 | [0.0701, 0.2042] |

Positive headroom is observed at each measured size. Wide intervals and four
parents per stratum do not establish no decay, equality, or a monotone size trend.
The pooled statistic above answers a different question from these strata.

## Train split, in progress

85 of 144 records at the time of writing, and the same picture:

| physical qubits | records | mean headroom |
|---:|---:|---:|
| 10 | 36 | 0.1466 |
| 11 | 3 | 0.1411 |
| 12 | 30 | 0.1471 |
| 14 | 16 | 0.1321 |

## Control families

| family | mean restriction loss | wins |
|---|---:|---:|
| `two_window` | 0.0063 | 12 / 24 records |
| `one_window` | 0.0194 | 11 |
| `eight_bin` | 0.0254 | 1 |
| `linear` | 0.1441 | 0 |

The stored `best_family_counts` sum to 24 records: two_window wins 12,
one_window 11, and eight_bin 1. They are not parent-level win counts. The mean
restriction loss describes performance under this finite budget, not a proof
that a family is sufficient or its expressiveness is superior.

## What this does and does not establish

**Does.** Control headroom is present and numerically resolved at
10, 12 and 14 physical qubits on real device connectivity. The 3–10 qubit result
is not an artifact of working only at the smallest sizes.

**Does not — the oracle comparison.** The privileged spectral baselines cannot be
computed here. A full spectral teacher requires diagonalising the Hamiltonian at
every point of the path and is capped at 10 physical qubits by exponential cost,
so `teacher.mode=none` is forced. Nothing in `reports/teacher_baselines_2026-09-17/`
extends above 10 qubits and it must not be quoted as if it did.

**Does not — hardware.** Real connectivity is not a real device. These are
closed-system simulations on a real coupling graph: no noise, no calibration
drift, no readout error, no QPU job was ever submitted.

**Does not — a like-for-like budget comparison.** Budget 32 here against 64 in
the main campaign. The comparison across campaigns is qualitative.

**Small.** 12 parents in validation. The per-size intervals are wide. Neither "does not decay" nor
"is constant" is established by overlap of confidence intervals.
