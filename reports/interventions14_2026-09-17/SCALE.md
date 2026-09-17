# The embedding still decides the control at 14 physical qubits

The G3 result — changing one embedding factor reverses which control is
preferred — was measured at 8 physical qubits. This repeats it at **14**, with
the same factors, the same equal-budget search and the same censoring rules, and
nothing else changed but the size.

Chain lengths are `[4,4,3,3]` rather than `[3,3,2]` because a path and a star
coincide on three nodes: the geometry intervention would be vacuous and the
planner would refuse it, correctly, as a change that did not happen.

## Result

| | 8 qubits (main campaign) | **14 qubits** |
|---|---:|---:|
| pairs | 1060 | **228** |
| parents | 96 | **20** |
| censored | 24.0% | **21.9%** |
| decisive reversals | 610 (57.5%) | **123 (53.9%)** |
| verdict | preference_changes_measured | **preference_changes_measured** |

**The signal holds.** 53.9% against 57.5%, on a quarter the parents, at nearly
twice the physical size.

### By intervened factor

| factor | pairs | transfer penalty | reversals |
|---|---:|---:|---:|
| `chain_strength` | 80 | 0.0791 | 69/80 (86%) |
| `ports` | 50 | 0.0202 | 25/50 (50%) |
| `geometry` | 78 | 0.0089 | 28/78 (36%) |
| `field_allocation` | 20 | 0.0009 | 1/20 (5%) |

Chain strength dominates, as at 8 qubits. Field allocation barely moves the
preferred control at either size, and its 20 pairs are what survived after the
planner skipped every parent where re-allocating the field left all raw
coefficients identical.

## The composition guard fired again

| quantity | value |
|---|---:|
| matched pairs | 42 |
| `scale_controlled` mean | 0.0912 |
| `total_compiled_effect` mean | 0.0270 |
| within-pair difference | **+0.0453**, 95% CI [0.0207, 0.0704] |
| **matched ratio** | **1.70×** |
| confounded marginal ratio | 3.38× *(would have been wrong)* |

Holding the programmed scale fixed raises the transfer penalty by a factor of
**1.70** at 14 qubits, against **1.52** at 8 — consistent, and both far from the
**3.38×** that dividing the two marginal means would have produced here. That
error was made once already at 8 qubits and reported as 2.6×; the instrument now
computes the matched contrast automatically and flags
`unmatched_is_composition_confounded`, so the same mistake could not be made
twice.

## What this does not cover

**20 parents.** A quarter of the 8-qubit campaign's 96. The reversal rate is a
proportion over 228 pairs and is stable, but per-factor cells are small — 20
pairs for `field_allocation` — and should not be read as precise.

**One size, not a ladder.** This is 14 qubits against 8. Two points do not
establish a trend, only that the effect has not vanished between them.

**Simulator, not hardware.** Same closed-system model as everywhere else in this
project.

**Not the same instances.** The 8- and 14-qubit campaigns generate their own
parents. This is a replication of the *finding* on a comparable distribution, not
the same problems embedded at two sizes.
