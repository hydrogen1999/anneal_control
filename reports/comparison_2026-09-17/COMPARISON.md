# The comparison table

Every method below is measured on the **same 864 held-out records over 48 test
parents**. Intervals are parent bootstraps. Lower loss is better; loss = 1 −
decoded logical success.

Reproduce with:

```
annealctrl comparison-table \
    --records reports/heldout_2026-09-17/heldout_records.json \
    --reference-sweep <g2_frontier_test shards> \
    --output reports/comparison_2026-09-17/comparison_table.json
```

## Rows are grouped by what they consume, and ordered only within a group

| cost class | method | loss | 95% CI | vs linear | n |
|---|---|---:|---|---:|---:|
| fixed | linear | 0.6009 | [0.5356, 0.6630] | — | 864 |
| fixed | global | 0.5654 | [0.4991, 0.6291] | −0.0355 | 864 |
| privileged spectrum | `d2` | 0.5913 | [0.5179, 0.6612] | −0.0041 | 846 |
| privileged spectrum | `gap_inverse_square` | 0.7750 | [0.7362, 0.8109] | +0.0328 | 432 |
| **amortised** | **summary / bank** | **0.5447** | [0.4777, 0.6088] | **−0.0563** | 864 |
| amortised | hierarchy_physics / bank | 0.5450 | [0.4784, 0.6088] | −0.0559 | 864 |
| amortised | hierarchy_outcome / bank | 0.5456 | [0.4788, 0.6095] | −0.0554 | 864 |
| amortised | physical / bank | 0.5462 | [0.4797, 0.6096] | −0.0547 | 864 |
| amortised | logical / bank | 0.5532 | [0.4859, 0.6175] | −0.0477 | 864 |
| amortised | best direct proposal | 0.5898–0.5978 | — | −0.0111…−0.0032 | 864 |
| online adaptation | search (best found) | 0.5074 | [0.4391, 0.5728] | −0.0936 | 864 |

There is deliberately **no global rank**. A linear ramp, an oracle whose
construction is exponential in the number of qubits, a network that consults no
outcome, and a search spending 257 true simulator evaluations per instance are
not competing for the same prize, and a single column sorted best-first would
make whichever row came first read as "the best method".

## What the table says

**The amortised learned selector beats both privileged spectral oracles.**
0.5447 against `d2`'s 0.5913 and `gap_inverse_square`'s 0.7750 — while the
oracles require the exact instantaneous spectrum at every point on the path and
the learned selector requires nothing at deployment beyond a forward pass. This
is the paper's central positive result, and it holds on held-out parents.

**It also beats the best single fixed schedule.** Global is 0.5654; bank
selection is 0.5447. Choosing per instance is worth 0.021 more than choosing one
good schedule for everything.

**Online adaptation still wins, and by how much is stated.** Search reaches
0.5074, which is 0.037 better than the best amortised method. That gap is the
honest price of not consulting outcomes at deployment. It is not hidden, and no
amortised row is described as matching search.

**Which encoder does not matter; seeing the embedding does.** The four
embedding-aware encoders land within 0.0015 of each other — inside their own
seed noise — while the embedding-blind `logical` sits 0.008 behind them. This
reproduces on this table what `method-contrast` established with Holm-corrected
paired tests, and it is why the paper does not claim an architecture.

**Direct generation is the weak mode.** Every method's direct proposal sits near
0.590, barely ahead of linear. One round of dataset aggregation moves it to
0.5547 on seed 0 — past global, and most of the way to bank selection — but that
is one seed with its matched control and it is reported in the DAgger results,
not here.

## What the table does not say

**`gap_inverse_square` is conditional on half the population.** It resolves on
432 of 864 records, because a degenerate or unresolved first gap leaves it with
no schedule to build. That is not missing at random: a vanishing gap is a
property of the instance. Its row is marked `measured_on_full_population: false`,
its paired difference is computed against linear on its own subset, and its loss
must not be read against rows measured on all 864.

**`d2` is conditional on 846 of 864**, which is close to complete but not
complete.

**Nothing here is a hardware result.** Closed-system simulation, at most 10
physical qubits, no noise and no calibration drift.

**The search row is a method, not a ceiling.** It consults true outcomes 257
times per instance. Treating it as an upper bound that amortised methods
approach would misdescribe what it is: a different cost class that a deployed
system usually cannot pay.
