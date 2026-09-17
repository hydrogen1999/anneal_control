# Evidence audit of upstream main at 2d48229

The upstream main branch advanced while the controlled-acquisition revision was
being developed. These results and the new 60-parent pilot use different recipes
and must remain separately identified.

## What upstream adds

The summary encoder's three-seed aggregation report contains 864 held-out records
from 48 logical parents. Direct loss is 0.593482 BEFORE, 0.587285 BANKEXT and
0.564730 DAGGER. DAGGER minus BANKEXT is -0.022555, negative in all three seeds.
Best-proposal loss improves by approximately 0.0194; ranking regret improves by
approximately 0.0031; mean critic rank correlation changes by only approximately
0.003. This supports an empirical proposal-quality gain in that experiment,
not the originally proposed explanation of strong critic recalibration.

DAGGER direct loss still exceeds its bank-selection loss (0.543728) by 0.021002.
The report augments validation candidates as well as training candidates, so
the checkpoint-selection bank also changes. Archived artifacts are aggregate
summaries; they do not include paired per-parent DAGGER/BANKEXT rows or their
confidence interval. Existing acquisition code in upstream also imputes appended
numerical diagnostics. The controlled study in this revision fixes those
measurement issues; its separate smaller pilot does not reproduce the upstream
gain and is retained in full.

The P3-to-P16 comparison reports a pooled direct-loss difference of -0.003801,
parent interval [-0.008346, 0.000318], and 26/48 parents favouring P16. Summary
alone worsens by 0.001650. The pooled analysis weights five encoders equally,
although hierarchy_physics has one matched seed and the others have three.
This is insufficient evidence of an improvement from this particular proposal
increase. Nonsignificance does not show equivalence or exhaust proposal count
as a design choice. Changing proposal count also changes training; it is not
solely additional inference computation.

## Comparators and scale

The full-population table places summary/bank at 0.544662, global at approximately
0.565437, and online search at 0.507352. Search spends 257 objective calls per
instance, whereas bank deployment uses zero online calls. Offline generation and
training must still be charged to the learned approach.

Upstream teacher rows include non-null losses from failed interpolation audits:
135 d2 and 93 gap-inverse-square rows. They also cover different subsets from
the learned full-population row. Claims against teachers require both audit
filtering and matched-population contrasts. The correction in this revision
preserves historical raw reports and writes a separate corrected report.

Commit 3f6e27d states a 12-qubit CuPy/NumPy throughput ratio of 2.83, but its
crossover JSON is absent from the inspected tree. No new 16/20-qubit GPU result
is archived there. The previously archived 10-qubit profile remains the available
raw GPU evidence. A commit message alone is insufficient to audit timing,
precision parity and resource isolation.

## Sources

- [Latest upstream commit](https://github.com/hydrogen1999/anneal_control/commit/2d48229bf3a7d2e869b3bdbf0290e06f6c9d14f3)
- [Aggregation report](https://github.com/hydrogen1999/anneal_control/blob/2d48229/reports/dagger_2026-09-17/DAGGER.md)
- [P16 report](https://github.com/hydrogen1999/anneal_control/blob/2d48229/reports/proposals16_2026-09-17/PROPOSALS16.md)
- [Comparison table](https://github.com/hydrogen1999/anneal_control/blob/2d48229/reports/comparison_2026-09-17/comparison_table.json)
- [GPU claim and configuration corrections](https://github.com/hydrogen1999/anneal_control/commit/3f6e27dcfbee263075a749f5f66ce2f96496dc98)

## Next claims to test

Retain the three-seed upstream gain as evidence with the stated limits. Complete
the fixed-validation acquisition design at research scale, add independent
random-acquisition seeds, and use the frozen-proposal diagnostic to separate
critic and policy changes. Develop any direct-selection checkpoint criterion
on validation only, then use a fresh frozen test population. Publish matched
literature-comparator curves and larger/OOD data before broadening scale or
method claims. Hardware claims require hardware observations.
