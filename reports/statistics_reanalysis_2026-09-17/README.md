# Corrected statistical reanalysis of the archived held-out evaluation

This analysis uses the same archived seed-averaged measurements as the original
2026-09-17 report. It performs no new training or simulation. The source data
and analysis module SHA-256 hashes, resampling seed, and 20,000-draw budget are
recorded in [contrasts.json](contrasts.json). The original reports remain intact.

The corrected implementation uses a centered-null bootstrap approximation for
p-values, applies Holm within each mode's ten pairwise comparisons, and lists
maximal sets of pairwise non-rejection. Pointwise percentile intervals are not
simultaneous intervals. See [statistical interpretation](../../docs/statistical_inference.md)
for assumptions, the correction to the original pooled-Holm wording, and the
limits of parent-only uncertainty.

| Analysis | Result |
|---|---|
| Bank, pairwise family | Four separated pairs: each embedding-aware encoder versus the logical encoder. No aware-to-aware separation. |
| Direct, pairwise family | Two separated pairs: physical versus each hierarchy variant. |
| Bank, pooled aware minus blind | -0.00781748; pointwise 95% parent interval [-0.01109282, -0.00447146]. Exploratory and unadjusted. |
| Direct, pooled aware minus blind | +0.00258717; pointwise 95% parent interval [-0.00507021, +0.00994535]. Exploratory and unadjusted. |

The pairwise rejection pattern is unchanged by this reanalysis. It does not
establish an architectural advantage or equivalence among the embedding-aware
encoders. In direct mode, the maximal non-rejection sets overlap: logical and
summary appear in both sets. A partition that combined all five methods would
incorrectly hide the two separated pairs.

The archived input contains averages over training seeds, so crossed parent-and-
seed uncertainty cannot be recovered from this file. New acquisition studies
must retain individual seed records; the new crossed bootstrap API consumes
those records without treating repeated parents or seeds as independent runs.

Reproduce from the repository root:

```bash
python scripts/reanalyze_heldout_statistics.py \
  --records reports/heldout_2026-09-17/heldout_records.json \
  --output reports/statistics_reanalysis_2026-09-17/contrasts.json
```
