# Audited evidence reanalysis — 2026-09-18

Authoritative replacement for the historical comparison table and historical contrast p-values. All values below are regenerated from committed record-level outcomes. Lower loss is better; loss is one minus decoded logical success. No training or new simulation is claimed.

## Common synthetic test population

864 records, 48 logical parents. Means give parents equal weight. Teacher rows are conditional and appear separately below. Search arms have matched record identities and per-record objective budgets; the archive does not retain search seeds, so seed matching is not asserted.

| Cost class | Method | Mean loss | 95% parent CI | Records |
|---|---|---:|---|---:|
| fixed | linear | 0.600929 | [+0.535595, +0.663042] | 864 |
| fixed | global | 0.565437 | [+0.499051, +0.629054] | 864 |
| amortised | hierarchy_outcome/bank | 0.545574 | [+0.478785, +0.609487] | 864 |
| amortised | hierarchy_outcome/direct | 0.590660 | [+0.525756, +0.651951] | 864 |
| amortised | hierarchy_physics/bank | 0.545025 | [+0.478357, +0.608790] | 864 |
| amortised | hierarchy_physics/direct | 0.590649 | [+0.526991, +0.650896] | 864 |
| amortised | logical/bank | 0.553182 | [+0.485946, +0.617472] | 864 |
| amortised | logical/direct | 0.589813 | [+0.522248, +0.653273] | 864 |
| amortised | physical/bank | 0.546196 | [+0.479682, +0.609621] | 864 |
| amortised | physical/direct | 0.597758 | [+0.534115, +0.658006] | 864 |
| amortised | summary/bank | 0.544662 | [+0.477669, +0.608827] | 864 |
| amortised | summary/direct | 0.590533 | [+0.526063, +0.651981] | 864 |
| online_adaptation | search_best_found | 0.507352 | [+0.439139, +0.572791] | 864 |
| online_adaptation | search_bayesian | 0.505460 | [+0.437440, +0.570880] | 864 |
| online_adaptation | search_policy_gradient | 0.511392 | [+0.442890, +0.576957] | 864 |

No global ranking across cost classes is defined. Offline training and label costs are not included in the zero deployment-query count. The 257-call references are budget-limited methods, not certified control optima.

## Audited teacher populations

Only finite teacher losses with `sampled_point_audit_passed` are eligible. Each contrast uses identical records for teacher and learned method. A sampled-point audit is a numerical diagnostic, not a continuous-path certificate.

| Teacher | Eligible records | Excluded records |
|---|---:|---:|
| d2 | 711 | 153 |
| gap_inverse_square | 339 | 525 |

| Learned method | Teacher | Parents | Learned − teacher | 95% parent CI |
|---|---|---:|---:|---|
| summary/bank | d2 | 45 | -0.050399 | [-0.069051, -0.032866] |
| summary/direct | d2 | 45 | -0.005901 | [-0.026212, +0.015268] |
| summary/bank | gap_inverse_square | 23 | -0.083240 | [-0.114283, -0.052837] |
| summary/direct | gap_inverse_square | 23 | -0.043319 | [-0.072295, -0.011880] |

These intervals are descriptive and unadjusted, conditional on teacher eligibility and archived learned seed averages. Different teacher rows concern different populations. They do not establish equal-cost superiority or hardware performance.

## Regenerated encoder contrasts

The ten encoder pairs form a separate Holm family for each dataset and mode. Approximate centered-null bootstrap p-values are corrected; intervals remain pointwise. Non-rejection does not establish equivalence. The pooled information effect is exploratory and outside those Holm families.

| Dataset | Contrast | Difference | 95% parent CI | Parents |
|---|---|---:|---|---:|
| synthetic | aware − blind (exploratory) | -0.007817 | [-0.011093, -0.004471] | 48 |
| synthetic | summary bank − global | -0.020775 | [-0.027734, -0.014167] | 48 |
| synthetic | summary direct − global | +0.025096 | [+0.016558, +0.034406] | 48 |
| pegasus | aware − blind (exploratory) | -0.008350 | [-0.020195, +0.001062] | 12 |
| pegasus | summary bank − global | -0.022905 | [-0.038362, -0.010572] | 12 |
| pegasus | summary direct − global | +0.055564 | [+0.029779, +0.082394] | 12 |
| pegasus240 | aware − blind (exploratory) | -0.014030 | [-0.019930, -0.008772] | 48 |
| pegasus240 | summary bank − global | -0.032458 | [-0.039710, -0.025580] | 48 |
| pegasus240 | summary direct − global | +0.052654 | [+0.037253, +0.068818] | 48 |

The two Pegasus hierarchy arms have no spectral-response labels in that dataset and are degenerate as an auxiliary-loss ablation. No architecture superiority, cross-topology transfer, equivalence, or minimum required sample size follows from these tables.

## Reproduction and remaining artifact limits

```bash
python scripts/rebuild_evidence.py
python scripts/rebuild_evidence.py --check
```

The command uses NumPy/SciPy and source code without importing PyTorch. JSON outputs include input/analysis hashes, all contrasts, exact teacher eligibility IDs, and search population audits. `--check` fails on any changed/missing output; it never silently refreshes artifacts.

Historical aggregation and proposal-count archives retain aggregate statistics but not the raw paired parent-by-seed panels needed to reconstruct new confidence intervals. The aggregation campaign also augmented validation; its gain is not isolated to train-only acquisition. The independent frozen-validation pilot remains a negative result. The regenerated comparisons do not repair missing raw training traces or prove data-regeneration identity. GPU crossover raw profiles and full manifest comparison inputs are not archived. Upstream commit 974207a now adds aggregate transfer measurements; see UPSTREAM_TRANSFER.md for their provisional interpretation and missing provenance. Budget-curve and new multi-seed acquisition campaigns must supply their own completed raw artifacts before being claimed.
