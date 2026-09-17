# Audit-corrected teacher comparison

This reanalysis excludes spectral controls without a passed sampled-point audit, including finite losses whose interpolation audit failed. Historical artifacts are preserved.

The learned and teacher means below use exactly the same teacher-specific records and equal logical-parent weight. Negative differences favour the learned method. Teachers require privileged spectral computation; learned policies require offline training. These are different cost classes, not equal-cost competitors.

| Teacher | Eligible records | Excluded records |
|---|---:|---:|
| d2 | 711 | 153 |
| gap_inverse_square | 339 | 525 |

| Learned mode | Teacher | Parents | Learned loss | Teacher loss | Difference | 95% parent CI |
|---|---|---:|---:|---:|---:|---|
| summary/bank | d2 | 45 | 0.526899 | 0.577298 | -0.050399 | [-0.069051, -0.032866] |
| summary/direct | d2 | 45 | 0.571397 | 0.577298 | -0.005901 | [-0.026212, +0.015268] |
| summary/bank | gap_inverse_square | 23 | 0.684249 | 0.767489 | -0.083240 | [-0.114283, -0.052837] |
| summary/direct | gap_inverse_square | 23 | 0.724170 | 0.767489 | -0.043319 | [-0.072295, -0.011880] |

All encoder/mode contrasts, eligible record IDs, status counts and input hashes are in `comparison_table.json`.

Intervals are descriptive, unadjusted parent bootstraps conditional on the archived learned seed averages; they do not include training-seed uncertainty. Passed sampled-point audits are numerical diagnostics, not continuous-path certificates. Conditional teacher rows are excluded from full-population rankings. This correction does not establish hardware performance or a general advantage over spectral schedules.

Reproduce from repository root:

```bash
PYTHONPATH=src python reports/comparison_audit_corrected_2026-09-17/reproduce.py
```
