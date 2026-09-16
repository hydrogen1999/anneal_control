# Control-complexity frontier (G2)

Software output of `annealctrl control-sweep`. Every quantity below is a **finite-budget best-found** reference, not a global control optimum, and the search family, seed and budget are part of its definition.

- Split: `train`
- Records: 2592 over 144 independent logical parents
- Censored by numerical resolution: 29 (1.1%)
- Low-headroom linear reference: 4.5% of records
- Records with audit failures: 0
- Failed units excluded: 0
- Verdict: **resolved_headroom_present**

## Headroom (linear loss − best found), parent-averaged

| n parents | mean | p10 | p25 | p50 | p75 | p90 | max |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 144 | 0.102471 | 0.065200 | 0.073809 | 0.092546 | 0.127582 | 0.151605 | 0.207497 |

Parent bootstrap 95% CI: [0.096466, 0.108680] over 10000 resamples, unit of independence = logical parent. This interval does **not** include training-seed uncertainty.

## Family restriction loss (family best − overall best), parent-averaged

| family | n parents | mean | p50 | p90 | wins |
|---|---:|---:|---:|---:|---:|
| `eight_bin` | 144 | 0.011551 | 0.009934 | 0.020887 | 247 |
| `linear` | 144 | 0.102471 | 0.092546 | 0.151605 | 0 |
| `one_window` | 144 | 0.009076 | 0.006542 | 0.020562 | 483 |
| `pause` | 144 | 0.017148 | 0.013682 | 0.036155 | 927 |
| `two_window` | 144 | 0.003236 | 0.002316 | 0.006559 | 906 |

## Reading this table

- Censored records are excluded from every headroom statistic above and counted separately. A fully censored population means the distribution offers no control signal above the integrator's own resolution, which is a result, not a bug.
- Families are nested **only** through the linear incumbent that each search evaluates first. A one-window waveform is not exactly representable in the eight-bin duration parameterisation, so the restriction losses above compare family outcomes at equal budget rather than positions on a nesting ladder.
- Increasing the budget can only lower the reference, so headroom is a lower bound; it is not a global control optimum.

Total objective calls charged: 666144. Total integrator steps: 720114519.
