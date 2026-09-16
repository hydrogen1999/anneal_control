# Control-complexity frontier (G2)

Software output of `annealctrl control-sweep`. Every quantity below is a **finite-budget best-found** reference, not a global control optimum, and the search family, seed and budget are part of its definition.

- Split: `validation`
- Records: 864 over 48 independent logical parents
- Censored by numerical resolution: 5 (0.6%)
- Low-headroom linear reference: 3.6% of records
- Records with audit failures: 0
- Failed units excluded: 0
- Verdict: **resolved_headroom_present**

## Headroom (linear loss − best found), parent-averaged

| n parents | mean | p10 | p25 | p50 | p75 | p90 | max |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 48 | 0.106127 | 0.064939 | 0.073242 | 0.112511 | 0.127359 | 0.155395 | 0.193434 |

Parent bootstrap 95% CI: [0.095320, 0.117165] over 10000 resamples, unit of independence = logical parent. This interval does **not** include training-seed uncertainty.

## Family restriction loss (family best − overall best), parent-averaged

| family | n parents | mean | p50 | p90 | wins |
|---|---:|---:|---:|---:|---:|
| `eight_bin` | 48 | 0.012955 | 0.010634 | 0.024509 | 68 |
| `linear` | 48 | 0.106127 | 0.112511 | 0.155395 | 0 |
| `one_window` | 48 | 0.009621 | 0.008760 | 0.018319 | 136 |
| `pause` | 48 | 0.019927 | 0.017841 | 0.035413 | 312 |
| `two_window` | 48 | 0.003394 | 0.002453 | 0.006175 | 343 |

## Reading this table

- Censored records are excluded from every headroom statistic above and counted separately. A fully censored population means the distribution offers no control signal above the integrator's own resolution, which is a result, not a bug.
- Families are nested **only** through the linear incumbent that each search evaluates first. A one-window waveform is not exactly representable in the eight-bin duration parameterisation, so the restriction losses above compare family outcomes at equal budget rather than positions on a nesting ladder.
- Increasing the budget can only lower the reference, so headroom is a lower bound; it is not a global control optimum.

Total objective calls charged: 222048. Total integrator steps: 254561145.
