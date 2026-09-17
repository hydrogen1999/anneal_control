# Control-complexity frontier (G2)

Software output of `annealctrl control-sweep`. Every quantity below is a **finite-budget best-found** reference, not a global control optimum, and the search family, seed and budget are part of its definition.

- Split: `validation`
- Records: 24 over 12 independent logical parents
- Censored by numerical resolution: 0 (0.0%)
- Low-headroom linear reference: 0.0% of records
- Records with audit failures: 0
- Failed units excluded: 0
- Verdict: **resolved_headroom_present**

## Headroom (linear loss − best found), parent-averaged

| n parents | mean | p10 | p25 | p50 | p75 | p90 | max |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | 0.144104 | 0.063834 | 0.083184 | 0.134317 | 0.193913 | 0.241888 | 0.305305 |

Parent bootstrap 95% CI: [0.104721, 0.189098] over 10000 resamples, unit of independence = logical parent. This interval does **not** include training-seed uncertainty.

## Family restriction loss (family best − overall best), parent-averaged

| family | n parents | mean | p50 | p90 | wins |
|---|---:|---:|---:|---:|---:|
| `eight_bin` | 12 | 0.025351 | 0.023706 | 0.044855 | 1 |
| `linear` | 12 | 0.144104 | 0.134317 | 0.241888 | 0 |
| `one_window` | 12 | 0.019356 | 0.008130 | 0.042251 | 11 |
| `two_window` | 12 | 0.006332 | 0.003233 | 0.020881 | 12 |

## Headroom by physical size

| physical qubits | records | censored | parents | mean | p50 | p90 | bootstrap CI |
|---:|---:|---:|---:|---:|---:|---:|---|
| 10 | 8 | 0 | 4 | 0.136183 | 0.134317 | 0.174561 | [0.0977, 0.1765] |
| 12 | 8 | 0 | 4 | 0.164887 | 0.150493 | 0.274382 | [0.0760, 0.2538] |
| 14 | 8 | 0 | 4 | 0.131242 | 0.108200 | 0.213965 | [0.0701, 0.2042] |

Pooling sizes would hide whether headroom survives as the system grows, so the ladder is reported stratified and never averaged across sizes.

## Reading this table

- Censored records are excluded from every headroom statistic above and counted separately. A fully censored population means the distribution offers no control signal above the integrator's own resolution, which is a result, not a bug.
- Families are nested **only** through the linear incumbent that each search evaluates first. A one-window waveform is not exactly representable in the eight-bin duration parameterisation, so the restriction losses above compare family outcomes at equal budget rather than positions on a nesting ladder.
- Increasing the budget can only lower the reference, so headroom is a lower bound; it is not a global control optimum.

Total objective calls charged: 2328. Total integrator steps: 6184473.
