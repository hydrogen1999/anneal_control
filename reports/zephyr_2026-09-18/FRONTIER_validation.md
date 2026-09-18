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
| 12 | 0.147939 | 0.097645 | 0.107550 | 0.125957 | 0.172243 | 0.225625 | 0.301136 |

Parent bootstrap 95% CI: [0.114998, 0.186664] over 10000 resamples, unit of independence = logical parent. This interval does **not** include training-seed uncertainty.

## Family restriction loss (family best − overall best), parent-averaged

| family | n parents | mean | p50 | p90 | wins |
|---|---:|---:|---:|---:|---:|
| `eight_bin` | 12 | 0.036286 | 0.024966 | 0.081093 | 3 |
| `linear` | 12 | 0.147939 | 0.125957 | 0.225625 | 0 |
| `one_window` | 12 | 0.023925 | 0.023066 | 0.035580 | 8 |
| `two_window` | 12 | 0.008391 | 0.007846 | 0.014071 | 13 |

## Headroom by physical size

| physical qubits | records | censored | parents | mean | p50 | p90 | bootstrap CI |
|---:|---:|---:|---:|---:|---:|---:|---|
| 10 | 8 | 0 | 4 | 0.117771 | 0.123851 | 0.150562 | [0.0779, 0.1516] |
| 12 | 8 | 0 | 4 | 0.183036 | 0.165729 | 0.272289 | [0.1130, 0.2575] |
| 14 | 8 | 0 | 4 | 0.143011 | 0.123346 | 0.200486 | [0.1038, 0.1985] |

Pooling sizes would hide whether headroom survives as the system grows, so the ladder is reported stratified and never averaged across sizes.

## Reading this table

- Censored records are excluded from every headroom statistic above and counted separately. A fully censored population means the distribution offers no control signal above the integrator's own resolution, which is a result, not a bug.
- Families are nested **only** through the linear incumbent that each search evaluates first. A one-window waveform is not exactly representable in the eight-bin duration parameterisation, so the restriction losses above compare family outcomes at equal budget rather than positions on a nesting ladder.
- Increasing the budget can only lower the reference, so headroom is a lower bound; it is not a global control optimum.

Total objective calls charged: 2328. Total integrator steps: 6691803.
