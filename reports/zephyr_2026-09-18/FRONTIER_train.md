# Control-complexity frontier (G2)

Software output of `annealctrl control-sweep`. Every quantity below is a **finite-budget best-found** reference, not a global control optimum, and the search family, seed and budget are part of its definition.

- Split: `train`
- Records: 144 over 72 independent logical parents
- Censored by numerical resolution: 0 (0.0%)
- Low-headroom linear reference: 0.0% of records
- Records with audit failures: 0
- Failed units excluded: 0
- Verdict: **resolved_headroom_present**

## Headroom (linear loss − best found), parent-averaged

| n parents | mean | p10 | p25 | p50 | p75 | p90 | max |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 72 | 0.133859 | 0.055948 | 0.081728 | 0.129975 | 0.173129 | 0.229894 | 0.294612 |

Parent bootstrap 95% CI: [0.118264, 0.149547] over 10000 resamples, unit of independence = logical parent. This interval does **not** include training-seed uncertainty.

## Family restriction loss (family best − overall best), parent-averaged

| family | n parents | mean | p50 | p90 | wins |
|---|---:|---:|---:|---:|---:|
| `eight_bin` | 72 | 0.022420 | 0.019333 | 0.042003 | 15 |
| `linear` | 72 | 0.133859 | 0.129975 | 0.229894 | 0 |
| `one_window` | 72 | 0.015575 | 0.012561 | 0.033709 | 46 |
| `two_window` | 72 | 0.006100 | 0.002407 | 0.016777 | 83 |

## Headroom by physical size

| physical qubits | records | censored | parents | mean | p50 | p90 | bootstrap CI |
|---:|---:|---:|---:|---:|---:|---:|---|
| 9 | 2 | 0 | 1 | 0.093168 | 0.093168 | 0.093168 | [n/a, n/a] |
| 10 | 46 | 0 | 23 | 0.155927 | 0.145498 | 0.239828 | [0.1293, 0.1824] |
| 11 | 2 | 0 | 1 | 0.081743 | 0.081743 | 0.081743 | [n/a, n/a] |
| 12 | 46 | 0 | 23 | 0.142621 | 0.131261 | 0.240135 | [0.1121, 0.1732] |
| 13 | 4 | 0 | 2 | 0.106322 | 0.106322 | 0.110280 | [0.1014, 0.1113] |
| 14 | 44 | 0 | 22 | 0.108351 | 0.112752 | 0.173965 | [0.0852, 0.1315] |

Pooling sizes would hide whether headroom survives as the system grows, so the ladder is reported stratified and never averaged across sizes.

## Reading this table

- Censored records are excluded from every headroom statistic above and counted separately. A fully censored population means the distribution offers no control signal above the integrator's own resolution, which is a result, not a bug.
- Families are nested **only** through the linear incumbent that each search evaluates first. A one-window waveform is not exactly representable in the eight-bin duration parameterisation, so the restriction losses above compare family outcomes at equal budget rather than positions on a nesting ladder.
- Increasing the budget can only lower the reference, so headroom is a lower bound; it is not a global control optimum.

Total objective calls charged: 13968. Total integrator steps: 36271986.
