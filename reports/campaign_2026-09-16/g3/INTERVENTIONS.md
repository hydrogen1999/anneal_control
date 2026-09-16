# Paired embedding interventions (G3)

Software output of `annealctrl intervention-sweep`. Every effect below is measured **inside the declared closed-system simulator**; none of it is evidence about unmeasured hardware. Transfer penalties are signed and unclipped.

- Pairs: 1060 over 96 independent logical parents
- Physical size matched: True
- Censored by numerical resolution: 254 (24.0%)
- One-sided (decisive in a single direction): 196
- Identical selected waveform on both arms: 239
- Pairs with a negative direction (search asymmetry, kept unclipped): 187 (17.6%)
- Decisive preference reversals: 610 of 1060 pairs (57.5%)
- Failed units excluded: 0
- Verdict: **preference_changes_measured**

## Transfer penalty by scale arm

| scale arm | pairs | incl. | reversals | mean | p50 | p90 | bootstrap CI |
|---|---:|---:|---:|---:|---:|---:|---|
| `scale_controlled` | 248 | 237 | 219/248 (88.3%) | 0.051464 | 0.051886 | 0.080909 | [0.0465, 0.0566] |
| `total_compiled_effect` | 812 | 569 | 391/812 (48.2%) | 0.020103 | 0.016548 | 0.042917 | [0.0172, 0.0233] |

## Transfer penalty by factor

| factor | pairs | incl. | reversals | mean | p90 |
|---|---:|---:|---:|---:|---:|
| `chain_strength` | 384 | 358 | 315/384 (82.0%) | 0.046444 | 0.071455 |
| `field_allocation` | 96 | 18 | 3/96 (3.1%) | 0.001244 | 0.002510 |
| `geometry` | 332 | 231 | 149/332 (44.9%) | 0.006301 | 0.018409 |
| `ports` | 248 | 199 | 143/248 (57.7%) | 0.014563 | 0.045037 |

## Reading this table

- A **decisive reversal** requires both directions to be decisive: importing B's control into A must cost more than A's own numerical ambiguity, and vice versa. A pair decisive in only one direction is reported as `one_sided`, not as a reversal.
- The reversal rate is quoted against **all** pairs. The conditional ratio P(reversal | resolved) is 1 by construction - both penalties exceeding their ambiguity already implies each arm's own control wins on its own arm - so it is kept only as an internal consistency check and never reported as a finding.
- `total_compiled_effect` uses each arm's own programmed scale, as a device would. `scale_controlled` compiles both arms at one conservative common scale so the intervened factor moves and the global H_Z scale does not. They answer different questions and are never pooled.
- **pairs** is every pair; **incl.** is those decisive in at least one direction, which is the population behind the penalty statistics; **swaps / resolved** counts pairs decisive in BOTH directions. The three columns are different populations and the swap fraction must be read against its own denominator, not against `incl.`.
- A zero reversal rate with a well-resolved population is a result: it says the intervention does not change which control is preferred on this distribution.

Total objective calls charged: 411280.
