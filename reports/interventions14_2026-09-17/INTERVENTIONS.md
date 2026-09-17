# Paired embedding interventions (G3)

Software output of `annealctrl intervention-sweep`. Every effect below is measured **inside the declared closed-system simulator**; none of it is evidence about unmeasured hardware. Transfer penalties are signed and unclipped.

- Pairs: 228 over 20 independent logical parents
- Physical size matched: True
- Censored by numerical resolution: 50 (21.9%)
- One-sided (decisive in a single direction): 55
- Identical selected waveform on both arms: 43
- Pairs with a negative direction (search asymmetry, kept unclipped): 53 (23.2%)
- Decisive preference reversals: 123 of 228 pairs (53.9%)
- Failed units excluded: 0
- Verdict: **preference_changes_measured**

## Transfer penalty by scale arm

| scale arm | pairs | incl. | reversals | mean | p50 | p90 | bootstrap CI |
|---|---:|---:|---:|---:|---:|---:|---|
| `scale_controlled` | 50 | 47 | 45/50 (90.0%) | 0.091186 | 0.069595 | 0.171298 | [0.0681, 0.1169] |
| `total_compiled_effect` | 178 | 131 | 78/178 (43.8%) | 0.026952 | 0.022327 | 0.054994 | [0.0179, 0.0368] |

## Transfer penalty by factor

| factor | pairs | incl. | reversals | mean | p90 |
|---|---:|---:|---:|---:|---:|
| `chain_strength` | 80 | 75 | 69/80 (86.2%) | 0.079125 | 0.176275 |
| `field_allocation` | 20 | 3 | 1/20 (5.0%) | 0.000867 | 0.001015 |
| `geometry` | 78 | 59 | 28/78 (35.9%) | 0.008918 | 0.021491 |
| `ports` | 50 | 41 | 25/50 (50.0%) | 0.020199 | 0.064259 |

## Reading this table

- A **decisive reversal** requires both directions to be decisive: importing B's control into A must cost more than A's own numerical ambiguity, and vice versa. A pair decisive in only one direction is reported as `one_sided`, not as a reversal.
- The reversal rate is quoted against **all** pairs. The conditional ratio P(reversal | resolved) is 1 by construction - both penalties exceeding their ambiguity already implies each arm's own control wins on its own arm - so it is kept only as an internal consistency check and never reported as a finding.
- `total_compiled_effect` uses each arm's own programmed scale, as a device would. `scale_controlled` compiles both arms at one conservative common scale so the intervened factor moves and the global H_Z scale does not. They answer different questions and are never pooled.
- **pairs** is every pair; **incl.** is those decisive in at least one direction, which is the population behind the penalty statistics; **swaps / resolved** counts pairs decisive in BOTH directions. The three columns are different populations and the swap fraction must be read against its own denominator, not against `incl.`.
- A zero reversal rate with a well-resolved population is a result: it says the intervention does not change which control is preferred on this distribution.

Total objective calls charged: 88464.
