# G2/G3 campaign results — apollo, 2026-09-16

Numbers from the run described in `CAMPAIGN_G2_G3.md`. Artifacts under
`campaign_2026-09-16/`. Every row of every sweep carries one source hash; the
campaign was re-run after the slope fix so nothing here mixes source revisions.

**Zero failed units across all three sweeps.**

---

## G2 — is there control headroom above the measurement's own resolution?

**Yes, decisively.** Headroom is `linear loss − best found` under a 257-call
budget (5 families, 64 per tunable family, exact switching waveforms).

| split | parents | records | censored | headroom mean | 95% CI (parent bootstrap) | p50 | p90 | max |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| validation | 48 | 864 | 5 (0.6%) | **0.1061** | [0.0953, 0.1172] | 0.1125 | 0.1554 | 0.1934 |
| train | 144 | 2592 | 29 (1.1%) | **0.1025** | [0.0965, 0.1087] | 0.0925 | 0.1516 | 0.2075 |

Train and validation agree to within their intervals, which is the distributional
consistency the split is supposed to show. Roughly **10 percentage points of
decoded success** separate the matched-duration linear schedule from the best
control found.

### The compression boundary: cost of restricting to one family

Parent-averaged `family best − overall best`, with the number of records each
family won:

| family | validation cost | wins (val) | train cost | wins (train) |
|---|---:|---:|---:|---:|
| `two_window` | **0.0034** | 343 | **0.0032** | 906 |
| `one_window` | 0.0096 | 136 | 0.0091 | 483 |
| `eight_bin` | 0.0130 | 68 | 0.0116 | 247 |
| `pause` | 0.0199 | 312 | 0.0171 | 927 |
| `linear` | 0.1061 | **0** | 0.1025 | **0** |

Two readings, both relevant to the paper's compression claim:

1. **Linear never wins** — 0 of 864 and 0 of 2592 records. The matched-duration
   linear schedule is beaten on every single instance in this distribution.
2. **The frontier is steep then flat.** Restricting to `two_window` costs ~3% of
   the available headroom; restricting to `linear` costs 100%. The richer
   `eight_bin` family is *worse* than `two_window` at equal budget — expressivity
   bought optimisation difficulty, not better controls, which is exactly the
   confound `docs/paper_protocol.md` §5.3 warns about.

The budget-sensitivity panel of `figure3_frontier.pdf` shows every family still
descending slightly at 64 calls, so these headrooms remain **lower bounds**.

### Screening (G2b) — fitted on train, applied to validation

| | |
|---|---|
| quantity | `headroom` (measured control gain; a model's advantage is not an allowed quantity) |
| threshold | 0.1276 = 75th percentile of the 144 train parents |
| applied to | 48 validation parents, **parent overlap 0** — genuinely out of sample |
| selected | **12/48 parents (25.0%)** |
| screening cost | 888,192 objective calls |

Labelled `is_unbiased_deployment_sample: false`: a conditional stress benchmark,
not a claim about how often such instances occur.

---

## G3 — does an embedding change alter the *preferred* control?

**Yes**, on the majority of pairs. 1060 pairs over 96 independent parents.

| | |
|---|---|
| censored by numerical resolution | 254 (24.0%) |
| decisive in one direction only (`one_sided`) | 196 |
| identical selected waveform on both arms | 239 |
| **decisive preference reversals** | **610 of 1060 (57.5%)** |
| pairs with a negative direction (search asymmetry) | 187 (17.6%) |

A reversal requires **both** directions to be decisive against their own
ambiguity. The conditional ratio P(reversal | resolved) is 1 by construction and
is not a finding; the rate above is against all pairs.

### The scale-controlled arm changes the conclusion

| scale arm | pairs | transfer penalty mean | 95% CI | reversals |
|---|---:|---:|---|---:|
| `scale_controlled` | 248 | **0.0515** | [0.0465, 0.0566] | 219/248 (88.3%) |
| `total_compiled_effect` | 812 | 0.0201 | [0.0172, 0.0233] | 391/812 (48.2%) |

**These two marginal means must not be divided by each other.** An earlier
revision of this report did exactly that and claimed holding scale fixed makes
the effect "2.6× larger". That ratio is a composition artefact. A
`scale_controlled` arm is only constructed when the intervention actually moves
α, so the controlled column contains only `chain_strength` and `ports`, while
the 812-pair column is 231 `geometry` and 18 `field_allocation` pairs as well.
Those factors carry much smaller penalties and are absent from the numerator, so
the ratio charges the difference between factors to the scale control.

The matched contrast pairs each intervention with its own controlled twin, so
factor composition cancels:

| quantity | value |
|---|---:|
| matched pairs (parents) | 214 (95) |
| matched censored, excluded | 34 |
| factors matched | `chain_strength` 167, `ports` 47 |
| `scale_controlled` | 0.0583 |
| `total_compiled_effect` | 0.0384 |
| within-pair difference | **+0.0199**, 95% CI [0.0132, 0.0267] |
| matched ratio | **1.52×** |
| confounded marginal ratio | 2.56× *(superseded)* |
| pairs where holding scale raises the penalty | 130/214 (60.7%) |

Holding the global `H_Z` scale fixed raises the measured transfer penalty by
**0.0199 [0.0132, 0.0267]**, a **1.52×** effect rather than 2.6×. The finding
survives the correction — the interval excludes zero and a clear majority of
matched pairs move the same way — but it is a little over half the size the
uncorrected ratio suggested, and it is no longer unanimous: 84 of 214 matched
pairs move the other way. Raising κ lowers α, and that drop partially
*compensates* the penalty change. Reporting only the total compiled effect still
understates the causal role of chain strength, which is why ADR-0003 requires
both arms and why `aggregate_interventions` refuses to pool them — but the size
of that understatement is 1.52×, on matched pairs, weighted by parent.

`scale_arm_matched_contrast` now computes this automatically and
`aggregate_interventions` reports it beside the arms, with
`unmatched_is_composition_confounded` set whenever the two arms' factor
compositions differ. Every number above is weighted by parent, the unit of
independence, so the point estimate and its interval describe the same
estimand.

### By intervened factor

| factor | pairs | penalty mean | p90 | reversals |
|---|---:|---:|---:|---:|
| `chain_strength` | 384 | 0.0464 | 0.0715 | 315/384 (82.0%) |
| `ports` | 248 | 0.0146 | 0.0450 | 143/248 (57.7%) |
| `geometry` | 332 | 0.0063 | 0.0184 | 149/332 (44.9%) |
| `field_allocation` | 96 | 0.0012 | 0.0025 | 3/96 (3.1%) |

Ordering: **chain strength ≫ ports > geometry ≫ field allocation**. Field
allocation is essentially inert here, and the reason is visible in the generator:
two of the four families (`weighted_maxcut`, `planted_loops`) have zero logical
fields, so 148 further candidate pairs were skipped as vacuous before the run.

### A worked pair

`parent_0086__chain_strength_3__t8__scale_controlled` — same logical objective,
same α = 0.3333, only κ differs (1.5 vs 3.0):

```
            executed on A   executed on B
control A       0.0676          0.4424
control B       0.3570          0.0730
```

Arm A prefers `eight_bin`, arm B prefers `two_window`. Transferring the wrong
arm's control costs 0.289 and 0.369 — about 550× the combined numerical
ambiguity (5.3e-4 and 2.7e-4).

---

## GPU

`profile-generation`, 10 physical qubits, 4608 accepted labels, one worker each:

| backend | wall | accepted labels/s |
|---|---:|---:|
| `numpy` | 243.48 s | 18.93 |
| `cupy` | 225.30 s | 20.45 |

**1.08×**, with CPU/GPU outcome parity at **1.22e-15**. CuPy generation is
restricted to one parent worker while NumPy runs many, so twelve CPU shards beat
one GPU process by roughly twelve. The campaign ran on CPU for that reason.

Installing CuPy did close a real gate: the suite on the host is **512 passed,
0 skipped** — the two CuPy/CUDA parity tests that `V02_VERIFICATION.md` records
as never executed now run and pass on the device.

---

## Cost

| stage | objective calls | integrator steps |
|---|---:|---:|
| G2 validation | 222,048 | 254,561,145 |
| G2 train | 666,144 | 720,114,519 |
| G3 interventions | 411,280 | — |
| **total** | **1,299,472** | — |

Wall clock on 12 concurrent shards at `nice 10`, sharing a 32-core machine with
~20 other users: G3 ≈ 30 min, G2 validation ≈ 20 min, G2 train ≈ 50 min.

---

## What these results do not establish

- **No hardness claim.** Headroom is a property of the declared control families,
  the runtimes, and the closed-system simulator.
- **No global optimum.** Every reference is best-found under a stated budget and
  seed, and the budget curves were still descending.
- **No hardware claim.** Toy lifted graphs are not a commercial topology; every
  causal statement is inside the declared simulator.
- **Nothing about a learned model.** G2 measures the *opportunity* and G3 measures
  that embedding information is *decision-relevant*. Whether a model can exploit
  either is the next gate, and a large headroom with no learnable structure
  remains a possible outcome.
- **No deployment frequency.** Factors, magnitudes, families and runtimes are
  declared choices in the config files.
