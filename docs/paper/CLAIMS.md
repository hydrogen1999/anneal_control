# Every number the paper may use, and where it comes from

No figure or sentence in the draft may cite a quantity absent from this table.
This file exists because the one paper-facing artifact that lacked a generator
is also the one that carried a wrong number for a week: a share computed
against one denominator and described as another. Prose is not a source.

Regenerate the audited block with `python scripts/rebuild_evidence.py`;
`--check` fails on drift.

## Headline: bank selection against a matched linear ramp

| quantity | synthetic (48 parents, 5 seeds) | Pegasus-240 (48 parents, 3 seeds) | source |
|---|---|---|---|
| linear − learned | +0.05627 [+0.04943, +0.06356] | **+0.08631 [+0.07216, +0.10086]** | `effect_size_2026-09-19/artifacts/` |
| Cohen's *d* (pooled over seeds) | **2.24** | **1.68** | same |
| parents won, pooled | 48 / 48 | 48 / 48 | same |
| parents won, worst single seed | 47 / 48 | **48 / 48** | same |
| tuned global − learned | +0.02077 [+0.01417, +0.02773], *d* 0.87 | +0.03246 [+0.02570, +0.03967], *d* **1.30** | same |

## Reads to 99 % confidence

| | linear | tuned global | learned | ratio vs linear | source |
|---|---:|---:|---:|---|---|
| synthetic | 22.3 | 19.2 | 17.5 | 1.279× [1.243, 1.319], 47/48 | `artifacts/tts_synth.json` |
| Pegasus-240 | 47.1 | 36.0 | 32.9 | **1.589× [1.513, 1.667]**, 47/48 | `artifacts/time_to_solution.json` |

The 47/48 on Pegasus is a **tie**, not a loss: `parent_0186` needs 3.00 reads
under both schedules, and *n*<sub>q</sub> is a ceiling. The selector has the
lower loss on all 48 and is worse on reads on none.

## The central claim: embedding information

Holm-corrected over all ten pairwise comparisons, bank mode.

| | separated pairs | the separated set | pooled aware − blind | source |
|---|---:|---|---|---|
| synthetic | 4 / 10 | logical vs all four aware, Holm 0.0005–0.0049 | −0.007817 [−0.011093, −0.004471] | `evidence_audit_2026-09-18/contrasts.json` |
| Pegasus-12 | **0 / 10** | nothing separates | −0.008350 [−0.020195, +0.001062] | same |
| **Pegasus-240** | **4 / 10** | logical vs all four aware, Holm **0.0005–0.0014** | **−0.014030 [−0.019930, −0.008772]**, 43/48 parents | same |

Non-rejection sets on Pegasus-240: {hierarchy_outcome, hierarchy_physics,
physical, summary} and {logical} **alone**. No two aware encoders separate.

**Caveat that must travel with this table.** `hierarchy_physics` and
`hierarchy_outcome` are the *same model* on Pegasus — `resolved_response_
fraction` is 0.0 there, so `response_weight` has nothing to weight and the
pair is degenerate by construction. The Holm family therefore has ten nominal
members and nine informative ones, which makes the correction conservative,
not lenient. The pooled aware mean also gives the hierarchy architecture half
its weight for the same reason.

## One forward pass, priced in simulator calls

| | synthetic | Pegasus-240 | source |
|---|---|---|---|
| bank selection (64) | 17.9 [16.2, 19.8], median 17 | **23.8 [21.5, 26.2]**, median 25 | `artifacts/*equiv*` |
| a perfect ranker, same bank | 25.2 [22.7, 27.9] | 35.5 [32.7, 38.2] | same |
| direct policy | wins 35/48, 10.0 [9.2, 11.1] | wins **26/48**, 13.5 [10.4, 17.3] | same |

Strategy-dependence, synthetic: policy gradient 15.5 [13.2, 18.1],
Sobol-local 17.9 [16.2, 19.8], Bayesian GP-EI 23.3 [20.2, 26.8], over
identical records and families. **The conservative claim is "≥ 15 calls".**

A parent the method does not beat has *no* call equivalent and is counted
separately (`n_no_gain`), never averaged in at the grid's first budget.

## The pool-size axis

Mean gain over a linear ramp, by what the critic may choose from.

| pool | synthetic (5 seeds) | Pegasus-240 (3 seeds) | source |
|---|---:|---:|---|
| its own proposals only | −0.01027, separates 3/5 | −0.00119, separates **0/3** | `pool_selection_2026-09-20/artifacts/` |
| + the linear ramp | −0.01587, 4/5 | −0.01139, 2/3 | same |
| + designed library (8) | **−0.03787, 5/5** | **−0.03132, 3/3** | same |
| full 64-candidate bank | −0.05627 | −0.08631 | effect-size artifacts |

Fallback rate self-adjusts 4.6–17.9 % (synthetic) and 16–55 % (Pegasus),
tracking generator quality rather than being tuned.

**No-harm is scoped.** 0 parents harmed in 384 cells for the *two-element*
pool — structural, since the only alternative is the reference itself. The
designed pool harmed 3 in 240 (synthetic) and 1 in 144 (Pegasus). The claim
must not be extended to the library.

## The factorisation

`share of findable = selector efficiency × bank coverage`

| | selector efficiency | bank coverage | share | source |
|---|---:|---:|---:|---|
| synthetic | 81.4 % | 73.9 % | 60.1 % | `artifacts/synth_both.json` |
| Pegasus-240 | 84.8 % | **74.1 %** | 62.9 % | `pegasus240_2026-09-20/artifacts/effect_size_both.json` |

Bank coverage agreeing within 0.2 points across two topologies is the
interesting number: the menu's share of what is findable looks like a property
of using 64 fixed controls, not of the problem.

**The withdrawn claim.** A previously published "60 % synthetic vs 83 %
Pegasus" compared a 257-call frontier headroom against a 64-candidate
bank-oracle headroom under one column header. Measured against the same
reference the topologies differ by **2.8 points**, not 23. Do not reuse the
old framing.

## Controlled negatives the paper should report

| result | number | source |
|---|---|---|
| scalar bottleneck costs accuracy | +0.00566 [+0.00230, +0.00907], Holm 0.0029 | `bottleneck_2026-09-19/` |
| an 8-number profile does not | Holm 0.45 | same |
| spectral **resolution as a target** buys nothing | bins − moments +0.00001 [−0.00104, +0.00118], Holm 0.9937 | `spectral_rung3_2026-09-20/` |
| auxiliary spectral loss buys nothing | +0.00055 [−0.00049, +0.00166] | same |
| but resolution as a **teacher** matters | d2 − gap −0.01608 [−0.02293, −0.00962] | `spectral_ladder_2026-09-19/` |
| gauge augmentation trades accuracy for robustness | −0.00683 [−0.01329, −0.00115] Holm 0.028; penalty difference +0.01100 [+0.00598, +0.01667] | `gauge_2026-09-19/` |
| the advantage does **not** grow with size | ρ −0.1231 (record), −0.2493 (parent) | `effect_size_2026-09-19/` |
| a bigger fixed menu saturates | last doubling +0.00546 (synthetic), +0.01014 (Pegasus) | `artifacts/*menu*` |

## Cost classes — never ranked across

`fixed` · `privileged_spectrum` · `amortised` · `online_adaptation`.

The learned selector is **amortised**: one forward pass, no simulation at
deployment. The 257-call frontier is **online_adaptation** *and* a privileged
test-split reference — `online_adaptation: true` on every row — used only as a
denominator, never as a comparator. A pool containing D₂ would be
`privileged_spectrum`, because `main.tex:637` forbids recomputing the spectral
teacher at deployment unless charged separately.

## What the paper may not claim

- Any hardware result. Everything is exact closed-system propagation.
- Anything above **14 physical qubits**.
- Factor 5 (auxiliary physics) on real connectivity — vacuous there.
- Rung 4 of the spectral ladder — never run; needs intervention labels on
  these records.
- A covariant-model arm (Factor 6) — not implemented.
- That non-separation means equivalence. It does not.
