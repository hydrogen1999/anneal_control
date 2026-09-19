# What the design document specifies that the experiments do not deliver

Checked against
`Learning_Control_Relevant_Representations_for_Embedded_Ising_Annealing/main.tex`
(931 lines, 2026-09-16) on 2026-09-19. Every row names the section of the
document that requires it and the artifact that does or does not exist.

---

## 1. A gate the document predeclares, that was never evaluated — and it fires

**§Stage A** commits to a threshold *in advance*:

> "choose an acceptable application regret ε in advance, such as an absolute
> two-percentage-point success loss … If the best-found one-window family
> exceeds that tolerance on a substantial held-out fraction, introduce a
> two-window or bin-allocation fallback. … This stage can reject an inadequate
> output family before spending effort on network architecture."

`family_restriction_loss[f] = best_loss[f] − best_found_loss` is exactly that
quantity and has been archived in every frontier sweep since 2026-09-16. It was
never aggregated against the threshold. Doing so now, over all six archived
sweeps (three topologies, train and validation):

| family | mean | median | p90 | **fraction exceeding 2 pp** |
|---|---:|---:|---:|---:|
| two_window | 0.0049 – 0.0115 | 0.0000 – 0.0004 | 0.016 – 0.037 | **5.6 % – 20.8 %** |
| one_window | 0.0156 – 0.0325 | 0.0024 – 0.0089 | 0.040 – 0.102 | **16.7 % – 41.7 %** |
| eight_bin | 0.0224 – 0.0410 | 0.0130 – 0.0264 | 0.055 – 0.110 | **34.7 % – 58.3 %** |
| linear | 0.1339 – 0.1830 | 0.0930 – 0.1638 | 0.250 – 0.354 | 91.7 % – 100 % |

**The gate fires.** One-window exceeds the predeclared tolerance on 17–42 % of
held-out records — a substantial fraction by any reading — so the document's
own rule requires the fallback.

**And it selects the opposite fallback from the one the architecture uses.**
The document offers "a two-window *or* bin-allocation fallback". Two-window is
three to five times better than eight-bin on every sweep. Eight-bin is the
*worst* tunable family at equal search budget despite having the most
parameters, because a higher-dimensional family is harder to search at 32
calls.

**This matters for the model, not just for the search.** `models.monotone_samples`
decodes the policy head to eight equal-time increments through a capped-simplex
water-filling — that is the `eight_bin` family. **The learned policy's output
family is the one Stage A would have rejected.** Nothing in the archive tests a
two-window output head.

This is the single most actionable gap: it is a check the document asked to be
run *before* architecture work, the data to run it has existed for three days,
and the answer contradicts a choice already baked into the model.

---

## 2. Factorial ablations: four of eight are incomplete

**§Evaluation, Table "Controlled tests and the conclusions they can support".**
`docs/paper_protocol.md:157` already concedes three of these as "optional
extensions beyond this completed baseline workflow".

| # | Factor | Status |
|---|---|---|
| 1 | Physical input (logical / physical / hierarchy) | **Partial.** Aware − blind = −0.007817 [−0.011093, −0.004471] over 48 parents, but the ledger labels it *exploratory, not Holm-adjusted*, and the Pegasus interval [−0.020195, +0.001062] includes zero. The document's headline factor is not confirmatory. |
| 2 | Spectral target (first gap → all gaps → response bins → response + intervention) | **Missing as a ladder.** `gap_inverse_square` and `d2` exist as *teachers*; response bins exist as *features*. The four-rung comparison that isolates "whether couplings, frequency resolution, or finite-time response add decision value" was never run. |
| 3 | **Mandatory bottleneck** G→D₂→ϱ versus direct residual branch, identical encoder | **Missing.** This is the document's Round-1 central resolution — "replace the mandatory G→D→ρ chain by operator-resolved auxiliaries plus a direct policy". Our model simply *has no bottleneck*, so the claim that the bottleneck destroys information is **assumed, not measured**. The falsifying test the document names for its own core design decision has not been run. |
| 4 | Control family | **Done** — all four families searched, and see §1. |
| 5 | Learning objective (imitation vs outcome, ± auxiliary physics) | **Partial.** `hierarchy_physics` and `hierarchy_outcome` are both trained and both appear in the comparison table, but no *paired contrast between them* is archived; the ledger explicitly says "no between-method architecture claim is established". |
| 6 | Symmetry (signed baseline / gauge augmentation / covariant model) | **Missing as evidence.** An `invariant-gauge` variant exists in the model contract and gauge code exists in five modules; no archived ablation result. |
| 7 | Transfer (zero-shot / equal-budget refinement / device adapter) | **Mostly done.** Zero-shot transfer archived; equal-budget refinement is the closed-loop filter (+0.00485). Device adapter needs a QPU. |
| 8 | Classical assistance (no-sampling / sample-assisted / sampler alone) | **Missing.** SVMC was tested as a *ranker* and ruled out (ρ ≈ −0.09). "Sampler alone" is therefore covered; **"sample-assisted model" was never built or tested.** |

---

## 3. The baseline the document calls the strongest overlap does not exist

**§Evaluation** requires "an implementation faithful to the closest
spectral-learning method when feasible", and **§Closest work** identifies it:
Tx-NQDT (Lu et al. 2026), "the strongest overlap in spectrum-guided hardware
scheduling", which uses transition matrix elements, normalised time density and
inverse cumulative reconstruction.

We implemented **Finžgar et al.** instead — and validated it properly
(80 seeds, GP-UCB − uniform = +0.1068 [+0.0809, +0.1323], 64/80). But the
document lists Bayesian schedule search and the closest spectral-learning method
as *two separate requirements*. Tx-NQDT is not implemented, and it is the
comparison a reviewer of this paper will ask for first, because its published
D-Wave result (44 of 60 cases beating the default linear schedule) is the direct
competitor.

---

## 4. Falsification suite: analytic checks exist, decision tests do not

**§Falsification suite** names six families. The distinction that matters is
that `verify_claims.py` checks the *formulas* on toy systems — the appendix says
so explicitly — while the document asks for these as tests of the *learned
decision*.

| Test | Analytic check | Learned-decision test |
|---|---|---|
| Two-qubit dark gap, tunable weak field | ✓ `verify_claims.py`, `spectral.py`, tests | **✗** |
| Equal-magnitude signed loops (frustration retention) | `planted_loops` is a data family | **✗ no falsification contrast** |
| Re-embedding the same logical objective | — | **✓** (14-qubit intervention, 53.9 % reversals) |
| Common rescaling (runtime covariance) | ✓ gauge/scale check | **✓** (matched scale arm, 1.52×) |
| Gauge transformations | ✓ spectral error < 3e-15 | **✗** |
| **≥2 transition regions + dense runtime scan for interference** | — | **✗ not attempted** |

The last one is the document's designated test for *coherent* failure modes and
is the one that would distinguish "a scalar profile works after averaging" from
"it works for a sharply specified coherent experiment". It is missing entirely.

---

## 5. Evaluation machinery that exists but is never reported

- **Time-to-solution.** `evaluation.time_to_solution` and the `n_q` formula are
  implemented (`evaluation.py:110`, `benchmarking.py:158`) exactly as
  §Amortization specifies, including the censoring rule. **No archived artifact
  reports it.**
- **Failure tails.** §Primary outcome asks for "distributions and failure tails,
  not only average improvements". Quantiles are reported in some artifacts and
  not others; there is no systematic tail reporting.
- **Model-mismatch bound** (Eq. modelbound, ε_model): requires hardware.

---

## 6. Blocked on hardware, correctly

§Stage D (D-Wave), §Gate devices, and the ε_model bound all need a QPU. The
document's own framing — "the executed waveform is the experiment" — means none
of this can be faked, and the project is right to defer it.

---

## 7. Where the experiments exceed the document

Four substantial results are not in the plan at all, and should be added to it
rather than left as orphans:

- **Erosion.** The noiseless-optimal control is the one an environment erodes
  most: ρ = −0.63 after removing both the [0,1] ceiling and the shared bank's
  candidate identity, flat across 6–8 qubits, present under a second channel.
  The document's §Round 1 anticipates *open-system arrival populations* as a
  falsifying test but does not anticipate this.
- **The closed-loop surrogate filter** with a random-chooser control
  (+0.00485, the control absorbing 39 % of the naive figure) — a concrete
  instance of the document's "equal-budget local refinement" row, measured more
  carefully than that row asks.
- **The scale ceiling, measured**: GPU generation to 20 physical qubits,
  entanglement growth bounding MPS to 50–100, SVMC ruled out at ρ ≈ 0,
  Lindblad wall at 8 qubits.
- **The over-optimisation null**: more noiseless search never hurt, at four
  noise rates — which rules out the most dramatic reading of the erosion result.

---

## Priority

1. **Re-run Stage A's gate as a predeclared decision and act on it** (§1). It is
   free — the data exists — and it questions the model's output family.
2. **The bottleneck ablation** (§2, factor 3). It is the document's own central
   design claim and is currently assumed.
3. **Tx-NQDT** (§3). The competitor a reviewer names first.
4. Multi-crossing runtime scan (§4) and the gauge ablation (§2, factor 6).
5. Report time-to-solution from existing data (§5) — also free.
