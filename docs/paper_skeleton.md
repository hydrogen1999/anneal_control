# Paper skeleton: the logic chain, checked against the evidence ledger

Every cell below names the artifact that supports it. A cell with no artifact is
marked **GAP** rather than written persuasively. This is the planning layer, not
prose.

## 1. Paper-type positioning

**Technique paper.** The contribution is a measured property of schedule
optimisation plus a method that exploits it, on an existing task. It is not a
new problem setting: embedded annealing control is established, and the datasets
and metrics here are ours but the task is not new.

Consequence for the narrative: the **Key Idea carries it** and "our goal" is a
short bridge. That ordering matters because the strongest result is an
observation about the world, not an artifact we built.

## 2. Thinking template

| Stage | Content | Evidence |
|---|---|---|
| **Research background** | An annealer's control is the schedule s(t). Classical adiabatic theory prescribes slowing where the spectral gap is small. Deployed problems are *embedded* into a hardware graph, which changes the spectrum the prescription depends on — and the spectrum is unavailable at device scale anyway. | — |
| **Limitation 1** | The spectral prescription is not the best use of a budget. Equal-budget search beats the local-adiabatic oracle by +0.110, and the gap does not close over a runtime ladder to 108 (+0.1410 / +0.0840 / +0.1004, all excluding zero). | `EVIDENCE.md`, teacher baselines |
| **Limitation 2** | Learned control policies are fit and evaluated in noiseless simulation; robustness is asserted, not measured. Prior open-system checks here covered a *fixed searched* control, never the learned rule. | `open_system_2026-09-17/` |
| **Limitation 3** | Comparisons routinely mix cost classes — a spectral oracle, an online search, and an amortised learned model — in one ranking column. | `paper_table.py`, `COST_CLASSES` |
| **Key Idea** | **The control that is optimal in a noiseless simulator is systematically the one an environment erodes most** — so a closed-system search curve overstates what it buys, by a factor that grows with the noise, and a smoothed learned preference is more robust than exact noiseless optimisation. Its right use is to *order* a search's proposals, not to replace the search. | `erosion_2026-09-18/`, `learned_robustness_2026-09-18/`, `closed_loop_2026-09-18/` |
| **Challenge 1** | Two artefacts can manufacture the erosion result. A loss in [0,1] makes level and change negatively correlated for free; and the bank is **shared**, with two waveforms winning 80% of records, so two fragile waveforms could carry the whole correlation. | Both removed: ρ −0.9081 raw → −0.7925 (ceiling) → −0.9320 (candidate fixed effect) → **−0.6312 with both**, negative in 192/200 |
| **Challenge 2** | Any filter that scores many proposals also *sees* more proposals. A gain could be the wider stream rather than the model. | Random chooser on the identical stream: +0.00137, **changes sign** |
| **Challenge 3** | Claims need a scale at which exact simulation is possible, and an honest statement of where that stops. | GPU ladder to 20 q; entanglement → MPS 50–100 q; SVMC ruled out at ρ ≈ 0 |
| **Methodology topic sentence** | Three instruments, each built so that the confound it addresses is measured rather than argued. | — |
| **Module A** (Challenge 1) | One shared Lindblad loss table across all selectors; headroom normalisation; a candidate fixed effect; a second noise channel. Sharing the table makes selector comparisons exact, and a test pins the shared path against the direct path at rel=0, abs=0. | `open_system.bank_loss_table`, `scripts/erosion_control.py` |
| **Module B** (Challenge 2) | An in-loop surrogate filter whose budget is simulator calls, with a paired random-chooser control on the identical oversampled stream and a verified-identical baseline. | `search.optimize_control_family(surrogate=…)`, `closed_loop_summary.py` |
| **Module C** (Challenge 3) | A measured scale ladder rather than an assumed one: GPU throughput to 20 physical qubits, entanglement growth to bound MPS, SVMC closed by measurement. | `throughput_2026-09-18/`, `entanglement_2026-09-18/`, `svmc_2026-09-18/` |
| **Contribution 1** | **The erosion result, and the discount it implies.** Within a record, closed-system quality predicts degradation at **ρ = −0.63 after removing both the [0,1] ceiling and the shared bank's candidate fixed effect**, negative in 192/200; unchanged at 7 qubits (−0.5952, 38/40); present under amplitude relaxation (−0.39, 78/100) at half strength. Consequence, measured over a budget ladder: the fraction of a noiseless search's gain that survives an environment falls **86.5% → 50.4%** as the rate goes 0.1 → 0.5. **The stronger claim that more search eventually hurts is FALSE** — the noisy budget curve is monotone at every rate tested. Erosion is a discount, not a reversal. | §4 |
| **Contribution 2** | **A learned selector is more robust than exact noiseless optimisation.** Retention of the noiseless advantage at dephasing 0.1: noise-aware oracle 82.5%, learned 69.5–74.5%, exact noiseless argmax 61.5%. Paired retention contrast excludes zero for all three seeds. | §5 |
| **Contribution 3** | **Filtering a search with the critic improves a fixed simulator budget** by +0.00485 across five configurations, every interval excluding zero, after subtracting the random-chooser control — replicated on real Pegasus connectivity at +0.00530. Online overhead 0.91%. | §6 |
| **Contribution 4** | **A measured scale ceiling**, including three routes closed or bounded by measurement rather than by argument. | §7 |

## 3. Self-consistency checks

**Check 1 — Limitations → Key Idea: PASS.** L1 says the spectral prescription
loses to search; L2 says robustness is unmeasured. The Key Idea addresses both:
it explains *why* noiseless optimisation underperforms (erosion) and supplies a
rule that is measured under noise. L3 is addressed structurally by Module B's
cost-class discipline rather than by the Key Idea, which is a weaker link —
noted below.

**Check 2 — Key Idea → Challenges: PASS.** C1 is the artefact that would fake
the erosion result; C2 is the artefact that would fake the filtering result; C3
is the precondition for measuring either. Each arises from implementing the
idea, none is invented to justify a module.

**Check 3 — Challenges → Methodology: PASS.** A↔C1, B↔C2, C↔C3, one to one.

**Check 4 — Methodology → Contributions: PASS with one caveat.** A→1 and 2,
B→3, C→4. Contribution 2 draws on Module A plus the checkpoint evaluation, so
the mapping is not strictly one-to-one; that is acceptable because 1 and 2 are
the same instrument applied to different selectors.

## 4. Severity summary

**0 CRITICAL, 2 MAJOR, 3 MINOR** (a third major was resolved by measurement; the row is kept struck through rather than deleted).

| | Issue | Why it matters |
|---|---|---|
| MAJOR | **Effect sizes are small.** +0.00485 on a loss near 0.70 is 0.7%, and the erosion finding's most quotable consequence — "optimise less" — was tested and is false. What remains is a discount factor and a robustness ordering, both real and both modest. | This is the main obstacle to an oral, not to acceptance. |
| MAJOR | **No QPU.** Every claim is simulation. | A reviewer can discount the whole robustness story as model-dependent. |
| ~~MAJOR~~ **RESOLVED** | The fine ranking fails on Pegasus (ρ = +0.16 among the best 10%, positive in 28/42) yet the in-loop gain survives. | Tested with a third arm that keeps the critic's better half and then chooses at random. Tail-avoidance is **75%** of what the model buys (+0.00334 [+0.00219, +0.00459]) and fine ranking **25%** (+0.00111 [+0.00050, +0.00182]), both excluding zero. The explanation is now measured, not offered. |
| MINOR | Closed-loop evidence is `sobol_local` only, one instance family per dataset, ≤8 qubits synthetic / ≤14 Pegasus. | Narrows the scope statement, does not threaten it. |
| MINOR | **The erosion result cannot be replicated on Pegasus.** Its smallest records are 10 physical qubits and the density solver caps at 8 (measured: 1.16 s/solve at 6 qubits, 5.12 at 7, 30.4 at 8). | The Pegasus bank has 64 candidates across two distinct banks and would have been the strongest available independent control. Stated as a gap. |
| MINOR | L3 (cost classes) is addressed by infrastructure, not by the Key Idea, so it reads as a methods contribution rather than part of the chain. | Consider demoting it from Limitations to Methods. |

**Top three fixes, in order:**

1. ~~Test the Pegasus explanation directly.~~ **Done.** The reject-worst arm
   attributes 75% of the gain to tail-avoidance and 25% to fine ranking, both
   intervals excluding zero. A critic does not have to be a good regressor to
   be a useful filter; it has to be right about which proposals are bad.
2. ~~Broaden the erosion result beyond one channel and one rate.~~ **Done.**
   Three rates and two channels: the effect holds under amplitude relaxation at
   about half the strength, and unlike dephasing it does not grow with rate.
   Whether the effect is a small-system artefact is still open — runs at 7 and
   8 qubits are the last reachable sizes.
3. Obtain QPU access (author-supplied).

## 5. Next step

With the checks passing, the next skill in the chain is `intro-drafter` for the
Introduction paragraph outline. Do not draft prose until fix 2 lands, because
Contribution 1 is the lead and its generality is still being measured.
