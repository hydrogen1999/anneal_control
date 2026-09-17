# What is claimed, what measured it, and what it does not cover

One row per claim. If a claim is not in this table it is not supported by this
repository. Every number is reproducible by the command in its artifact
directory; every caveat is load-bearing, not decoration.

---

## 1. Choosing the control per instance is worth a lot

| | |
|---|---|
| **Claim** | Optimising the annealing schedule per instance beats a linear ramp by ≈10.6 percentage points of logical success. |
| **Evidence** | Validation headroom 0.1061, 95% parent CI [0.0953, 0.1172], 48 parents. Train 0.1025 over 144 parents. Linear wins 0 of 3456 records. |
| **Artifact** | `campaign_2026-09-16/g2_validation/`, `CAMPAIGN_G2_G3_RESULTS.md` |
| **Does not cover** | Simulator search, not a learned model. Budget-limited, so headroom is a lower bound. 3–10 physical qubits. |

## 2. The embedding decides which control wins

| | |
|---|---|
| **Claim** | Changing one embedding factor reverses which control is preferred in 57.5% of paired interventions. |
| **Evidence** | 610 of 1060 pairs show a decisive reversal, both directions decisive against their own numerical ambiguity. 24% censored and reported. |
| **Artifact** | `campaign_2026-09-16/g3/`, ADR-0003 |
| **Does not cover** | 8 physical qubits. `P(reversal \| resolved) ≡ 1` by construction and is **not** the finding; the rate is against all pairs. |
| **Correction** | The scale-arm effect was first reported as **2.6×** by dividing two marginal means with different factor compositions. Matched value is **1.52×**, +0.0199 [0.0132, 0.0267] over 214 pairs. ADR-0008. |

## 3. The privileged spectral schedule is not the ceiling

| | |
|---|---|
| **Claim** | The local-adiabatic rule, given the exact instantaneous gap, loses to a 64-candidate search by 0.110 and to a linear ramp by 0.023. |
| **Evidence** | 3447 units, 192 parents. `gap_inverse_square` 0.7159 vs linear 0.7013 vs search 0.6116. Against **search** the gap is +0.0540, +0.1671, +0.1105 at runtimes 1, 4, 12 — every interval excludes zero and the gap at 12 is twice the gap at 1. The oracle beats search on 0.7% of records (`d2`: 3.8%). Against **linear** it does catch up, reaching parity at runtime 12. |
| **Artifact** | `teacher_baselines_2026-09-17/` |
| **Does not cover** | **Cannot extend above 10 physical qubits at all**: a full spectral teacher needs the Hamiltonian diagonalised at every path point and is exponentially capped. The gap to search is **non-monotonic** in runtime, peaking at 4, so nothing here extrapolates beyond runtime 12 in either direction; a ladder at 12/36/108 is running to test it. `gap_inverse_square` resolves on only 44.7% of instances, and that subset is harder than the rest (linear 0.7085 vs 0.4565), so it is conditional. For `d2`, including audit failures flips its sign against linear. |

## 4. An amortised selector beats the privileged oracles

| | |
|---|---|
| **Claim** | On held-out parents, a learned bank selector reaches 0.5447 against the oracles' 0.5913 and 0.7750, needing only a forward pass at deployment. |
| **Evidence** | 864 records, 48 parents, all methods on the **same** records, grouped by cost class. |
| **Artifact** | `comparison_2026-09-17/` |
| **Does not cover** | Online adaptation still wins at 0.5055 — the honest gap is 0.039 and is printed. Oracle rows are measured on 846 and 432 records and are marked conditional. |

## 5. Embedding information matters; the architecture does not

| | |
|---|---|
| **Claim** | Embedding-aware encoders beat the blind one by 0.00782 [0.00447, 0.01109]; no aware encoder separates from any other. |
| **Evidence** | 10 pairwise contrasts, parent-paired, Holm-corrected. Every separation is the blind encoder losing. `summary`, the cheapest, ranks first. |
| **Artifact** | `heldout_2026-09-17/` |
| **Does not cover** | Failing to separate is not equality. With 48 parents the narrowest aware-vs-aware interval is ±0.0011; no equivalence test was run. |

## 6. Amortised generation fails for a fixable reason

| | |
|---|---|
| **Claim** | One round of dataset aggregation is worth −0.0226 in selected loss against a matched control; raising the proposal count 5.3× is worth −0.0038 and does not separate from zero. |
| **Evidence** | Four arms × three seeds. `CONTROL` reproduced `BEFORE` to every printed digit, so retrain noise is zero. `BANKEXT` isolates bank size: 22% of the raw gain. 16-proposal arm: −0.0038, CI [−0.00835, +0.00032], 26/48 parents. |
| **Artifact** | `dagger_2026-09-17/`, `proposals16_2026-09-17/` |
| **Correction** | The module was written to fix **critic distribution shift**. It does not: rho moves +0.003 against the control and is negative on one seed of three. What moves is **generation** (−0.0183 gap). The mechanism claim was wrong and is retracted in the module docstring. |

## 7. Headroom survives to 14 qubits on real device connectivity

| | |
|---|---|
| **Claim** | On genuine Pegasus P16 connectivity, headroom is 0.1362 / 0.1649 / 0.1312 at 10 / 12 / 14 physical qubits — no decay. |
| **Evidence** | 24 validation records over 12 parents, 0 censored, verdict `resolved_headroom_present`; train split complete at 144 records with the same picture (0.1466 / 0.1411 / 0.1471 / 0.1321 at 10 / 11 / 12 / 14). `two_window` wins 12 of 12 parents. |
| **Artifact** | `pegasus_2026-09-17/` |
| **Does not cover** | Real connectivity is **not a real device**: closed-system simulation, no noise, no calibration drift, no QPU job ever submitted. Budget 32 here against 64 in the main campaign. 12 parents give wide overlapping intervals — "does not decay" holds, "is constant" does not. |

## 8. The search baseline is not a straw man

| | |
|---|---|
| **Claim** | Bayesian optimisation at the same 257-call budget beats the quasi-random search by only 0.0019. |
| **Evidence** | 864 records; BO favoured on 47 of 48 parents. Consistent direction, small size. |
| **Artifact** | `comparison_2026-09-17/bayes_rows.json` |
| **Does not cover** | This is an *optimisation* baseline. There is still no comparison against published **learned** schedule methods (RL and similar), and none against published embedding-aware control methods. |

## 9. GPU throughput crosses over at 12 qubits

| | |
|---|---|
| **Claim** | CuPy reaches 2.83× NumPy at 12 physical qubits; at 10 the ratio is within noise of 1. |
| **Evidence** | numpy 2.62, cupy 7.41, repeat spread 3.6% and 2.1%, GPU exclusive with every other compute process paused, driver at nice 0. |
| **Artifact** | `backend_2026-09-17/crossover.json` |
| **Does not cover** | 14 and 16 qubits are **not measured**: repeated config failures, then the GPU was returned to the Pegasus sweep. The ratios 3.22×/10.67×/103.57× that appear in commit `eb9de9f` have no artifact and are withheld. The census in this artifact mislabels process ownership — see `census_labelling_defect` inside it. |

## 10. The advantage is not an artefact of simulating no environment

| | |
|---|---|
| **Claim** | Under local dephasing, the searched control still beats the linear ramp on every record tested, and the ordering never flips. |
| **Evidence** | Ten held-out records at 4–6 physical qubits. Mean headroom 0.1389 / 0.1193 / 0.0966 / 0.0708 at dephasing rate 0 / 0.02 / 0.05 / 0.1; best-found wins 10/10 at every rate. At rate zero the open-system pipeline reproduces the closed-system headroom to within 2 × 10⁻⁵, from a solver sharing no integration code with it. |
| **Artifact** | `open_system_2026-09-17/` |
| **Does not cover** | Local dephasing at a hand-chosen rate is **not a device model**: no thermal bath, no measured T1/T2, no per-qubit calibration, no readout error. Relaxation held at zero, so one noise axis only. 4–6 qubits — two records at N=7 were refused by the solver's own cap and are reported absent. Compares the *searched* control against linear; the amortised selector's degradation is untested. |

---

## Standing limitations that apply to everything above

1. **No hardware.** Every campaign number is a closed-system simulation; §10 adds an
   open-system check under hand-chosen local dephasing on ten small records, which
   is a perturbation study and not a device model. Real Pegasus connectivity is
   used in §7; a real Pegasus *machine* is not used anywhere.
2. **Scale.** §1–6 and §8 are at most 10 physical qubits, i.e. 1024 amplitudes. §7 reaches 14. §3 cannot be extended by construction.
3. **Budget-limited references.** Every "best found" is a finite-budget reference, never a global control optimum. Increasing the budget can only lower it, so headroom is a lower bound.
4. **Parent-level independence.** Every interval resamples logical parents, and none of them include training-seed uncertainty, which is reported separately.
