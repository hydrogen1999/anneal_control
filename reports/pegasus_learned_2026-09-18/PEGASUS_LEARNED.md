> Numerical update: use [the regenerated analysis](../evidence_audit_2026-09-18/RESULTS.md) for current confidence intervals and corrected tests. Historical raw outcomes are unchanged.

# The learned comparison on real Pegasus connectivity

5 encoders × 3 seeds = 15 runs, trained and evaluated on `data_pegasus_trainable.json`:
96 parents, 1152 records, 10–14 physical qubits on genuine Pegasus P16
connectivity, 64-candidate bank. Splits by parent: 72 train / 12 validation /
12 test. A historical regeneration report claims matching scientific arrays; its original
comparator excluded timing, provenance, and metadata and did not certify byte
equality. The two complete datasets and a fresh machine-readable audit are not
archived here, so the revised comparator has not independently verified that claim.
See `dataset_integrity_2026-09-18/INTEGRITY.md`.

## What replicates

**Bank selection beats the best single fixed schedule**, on every encoder, with
every interval excluding zero:

| method | bank loss | Δ vs global | 95% CI |
|---|---:|---:|---|
| `summary` | **0.62821** | −0.02291 | [−0.03830, −0.01058] |
| `hierarchy_outcome` | 0.63198 | −0.01913 | [−0.03453, −0.00693] |
| `physical` | 0.63391 | −0.01720 | [−0.03239, −0.00501] |
| `logical` | 0.63987 | −0.01124 | [−0.01827, −0.00492] |

The ordering is the same as on synthetic graphs: `summary` first, the
embedding-blind `logical` last. Direct proposals remain far behind (0.687–0.707).

## What remains unresolved

**The embedding-information effect does not separate here.**

| dataset | effect | 95% CI | parents | separated |
|---|---:|---|---:|:--:|
| synthetic, 3–10 qubits | −0.00782 | [−0.01109, −0.00447] | 48 | **yes** |
| **real Pegasus, 10–14 qubits** | **−0.00835** | **[−0.02019, +0.00106]** | **12** | **no** |

The point estimates are similar, but the interval is **3.2× wider** and crosses
zero. Twelve test parents leave substantial uncertainty. The data neither
establish replication of the information effect nor identify insufficient power
as the unique explanation. The former estimate of approximately 19 required
parents was calculated from the observed test effect and is not a prospective
power analysis or a guarantee of separation.

After Holm correction over all ten pairs, no pair separates and all five encoders
form one maximal set of pairwise non-rejections. This does not establish
equivalence. Fresh centered-null/Holm results are in
`reports/evidence_audit_2026-09-18/contrasts.json`; this directory retains its
historical contrast JSON unchanged.

**The honest statement is therefore:** the effect size seen on synthetic graphs
reappears on real connectivity at larger sizes, and this dataset cannot establish
it. Claiming the embedding result "holds on real hardware topology" would be
reading a point estimate as a finding.

## A degenerate arm that must not be reported as an ablation

`hierarchy_outcome` and `hierarchy_physics` returned **identical losses to six
decimals** (0.634078 on seed 0) with different checkpoint hashes, and their
contrast is exactly `+0.00000 [+0.00000, +0.00000]`.

They are the same model here. `hierarchy_physics` differs only by
`response_weight = 0.05`, an auxiliary loss on spectral response targets — and a
full spectral teacher caps at 10 physical qubits, so this dataset is forced to
`teacher.mode = none` and carries no such targets. The auxiliary term has nothing
to act on.

The current full-spectrum teacher configuration caps at 10 qubits. This is an
implementation and compute limitation, not a theorem prohibiting other spectral
approximations. In this campaign the arm is degenerate, not a valid null ablation.

## What this does not cover

**12 test parents.** Every interval here is wide. The one clearly supported claim
— bank selection beats a fixed schedule — survives it; the encoder comparison
does not.

**Connectivity, not a device.** No working-graph exclusions, no per-qubit
calibration, no noise, no QPU job.

**No oracle, no physics ablation.** Both need a spectral teacher, capped at 10
qubits.

**Next evaluation.** Freeze an expanded-parent protocol using a scientifically
meaningful effect or interval-width target before reading new test outcomes.
More independent parents can narrow uncertainty but cannot guarantee a positive
effect or repair an invalid ablation.
