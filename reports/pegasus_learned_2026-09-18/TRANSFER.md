# The architecture does not pay in-distribution. It pays for transfer.

Five encoders × three seeds, trained on **synthetic quotient graphs at 3–10
physical qubits**, evaluated on **144 real Pegasus records at 10–14 qubits** that
they never saw. Disjointness established by **content fingerprint** (overlap
zero), not by parent id — the two datasets both number parents from
`parent_0000` and seven collide, which is why the first run of this evaluation
refused every checkpoint.

Nothing is refitted. The checkpoint's own normalizer travels with it, because
refitting on the target is an adaptation and answers a different question.

## The selector transfers; the blind encoder does not

| trained on synthetic | loss on Pegasus | beats linear | verdict |
|---|---:|---:|---|
| `hierarchy_physics` | **0.65855** | 83% | beats linear |
| `hierarchy_outcome` | 0.66333 | 81% | beats linear |
| `physical` | 0.67082 | 73% | beats linear |
| `summary` | 0.67413 | 73% | beats linear |
| `logical` (embedding-blind) | 0.70584 | 72% | **indistinguishable from linear** |
| *linear reference* | *0.7071* | — | — |
| *bank-best ceiling* | *0.6116* | — | — |

## The ranking inverts, and it separates

Holm-corrected over all ten pairs:

| comparison | in-distribution | **out-of-distribution** |
|---|---|---|
| `hierarchy_outcome` vs `summary` | −0.00091, holm 1.000, **not separated** | **+0.01080** [+0.00276, +0.01894], holm **0.0498**, separated |
| `hierarchy_physics` vs `summary` | −0.00036, holm 1.000, not separated | **+0.01557** [+0.00884, +0.02313], holm **0.0010**, separated |
| `hierarchy_physics` vs `physical` | +0.00117, holm 0.912, not separated | **+0.01227** [+0.00755, +0.01720], holm **0.0010**, separated |
| aware vs blind | −0.00782 [−0.01109, −0.00447] | **−0.03913** [−0.07219, −0.00889] |

*(sign convention: positive means the second-named method is worse)*

**In distribution, `summary` — 64K parameters, no message passing — ranks first
and nothing separates from it. Out of distribution the sign flips and both
hierarchical encoders separate from it after correction.** The embedding
information effect is **5× larger** off-distribution (−0.039 against −0.0078).

Indistinguishable groups out of distribution: `{hierarchy_outcome,
hierarchy_physics, physical}` and **`{logical, summary}`**.

## What that second group means

`summary` encodes **pooled statistics** of the graph — moments that shift when
the graph distribution shifts. `hierarchical` encodes **structure**, through
message passing that does not depend on the moments being the same.

Under transfer, `summary` lands in the same group as the encoder that cannot see
the embedding at all. The natural reading is that pooled moments memorise a
distribution while message passing learns something that survives leaving it.
**That is a hypothesis consistent with this measurement, not a second
measurement**, and testing it would need an encoder ablation designed for the
purpose rather than reinterpreted after the fact.

## Why this matters for the paper

The architecture ablation was this project's clearest negative result: four
embedding-aware encoders mutually indistinguishable, the cheapest ranking first,
and therefore no support for the architecture at all. That negative stands **on
its own distribution**. It does not survive a transfer test, which this project
had never run.

The claim the paper can now make is narrower and more useful than either:

> The architecture buys nothing when train and test come from the same
> generator, and buys a separated advantage when they do not.

## What this does not cover

**12 test parents, one target distribution.** Separation after Holm with 12
parents means the effect is large, not that the test is powerful. A second target
distribution — Zephyr — has a headroom campaign but no trainable dataset yet.

**Fewer parents than the in-distribution comparison.** The in-distribution test
had 48 parents and found nothing; this has 12 and finds separation. That is
consistent with a larger effect, and it is not a like-for-like comparison of
statistical power.

**`hierarchy_outcome` and `hierarchy_physics` do not separate from each other**
(−0.00477, holm 0.228). The physics-auxiliary loss is not shown to help here; the
two are reported as one architectural family.

**Bank selection only.** Direct generation was not evaluated under transfer.

**Still no hardware.** Real connectivity, closed-system simulation.
