# The embedding-information claim is confirmatory, not exploratory

## What the ledger said, and why it understated the evidence

The evidence ledger carried the encoder-information result as

> Exploratory pooled contrast, not Holm-adjusted.

That label is correct for the *pooled* aware-minus-blind statistic, which the
contrast tool itself marks `correction: none` and
`inference_role: exploratory_pooled_information_effect`. But the same tool also
produces a **Holm-corrected pairwise matrix over all ten comparisons**, and
that matrix had never been reported. It reaches the same conclusion four times,
each surviving correction on its own.

## Result

Held-out bank selection, `research_v1`, Holm-corrected across ten pairwise
comparisons:

| contrast | difference | Holm p | |
|---|---|---:|---|
| logical − hierarchy_outcome | +0.00761 [+0.00461, +0.01056] | **0.0005** | separated |
| logical − hierarchy_physics | +0.00816 [+0.00475, +0.01156] | **0.0005** | separated |
| summary − logical | −0.00852 [−0.01205, −0.00497] | **0.0008** | separated |
| physical − logical | −0.00699 [−0.01100, −0.00293] | **0.0049** | separated |
| physical − hierarchy_physics | +0.00117 [−0.00041, +0.00285] | 0.9594 | — |
| hierarchy_physics − hierarchy_outcome | −0.00055 [−0.00166, +0.00050] | 1.0000 | — |
| physical − hierarchy_outcome | +0.00062 [−0.00114, +0.00253] | 1.0000 | — |
| summary − hierarchy_outcome | −0.00091 [−0.00262, +0.00080] | 1.0000 | — |
| summary − hierarchy_physics | −0.00036 [−0.00236, +0.00167] | 1.0000 | — |
| summary − physical | −0.00153 [−0.00411, +0.00082] | 1.0000 | — |

Pairwise non-rejection sets:
**{hierarchy_outcome, hierarchy_physics, physical, summary}** and
**{logical}**.

## What it establishes, and what it does not

**Establishes.** The encoder that cannot see the embedding is separated from
**every** encoder that can, after correction for all ten comparisons, with the
four contrasts agreeing in direction and size (0.0070 to 0.0085). This is the
design document's first factorial row — "logical graph versus physical graph,
then hierarchy" — and the answer is confirmatory rather than exploratory.

**Does not establish.** No two embedding-aware encoders separate from each
other; the smallest Holm p among them is 0.96. The information helps; the
architecture that consumes it does not measurably matter here. And a
non-rejection set is **not** an equivalence class — the tool labels it so, and
failing to separate four encoders at 48 parents does not make them equal.

The pooled aware-minus-blind figure of −0.00782 [−0.01109, −0.00447] remains
exploratory and unadjusted, and is not the basis of the claim above.

## Limits

- One dataset, bank-selection mode, 48 held-out parents, intervals conditional
  on archived training-seed averages.
- On Pegasus the same pooled contrast has an interval including zero
  ([ledger](../EVIDENCE.md)); this result is synthetic-only and does not
  transfer automatically.
