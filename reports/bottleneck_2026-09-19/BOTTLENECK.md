# The mandatory scalar chain costs; an eight-dimensional profile does not

## The claim being tested

The design document's Round-1 resolution is to **reject** the mandatory
G → D₂ → ϱ chain — "replace the mandatory G→D→ρ chain by operator-resolved
auxiliaries plus a direct policy" — and its factorial table asks for the
ablation that justifies that:

> Mandatory bottleneck | Force G→D₂→ϱ versus direct residual branch with
> identical encoder | **Loss caused by scalar compression, not merely model
> size.**

Until now the rejection rested on an oracle argument: the `d2` teacher, handed
a *perfect* D₂, already loses to the learned selector by 0.050. Strong, but not
the ablation.

## Design

`AnnealController(bottleneck_dim=k)` forces every route from the graph to a
control through `k` nonnegative numbers, and collapses the token bank to a
single token so attention cannot carry per-node information around the
constriction. The encoder and all three heads are untouched.

Three arms, three seeds each, one experiment so that data, splits and seeds are
shared: `unconstrained`, `bottleneck_1` (the literal scalar chain),
`bottleneck_8` (a coarse profile). All use the hierarchical encoder, width 64,
`response_weight` 0.0.

**The constricted models are the larger ones** — 230 176 and 231 079 parameters
against 225 823 — so a loss cannot be attributed to capacity. That is precisely
what the table asks to rule out.

## Result

864 held-out records, 48 parents, three seeds averaged per record.

| arm | mean loss |
|---|---:|
| bottleneck_8 | **0.544449** |
| unconstrained | **0.545347** |
| bottleneck_1 | **0.551003** |
| tuned global schedule | 0.565437 |
| linear at matched duration | 0.600929 |

Holm-corrected across all three pairs (the project's own `method-contrast`):

| contrast | difference | Holm p | |
|---|---|---:|---|
| bottleneck_8 − bottleneck_1 | −0.00655 [−0.00981, −0.00324] | **0.0003** | separated |
| unconstrained − bottleneck_1 | −0.00566 [−0.00907, −0.00230] | **0.0029** | separated |
| unconstrained − bottleneck_8 | +0.00090 [−0.00145, +0.00322] | 0.4461 | not separated |

Pairwise non-rejection sets: **{bottleneck_1}**, **{bottleneck_8,
unconstrained}**.

## What it says

**The mandatory scalar chain measurably costs decision quality** — +0.00566
against the unconstrained model, separated after Holm correction, with more
parameters rather than fewer. The document's rejection of it is now an
ablation, not an argument.

**But compression itself is not the problem; dimensionality is.** An
eight-number profile is statistically indistinguishable from the unconstrained
model (Holm p = 0.45) and is marginally ahead on the point estimate. One number
is too few; eight is enough on this task.

That refines the document's resolution rather than merely confirming it. The
right statement is not "do not compress" but **"do not compress to a
scalar"** — an operator-resolved profile of modest width loses nothing here,
which is what the document's own "operator-resolved auxiliaries" language
anticipated.

All three arms still beat the tuned global schedule, by 0.014 to 0.021.

## Limits

- "Not separated" is **not equivalence**, and the contrast tool labels its
  non-rejection sets accordingly. Failing to reject at 48 parents does not
  establish that `bottleneck_8` is as good as unconstrained.
- Three seeds, one dataset, one encoder, bank selection only. The direct policy
  head is not evaluated separately here.
- `k = 8` was chosen once, to match the eight-bin control resolution. The
  threshold between 1 and 8 is not located.
