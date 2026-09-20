# Classical assistance: the sampler does know something, and there is nothing left to tell

## The question

> Classical assistance | No-sampling model versus sample-assisted model versus
> sampler alone | Benefit relative to the cost of extra classical information.

One arm was already archived: **sampler alone** is useless as a ranker —
Spearman −0.091 against exact loss, picking the exact best candidate in 0 of 40
records. The **sample-assisted** arm was never built, and building it means new
record fields, a wider context vector and a training run.

A three-tier gate decides whether that is worth it, because each tier is
meaningless without the one before.

## Tier 1: the sampler does know the instance

40 held-out parents at ≤10 physical qubits, SVMC annealed along a linear
schedule with the exact pipeline's own accepted set — the basis-index-to-spin
convention taken from `physics.py:191` and checked against the energy that same
expression produces, rather than guessed.

| correlation | value |
|---|---:|
| SVMC linear loss against **true** linear loss | **+0.6618** |
| SVMC linear loss against true best-in-bank loss | **+0.6685** |

**This corrects an over-broad reading of the archived SVMC report.** That study
showed the surrogate cannot rank controls *within* an instance, and it was
right. It does not follow that the surrogate knows nothing — at instance level
it tracks true difficulty at ρ ≈ 0.66. Those are different questions and only
the first had been asked.

## Tier 2: the model has nothing left to learn from it

| quantity | value |
|---|---:|
| SVMC linear loss against the learned model's **bank regret** | −0.1969 |
| model's bank regret, mean | **0.00073** |
| records where the model already picks the exact best candidate | **39 / 40** |

The learned selector picks the exact best bank candidate in **97.5 %** of these
records. There is no regret for sampler features to reduce: the quantity a
sample-assisted model would improve is already at its floor on this population.

## Conclusion

**Factor 8 is closed for bank selection on this population, by measurement
rather than by training.** A sample-assisted model would be optimising a
residual of 0.0007 that is zero in 39 of 40 records, and the cost side of the
document's question — "benefit relative to the cost of extra classical
information" — cannot come out favourable against a benefit bounded that
tightly.

## Where it stays open, and these are not small

- **The direct policy.** Its regret is real (+0.025 against the tuned global
  schedule), so sampler features have something to work on there. This gate
  measured bank selection only.
- **Harder instances.** ≤10 physical qubits with a 8-candidate bank is where
  the selector saturates. On the 64-candidate Pegasus banks, or at larger
  sizes, the regret floor may not be zero.
- **Other sampler features.** Only SVMC success under a linear schedule was
  tested. A richer summary — the spread across the bank, rotor statistics —
  could carry information this one does not.

## Limits

- 40 parents, ≤10 physical qubits, one checkpoint, one sampler configuration
  (200 steps, 8 sweeps, 256 restarts).
- Correlations are over records, not parent-resampled; they gate a decision
  rather than support a claim.
