# The half of the pool that was never built, and what it repairs

## What was missing

The design document specifies three branches: spectral auxiliaries, a schedule
proposed directly from structural tokens, and a critic that evaluates
*"proposed schedules ... alongside the direct proposal and **simple
baselines**"* (`main.tex:572`).

The project built the first two and a critic — but the critic only ever ranked
the policy's **own proposals**. The "simple baselines" half of the pool was
never there, and so the document's method had never actually been evaluated.

That omission was not cosmetic. On the 48-parent Pegasus arm the direct policy
loses to a linear ramp on 22 of 48 parents and **changes sign across training
seeds** (−0.00890, +0.00230, +0.00302). Every one of those is a case a pool
containing the linear ramp could recover — *if* the critic can tell that its
own proposal is the worse of the two.

## Result: it can, and the instability disappears

Pool = the policy's proposals plus the fixed linear ramp, ranked by the critic
in one forward pass. 48 held-out Pegasus parents, per training seed, paired at
parent level (negative favours the first named).

| seed | proposal only − linear | **pool − linear** | pool − proposal only | fallback rate | rescued / harmed |
|---|---|---|---|---:|---:|
| 0 | −0.00890 [−0.02213, +0.00460] | **−0.01609 [−0.02811, −0.00412]** | −0.00719 [−0.01413, −0.00177] | 3.5 % | 4 / **0** |
| 1 | **+0.00230** [−0.01318, +0.01847] | **−0.01707 [−0.02903, −0.00520]** | −0.01937 [−0.03267, −0.00854] | 9.2 % | 8 / **0** |
| 2 | **+0.00302** [−0.01004, +0.01655] | −0.00100 [−0.01380, +0.01216] | −0.00402 [−0.00894, −0.00018] | 2.4 % | 2 / **0** |

Four things, in order of importance.

**The sign instability is gone.** Proposal-only runs [−0.009, **+0.002,
+0.003**] — worse than a linear ramp on two of three seeds. With the fallback
the same three checkpoints give [−0.016, −0.017, −0.001], all negative.

**Two of three seeds now separate from linear.** Proposal-only separated on
none.

**The fallback helps on every seed**, `pool − proposal only` excluding zero in
all three.

**It never harms.** Zero parents made worse, across all three seeds and 144
parent-seed cells. The critic's fallback decisions are precise, not merely
frequent — it fires on 2.4–9.2 % of records and is right essentially every
time it does.

## It is not a Pegasus effect

Five `hierarchy_outcome` checkpoints on the synthetic set, same instrument,
48 held-out parents.

| seed | proposal only − linear | **pool − linear** | pool − proposal only | fallback | rescued / harmed |
|---|---|---|---|---:|---:|
| 0 | −0.00579 [−0.01305, +0.00191] | **−0.01242 [−0.01927, −0.00502]** | −0.00662 [−0.01094, −0.00309] | 10.9 % | 5 / **0** |
| 1 | −0.01422 [−0.02237, −0.00660] | **−0.01742 [−0.02482, −0.01089]** | −0.00320 [−0.00608, −0.00088] | 4.6 % | 3 / **0** |
| 2 | −0.01224 [−0.02175, −0.00266] | **−0.02075 [−0.02805, −0.01337]** | −0.00851 [−0.01558, −0.00170] | 16.1 % | 8 / **0** |
| 3 | **+0.00026** [−0.00978, +0.01025] | −0.00386 [−0.01153, +0.00431] | −0.00412 [−0.00792, +0.00062] | 10.6 % | 2 / **0** |
| 4 | −0.01934 [−0.02857, −0.00983] | **−0.02492 [−0.03286, −0.01679]** | −0.00558 [−0.01123, −0.00064] | 17.9 % | 4 / **0** |

The same three properties hold. The pool is negative on **5 of 5** seeds where
proposal-only was negative on 4 and one seed sat at +0.00026; it separates
from linear on **4 of 5** against proposal-only's 3; and `pool − proposal
only` is negative on every seed.

Taken with the Pegasus arm, the no-harm property now covers **384
parent-seed cells across two topologies with zero parents made worse**. The
fallback rate varies widely (2.4 – 17.9 %) without ever costing anything,
which is the signature of a decision that is being made on evidence rather
than fired at a fixed rate.

The synthetic set is also the easier case for the policy — it already cleared
the ramp on most seeds — so the gain is smaller in relative terms than on
Pegasus. The repair is worth most exactly where the policy is weakest.

## The pool-size axis, measured

The repaired method is still far from bank selection over 64 candidates. That
is not two different methods but one mechanism at different pool sizes, and
the axis can be walked directly by choosing which bank members the critic is
allowed to see. On the synthetic set, five seeds, 48 held-out parents:

| what the critic may choose from | mean gain over linear | seeds separating |
|---|---:|---:|
| its own proposals only | −0.01027 | 3 / 5 |
| + the linear ramp — *the document's "simple baseline"* | −0.01587 | 4 / 5 |
| + the designed library: linear, windows, pauses (8) | **−0.03787** | **5 / 5** |
| the full 64-candidate bank | −0.05627 | — |

Monotone across four points spanning a factor of thirty in pool size. **The
critic's value is set by what it is given to choose between**, and roughly
two thirds of the full library's advantage is already reached by its eight
*designed* members — the 56 low-discrepancy samples supply the rest.

That decomposition matters for the bank-coverage result. The library is not
an undifferentiated blob of 64: a small designed core does most of the work,
which is why a fixed menu is worth so much less than an instance-specific
search of comparable size.

### The same axis on real connectivity

Three `summary` checkpoints, 48 held-out Pegasus parents:

| what the critic may choose from | mean gain over linear | seeds separating |
|---|---:|---:|
| its own proposals only | −0.00119 | **0 / 3** |
| + the linear ramp | −0.01139 | 2 / 3 |
| + the designed library (8) | **−0.03132** | **3 / 3** |
| the full 64-candidate bank | −0.08631 | 3 / 3 (48/48 each) |

Monotone again, and starker: proposal-only separates from a linear ramp on
**none** of the three seeds, the designed pool on **all three**.

Seed 1 is the clearest single case in the study. Its policy was the worst of
the three — proposal-only **+0.00230**, i.e. losing to the ramp. The critic
fired the fallback on **55.4 %** of records, rescued **24 parents**, and the
pool finished at **−0.05080 [−0.06289, −0.03906] on 46 of 48**. The fallback
rate is not a constant being tuned: it ranges 16 – 55 % across three seeds of
the same configuration, tracking how bad that seed's generator is. When the
proposal is good the critic keeps it; when it is bad the critic routes around
it.

That is the property worth putting in the paper. It is not "the policy works"
— on Pegasus, alone, it does not. It is that **a critic good enough to rank a
library is also good enough to know when its own generator should be
overruled**, and the two capabilities come from the same trained model.

### The no-harm property is structural, and only for the smallest pool

Linear-only harmed **zero** parents in 384 parent-seed cells. The designed
pool harmed **3 in 240**. That is not noise and not a regression — it follows
from what the pools contain. When the only alternative is the reference
itself, choosing it yields exactly the reference and cannot lose; once the
pool holds windows and pauses, the critic can prefer one that is worse than
linear, and occasionally does.

So the graceful-degradation claim belongs to the two-element pool
specifically, and the paper should say so rather than extend it to the
library. Bought against that, the designed pool more than doubles the gain.

## Cost class, stated rather than assumed

This pool is **amortised**: the policy's proposals and one fixed waveform,
ranked in a single forward pass, no simulation at deployment.

The document also places physics-derived schedules (normalized D₂, gap-only)
in the pool. Those are deliberately **excluded here**, because `main.tex:637`
forbids recomputing the spectral teacher for test instances at deployment
unless the cost is charged as a separate solver-assisted method. A pool
containing D₂ is `privileged_spectrum`, and this project does not rank across
cost classes. The document's pool as literally written mixes the two; that
tension is the document's, and it is resolved here by measuring only the
deployable half.

## Scope

One topology, three seeds, one fallback candidate. The fallback is the linear
ramp because that is the reference every other number in the project is
measured against; a richer amortised pool (windows, pauses) is untested and
would sit further along the same axis.
