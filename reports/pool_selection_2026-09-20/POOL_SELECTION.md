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

## Where this sits on the pool-size axis

The repaired method is still far from bank selection over 64 candidates
(−0.086, 48/48 every seed). That is not two different methods; it is one
mechanism at two pool sizes, and the numbers agree with the independently
measured menu-size curve:

| pool | measured gain over linear |
|---|---:|
| proposal + linear (2 members) | 0.010 – 0.017 |
| random menu of 1, from the curve | 0.0147 |
| random menu of 2, from the curve | 0.0225 |
| the full 64-candidate library | 0.1018 |

A two-element pool achieves roughly what a one-to-two element menu achieves.
The critic's value scales with what it is given to choose between, and that
axis is measurable end to end.

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
