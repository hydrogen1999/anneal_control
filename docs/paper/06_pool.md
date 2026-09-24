# 6. What the critic is allowed to choose between

§5 gave the critic a 64-candidate library. This section varies that and finds
the axis it sits on — and, on the way, that the critic does something we did
not ask it to.

## 6.1 A generator alone is not enough

The policy head decodes a schedule directly from the tokens. Ranked only
against its own proposals, on 48 held-out Pegasus parents:

| training seed | direct − linear | parents beaten |
|---|---:|---:|
| 0 | −0.00890 | 28 / 48 |
| 1 | **+0.00230** | 22 / 48 |
| 2 | **+0.00302** | 22 / 48 |

It **changes sign across training seeds** and separates from a linear ramp on
**none** of them. Pooled, −0.00119 [−0.01380, +0.01202], an interval centred
almost exactly on zero. Against a single tuned global schedule it is clearly
worse, +0.04495 [+0.03062, +0.05929].

Reported alone this would be a failed method. It is instead a missing pool.

## 6.2 One fallback candidate

Admitting the matched linear ramp to the pool — the critic may now rank its own
proposals *against a baseline* — on the same three checkpoints:

| seed | proposal only − linear | **pool − linear** | pool − proposal only | fallback rate | rescued / harmed |
|---|---|---|---|---:|---:|
| 0 | −0.00890 [−0.02213, +0.00460] | **−0.01609 [−0.02811, −0.00412]** | −0.00719 [−0.01413, −0.00177] | 3.5 % | 4 / **0** |
| 1 | +0.00230 [−0.01318, +0.01847] | **−0.01707 [−0.02903, −0.00520]** | −0.01937 [−0.03267, −0.00854] | 9.2 % | 8 / **0** |
| 2 | +0.00302 [−0.01004, +0.01655] | −0.00100 [−0.01380, +0.01216] | −0.00402 [−0.00894, −0.00018] | 2.4 % | 2 / **0** |

The sign instability is gone: all three negative, two separating where
proposal-only separated on none, `pool − proposal` excluding zero on every
seed, and **zero parents harmed**. Five synthetic seeds reproduce it (pool
negative on 5/5, separating on 4/5 against proposal-only's 3/5, 0 harmed).

**No harm here is structural, not luck.** When the only alternative is the
reference itself, choosing it returns exactly the reference and cannot lose.
The property therefore belongs to the two-element pool and we do not extend it:
a richer pool harmed 3 parents in 240 synthetic cells and 1 in 144 on Pegasus.

## 6.3 The critic polices its own generator

The fallback rate is not a tuned constant. Across three seeds of one
configuration it ranges **2.4 % to 55 %**, tracking how poor that seed's
generator is.

Seed 1 is the clearest case in the study. Its policy was the worst of the
three — proposal-only +0.00230, losing to the ramp outright. With the designed
library available the critic fired the fallback on **55.4 %** of records,
rescued **24 parents**, and finished at **−0.05080 [−0.06289, −0.03906] on 46
of 48**.

The claim worth making is therefore not "the direct policy works" — on Pegasus,
alone, it does not. It is that **a critic good enough to rank a library is also
good enough to know when its own generator should be overruled**, and both
capabilities come from one trained model with no extra supervision for the
second.

## 6.4 The pool-size axis

Mean gain over a linear ramp, by what the critic may see:

| pool | synthetic (5 seeds) | Pegasus (3 seeds) |
|---|---:|---:|
| its own proposals only | −0.01027, separates 3/5 | −0.00119, separates **0/3** |
| + the linear ramp | −0.01587, 4/5 | −0.01139, 2/3 |
| + the designed library (8 fixed waveforms) | **−0.03787, 5/5** | **−0.03132, 3/3** |
| the full 64-candidate library | −0.05627 | −0.08631 |

Monotone on both topologies across a factor of thirty in pool size. Roughly
**two thirds of the full library's advantage is already reached by its eight
*designed* members**; the 56 low-discrepancy samples supply the remaining
third.

That decomposes the coverage result of §5.2. The library is not an
undifferentiated set of 64: a small designed core does most of the work, which
is also why a fixed menu is worth so much less than an instance-specific search
of comparable size — the menu cannot adapt, and adding random members to it
buys progressively less (a fixed menu's last doubling, 32 → 64, returns
+0.00546 on synthetic and +0.01014 on Pegasus against remaining gaps several
times larger).
