# 5. Does the embedding carry decision-relevant information?

## 5.1 The encoder contrast

All ten pairs of the five encoders, Holm-corrected as one family, bank
selection, held-out parents. Differences are stated so that **positive means
the embedding-blind encoder is worse**.

### Pegasus, 48 held-out parents

| contrast | difference | 95 % CI | Holm *p* | |
|---|---:|---|---:|---|
| logical − hierarchy_outcome | +0.01401 | [+0.00861, +0.02017] | **0.0005** | separated |
| logical − summary | +0.01559 | [+0.01011, +0.02179] | **0.0005** | separated |
| logical − hierarchy_physics | +0.01401 | [+0.00858, +0.02021] | **0.0008** | separated |
| logical − physical | +0.01252 | [+0.00678, +0.01906] | **0.0014** | separated |
| summary − physical | −0.00307 | [−0.00672, +0.00058] | 0.5991 | — |
| physical − hierarchy_* | +0.00149 | [−0.00040, +0.00352] | 0.6562 | — |
| summary − hierarchy_* | −0.00159 | [−0.00463, +0.00162] | 0.9489 | — |
| hierarchy_physics − hierarchy_outcome | +0.00000 | [0, 0] | 1.0000 | *degenerate* |

Non-rejection sets: **{hierarchy_outcome, hierarchy_physics, physical,
summary}** and **{logical}**, alone. Pooled aware − blind:
**−0.014030 [−0.019930, −0.008772]**, favouring the aware encoders in **43 of
48** parents.

Four of ten pairs separate and they are **exactly** the four
logical-versus-aware contrasts. No aware pair separates from another.

### Two readings, and only one of them is about architecture

The information is decision-relevant. Which architecture consumes it is not
measurably relevant — and the strongest form of that statement is that
`summary`, which does no message passing and sees only pooled statistics of
both graphs, is not separated from the full hierarchical token bank, and has
the lowest mean loss of the four. **Coarse embedding statistics carry most of
what fine graph structure carries.**

That is a negative result, and it constrains the design space more usefully
than the positive one: effort spent on representation machinery is not where
the remaining gain is.

### The degenerate cell, stated rather than quietly dropped

`hierarchy_physics` and `hierarchy_outcome` differ only in `response_weight`,
which multiplies a loss supervised by resolved spectral response. Pegasus has
`resolved_response_fraction = 0.0`, so there is nothing to weight and the two
are **the same model**, agreeing to eight decimals on every seed. The
+0.00000 entry is an identity, not a tight null.

Two consequences: the Holm family has ten nominal members and nine informative
ones, which makes the correction **conservative** rather than lenient, so the
four separations survive it; and the pooled aware mean gives the hierarchy
architecture half its weight, so the pooled figure is not a clean four-way
average. Every aware encoder separates from `logical` individually, so the
direction does not depend on either.

## 5.2 How large is the effect, and against what

Bank selection against a matched linear ramp, held-out parents:

| | synthetic (48 parents, 5 seeds) | Pegasus (48 parents, 3 seeds) |
|---|---|---|
| linear − learned | +0.05627 [+0.04943, +0.06356] | **+0.08631 [+0.07216, +0.10086]** |
| Cohen's *d* | **2.24** | **1.68** |
| parents won | 48 / 48 (47/48 worst seed) | **48 / 48 in every seed** |
| tuned global − learned | +0.02077, *d* 0.87 | +0.03246, *d* **1.30** |
| reads to 99 % confidence vs linear | 1.279× [1.243, 1.319] | **1.589× [1.513, 1.667]** |

On Pegasus a linear ramp needs about 1.6× as many reads — a **37 % reduction**
— in 47 of 48 parents. The 48th is a **tie**, not a loss: `parent_0186` needs
3.00 reads under both schedules and $n_q$ is a ceiling, so an instance already
solved in three reads leaves no integer room. The selector has the lower loss
on all 48 and is worse on reads on none.

### A unit that does not depend on a chosen budget

A "share of what a search finds" moves with the search's budget. Walking the
search's own incumbent curve shows how much: it reaches 96.7 % of its final
headroom by 129 calls and 92.6 % by 65, so 257 was a stopping point rather
than a natural unit. We therefore price one forward pass in **instance-specific
simulator calls**:

| | synthetic | Pegasus |
|---|---|---|
| bank selection, one forward pass | 17.9 [16.2, 19.8] | **23.8 [21.5, 26.2]** |
| a *perfect* ranker on the same library | 25.2 [22.7, 27.9] | 35.5 [32.7, 38.2] |

All 48 parents resolve on both topologies, none censored. The gap between the
deployed critic and a perfect one is about seven calls on synthetic and twelve
on Pegasus — that is what better ranking alone would buy.

The unit is not invariant to the search algorithm, so we report the range
rather than one figure: over identical records and families, policy gradient
gives 15.5 [13.2, 18.1], Sobol-local 17.9 [16.2, 19.8], and Bayesian GP-EI
23.3 [20.2, 26.8]. GP-EI explores early and ends best, so it is slowest to
reach our gain. **The conservative claim across all three is "at least 15
calls."**

### What the denominator was hiding

An earlier version of this analysis reported "60 % of achievable on synthetic,
83 % on Pegasus" and read the gap as a device-connectivity effect. It was not.
The two rows used different references — a 257-call frontier search and a
64-candidate bank oracle — under one column header. Measured against the same
reference the topologies differ by **2.8 points**:

| | selector efficiency | bank coverage | share of findable |
|---|---:|---:|---:|
| synthetic | 81.4 % | 73.9 % | 60.1 % |
| Pegasus | 84.8 % | **74.1 %** | 62.9 % |

because `share of findable = selector efficiency × bank coverage`, and the two
factors move in opposite directions: the critic is *better* on real
connectivity while the library covers almost exactly as much. Bank coverage
agreeing within 0.2 points across two topologies, two instance populations and
a different difficulty scale suggests the library's share of what is findable
is a property of **using 64 fixed controls**, not of the problem.

## 5.3 Replication, including where it is imperfect

The Pegasus arm is an *independent* draw: an identically configured generator
differing only in seed, 240 parents against 96, and held-out sets sharing **no
parent**. So the 48-parent result reproduces the 12-parent one on fresh
instances at four times the scale rather than absorbing it.

| quantity | 12 parents | 48 parents | agreement |
|---|---|---|---|
| linear − learned | +0.07885 [+0.05301, +0.10581] | +0.08631 [+0.07216, +0.10086] | overlap; each point inside the other's interval |
| tuned global − learned | +0.02291 [+0.01057, +0.03836] | +0.03246 [+0.02570, +0.03967] | overlap; **12-parent point just outside** |
| reads vs linear | 1.509× [1.390, 1.630] | 1.589× [1.513, 1.667] | overlap; **12-parent point just outside** |

All three overlap, but the 48-parent estimates are **uniformly larger** and in
two of three the earlier point falls marginally outside the new, narrower
interval. That is the shape of an independent draw plus a tighter interval
rather than a contradiction, and the honest reading is that the 12-parent arm
sat at the low end of what this generator produces.

The claim this section set out to test was, before this replication,
**synthetic-only**: at 12 Pegasus parents nothing separated at all.
