# How large is the effect? Three units, and one framing that is not available

## The problem with the loss unit

Bank selection beats a matched linear ramp by **0.056** on the synthetic set.
Read alone, that is a small number on a base of 0.60, and it is the number
this project has led with. It understates the result, because the whole
achievable range is also small: a 257-call search only finds **0.094**. That
search is budgeted Sobol-local, not exhaustive — its own scope note reads
"best found under a declared budget; not a global control optimum" — so 0.094
is a floor on what is achievable, not a ceiling.

Three units say it better, and all three are measured on held-out parents with
parent-level bootstrap.

## 1. Effect size and win rate

| comparison | difference | Cohen's *d* | parents won |
|---|---|---:|---:|
| synthetic, linear − learned | +0.05627 [+0.04943, +0.06356] | **2.24** | **48 / 48** |
| synthetic, tuned global − learned | +0.02077 [+0.01417, +0.02773] | 0.87 | 43 / 48 |
| **Pegasus**, linear − learned | **+0.07885** [+0.05301, +0.10581] | **1.62** | **12 / 12** |
| Pegasus, tuned global − learned | +0.02291 [+0.01057, +0.03836] | 0.89 | 12 / 12 |

*d* = 2.24 over logical parents, with every parent won, is not a small effect
by any conventional standard — 0.8 is the usual threshold for "large". The
against-global figures, *d* ≈ 0.88 on both topologies, clear it too.

## 2. Share of what is achievable

The learned selector consumes **one forward pass**. What it should be compared
against is a search that pays for every instance. There are two such references
in this project, and **they are not interchangeable** — an earlier version of
this table put one of each in the same column under a single "257-call" header,
which is corrected here.

| reference | what it costs per instance | what it is |
|---|---|---|
| bank oracle | 64 propagations, scored once at dataset build | best of the **same 64-candidate menu the selector chooses from** |
| frontier search | 257 propagations, adaptive, 5 control families | best found by an instance-specific Sobol-local search |

The bank oracle is the weaker reference by construction: it is the ceiling on
the selector's own menu, so a high share against it says the critic is ranking
well, **not** that the control is near optimal. The frontier search is the one
that answers "how much of what is findable did we get?".

Measured with the reference held fixed:

| | reference | headroom | learned gain | share |
|---|---|---:|---:|---:|
| synthetic (48 parents) | bank oracle | 0.06912 | +0.05627 | **81.4 %** |
| Pegasus (12 parents) | bank oracle | 0.09547 | +0.07885 | **82.6 %** |
| synthetic (48 parents) | 257-call frontier | 0.09358 | +0.05627 | **60.1 %** |
| Pegasus (12 parents) | 257-call frontier | *pending* | +0.07885 | *pending* |

**The apparent 60 % vs 83 % gap between the two topologies was an artifact of
the two denominators, and is withdrawn.** Against a matched reference the two
are the same to within a point (81.4 % vs 82.6 %). Nothing in the data
supports "the effect is sharper on real connectivity" stated in this unit; the
reads unit in section 3 is where the topologies genuinely differ, and it uses
one reference throughout.

The Pegasus frontier cell is blank because the 257-call search had never been
run on Pegasus test parents — only a 97-call, 4-family search on *validation*
parents exists (headroom 0.14410, different parents, not substitutable). That
sweep is running; this row will be filled from measurement, not inferred.

Against the frontier reference the share does not depend on how much there is
to get:

| synthetic parents | headroom | gain | share |
|---|---:|---:|---:|
| below-median headroom (24) | 0.0656 | +0.03951 | **60 %** |
| above-median headroom (24) | 0.1216 | +0.07302 | **60 %** |

Double the available headroom, double the gain, same fraction. **The effect is
proportional, not small** — what is modest is the achievable range itself.

## 2b. The same result with no denominator at all: simulator calls

A share depends on a budget someone chose. Measuring the search's own
incumbent curve shows how much: it reaches **96.7 %** of its final headroom by
129 calls and 92.6 % by 65, so 257 is a stopping point, not a natural unit.

| total calls | 5 | 9 | 17 | 33 | 65 | 129 | 257 |
|---|---:|---:|---:|---:|---:|---:|---:|
| mean parent headroom | 0.00000 | 0.03947 | 0.06478 | 0.07708 | 0.08669 | 0.09046 | 0.09358 |

(The 5-call point is the linear reference alone, before any tunable family has
been evaluated twice — it is zero by construction, not a finding.)

Priced in calls instead, per parent, bootstrapped over parents, on the
synthetic set:

| | equivalent instance-specific simulator calls |
|---|---|
| direct policy — **generates** a control | **8.7** [7.8, 9.7], median 9 |
| bank selection — **picks** from 64 | **17.9** [16.2, 19.8], median 17 |
| a *perfect* ranker on that same bank | 25.2 [22.7, 27.9], median 25 |

All 48 parents resolved, none censored. Two things follow that the percentage
did not show. **Selecting is worth about twice generating**, which is the
project's central design choice stated as a measurement. And the gap from 17.9
to 25.2 is what better ranking alone would buy — about seven calls.

This unit does not depend on which search spends the budget, but it is not
invariant either, so the range is reported rather than a single figure. Over
the same 864 records, same five families, same budget, differing only in
strategy:

| search | equivalent calls | its own final headroom |
|---|---:|---:|
| policy gradient | 15.5 [13.2, 18.1] | 0.08954 |
| Sobol-local | 17.9 [16.2, 19.8] | 0.09358 |
| Bayesian GP-EI | 23.3 [20.2, 26.8] | 0.09547 |

GP-EI explores early and ends best, so it is slowest to reach the learned
gain; policy gradient exploits early and plateaus. **The conservative claim
across all three is "at least 15 calls"**, and the share unit is correspondingly
stable at 58.9–62.8 %.

### Would a bigger menu close the gap?

`bank_coverage` is the binding factor, so the obvious reply is to enlarge the
bank. Measured from candidate losses already stored with every record, at no
simulation cost — the bank is shared across records, so a menu of size *k* is
one fixed subset applied everywhere:

| menu size | 1 | 2 | 4 | 8 | 16 | 32 | 64 | search |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| headroom | 0.01065 | 0.01888 | 0.03207 | 0.04431 | 0.05662 | 0.06366 | **0.06912** | **0.09358** |

Strongly saturating: the last doubling bought **+0.00546** while **0.02446** of
gap remains, and each doubling buys less than the one before. A bigger fixed
menu is not the cheap fix — what a search has and a menu cannot have is the
ability to adapt to the instance. Subsets below 64 are random, so they
understate a purpose-designed menu of that size; since the endpoint is the
deployed bank itself, a designed curve would sit above these and end in the
same place, making this a conservative reading of the saturation.

## 3. Reads to 99 % confidence

The unit a practitioner budgets in. Success probability from exact propagation,
so this is a projection onto independent reads rather than a hardware
measurement; the derivation and its assumptions are in
[the time-to-solution report](../time_to_solution_2026-09-19/TIME_TO_SOLUTION.md).

| | linear | tuned global | learned | linear / learned |
|---|---:|---:|---:|---|
| synthetic | 22.3 | 19.2 | 17.5 | **1.279×** [1.242, 1.319], 47/48 parents |
| **Pegasus** | 47.8 | 38.2 | 34.0 | **1.509×** [1.390, 1.630], **12/12 parents** |

**On real device connectivity a linear ramp needs half again as many reads**,
in every held-out parent. That is a 34 % reduction, and it is the sharpest
honest statement of the result.

## The framing that is NOT available

It is tempting to add "and the advantage grows with problem size". **It does
not**, and this was tested rather than assumed:

    rank correlation, physical qubits against gain      -0.1231
    rank correlation, physical qubits against headroom  -0.1812

Across 3 to 9 physical qubits the per-size gain is flat within noise (+0.087,
+0.063, +0.055, +0.054, +0.049, +0.063, +0.033) and so is the headroom. The
larger Pegasus effect is therefore attributable to **the setting** — a
64-candidate bank, real connectivity, a different instance family — and not to
size. Claiming a scaling trend here would be unsupported.

## What to lead with

> On real device connectivity, a learned selector consuming one forward pass
> cuts the reads needed for 99 % confidence by **34 %** against a matched
> linear ramp — in **every one of 12 held-out parents**. On the synthetic set
> the same selector wins **48 of 48** parents at *d* = 2.24, capturing a
> **constant 60 %** of the headroom a 257-call instance-specific search finds,
> whether that headroom is large or small.

Both sentences are paired, parent-bootstrapped, held-out, and carry their
sample sizes. The Pegasus arm rests on **12 parents**, which is the number to
watch.

The earlier version of this lead put "captures 83 % of what a 257-call search
finds" on the *Pegasus* clause. That was wrong on both counts — the Pegasus
denominator was the 64-candidate bank oracle, not a 257-call search — and the
83 % is not a device-connectivity result at all, since the synthetic set scores
81.4 % against the same reference. The share claim is therefore made only where
it is measured against the frontier, which today is the synthetic set. It moves
to the Pegasus clause if and when the Pegasus frontier sweep supports it.

## Limits

- Pegasus: 12 held-out parents. A 240-parent dataset with 48 held-out parents
  is training as of 2026-09-20, and a matched 257-call 5-family frontier over
  those 48 test parents is running beside it. Until both land, **every number
  in sections 2b is synthetic-only**, and the Pegasus frontier cell in
  section 2 stays blank.
- The call equivalent prices the **online** cost only. One forward pass is one
  forward pass because the bank is fixed and the critic predicts its losses
  without simulating; building the bank and training the model are offline and
  accounted separately in `costs.py`. "Worth 17.9 calls" is not "cheaper than
  17.9 calls" until the deployment count is fixed.
- The call equivalent is the smallest grid budget whose headroom reaches the
  gain, so it is an upper bound between grid points; the per-parent lower
  bound is recorded alongside in the artifact. It also depends on the search
  strategy, which is why the range 15.5–23.3 is reported rather than one
  figure.
- Cost class: the learned selector is `amortised` — one forward pass at
  deployment, but `C_data` and `C_training` are real and are accounted for
  separately in `costs.py`. A 34 % read reduction is not a 34 % cost reduction
  until the deployment count is fixed.
- Reads assume independent identically distributed trials at fixed per-read
  cost, from a simulated success probability. No hardware.
