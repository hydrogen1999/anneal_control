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
| **Pegasus**, linear − learned | **+0.08631** [+0.07216, +0.10086] | **1.68** | **48 / 48** |
| Pegasus, tuned global − learned | +0.03246 [+0.02570, +0.03967] | **1.30** | 47 / 48 |

*d* = 2.24 over logical parents, with every parent won, is not a small effect
by any conventional standard — 0.8 is the usual threshold for "large". The
against-global figures clear it too, and on Pegasus comfortably.

The Pegasus rows are the **48-parent** arm (2026-09-20), which supersedes the
12-parent one on the same topology and instance families. Quadrupling the
held-out set moved the effect slightly up and the interval sharply in:

| | 12 parents | 48 parents |
|---|---|---|
| linear − learned | +0.07885 [+0.05301, +0.10581], *d* 1.62, 12/12 | +0.08631 [+0.07216, +0.10086], *d* **1.68**, **48/48** |
| tuned global − learned | +0.02291 [+0.01057, +0.03836], *d* 0.89, 12/12 | +0.03246 [+0.02570, +0.03967], *d* **1.30**, 47/48 |

The linear interval is **37 % narrower** and the against-global interval
**half** the width. The 48/48 holds in every training seed individually, not
only pooled.

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
| Pegasus (48 parents) | bank oracle | 0.10176 | +0.08631 | **84.8 %** |
| synthetic (48 parents) | 257-call frontier | 0.09358 | +0.05627 | **60.1 %** |
| Pegasus (48 parents) | 257-call frontier | *running* | +0.08631 | *running* |

**The apparent 60 % vs 83 % gap between the two topologies was an artifact of
the two denominators, and is withdrawn.** Against a matched reference the two
are within a few points (81.4 % vs 84.8 % on the bank oracle). Nothing in the data
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

(The 5-call point is zero by construction, not a finding: every tunable
family's *first* trial is exactly the linear control — checked on 200 records,
where the first-evaluation best equals the linear loss to 0.000000 in all
200 — so the first call of each family re-scores the reference. The search's
first informative budget is 9 calls.)

Priced in calls instead, per parent, bootstrapped over parents, on the
synthetic set:

| | parents it beats linear on | equivalent calls, where it wins |
|---|---:|---|
| direct policy — **generates** a control | **35 / 48** | 10.0 [9.2, 11.1], median 9 |
| bank selection — **picks** from 64 | **48 / 48** | **17.9** [16.2, 19.8], median 17 |
| a *perfect* ranker on that same bank | 48 / 48 | 25.2 [22.7, 27.9], median 25 |

No parent is censored: the search reaches every method's gain inside 257
calls. The middle column matters as much as the right one — a parent the
method does not beat has **no** call equivalent, because every curve starts at
zero headroom and a non-positive gain would otherwise be "matched" by the
smallest budget on the grid and priced as if the search had needed it. That is
a floor, not a price; excluding those 13 parents moves the direct policy from
an apparent 8.7 calls to 10.0 over the 35 it actually wins.

Two things follow that the percentage did not show. **Selecting beats
generating on both axes** — 48/48 parents against 35/48, and 17.9 calls
against 10.0 — which is the project's central design choice stated as a
measurement rather than a preference. And the gap from 17.9 to 25.2 is what
better ranking alone would buy: about seven calls.

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

### On Pegasus there is no direct-policy gain to price

The ladder above is synthetic. On the 48-parent Pegasus arm the direct policy
cannot be placed on it at all, because it has no reliable gain over a linear
ramp to convert into calls. Paired at parent level, per training seed
(negative favours the policy):

| training seed | direct − linear | parents beaten | bank selection, same seed |
|---|---:|---:|---|
| 0 | −0.00890 | 28 / 48 | −0.08761, **48 / 48** |
| 1 | **+0.00230** | 22 / 48 | −0.08541, **48 / 48** |
| 2 | **+0.00302** | 22 / 48 | −0.08590, **48 / 48** |

**The direct policy changes sign across training seeds** — it beats the ramp
on one and loses on two — and pooled over seeds its interval against linear
crosses zero, −0.00890 [−0.02204, +0.00447]. Against the tuned global
schedule it is clearly worse, +0.04495 [+0.03062, +0.05929]. Bank selection,
on the same records and the same seeds, wins every parent in every seed at
roughly ten times the magnitude.

This is the strongest available form of the project's design choice: on real
device connectivity, **generating a control is not reliably better than not
trying, while selecting from a fixed menu is worth *d* = 1.68 and 48/48**.
It also relocates where the direct policy's residual sits — the synthetic set,
where it does clear the ramp (35/48, about 10 calls), is the optimistic case,
not the representative one.

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
| **Pegasus** (48 parents) | 47.1 | 36.0 | 32.9 | **1.589×** [1.513, 1.667], **47/48 parents** |

**On real device connectivity a linear ramp needs about 1.6× as many reads** —
a **37 %** reduction, and the sharpest honest statement of the result. At 12
parents this read 1.509× [1.390, 1.630] in 12/12; the 48-parent arm moves the
ratio up and cuts the interval width by **37 %**. The win count falls to
47/48 because one parent (`parent_0186`) **ties** at 3.00 reads for both
schedules — it is solved in three reads either way, and *n*<sub>q</sub> is a
ceiling, so there is no integer room left. No parent is worse.

## The framing that is NOT available

It is tempting to add "and the advantage grows with problem size". **It does
not**, and this was tested rather than assumed:

    unit      rank correlation of physical qubits against
              gain       headroom
    record    -0.1231    -0.1812     (n = 864)
    parent    -0.2493    -0.3597     (n = 48)

Both units are shown because the rest of this document treats the **logical
parent** as the unit of independence, and records within a parent are not
independent — a record-level Spearman quietly claims n = 864 where there are
48 independent units. It does not change the conclusion here: both are
negative, and the parent-level figure is the more negative of the two, so
the "grows with size" hypothesis fares worse under the stricter unit, not
better.

Across 3 to 9 physical qubits the per-size gain is flat within noise (+0.087,
+0.063, +0.055, +0.054, +0.049, +0.063, +0.033) and so is the headroom. The
larger Pegasus effect is therefore attributable to **the setting** — a
64-candidate bank, real connectivity, a different instance family — and not to
size. Claiming a scaling trend here would be unsupported.

## What to lead with

> On real device connectivity, a learned selector consuming one forward pass
> cuts the reads needed for 99 % confidence by **37 %** against a matched
> linear ramp (1.589× [1.513, 1.667]) in **47 of 48 held-out parents**, and
> beats that ramp on **48 of 48** at *d* = 1.68. On the synthetic set the same
> selector wins **48 of 48** at *d* = 2.24, delivering what an instance-specific
> search needs **about 18 simulator calls** to find, and capturing a
> **constant 60 %** of that search's headroom whether the headroom is large or
> small.

Both sentences are paired, parent-bootstrapped, held-out, and carry their
sample sizes. Both arms now rest on **48 parents**. The two Pegasus counts
differ on purpose, and the difference is a tie rather than a loss: the
selector has the lower loss on all 48, but on `parent_0186` both schedules
need **3.00** reads. *n*<sub>q</sub> is a ceiling, and a parent that is
already solved in three reads leaves no integer room to improve. The selector
is never *worse* on reads on any parent.

The earlier version of this lead put "captures 83 % of what a 257-call search
finds" on the *Pegasus* clause. That was wrong on both counts — the Pegasus
denominator was the 64-candidate bank oracle, not a 257-call search — and the
83 % is not a device-connectivity result at all, since the synthetic set scores
81.4 % against the same reference. The share claim is therefore made only where
it is measured against the frontier, which today is the synthetic set. It moves
to the Pegasus clause if and when the Pegasus frontier sweep supports it.

## Limits

- Pegasus: **48 held-out parents** as of 2026-09-20. The matched 257-call
  5-family frontier over those same 48 test parents is still running, so
  **every number in section 2b is synthetic-only** and the Pegasus frontier
  cell in section 2 stays open. The runtime-4 half of that sweep is complete
  (288/288 records) and gives 0.11139 mean parent headroom against synthetic's
  0.09358, so the Pegasus share against a matched frontier will land well
  below its 84.8 % against the bank oracle.
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
