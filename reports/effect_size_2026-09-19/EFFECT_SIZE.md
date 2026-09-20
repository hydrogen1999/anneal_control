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

- Pegasus: 12 held-out parents. The 240-parent dataset generated on
  2026-09-19 would quadruple that and has not been trained on.
- Cost class: the learned selector is `amortised` — one forward pass at
  deployment, but `C_data` and `C_training` are real and are accounted for
  separately in `costs.py`. A 34 % read reduction is not a 34 % cost reduction
  until the deployment count is fixed.
- Reads assume independent identically distributed trials at fixed per-read
  cost, from a simulated success probability. No hardware.
