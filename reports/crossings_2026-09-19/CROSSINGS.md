# How many transition regions do these instances have? Half have none

## Why this matters

The design document reserves its coherent-failure test for *"paths with two or
more transition regions"*, scanned densely in runtime, and warns that a scalar
profile can succeed after averaging while failing on a sharply specified
coherent experiment. Whether that test is runnable at all is a property of the
dataset, and nobody had checked.

The answer bounds every representation claim made on this evidence base, and it
turned out to explain a null elsewhere.

## Census

48 held-out parents, one record each, ≤10 physical qubits, exact first gap on a
**129-point** s grid. A transition region is an interior local minimum of the
gap.

| interior gap minima | records | share |
|---:|---:|---:|
| 0 | 24 | **50.0 %** |
| 1 | 20 | 41.7 % |
| ≥2 | 4 | **8.3 %** |

| record | minima | min gap | locations |
|---|---:|---:|---|
| parent_0016_e0_k0_t0 | 2 | 0.3252 | s = 0.625, 0.766 |
| parent_0048_e0_k0_t0 | 2 | 0.1312 | s = 0.602, **0.992** |
| parent_0104_e0_k0_t0 | 2 | 0.0982 | s = 0.688, 0.867 |
| parent_0124_e0_k0_t0 | 2 | 0.0789 | s = 0.570, **0.969** |

Two of the four place their second minimum within 0.04 of the endpoint, where a
finite grid cannot distinguish a genuine avoided crossing from boundary
behaviour. The defensible count of interior multi-crossing instances is
therefore **2 to 4 of 48**.

## Replication on real Pegasus connectivity

The same census on the new 240-parent Pegasus dataset, test split. Only **16 of
48 parents** have a record at ≤10 physical qubits, which is where the exact
spectral profile is capped, so this is a **size-selected subsample** — the 16
parents with the most compact embeddings.

| interior gap minima | synthetic (n=48) | Pegasus (n=16) |
|---:|---:|---:|
| 0 | 50.0 % | **50.0 %** |
| 1 | 41.7 % | 25.0 % |
| ≥2 | 8.3 % | **25.0 %** |

**The monotone-gap half replicates exactly** — 50.0 % on both topologies. That
is the fraction that drives the teacher stratification below, and it is not an
artefact of the synthetic generator.

The multi-crossing fraction is three times higher on Pegasus, and the instances
are harder:

| record | minima | min gap | locations |
|---|---:|---:|---|
| parent_0000_e0_k0_t0 | 2 | 0.0920 | s = 0.531, 0.859 |
| parent_0012_e0_k0_t0 | 2 | 0.1209 | s = 0.664, 0.922 |
| parent_0096_e0_k0_t0 | 2 | 0.1718 | s = 0.719, 0.875 |
| parent_0219_e0_k0_t0 | 2 | **0.0020** | s = 0.773, **0.992** |

Three of the four have both minima clearly interior, against two of four on the
synthetic set, and one reaches a minimum gap of 0.0020 — two orders of
magnitude smaller than anything in the synthetic multi-crossing group, whose
smallest was 0.0789.

Read this as a direction, not a rate: **4 of 16 is a wide interval**, and the
subsample is selected by embedding compactness. What it does establish is that
chains on real device connectivity do produce genuinely multi-crossing,
small-gap instances, which the synthetic generator largely does not.

## Consequence 1: the coherent-failure test cannot be run here

At most four usable instances on the synthetic set, and the generator does not
produce multi-crossing paths by design. **Every synthetic representation claim
in this project therefore rests on instances with at most one transition region
in 92 % of cases** — a scope statement the paper needs and did not have.

Pegasus is more promising: four usable instances in a 16-parent subsample, one
of them with a minimum gap of 0.0020. A purpose-built family is still the right
answer, but real device connectivity may supply enough instances without one,
and the 12–14-qubit Pegasus records — which the 10-qubit spectral cap excludes
here — have not been examined at all.

## Consequence 2: it explains why the spectral teachers look useless

Half of these instances have a *monotone* gap, and a time density derived from
D₂ has nothing to localise there. That motivated splitting the `d2` teacher on
the mechanism — and the split behaves very differently at different runtimes,
which an earlier version of this report did not check.

**Teacher − linear, by subgroup and runtime:**

| runtime | 0 interior minima | ≥1 interior minimum |
|---:|---|---|
| 1 | +0.01065 [−0.00093, +0.02183], 6/23 | **−0.01291** [−0.01815, −0.00768], 21/24 |
| 4 | −0.03788 [−0.08203, +0.00375], 12/23 | **+0.05777** [+0.01517, +0.10218], 7/24 |
| 12 | **−0.04834** [−0.08882, −0.01220], 15/23 | +0.02544 [−0.02310, +0.07425], 9/24 |

**The stratification reverses.** At runtime 1 the teacher helps the
interior-minimum group; at runtime 4 it *hurts* that same group, with the
interval clearing zero in the opposite direction. Pooled correctly — averaging
within parent across all three runtimes, then resampling parents — it helps the
**monotone** group (−0.02519 [−0.04990, −0.00060], 23 parents) and not the
other (+0.02343 [−0.00587, +0.05259]).

So the earlier claim here, that the teacher "works where it has a bottleneck to
work with", was a **runtime-1 result reported as a general one, and is
withdrawn**. What is true is narrower and less tidy: the teacher's value
depends on runtime at least as strongly as on gap structure, and the two
interact with opposite signs.

The rank correlation of −0.399 between minimum gap and teacher advantage was
also computed at runtime 1 only and inherits the same limitation.

What does survive the pooling is the related-work contrast: on the
interior-minimum subgroup the physical embedded spectrum beats the logical one
by **+0.01137 [+0.00320, +0.01975]** over 24 parents, pooled across all three
runtimes, with the same sign at each of them individually (+0.0087, +0.0135,
+0.0120). See
[the logical-spectrum report](../logical_spectrum_2026-09-19/LOGICAL_SPECTRUM.md).

## Limits

- Interior local minima on a finite grid: a coarser grid hides minima, and this
  does not filter degenerate or unresolved gaps, so the count is a bound rather
  than a certificate.
- One dataset, test split, ≤10 physical qubits, 48 parents.
- The subgroup analyses above are **one** stratification, chosen from a stated
  mechanism before the split was examined, on a variable that is a property of
  the instance and not of the outcome. They need replication and are not
  confirmatory.
