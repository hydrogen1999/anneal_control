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

## Consequence 1: the coherent-failure test cannot be run here

At most four usable instances, and the data generator does not produce
multi-crossing paths by design. The document's interference scan needs a
purpose-built family, which does not exist. Until it does, **every
representation claim in this project rests on instances with at most one
transition region in 92 % of cases** — a scope statement the paper needs and
did not have.

## Consequence 2: it explains why the spectral teachers look useless

Half of these instances have a *monotone* gap. A time density derived from D₂
has nothing to localise on a monotone gap, so the construction reduces to
something close to linear there. Splitting the `d2` teacher on that mechanism:

| subgroup | n | teacher − linear | beats linear |
|---|---:|---|---:|
| 0 interior minima | 23 | **+0.01065** [−0.00093, +0.02183] | 6/23 |
| ≥1 interior minimum | 24 | **−0.01291** [−0.01815, −0.00768] | **21/24** |

The teacher is not useless. It works where it has a bottleneck to work with and
does nothing where it does not, and rank correlation between the minimum gap
and its advantage over linear is **−0.399** — smaller gap, larger benefit,
which is what adiabatic theory predicts.

The pooled "spectral teachers barely beat linear" figure on this population was
a **mixture of two opposite subgroups**, in the same way that a claimed 2.6×
scale-arm effect was a mixture of factor populations and a +0.00729 filtering
gain was 39 % proposal-stream coverage.

It also sharpens the related-work comparison: on the working half, the physical
embedded spectrum beats the logical one by +0.00871 [+0.00450, +0.01304], 19/24
parents. See
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
