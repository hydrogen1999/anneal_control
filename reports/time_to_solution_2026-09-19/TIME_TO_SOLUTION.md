# The same result, in the unit a practitioner budgets in

## Why this was missing

The design document's amortization section specifies

    n_q = ceil( log(1 - q) / log(1 - p) )

with two conditions: state the independence assumption, and never turn a
zero-success observation into a finite number. `evaluation.time_to_solution`
implements both, including Clopper–Pearson bounds and a `censored` flag. **No
archived artifact had ever reported it**, so every loss difference in this
project had only ever been expressed in loss units.

## Result

864 held-out records, `summary` encoder averaged over five training seeds
before any aggregation, 99 % target confidence:

| method | mean loss | parent-mean reads | p50 | p90 | p99 | censored |
|---|---:|---:|---:|---:|---:|---:|
| learned bank selection | 0.5318 | **16.9** | 9 | 39 | 115 | 0 |
| tuned global schedule | 0.5654 | 19.2 | 10 | 43 | 124 | 0 |
| linear at matched duration | 0.6009 | 22.3 | 12 | 49 | 148 | 0 |

Paired per-record read ratios, parent bootstrap:

    linear / learned   1.361x  [1.319, 1.408]   median 1.286x   48/48 parents
    global / learned   1.206x  [1.164, 1.253]   median 1.071x   48/48 parents

**A loss difference of 0.069 against linear is a 36 % reduction in reads**, and
the learned selector needs fewer reads on every one of 48 held-out parents
against both references. No record is censored: every method reaches a positive
success probability everywhere.

The tail is reported because the document asks for it and because `n_q` is
convex in the loss, so the mean is dominated by the worst records. The
hardest 1 % of records need 115 reads under the learned selector and 148 under
linear — the ordering holds in the tail, and the tail is where a read budget
actually gets spent.

## What this is not

- **p here is a simulated success probability** from exact propagation, not a
  fraction of observed reads. The binomial Clopper–Pearson interval that
  `time_to_solution` computes for hardware counts therefore does not apply and
  is not used; the uncertainty above resamples **logical parents**, the unit
  the rest of this project resamples.
- **Reads-to-target, not a certified optimum.** "Success" is the decoded
  logical indicator, as everywhere else in this project.
- **Independent identically distributed trials at fixed per-read cost.** Real
  devices have programming cycles, drift and correlated reads; none of that is
  modelled.
- **Execution cost only.** This is the `C_execution` term of the document's
  amortization equation. The learned method also carries `C_data` and
  `C_training`, which `costs.py` accounts for separately and which do not
  disappear because the read count improved. A 36 % read reduction is not a
  36 % cost reduction until the deployment count `M` is fixed.
