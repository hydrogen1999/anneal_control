# The same result, in the unit a practitioner budgets in

## Why this was missing

The design document's amortization section specifies

    n_q = ceil( log(1 - q) / log(1 - p) )

with two conditions: state the independence assumption, and never turn a
zero-success observation into a finite number. `evaluation.time_to_solution`
implements both, including Clopper–Pearson bounds and a `censored` flag. **No
archived artifact had ever reported it**, so every loss difference in this
project had only ever been expressed in loss units.

## A correction to the first version of this report

The evaluation records carry two fields with nearly the same name and the same
value — `bank_best_loss` and `best_bank_loss` — and **both are the oracle best
over the bank**, a property of the record rather than of the method
(`benchmarking.py:391-392`). The method's own selection is
`bank_best_loss + bank_regret`.

The first version of this report read `bank_best_loss` as the learned
selector's loss. It therefore reported the **bank oracle** under the learned
method's name, and overstated the read reduction against linear as 36 % when
it is 22 %. The corrected table is below; the oracle is kept as its own row,
which is where it belongs.

The script now reconstructs the method loss and **checks it against each
evaluation's own `mean_loss`**, refusing to continue if they disagree. They
agree to exactly 0.0, which is the check that would have caught the original
error.

## Result

864 held-out records, `summary` encoder averaged over five training seeds
before any aggregation, 99 % target confidence:

| method | mean loss | parent-mean reads | p50 | p90 | p99 | censored |
|---|---:|---:|---:|---:|---:|---:|
| linear at matched duration | 0.6009 | 22.3 | 12 | 49 | 148 | 0 |
| tuned global schedule | 0.5654 | 19.2 | 10 | 43 | 124 | 0 |
| **learned bank selection** | **0.5447** | **17.5** | 9 | 40 | 115 | 0 |
| *bank oracle (ceiling, not a method)* | *0.5318* | *16.9* | *9* | *39* | *115* | *0* |

Paired per-record read ratios, parent bootstrap:

    linear / learned        1.279x  [1.243, 1.319]   median 1.227x   47/48 parents
    global / learned        1.128x  [1.092, 1.167]   median 1.034x   44/48 parents
    bank oracle / learned   0.952x  [0.933, 0.966]   median 1.000x    0/48 parents

**A loss difference of 0.056 against linear is a 22 % reduction in reads**, and
the learned selector needs fewer reads on 47 of 48 held-out parents against
linear and 44 of 48 against the tuned global schedule. Nothing is censored.

The last row is a useful diagnostic rather than a competitor: the selector
gives up **4.8 %** of reads against a perfect choice from the same bank, and on
the median record gives up none. Most of what the bank can offer is being
taken.

The tail is reported because the document asks for it and because `n_q` is
convex in the loss, so the mean is dominated by the worst records. The hardest
1 % need 115 reads under the learned selector against linear's 148 — the
ordering holds where a read budget is actually spent.

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
