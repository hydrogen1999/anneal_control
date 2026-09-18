# Statistical interpretation and audit corrections

The experimental unit is the logical parent. Different embeddings, chain
strengths, runtimes, and schedules from one parent are repeated measurements,
not independent samples. Report effects with the explicit sign convention
**method B minus method A**; negative loss differences favour B.

## Parent and training-seed uncertainty

`paired_method_contrast` first averages each method's measurements within each
parent and then forms paired parent differences. Its percentile bootstrap
interval resamples parents and is conditional on the training seeds that were
actually evaluated. Adding many interventions or evaluating more records with
the same trained model does not increase the number of independent training
runs. Use complete, balanced seed-by-parent evaluation panels for this analysis.

`paired_parent_seed_contrast` is an additional sensitivity analysis for repeated
training runs. It requires matching records in a complete parent-by-seed panel,
averages interventions within each cell, and independently resamples the parent
and seed axes. Pairing is retained between methods in every draw. This is the
two-factor pigeonhole bootstrap described by [Owen (2007)](https://arxiv.org/abs/0712.1111).
It accounts for a seed affecting all test parents rather than pretending that
each parent received a separately trained model. The interval is approximate;
with three seeds, the distribution of training variability remains poorly
resolved. This procedure does not integrate over alternate training datasets,
hyperparameter searches, or adaptive choices made after viewing test outcomes.
It emits no p-value and no equivalence decision. With fewer than two parents or
two seeds it returns an explicit insufficient-data status instead of a joint CI.

## Tests, multiplicity, and non-rejection

From contrast schema version 2, the parent-level test resamples differences
centered to satisfy a zero-mean null, and compares the absolute resampled mean
with the absolute observed mean. The returned `p_value` is an approximate
centered-null bootstrap p-value with a finite Monte Carlo correction. This is
not an exact permutation test, nor does the implementation promise finite-sample
type-I error control. The percentile interval and null test are distinct
summaries. Historical schema version 1 used an uncentered percentile-tail
quantity; archived numerical p-values therefore cannot be relabeled as outputs
of the new test without rerunning the analysis from the original rows.

`contrast_matrix` applies Holm adjustment to all pairwise p-values in the
declared method family for one evaluation mode. A reported separation also
requires the pointwise interval to exclude zero. These intervals are **not**
simultaneous confidence intervals, and Holm does not correct other analyses,
other modes, or later decisions to add comparisons. Report the complete family
and all planned primary comparisons before running the confirmatory evaluation.

Non-rejection is not transitive. If A and B are not separated, and A and C are
not separated, B and C may still be separated. `nonseparated_maximal_sets`
therefore enumerates maximal sets whose every pair is not separated; these sets
can overlap. The legacy `indistinguishable_groups` field is a deprecated alias.
Neither field establishes equality or practical equivalence. An equivalence
claim would require a scientifically justified margin declared before viewing
the results and a suitable equivalence analysis.

## Correction to the 2026-09-17 held-out narrative

The historical `reports/heldout_2026-09-17/HELDOUT.md` calls the pooled
embedding-information effect Holm-corrected. That wording is incorrect. The
reported pooled aware-minus-blind effect of approximately -0.00782 and its
parent-bootstrap interval [-0.01109, -0.00447] are an **unadjusted exploratory
pooled contrast**, conditional on the evaluated seeds. The separate ten-pair
comparison matrix applied Holm adjustment to its own family; the pooled
contrast was not a member of that family. Historical raw records and report
artifacts are retained unchanged to preserve their provenance. Any paper using
them must use this corrected interpretation and archive a fresh analysis if
using schema version 2 p-values.

## Proposal diagnostics

Spearman correlation now gives tied values average ranks and is undefined for
constant vectors. Breaking ties by array position can create a spurious
correlation and makes the result depend on proposal ordering. The decomposition
`direct loss - bank-selected loss = ranking regret + generation gap` remains an
identity for the proposals actually scored. A large ranking term does not by
itself identify distribution shift: limited critic capacity, optimization
failure, and label uncertainty can also produce ranking errors. Compare the
same frozen proposals under old and refitted critics, and use matched bank and
decoder-random acquisition controls before making a causal explanation.

## Authoritative archived-outcome reanalysis

`python scripts/rebuild_evidence.py` rebuilds the current comparison and both
synthetic/Pegasus contrast analyses from committed record-level inputs, without
PyTorch or private server directories. `--check` compares every generated byte
against the archived output and fails on drift. The output records source and
input hashes, bootstrap settings, teacher eligibility IDs, matched teacher
contrasts, and per-search population/budget validation. Historical JSON and
figures are kept as superseded artifacts, not silently overwritten.

These inputs contain learned seed averages, not full seed-by-parent panels.
The reanalysis therefore cannot quantify training-seed uncertainty, recover
checkpoint choices, or turn the historical aggregation summary into a paired
confidence interval. Search-seed fields are also absent from the archived
optimizer rows. Equal record identity and objective budgets are verified;
matching or robustness across search seeds is not inferred.

Do not convert a post-hoc observed effect and variance into a statement that
19, 30, or any fixed number of new parents will guarantee significance. Plan a
future sample using a prespecified meaningful effect or interval precision and
report its assumptions. Overlapping size-stratum intervals do not establish
no decay or equivalence. A replay at the same training seed verifies
determinism, not zero training uncertainty.
