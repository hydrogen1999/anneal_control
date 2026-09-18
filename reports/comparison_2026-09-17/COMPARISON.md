# Comparison archive and corrected analysis

Use **[the audited 2026-09-18 table](../evidence_audit_2026-09-18/RESULTS.md)** for
paper text and **[its machine-readable table](../evidence_audit_2026-09-18/comparison_table.json)**
for plotting. Reproduce from the repository root:

```bash
python scripts/rebuild_evidence.py
python scripts/rebuild_evidence.py --check
```

This directory preserves the original record-level inputs: `testref_rows.json`,
`bayes_rows.json`, and the corrected `policy_gradient_rows.json`. Historical
`comparison_table.json` and `figure1_comparison.*` are **superseded**: their
teacher eligibility included failed interpolation audits and must not be used
as current paper results. Raw numeric data are not rewritten to conceal the
correction.

The audited full-population means are summary/bank 0.544662, global 0.565437,
Bayesian search 0.505460, Sobol-local 0.507352, and corrected policy-gradient
search 0.511392. Each search uses 257 objective calls per record on the same
864 records/48 parents. Learned inference uses no online simulator queries;
offline label and training costs still need to enter an amortization claim.
These are distinct cost classes.

Teacher comparisons require separate matched populations: d2 has 711 eligible
records/45 parents, and gap_inverse_square has 339/23. Summary/bank minus teacher
is respectively −0.050399 [−0.069051, −0.032866] and −0.083240
[−0.114283, −0.052837]. These are unadjusted parent intervals conditional on
passed sampled-point audits and archived seed averages. Do not compare teacher
subset means against the full-population learned mean.

The first policy-gradient run used an incorrect Gaussian score and reported
0.5130; that result is withdrawn. The corrected score uses (z − μ)/σ². The
corrected 0.511392 remains worse than both search alternatives. The methods are
generic optimizer baselines; calling REINFORCE a reproduction of a published
quantum-control method would be inaccurate.

See the [evidence ledger](../EVIDENCE.md) for aggregation, Pegasus, transfer, and
artifact limitations. No table in this directory establishes hardware results,
architecture superiority, or a globally optimal control reference.
