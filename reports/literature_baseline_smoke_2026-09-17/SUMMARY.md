# Literature baseline implementation checks — 2026-09-17

These are bounded execution and physics checks, **not paper-level comparative
evidence**. Both successful and unfavorable results are retained.

## Embedded-task adaptation

`embedded_task_reference.json` contains all 360 queries: two validation records
from **one** logical parent, three seeds, 60 queries per record/seed. The parent
has three logical and six physical qubits. The two records differ in runtime.
There were zero numerical failures. Settings, source fingerprint, manifest hash,
record payload fingerprints, waveforms, diagnostics and raw query costs are
included in the archive.

| Runtime | Linear loss | BO best loss, seed 0 | Seed 1 | Seed 2 |
| --- | ---: | ---: | ---: | ---: |
| 2 | 0.69667247 | 0.65288603 | 0.66045169 | 0.64697552 |
| 8 | 0.50089090 | 0.27995410 | 0.36830972 | 0.27072904 |

The optimizer can find improved controls in these examples. One parent provides
no basis for a population confidence interval or a comparison against trained
neural policies. The control parameterization is the capped-simplex policy
decoder; this is explicitly the task-constrained adaptation.

## Original-system reference

`pspin_reference.json` contains all 360 queries for the independent p-spin check:
`N=15`, `p=3`, `Gamma=5`, `T=3`, four real-space parameters, `zeta=2`, and 60
charged queries per method and seed. GP-UCB and uniform search share exactly
their first ten parameter points within each seed. All queries passed the
two-tolerance numerical gate.

| Seed | GP-UCB best fidelity | Uniform best fidelity | BO minus uniform |
| --- | ---: | ---: | ---: |
| 0 | 0.13133367 | 0.19913906 | -0.06780538 |
| 1 | 0.19956293 | 0.22521557 | -0.02565263 |
| 2 | 0.29064067 | 0.19915096 | 0.09148971 |

Linear fidelity is `0.02346844266`. GP-UCB wins only one of three seeds here;
this small run does not demonstrate superiority over uniform search. It is not
a quantitative reproduction of the published figure's repetition study or its
tuned optimizer comparison. The observed mixed result is preserved, not tuned
away by seed selection.

The solver uses the permutation-symmetric sector, whose dimension is `N+1`.
This reference therefore does not establish scaling to 15 physical qubits in
the embedded pairwise-Ising task. The test suite compares this reduced solver
against independent full-Hilbert-space evolution for a smaller nonmonotone
example. Sixteen tests cover both baseline modules and pass.

The exact source correspondence, changed implementation choices, constraints,
and commands are in [the baseline protocol](../../docs/literature_baseline.md).
The published source is [Finžgar et al., PR Research 6, 023063](https://doi.org/10.1103/PhysRevResearch.6.023063).
