# Literature baseline: task-constrained GP-UCB schedule search

`annealctrl.literature_baselines` implements a **task-constrained adaptation** of
Finžgar et al., *Designing quantum annealing schedules using Bayesian
optimization*, Physical Review Research 6, 023063 (2024).
It is separate from the existing internal `bayesopt.py` expected-improvement
optimizer. Neither is evidence that a neural method outperforms the literature
until the matched experiments have actually run.

## Source correspondence

The [published article](https://doi.org/10.1103/PhysRevResearch.6.023063),
[published PDF](https://link.aps.org/pdf/10.1103/PhysRevResearch.6.023063), and
[author preprint](https://arxiv.org/abs/2305.13365) specify an isotropic Matérn-5/2
GP, confidence-bound acquisition, a linear plus nine random initial queries,
and decreasing exploration weight. The reported applications concern p-spin
and Rydberg systems. Some improvements depend on nonmonotone schedules.

Our implementation retains that surrogate/acquisition structure and initial
design. It minimizes loss using `mean - kappa * standard_deviation`, equivalent
to maximizing the confidence bound on success. Fifty adaptive steps hold
`kappa=2` for 25 steps and geometrically reduce it to `.01` over the remaining
25. All ten initial queries count: the default budget is **60 total calls**.
Other total budgets retain ten initial calls (or fewer if necessary), with
decay rescaled over the remaining adaptive steps.

## Explicit changes and limits

| Item | This implementation |
| --- | --- |
| Problem | Embedded pairwise Ising Hamiltonian and decoded logical success; not the original experimental systems. |
| Dynamics | Existing switch-aligned, convergence-checked closed-system simulator; no shot noise or hardware claim. |
| Main control family | Eight bounded logits mapped through the policy's capped-simplex decoder; nine equally spaced time knots, monotone, slope at most four. The finite default logit box is `[-6,6]`; it approximates the decoder family and is not every possible neural logit. |
| Real-space sensitivity variant | Seven interior knot heights with `zeta=.5`. This narrower box guarantees monotonicity; it is **not** the paper's usual `zeta=2` box. Nonmonotone controls are excluded by this project's task definition. |
| Kernel fitting | Independent NumPy/SciPy implementation, standardized labels, fixed `1e-6` noise variance plus `1e-10` Cholesky jitter, continuous marginal-likelihood optimization over log length scale `[-11.5,11.5]`, three starts. This is not the authors' library/version. |
| Acquisition optimization | 1,024 uniform candidates followed by three L-BFGS-B starts. Repeated parameter points are excluded. These computational choices are disclosed adaptations. |
| Unresolved simulation | A failed query consumes budget, appears in history, and contributes no fabricated loss. Known inner cost is a lower bound if a failed solver call cannot report its steps. |

This restricted real-space variant must not be the only comparator: doing so
would give the baseline less waveform freedom than the policy. Use the policy
decoder variant for the primary comparison and the real-space variant as a
sensitivity check. No claim of reproducing the original paper's numerical
results, nonmonotone mechanisms, or hardware findings is justified.

## Running the comparison

From the repository root, with the project installed:

```bash
python -m annealctrl.literature_baselines \
  --data runs/g2/data --split validation \
  --budget 60 --seeds 0 1 2 \
  --parameterization policy_decoder \
  --out runs/finzgar_validation_policy.jsonl
```

The data argument is the generated data directory, not its parent run
directory. Each JSONL row contains a record identity, all settings, all queried
waveforms, individual diagnostics, failures, objective-call counts, and the
complete best-so-far trace. Existing outputs are never overwritten. A bounded
smoke run can use `--max-records 2 --seeds 0 --budget 12`.
Actual test-time search requires `--split test --allow-test-adaptation`; this
is explicitly marked online adaptation. Select settings on validation first.

## Evaluation protocol

1. Freeze the dataset manifest, parent splits, normalization, control constraints,
   objective, tolerance, initial/max integrator steps, and runtime. Store their
   identifiers with the experiment. Never use the final test set to tune BO.
2. Prespecify a budget curve (for example 10, 20, 40, 60, 100 total queries).
   Budgets have different exploration decay schedules; running one 100-query
   trajectory and slicing prefixes is a different experiment and must be labeled.
3. Pair BO seeds and task records with equally charged random/Sobol and existing
   local-search baselines. Include the initialization, rejected/numerically
   unresolved queries, construction cost, optimizer wall time, and propagation
   work. Compare successful-incumbent loss alongside failure rates.
4. Evaluate neural bank selection and direct policy selection with zero online
   outcome queries, then show optional adaptation curves separately. A method
   that spends 60 simulator calls is not a zero-query deployment comparator.
5. Aggregate paired differences over logical parents before intervals. Report
   optimizer/training seed variation separately, all families and sizes, and
   total offline training/acquisition cost. Do not pick the best seed.
6. A claim that BO is surpassed requires the completed matched results. Closing
   the stronger *faithful published-experiment reproduction* requirement also
   requires quantitative agreement with the original experiment, beyond the
   independent original-system check below.

The implementation tests verify search semantics, exact query charging,
failure behavior, test-set gates, and physical scoring integration. A smoke
experiment verifies execution only; it cannot establish comparative efficacy.

## Independent check in the original p-spin setting

`annealctrl.finzgar_reference` separately implements the original closed-system
Hamiltonian in its permutation-symmetric basis. For `k` down spins, the problem
diagonal is `-N*((N-2*k)/N)**p` and the adjacent driver matrix element is
`-Gamma*sqrt((k+1)*(N-k))`. The initial amplitudes are the square roots of the
binomial probabilities. Evolution uses `H(t)=(1-u(t))*H_driver+u(t)*H_problem`.
The fidelity is the final probability of the all-up state. This is a p-body
reference problem; it does not increase the size of the embedded pairwise-Ising
dataset.

The [published PDF, Eqs. (3)–(6), Table I and Fig. 4(b)](https://link.aps.org/pdf/10.1103/PhysRevResearch.6.023063)
specifies the reference setting used here: `p=3`, `N=15`, `T=3`, four real-space
parameters, with the stated default `Gamma=5`. The `zeta=2` parameter box is
retained, including nonmonotone and out-of-unit-interval intermediate controls.
These controls use a separate `ReferenceSchedule`; the embedded-task hardware
constraints are never relaxed.
Specifically, Table I gives `[(j-zeta)/(n+1),(j+zeta)/(n+1)]` without an
intersection with `[0,1]`. We implement those printed bounds literally, without
clipping. This is a documented interpretation of the article, not verification
of the authors' unavailable implementation choices. The rising target control
`u` is the complement of the falling driver control `s` in Eq. (6).

```bash
python -m annealctrl.finzgar_reference \
  --budget 60 --seeds 0 1 2 \
  --out runs/finzgar_pspin_reference.json
```

The harness compares GP-UCB and uniform search using the same first ten queried
parameter points, with every call charged in both arms. It archives all queried
controls, losses, best-so-far traces, wall times and integration diagnostics.
Piecewise DOP853 evolution is repeated with tenfold tighter tolerances for every
query; the two final states must agree within `5e-7` and norm error must be below
`1e-6`. These numerical diagnostics are not certified error bounds.

The symmetric-sector solver is checked against an independently constructed full
Hilbert-space solver for a small nonmonotone example. The default linear control
gives fidelity approximately `0.02346844`; the published caption quotes roughly
`0.03`. This approximate reference point, independent GP implementation, and
three seeds do **not** establish numerical reproduction of the published
distribution. In particular, our bounded check does not replace the paper's
larger repetition study, tuned optimizer comparison, or experimental results.
