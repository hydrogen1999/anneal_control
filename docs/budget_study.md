# Learned warm starts and finite-budget control search

The question is whether a frozen learned selector reaches a useful **common
quality target with fewer online simulator calls**, after accounting for offline
cost. This campaign does not establish a globally optimal control, hardware
speedup, or an architectural advantage.

## Methods and fairness

The zero-query methods are the original learned bank selection, direct policy
selection, a single source-validation-selected waveform, and linear. Each choice
is frozen before its diagnostic simulator outcome is read. Diagnostic evaluations
are recorded as offline measurement work and cannot initialize the online
optimizers with a free known loss.

Every nonzero-query method optimizes the same `eight_bin` family: piecewise-linear
controls with **fixed path-coordinate knots**, and durations allocated by the
same bounded residual-softmax decoder. For Sobol/local search, GP expected
improvement, and the Finzgar-inspired GP confidence-bound adaptation, trial one
is a charged linear reference. A warm method uses trial two for its charged hint;
a cold method uses it for exploration. Warm hints come from bank, direct, or the
source-global schedule. The latter tests whether a good fixed source schedule is
already sufficient. Remaining proposals and all failed outcomes consume budget.

`finzgar_gp_ucb` here uses the identical eight-bin search space, in addition to
the disclosed task-specific changes documented in `literature_baseline.md`.
It is an adaptation, not a reproduction of the original nonmonotone p-spin
experiment. The separate original-system reference harness supplies that check.

Search streams are keyed by `(seed, logical_fingerprint)`; cold and warm arms
share the same seed. One trajectory runs to the preregistered maximum horizon.
All plotted budgets are prefixes of that trajectory, not separately retuned runs.
In particular the confidence-bound exploration schedule knows the fixed maximum
horizon. Changing that horizon requires a new campaign.

## The waveform conversion is not exact in general

The direct policy outputs values at **fixed time knots**. Eight-bin search fixes
path values. These are different sets of functions. A sampled bank can also
contain windows or pauses that eight-bin controls cannot exactly represent.

`eight_bin_hint` evaluates the inverse `tau(s)` at nine fixed path knots, uses the
midpoint inverse at a pause, and maps the residual durations back to unit-box
parameters. Clipping needed for finite logits is included in the error. The
maximum difference is measured exactly for the two piecewise-linear waveforms
on the union of their switching knots. This is a deterministic projection, not a
minimum-error projection. `mode=reject` refuses nonrepresentable hints;
`mode=project` explicitly records the original waveform, projected waveform,
maximum error and tolerance. Every projected waveform is scored anew. The
zero-query curve always reports the unprojected deployed control.

## Provenance and outcomes

The checkpoint must prove independence from the target at the logical Hamiltonian
level. Changed runtime, embedding, source version, or dataset-local parent names
do not establish independence. Legacy checkpoints require verified source records;
source data must match the checkpoint before choosing the global baseline.
Exact labelled logical fingerprints do not certify gauge/isomorphism independence.

The frozen manifest includes source/config/checkpoint/dataset/record hashes, exact
source-global selection, and offline-cost declarations. Completed unit artifacts
and query ledgers are hashed. Resume rejects changed inputs or artifacts. Every
simulator call has a durable requested receipt followed by an outcome receipt.
Replay regenerates deterministic proposals and returns stored outcomes without
rescoring. Different proposals fail the resume check. Completed numerical failures
remain failures. An interrupted request is retained, retried with a new attempt,
and its unknown inner work is marked as a lower bound. A retried trajectory cannot
enter the equal-budget Pareto set as though that work were free. Resumed partial
trajectories have unavailable deployment wall time, because replay overhead is not
ordinary deployment work.

Search convergence failures produce censored rows, not invented penalty labels.
Reports expose coverage and paired differences on jointly valid records. Pareto
membership requires the complete common record/seed population. Seeds are averaged
within logical parents for descriptive parent-bootstrap intervals. These intervals
are conditional on the chosen checkpoints; search seeds are **not** independent
model-training replications. Run every frozen training checkpoint separately and
use a preregistered parent-by-training-seed analysis for claims across training.
The output makes no equivalence or multiple-testing-adjusted significance claim.

## Wall time and amortization

Bank/direct latency includes graph creation, normalization, tensor transfers,
inference, selection and host copy; device synchronization and warmup exclusions
are explicit. Search wall time includes hint projection/inference, physical
observable construction, objective calls and optimizer overhead. Plot only
measurements taken under the same host/load contract; a seed-averaged walltime
curve is not a hardware-independent speedup.

Offline costs can be collected automatically with
`"offline_cost": {"acquisition_study": "runs/source", "fixed_recipe": true}`.
The runner verifies the source study/dataset/fit/acquisition receipts, sums source
data generation, all actual fits and each unique acquisition attempt once, and
hashes the supporting artifacts. It conservatively charges the entire frozen
study preparation, including comparator and mechanism arms. It is not the minimal
cost of training a single deployed method. Incomplete generation, restarted fit
costs, unknown acquisition work or an undeclared tuning recipe suppress break-even.
Held-out scientific evaluation, historical project development and external tuning
are excluded explicitly; `fixed_recipe` is an operator declaration about this
campaign, not an assertion that no earlier scientific exploration ever occurred.

Unknown offline data generation/training/acquisition/tuning cost remains **unknown**,
never zero. To declare complete offline cost, set `offline_cost.status=complete`
and supply all four `components`, each with nonnegative `seconds` and a `receipt`
file path. The receipt hashes and declared times enter the frozen manifest. Zero
cost for an unused stage still needs a receipt explicitly declaring it unused.
These are operator-declared walltimes, not automatic accounting of shared cluster
rental or all exploratory scientific effort.

`quality_thresholds` must be declared before looking at target outcomes. Conditional
break-even arithmetic is emitted only when both compared fully observed methods
meet the same threshold and online savings are positive. Unknown offline costs
produce no break-even estimate. Choosing the fastest measured grid point after
evaluation is labelled descriptive, and its break-even is not an independently
validated deployment policy or a confidence bound.

## Run

After data generation and training, edit paths in `configs/budget_study_smoke.json`
or `configs/budget_study_research.json`. Omitting `target_data` uses held-out source
test records; setting it enables independently generated transfer targets.

```bash
annealctrl budget-study --config configs/budget_study_smoke.json --output runs/budget_smoke
annealctrl budget-study --config configs/budget_study_smoke.json --output runs/budget_smoke --resume
python scripts/plot_budget_study.py --report runs/budget_smoke/report.json --out runs/budget_smoke/curves.pdf
```

The root contains `manifest.json`, `rows.json`, `report.json`, and per-record
selection, unit results and objective ledgers. Keep all receipts when archiving a
claim. Smoke settings validate software only. A research campaign is evidence
only after it finishes, passes coverage checks, and is analyzed as preregistered.
