# Evaluation and exact-control benchmarks

`annealctrl.benchmarking` is the reusable evaluation layer; the CLI delegates to
it. Evaluation results are JSON-safe and use `null`, never `NaN`/`Infinity`, for
undefined quantities. Simulator time is dimensionless physical evolution time;
measured `*_seconds` fields are actual CPU/GPU wall time. They are not comparable
units and are never silently combined.

## Held-out checkpoint evaluation

```python
from annealctrl.benchmarking import evaluate_checkpoint

result = evaluate_checkpoint(
    "runs/data", "runs/model.pt", device="cpu", direct=True,
    backend="numpy", tolerance=5e-4, initial_steps=128, max_steps=8192,
    n_resamples=2000, seed=0,
)
```

The dataset must contain explicit train, validation and test parent splits.
Checkpoint parent provenance is checked against all three splits. New v0.2
checkpoints also provide training/validation record fingerprints and IDs;
legacy checkpoints explicitly report their weaker parent-only provenance. Use
the validation-selected **best** checkpoint, not a `.latest` resume snapshot.

The common bank check compares IDs, time knots and every actual normalized
waveform across all three splits. Matching bank lengths alone is not enough.
The linear waveform is identified by its values, not assumed to occupy index 0.
One global candidate is selected using the mean of within-parent validation
means; no test outcome is involved.

For each test record the bank critic chooses an index before inspecting its true
loss. The optional direct policy first proposes waveforms, then its critic selects
one; only that frozen waveform is scored by the simulator. Unselected proposals
are not simulated. The evaluator never interpolates losses from the training
bank to label a new waveform.

Outputs include per-record success/loss, decoded expected energy, chain-break
probability and fraction when available; record/parent IDs, family, sizes,
runtime; linear, global-validation and best-bank reference losses; inference and
offline scoring cost. `reference_bank_regret` is not clipped: a direct proposal
may beat the finite reference bank. This does not prove a global control optimum.

Headline CIs use paired equal-parent bootstrap, not independently resampled
variants. One-parent evaluation remains usable but reports `null` CI endpoints.
The bootstrap does not include training-seed uncertainty. Seed replication must
be reported separately or analyzed using a prespecified multilevel design.

## Exact-waveform family comparison

```python
from annealctrl.pipeline import load_records
from annealctrl.benchmarking import benchmark_record_controls

record = load_records("runs/data", "validation")[0]
result = benchmark_record_controls(
    record, budget_per_family=32,
    families=("linear", "one_window", "two_window", "eight_bin", "pause"),
    seed=0, backend="numpy",
)
```

Each tunable family consumes exactly the same number of objective evaluations.
The initial linear incumbent counts toward that budget; rejected proposals also
count. Odd trials use scrambled Sobol exploration; even trials use clipped local
perturbations around the best parameters observed so far. Every trial, parameter,
waveform, accepted-state diagnostic and running incumbent is retained. This is
an explicit reproducible derivative-free search baseline, **not** a claim to
globally optimize a control family. Linear is parameter-free and consumes one
evaluation, not budget-padding repeats.

The one-/two-window and pause controls retain their exact switching knots.
They are **not** resampled onto the model's nine-point waveform representation.
Eight-bin controls allocate traversal durations over eight equal path intervals.
Pauses have a genuinely flat segment. All families obey the same runtime and
normalized slew constraint (`max_ds_dtau=4` by default). For fairness at a fixed
physical slew across different runtimes, explicitly set `max_ds_dtau=T*vmax` for
each record; the default is a normalized-time contract, not a hardware limit.

Test-instance search requires `allow_test_adaptation=True`. It is then labeled
online adaptation and all objective calls are charged. A search reference is not
the zero-search learned policy, and its test outcomes must not train or tune that
policy. Dataset-level aggregation should preserve each logical parent's weight.

## Numerical validity and paths

`score_schedule` reconstructs the stored Hamiltonian including optional XX
edges/weights, catalyst strength and whole-path energy scale. A nonzero catalyst
without its driver graph is rejected. The graph model's supported input path may
be narrower than the physics evaluator; unsupported model paths must fail rather
than be silently mapped to a standard X-driver path.

Every switching knot is aligned exactly by integrating schedule segments with
their own step counts. A coarse trajectory and a trajectory with twice the steps
are compared. The fine-state diagnostic `||psi_fine-psi_coarse||/3` must pass the
requested tolerance; norm preservation must pass separately. Failed attempts are
charged, and exceeding the maximum fine-trajectory step budget raises an error.
States are never silently renormalized. The loss ambiguity indicator
`2*delta+delta**2` is useful for suppressing unresolved comparisons but is **not a
certified error bound**. Independent dense-solver tests cover irregular pauses
and nonzero catalysts.

For simulated zero success probability, ideal IID time-to-solution is unbounded:
the JSON contains `null` time/reads and a censoring flag, not a pseudocount-derived
finite value. These simulated probabilities have numerical uncertainty, not a
binomial shot-count CI. Use `evaluation.time_to_solution` for actual success/read
counts, with all recurring per-read costs specified separately.

## Privileged spectral baselines

Pass `teacher_methods=("gap_inverse_square", "d2")` explicitly to the family
benchmark. These baselines construct their own spectrum and are reported in
`privileged_teachers`, outside the equal-budget family claim. Their spectral
teacher evaluations/time and subsequent outcome calls are separate. Unresolved
profiles may return no schedule; interpolation-audit failures remain visible even
when the tentative waveform is scored. A positive `teacher_gap_epsilon` explicitly
changes the method to a regularized variant. No spectral input is given to the
zero-shot learned policy at deployment.
