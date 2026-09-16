# Training and ablation contract

The training implementation is complete for the repository's **closed-system,
finite-bank-labelled task**. It does not imply that simulator labels transfer to
a physical annealer, or that a neural policy beats a validation-tuned classical
schedule. Those are experimental questions.

## What the controller consumes

`graph_from_record` reads the programmed physical Ising, original logical Ising,
chain membership, positive runtime, programmed scale, and path-query coordinates.
It does **not** read candidate losses, ground states, spectral responses, split
names, record IDs, or parent IDs into model features. Fixed or adaptive teacher
coordinates enter only the auxiliary response branch; the policy and critic do
not consume query positions. Adaptive coordinates are privileged training queries,
not free deployment measurements. At deployment, public fixed queries suffice;
neither their responses nor adaptive locations are needed for action selection.

The implementation currently rejects a nonzero catalyst or nonunit global
`energy_scale` in this graph adapter: silently hiding a driver graph or altered
Hamiltonian path would invalidate the learned representation.
The simulator's broader Hamiltonian support is not a claim that this encoder
already supports every driver.

## Four selectable encoders

Use `AnnealController(encoder_variant=...)`. All variants share the same response,
proposal, and waveform-critic head architecture, with independently trained
weights. Only modules used by a variant are instantiated.

| Variant | Graph information actually used | Excluded information |
|---|---|---|
| `hierarchical` (default) | Signed physical message passing → chain sum/mean/count → signed logical message passing; role-tagged multilevel tokens; runtime and scale | All expensive labels and identifiers |
| `physical` | Flat signed physical message passing; physical node/coupler features; runtime and scale | Logical graph coefficients and hierarchical chain pooling |
| `logical` | Original logical fields/couplers, logical message passing, logical node count, runtime | **Every physical/chain feature, programmed scale, embedding sizes and physical tokens** |
| `summary` | Mean and population-standard-deviation of physical/logical node/edge feature channels; node/edge counts; mean/max chain length; runtime and scale | Message passing and node-token topology |

`physical` still sees the public `same_chain` edge channel and chain-related
local physical features. It is a flat physical representation ablation, **not**
an embedding-blind baseline. `summary` is physically informed but topology is
visible only through its stated aggregate statistics. These distinctions should
remain explicit in paper tables.

The logical branch discards the second context coordinate even after train-only
normalization; changing compiled couplings, memberships, physical features, or
programmed scale cannot change its prediction. Tests verify this isolation.

For fair comparisons, report parameter counts and tune width/depth on validation
under a common budget. Equal width does not imply equal parameter count or cost.

## A complete Python training call

```python
from annealctrl.learning import fit_records
from annealctrl.pipeline import load_records

records = load_records("runs/pilot_data")
train = [r for r in records if str(r["split"]) == "train"]
validation = [r for r in records if str(r["split"]) == "validation"]

fit = fit_records(
    train,
    validation,
    model_config={
        "encoder_variant": "hierarchical",
        "width": 64,
        "physical_layers": 3,
        "logical_layers": 2,
        "proposals": 3,
        "schedule_points": 9,
        "max_ds_dtau": 4.0,
    },
    epochs=100,
    patience=15,
    learning_rate=1e-3,
    weight_decay=1e-4,
    batch_size=4,
    accumulation_steps=2,
    policy_weight=0.2,
    ranking_weight=0.1,
    response_weight=0.05,
    label_temperature=0.05,
    bandwidth=0.1,
    ranking_tolerance=1e-5,
    max_grad_norm=1.0,
    seed=0,
    device="cuda",
    deterministic=True,
    checkpoint="runs/model/best.pt",
    latest_checkpoint="runs/model/latest.pt",
)
```

The example selects explicit CUDA. An unavailable CUDA device raises an error;
choose `device="cpu"` explicitly for a CPU run. The CLI/experiment configuration
provides the standard workflow; this API is useful for custom experiments.

`model_config` is instantiated **after** seeding. When supplying an existing
`model=...` instead, its supplied weights are preserved; for repeatable initial
weights call `seed_everything(seed)` before constructing it. Do not supply both
`model` and `model_config`.

## Loss, batching, and validation

For each graph, training combines:

1. Huber regression of candidate outcome losses.
2. Pairwise ranking only when label differences exceed the configured tolerance
   plus the candidates' numerical-uncertainty indicators.
3. Multimodal policy distillation: a mixture kernel in waveform space is matched
   to soft labels on the evaluated candidate bank. This is **not** a differentiable
   simulator loss and does not interpolate labels for off-bank controls.
4. Masked, log-transformed response-moment regression; unresolved labels do not
   contribute.

Setting an auxiliary weight to zero disables its gradient contribution. The
outcome term has unit weight. The response head remains available for evaluation
of the corresponding ablation. Per-epoch history records every component loss,
validation bank regret, validation mean outcome loss, and optimizer update count.

The effective optimizer group has at most
`batch_size * accumulation_steps` graphs. Each graph contributes equally to the
group mean. A final short group is divided by its **actual** cardinality. Gradients
are clipped once per optimizer update. Graph forwards/backwards are processed
one at a time, so this is memory-bounded accumulation, not a claim of vectorized
graph batching or linear GPU speedup. CPU graph features are cached and moved to
the accelerator only for each forward; candidate banks move with the current
record. The in-memory record loader is still a host-RAM scaling consideration.

Train and validation must be nonempty, explicitly labelled, and logical-parent
disjoint. Test records cannot be passed into `fit_records`. Normalization is fit
only on training graphs. Best epoch selection minimizes mean validation
finite-bank regret with strict improvement; ties keep the earlier epoch. This is
a task-weighted validation mean, so balance variant counts across parents when
designing the training distribution. Final paper uncertainty should use paired
logical-parent resampling, not treat variants as independent replicates.

## Feasible proposals and the knot contract

`proposals`, `schedule_points`, and `max_ds_dtau` are configurable. The capped
simplex decoder yields monotone equal-time piecewise-linear waveforms with exact
endpoints and normalized slope bound `ds/dtau <= max_ds_dtau` (up to floating
point tolerance). Physical slope is `ds/dt = (ds/dtau)/runtime`; these are not
device-specific hardware approval limits.

Candidate waveform arrays must use exactly the model's uniform knot grid.
Training rejects a knot-count mismatch. Generate and simulate labels for the new
representation if changing the grid: resampling controls without recomputing
their outcomes would change the supervised task. Proposal count or slope cap
can change without changing the bank knot count, but a new run is then required.

## Checkpoints and exact resume

Every completed epoch atomically updates two optional artifacts:

- **`best.pt`**: top-level `model_state` and `optimizer_state` correspond to the
  best validation epoch. Use this file for held-out evaluation.
- **`latest.pt`**: top-level states correspond to the latest completed epoch.
  Use this file to resume interrupted training.

Both contain the full latest `resume_state`, best model/optimizer states, current
epoch, early-stopping counter, full history, Python/NumPy/Torch RNG states,
architecture and training configurations, train-only normalization, parent/record
IDs, available dataset fingerprints, and SHA-256 digests of the **actual ordered
training and validation record contents**. Digests include labels and coefficient
arrays; retaining IDs after editing labels does not evade the lineage check.

```python
resumed = fit_records(
    train,
    validation,
    epochs=150,                 # Total target, not +150 additional epochs.
    patience=15,
    learning_rate=1e-3,
    weight_decay=1e-4,
    batch_size=4,
    accumulation_steps=2,
    seed=0,
    device="cuda",
    resume_from="runs/model/latest.pt",
    checkpoint="runs/model/best.pt",
    latest_checkpoint="runs/model/latest.pt",
)
```

The example resumes the earlier call because every omitted loss/gradient option
equals its earlier default; pass all customized options unchanged. Architecture
is read from the checkpoint when `model` and `model_config` are omitted. Resume
rejects changes to training settings, model configuration, data contents/order,
or optional external dataset fingerprint. Only increasing the total epoch target
and changing device are permitted; cross-device bitwise equality is not promised.
The same ordered CPU data, settings, and implementation produce exact
epoch-boundary continuation, verified against uninterrupted training in tests.

An early-stopped run stays stopped when resumed: increasing its epoch limit does
not erase an already-triggered stopping decision. For an intentional new
fine-tuning experiment, explicitly use `load_checkpoint(...)` to obtain a model,
then start a new `fit_records(model=model, ...)` without `resume_from`; this
refits training normalization and resets optimizer/early-stopping state. Record
that lineage and do not describe it as an exact resume.

Legacy v0.1 checkpoints remain inference-loadable with `load_checkpoint`, but
cannot provide exact resume because they lack current-epoch RNG/history state.
Checkpoint deserialization uses `weights_only=True`, not arbitrary unpickling.

Store the dataset manifest and source release/checksums with each study. Content
digests do not certify that two different source implementations have identical
semantics. CUDA deterministic kernels may require
`CUBLAS_WORKSPACE_CONFIG=:4096:8` before launching Python; unsupported deterministic
operations fail rather than silently accepting nondeterminism. GPU availability
and performance must be checked on the actual training host.

## Limits of the completed training implementation

- Single-process AdamW; no DDP, mixed precision, or fused graph batching is
  claimed. These are performance extensions, not missing scientific labels.
- The response head has three moments under the existing dataset contract.
- Finite-bank validation measures critic selection on actual evaluated controls;
  it is not a continuous-control optimum. Direct proposals need independent
  physical simulation or hardware execution.
- Curriculum, hyperparameter search, and seed/family/size sweeps are orchestration
  choices outside `fit_records`; do not change settings after looking at test
  results.
- Closed-system dynamics and toy hardware scaling are not a calibrated
  commercial-annealer noise or timing model.
