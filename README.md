# annealctrl v0.2 — config-driven embedded Ising control

Runnable research pipeline: compositional Hamiltonians, audited labels,
model/ablation training, exact checkpoint resume, held-out evaluation and paper
tables/figures. Working software is not evidence of superiority or A* acceptance.

**Start with `RUNBOOK_VI.md`** for the complete Vietnamese operating guide.

## Controlled acquisition and paper evidence

The acquisition study tests whether labelling a frozen policy's proposals helps
beyond more bank labels and random waveforms from the same decoder. It completes
all training arms before reading test outcomes, preserves real numerical
diagnostics, and compares direct deployment with the same untouched bank.

```bash
python -m annealctrl acquisition-study --config configs/acquisition_smoke.json --output runs/acquisition_smoke
python -m annealctrl acquisition-study --config configs/acquisition_research.json --output runs/acquisition_research --dry-run
```

- [Method, architecture, conditional bounds and experiment contract](docs/acquisition_methodology.md)
- [Literature BO baseline and disclosed adaptations](docs/literature_baseline.md)
- [Repeated scaling benchmark and OOD protocol](docs/scaling_and_ood_protocol.md)
- [Statistical corrections and parent/seed uncertainty](docs/statistical_inference.md)
- [Fresh reanalysis of archived held-out results](reports/statistics_reanalysis_2026-09-17/README.md)
- [Completed 60-parent, three-seed acquisition pilot, including negative results](reports/acquisition_pilot_2026-09-17/README.md)
- [Audit of newer upstream results](docs/latest_evidence_audit_2026-09-17.md)
- [Corrected teacher comparisons on audited, matched populations](reports/comparison_audit_corrected_2026-09-17/RESULTS.md)

Summary moments remain the inexpensive reference encoder. The archived results
support embedding information in bank selection, but do not establish a bank
advantage for hierarchy. Pooled information effects are exploratory and outside
the pairwise Holm correction family. New acquisition runs must establish their
own benefit; no architecture or A* acceptance claim follows from implementation.

## Install and run one complete small experiment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,plots]'
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
python -m annealctrl doctor
python -m pytest -q
python -m annealctrl run --config configs/experiment_smoke.json --output runs/my_smoke --dry-run
python -m annealctrl run --config configs/experiment_smoke.json --output runs/my_smoke
```

Smoke: 12 parents, two sizes, 24 tasks, 192 candidate outcomes, five methods × two
seeds. Three epochs exercise the pipeline; they are not research hyperparameters.

| Run output | Contents |
|---|---|
| `experiment.json`, `resolved_config.json` | Frozen config/source, environment, stage/run states |
| `data/` | Parent-split records, teacher caches, coefficient and label checksums |
| `data_audit_train.json` | Training-only diagnostics, not test headroom |
| `models/METHOD/seed_N/` | Best/latest checkpoints and training/validation history |
| `evaluations/` | Bank/direct selections, true simulator scores and costs |
| `paper/` | Markdown/LaTeX tables, JSON values, editable SVG/PNG figures |

## Measurement gates G2 and G3 (v0.3)

Two instruments decide whether the research claims survive, and both are built so
that "no measurable effect" is a reachable, reportable outcome.

```bash
# G2: does instance-specific control choice buy anything above the numerics?
python -m annealctrl control-sweep --data runs/my_smoke/data \
  --config configs/frontier_smoke.json --output runs/frontier_val --report

# G2b: qualify a stress subset from train/validation headroom only.
python -m annealctrl screen --fit-sweep runs/frontier_train runs/frontier_val \
  --apply-sweep runs/frontier_test --output runs/screen.json

# G3: does changing one embedding factor change the PREFERRED control?
python -m annealctrl intervention-sweep \
  --config configs/intervention_smoke.json --output runs/interventions --report

# Whole campaign, resumable, on a remote host.
bash scripts/launch_apollo.sh ~/runs/campaign_v1            # no scheduler
sbatch --export=ALL,OUTPUT=$HOME/runs/campaign_v1 scripts/launch_goose.slurm
```

Headroom smaller than the integrator's own loss ambiguity is **censored**, not
reported as a small positive effect. A swap of the preferred control requires
both transfer directions to be decisive. Chain-strength interventions always
report a scale-controlled arm beside the total compiled effect, because chain
strength moves the programmed scale as well as the penalty.

Read `docs/g2_headroom.md`, `docs/g3_interventions.md` and `docs/observability.md`
before launching a campaign; `SPEC.md`, `PLAN.md` and `docs/decisions/` record
what was built and why.

## Configurations and stages

Edit `configs/data_research.json` for distributions/physics/label budgets and
`configs/experiment_research.json` for methods, seeds, losses, device and reporting.
The research template requests 4,320 tasks / 276,480 candidate outcomes.
Archived held-out results are under `reports/heldout_2026-09-17`; they are distinct
from the newly added controlled-acquisition study. Profile a subset before
spending a new generation budget.

```bash
# Same frozen experiment, reusing completed stages.
python -m annealctrl run --config configs/experiment_smoke.json --output runs/my_smoke --resume
# Add --stage generate|train|evaluate|report for separate stages.
python -m annealctrl tune --config configs/tuning_smoke.json --output runs/tuning
python -m annealctrl train-config --config configs/training.json --data runs/my_smoke/data \
  --output runs/custom/best.pt --seed 0
```

Tuning is a bounded validation-only grid with shared generated data. Checkpoint
resume restores optimizer/RNG/history at epoch boundaries. `train-config` can
resume with `--resume-from runs/custom/best.latest.pt` and extend total epochs;
other settings must match. Changed hyperparameters require a new run. No automatic
stale-lock takeover occurs. Verify no live writer before manually removing a lock.

Training uses graphwise accumulation, not fused graph batching or multi-GPU DDP.
Methods/seeds run sequentially. Equal width does not imply equal parameter count;
checkpoints retain parameter counts for capacity-matched studies.

## GPU/server

Use CUDA PyTorch and one compatible CuPy package, following
[CuPy installation](https://docs.cupy.dev/en/stable/install.html).
Do not blindly install a CUDA-12 wheel on a CUDA-11 server. No sudo is needed.

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m annealctrl doctor --require-gpu
python -m pytest -q tests/test_physics.py tests/test_adaptive_teacher.py
python -m annealctrl profile-generation --config configs/data_smoke_v2.json \
  --backends numpy cupy --output runs/gpu_profile
```

Only after parity/memory checks run `experiment_research.json` without `--dry-run`.
`scripts/train.slurm` is an unsubmitted editable cluster template. No GPU path
silently falls back to CPU. Parent workers are CPU-only; CuPy uses one parent
worker and can batch candidates. Full-spectrum teachers remain a CPU cost.
Cross-version/device bitwise reproducibility is not guaranteed; see
[PyTorch reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html).

## Commands

`python -m annealctrl COMMAND --help` provides arguments:

- `run`, `tune`, `train-config`, `train`: experiment/training workflows.
- `generate`, `hardware-grow`: data or one connected partition.
- `evaluate`, `infer`: audited test scoring or outcome-free waveform prediction.
- `control-benchmark`: exact-window/eight-bin/pause best-found search; test-time
  search requires explicit `--allow-test-adaptation`.
- `audit`, `report`, `doctor`, `profile-generation`, `benchmark`: diagnostics.
- `simulate-open`: independent small Lindblad density-matrix calculation.
- `export-hardware`, `ingest-hardware`: constrained offline payload and supplied
  sample aggregation. **No QPU submission, credentials, or network execution.**

## Important contracts

Compiled HZ contains problem and chain terms once, with declared scaling. Bit i
is LSB-indexed; bit zero means spin +1. Success accepts all decoded logical
optima under the recorded tie rule, not just one planted string.

Split logical parents before variants. Exact labelled coefficient duplicates
are audited, not full graph/gauge isomorphism. Split-specific loading does not
read other outcome files. Policy/critic never consume response labels/test losses.

Dense full-spectrum teacher <=10 physical qubits. Outcome-only exact acceptance
<=20 with explicit memory budgets above10. State/endpoint costs remain exponential;
incomplete sparse eigenpairs are not a complete response teacher. Adaptive random
audits and step-doubling are diagnostics, not uniform certificates.

ML helpers require canonical X/Z (no nonzero catalyst/nonunit global energy_scale).
Broader physics APIs do not imply broader model support. Nine-knot stored controls
are simulated after resampling; exact-family frontiers use separate switch-aligned
search. Best-found-relative regret may be negative and is not clipped.

Parent bootstrap and seed variability stay separate. All predeclared methods/seeds
must finish before a paper table is produced. Resumed costs may omit time before
a hard kill and are flagged incomplete. Negative results are retained.

Lindblad/calibration utilities do not establish device fidelity or replace the
closed-system training contract. No CUDA-Q/QuTiP adapter, tensor-network teacher,
multi-GPU trainer, actual GPU speedup, QPU result or publication result is claimed.
The exact GPU package named in the meeting remains unresolved.

## Module documentation

See `docs/data.md`, `teacher.md`, `training.md`, `benchmarking.md`, `adapters.md`,
`paper_protocol.md`, `algorithms.tex`. Old v0.1 evidence is retained under
`reports/smoke/`; release verification is `reports/V02_VERIFICATION.md`.
CI is supplied as source, not published. Choose authorship/license/anonymity
metadata before public release.
