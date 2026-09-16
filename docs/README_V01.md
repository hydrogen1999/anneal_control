# Historical v0.1 README — superseded by the root README and RUNBOOK_VI.md

A modular, executable research scaffold based on the supplied meeting/document.
It is **not** a reproduced A* paper, a calibrated QPU simulator, or a GPU benchmark.

Start with `IMPLEMENTATION_PLAN_VI.md` (Vietnamese implementation plan),
`docs/paper_protocol.md` (claim/experiment protocol), and `docs/algorithms.tex`
(mathematical algorithms with source mapping).

## What works

- Compositional logical/physical Hamiltonian generation, connected chain growth,
  synthetic controlled lifts, planted-cycle endpoint certificates, programmed
  coefficient provenance, exact small-system checks, and declared decoding.
- Matrix-free NumPy propagation, optional CuPy execution, X/XX driver support,
  capped dense spectral teachers, projector-aware response, and incomplete
  sparse low-energy eigensolver with honest diagnostics.
- Feasible schedules, real pauses, shared candidate banks, budgeted local
  refinement, finite-difference control interventions, and opt-in privileged
  spectral baselines.
- Signed hierarchical PyTorch model with physical/chain/logical tokens, response
  auxiliary head, multimodal feasible policy and schedule-conditioned critic.
- Versioned NPZ/JSON generation, parent splits, convergence acceptance gates,
  resume, training/checkpoints, held-out bank and direct-policy scoring, bootstrap
  and cost utilities. CPU CI configuration and regression tests are included.

## Install and run

Use a new Python environment. CPU physics has no CUDA requirement.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m annealctrl generate \
  --config configs/smoke.json --output runs/my_smoke_data
python -m annealctrl train --data runs/my_smoke_data \
  --output runs/my_model.pt --epochs 8 --patience 4 --width 32 --seed 0
python -m annealctrl evaluate --data runs/my_smoke_data \
  --checkpoint runs/my_model.pt --output runs/my_evaluation.json --direct
```

The new-run paths above intentionally differ from archived smoke evidence.
Commands refuse to overwrite existing output files. Resume generation only with
the exact matching configuration using `--resume`. Keep source revision with
each dataset: configuration fingerprints do not prove identical code revisions.
Do not run concurrent writers in the same dataset directory.

`configs/pilot.json` is **not** a final benchmark. It requests 120 parents,
480 physical paths, 1,440 runtime tasks and 92,160 candidate outcomes at one
small logical size. Profile a subset before running it. Dense CLI teacher size
is capped at ten **physical** qubits. A scalability claim needs an audited
approximate-label contract and multi-size experiments.

## Fixed hardware construction

```bash
python -m annealctrl hardware-grow --hardware configs/toy_hardware.json \
  --lengths 2,2,2 --family spin_glass --chain-strength 1.5 \
  --output runs/my_hardware_instance.json
```

Supply your verified hardware graph as JSON with `n_qubits` and undirected
`edges` using contiguous indices. The included graph is a toy grid, not a
commercial annealer. Growth can jam: inspect `achieved_lengths` and `target_met`.
This command outputs one compiled Hamiltonian, not a trained dataset or an
embedding optimizer. Exhaustive endpoint validation is not attempted above
16 active qubits in this command.

## GPU backend

Exact identification of the tool mentioned in the recording is unresolved;
Q-GPU, CUDA-Q and cuQuantum are not synonyms. See `docs/gpu_backend_evidence.md`.

Install the appropriate CuPy binary on an actual supported GPU machine following
the official documentation, then run:

```bash
python -m pytest -q tests/test_physics.py
python -m annealctrl benchmark --backend cupy --qubits 10 --steps 256 \
  --output runs/my_gpu_parity.json
python -m annealctrl generate --config configs/smoke.json \
  --backend cupy --output runs/my_gpu_smoke_data
```

`generate --backend cupy` accelerates propagation, **not** its capped dense CPU
spectral teacher. `sparse_low_energy(..., backend='cupy')` is a separate tested-on-
CPU/prototyped-on-GPU API whose incomplete eigenpairs are not a complete response
teacher. No missing-GPU path silently falls back to CPU. CUDA-Q/QuTiP adapters,
device calibration, and actual QPU submission are not implemented.

## Scientific contracts

1. Bit i is the least-significant-bit-indexed qubit; bit0 represents spin +1.
2. Programmed HZ is composed once from problem and chain terms. Common caps scale
   HZ only, not the driver/runtime. These are toy symmetric caps, not vendor autoscale.
3. Success accepts **all physical strings whose declared decoding is a logical
   optimum**, not one planted string and not necessarily only physical ground states.
   Majority ties use +1; gauge augmentation must transform the decoder too.
4. Spectral targets are privileged training labels. Neither them nor test losses
   enter deployment features. The fixed candidate bank is independent of spectra.
5. Pilot candidates are resampled to nine uniform time knots **before simulation**;
   the critic sees the exact executed waveform. Original candidate family names
   are provenance, not proof of an exact control-family frontier after resampling.
6. The normalized pilot slope cap is ds/dtau <=4, equivalent to ds/dt <=4/T;
   it is not a fixed hardware slew limit across runtimes.
7. Step doubling is an asymptotic error diagnostic, not a rigorous certificate.
   Independent dense dynamics are audited on selected small cases. Spectral
   fixed-grid interpolation is not certified.
8. Primary splits are by parent before all variants. IDs alone do not detect
   gauge-equivalent or duplicate parents; canonical duplicate audits remain a
   required full-benchmark milestone.
9. References are best in a finite bank or best found with stated search budget,
   never asserted globally optimal controls. Direct-proposal scores use fresh
   true simulation after model selection, not interpolated bank outcomes.
10. An exact initial/problem Hamiltonian does not validate an open-system noise
    model. No quantum speedup, GPU speedup or conference-readiness is claimed.

## Reproducibility evidence

See `reports/SMOKE_REPORT.md` and `reports/smoke/` for the saved CPU run: complete
small dataset, checkpoint, training history and evaluation JSON. The report
includes negative comparisons rather than selecting only favorable results.
Full-test counts and runtime dependencies are recorded there. The archive
includes source checksums in `reports/release_source_checksums.json`.

The included GitHub Actions workflow is supplied as source only; it was not
published or run on an external repository. Select authorship, license, citation
and anonymous-submission metadata before a public release.

## Highest-value next work

Data distribution/hardness qualification; adaptive validated spectral labels;
logical-only/flat/summary matched baselines; exact control-family optimization;
parent/family/size transfer; GPU throughput profiling; calibrated hardware study.
The roadmap intentionally distinguishes these missing experiments from working
software.
