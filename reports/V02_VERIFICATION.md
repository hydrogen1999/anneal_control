# v0.2 verification — executable software, not paper superiority

Date: 2026-09-16. No GPU, QPU, paid job or external repository action was used.

## Verified release

- Full tests: **340 passed, 2 skipped**, 21.17 seconds in the recorded CPU runtime.
- Skips: optional CuPy/CUDA parity; no available GPU/CuPy installation.
- Package installed as `annealctrl 0.2.0` with editable local installation.
- Actual end-to-end run: `configs/experiment_smoke.json`, output archived under
  `reports/v02/experiment/`.
- Frozen runtime-source SHA256:
  `7ac72e2b8325a86a75c70a2f95be07255b3ec3649a3dd959cc72fa3df373f4b1`.
- Algorithm LaTeX compiled twice in draft mode with no remaining warnings or
  overfull boxes. Source, not a rendered PDF, is supplied.

## Data and training

Twelve independent logical parents (sizes 3 and 4), two families, physical sizes
3–7; four parents in each split. Twenty-four tasks, eight schedules/task,
**192 accepted candidate outcomes**. Candidate batch size four. Adaptive teacher
budget17 includes three independent audit queries.

Five methods × two seeds completed (ten training runs, three epochs each),
with best/latest checkpoints, provenance, history, bank and direct evaluation.
Test has eight tasks but only **four independent parents**. The result is a
software smoke test, not a reliable large-sample scientific comparison.

Measured generation stage: 4.29 seconds, including training-only audit.
Sum of the ten recorded training invocation durations: 9.15 seconds.
Sum of the ten evaluation durations: 6.21 seconds.
These are observed runtime timings, not server performance guarantees.

The validation-tuned global baseline had mean loss0.668796. Hierarchy+physics
had bank loss0.694096 and direct loss0.669774. Thus this run does **not** establish
superiority of the proposed hierarchical/physics architecture. All methods,
unfavorable results and parent intervals are retained in the report.

Generated SVG/PNG plots were visually checked. The report keeps training-seed SD
separate from parent bootstrap uncertainty; it requires every predeclared method
and seed before generating its summary table.

## Independent operation checks

- Full experiment resume succeeded without retraining completed models.
- `configs/tuning_smoke.json`: actual two-trial validation-only run, shared data
  generated once; no test scoring. Both trials tied at validation regret0.0717233;
  deterministic first-trial tie breaking selected trial0.
- CPU `profile-generation` on the smoke distribution: 192 accepted labels in
  3.359 seconds, **57.16 labels/s**, including teacher/audits/I/O. This is one
  CPU run, no GPU speedup comparison.
- Profile maximum norm error1.887e-14; maximum step-doubling diagnostic4.991e-4;
  maximum independent dense-state difference4.907e-4; maximum accepted steps256.
- All12 adaptive paths had failed sampled profile checks at this deliberately
  small query budget. Individually resolved auxiliary moments were retained,
  but **no continuous spectral-curve accuracy claim** follows. Budget/audit status
  remains explicit in every path's metadata.
- Toy hardware export CLI ran on a three-physical-qubit instance with explicitly
  supplied device constraints, gauge and microsecond runtime. No submission.
- Independent Lindblad CLI ran local dephasing0.01 in dimensionless units on
  three qubits; maximum trace error4.441e-16, Hermiticity error0, physicality checks
  passed on17 sampled states. This is not device calibration.

Automated tests additionally cover serial/parallel and scalar/batch numerical
equivalence, exact optimizer/RNG resume, label leakage barriers, content
tampering, split-specific file isolation, exact pause/switch simulation, offline
sample ingestion, source drift, locks and invalid configuration.

## Environment and interpretation

Python3.12.14; NumPy2.3.5; SciPy1.17.0; NetworkX3.6.1; PyTorch2.14.0+cpu;
pytest9.1.1. CPU runs set OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1.

No calibrated open-system training dataset, hardware transfer experiment,
large-scale tensor-network teacher, actual CUDA-Q/QuTiP backend, fused graph
batching or DDP is claimed. Scientific data/ablation/transfer gates remain the
user's research experiments; software completion does not pre-decide them.
