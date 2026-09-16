# CPU end-to-end smoke report — not a paper benchmark

Run date: 2026-09-16. Purpose: verify generation, numerical labels, tensor features,
gradients, checkpoints, independent splits and true evaluation of new proposals.
No configurations were retuned after inspecting these test results.

## Dataset and environment

- Config: `configs/smoke.json`, master seed 20260911.
- 12 logical parents, four families, 3 logical / 6 physical qubits.
- Two port-layout variants, one chain strength, runtimes 2 and 8 (dimensionless).
- 48 control tasks; 12 schedules each; **576 accepted candidate outcomes**.
- Four parents in each of train/validation/test; test has 16 tasks but only 4 parents.
- Nine interior spectral points per physical path; candidate waveforms use nine
  time knots and are simulated after resampling.
- CPU dataset generation: 32.67 seconds in this runtime. This is not a projected
  server/GPU speedup. Thread settings: OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1.
- Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, NetworkX 3.6.1, PyTorch 2.14.0+cpu,
  pytest 9.1.1, as reported by the executed environment.

These few logical parents on one tiny size are not sufficient for novelty,
statistical power, graph generalization, hardware transfer, or an A* claim.

## Numerical and software evidence

| Check | Observed result |
|---|---:|
| Complete test suite at release verification | 131 passed, 1 skipped |
| Skipped test | Optional CuPy/CUDA parity; dependency unavailable |
| Maximum candidate norm error | 2.354e-14 |
| Maximum step-doubling state-error diagnostic | 4.970e-4 |
| Accepted propagation step counts | 128–512 |
| Maximum dense eigenpair residual | 1.250e-14 |
| Maximum state difference in independent DOP853 audits | 1.472e-4 |

The step-doubling estimate is not a rigorous global bound. The independent audit
covers selected linear controls, not every candidate. Full tests include nonlinear
controls, Pauli/sign conventions, X/XX drivers, higher bright modes, degeneracy,
planting certificates, energy-preserving compilation, weak-chain failures,
parent splits, resume, permutations, gradients and checkpoint round-trip.

The fixed-hardware CLI also ran on the bundled toy 3x4 grid: six active qubits
with achieved target lengths [2,2,2]. This is a construction test, not an annealer
benchmark. A CPU propagation microbenchmark at 8 qubits is saved separately.

## Training and held-out scores

One seed (0), width 32, maximum 8 epochs, patience 4. Five epochs executed; validation
bank regret did not improve after epoch 0, which was retained. Training took 3.53s.
The checkpoint is an integration example, not a tuned model.

Lower loss is better; loss=1−decoded logical ground-state success.

| Method | Mean test loss | Mean success |
|---|---:|---:|
| Matched-duration linear | 0.552373 | 44.7627% |
| Global shared-bank candidate chosen on validation only | **0.541280** | **45.8719%** |
| Learned critic selecting from the fixed bank | 0.559213 | 44.0787% |
| Learned direct proposal selected by critic, then true simulator scored | 0.546347 | 45.3653% |

The fixed-bank learned policy is worse than linear and the tuned global
candidate. The direct policy is slightly better than linear in this tiny run,
but also worse than the tuned global candidate. Therefore the smoke provides
**no evidence of superiority over a credible baseline**.

The bank-selected model's mean regret to the finite-bank best is 0.049980. Paired
parent bootstrap gives loss difference versus linear +0.006840, interval
[-0.000759,0.016283]. These intervals are software output on four parents, not
reliable full-benchmark uncertainty; they exclude training-seed uncertainty.
Complete numbers and direct-proposal waveforms are in the evaluation JSON.

The finite-bank inference measured 0.0471s over 16 test tasks. This timing excludes
offline reference generation and is not a device throughput measurement. Direct
proposal simulator calls are offline scoring only: the model selected its action
before observing that action's true loss.

## What must not be concluded

- No GPU or QPU was used; optional backend code has not demonstrated speedup.
- No full spectral-response scalability has been established by sparse eigenpairs.
- Neither toy embeddings nor planted endpoints establish annealing hardness.
- No one-window/BAB compression frontier is measured by the resampled bank.
- No paper-level superiority, transfer, novelty priority or acceptance is shown.

Next evidence should answer whether a broader, controlled dataset has meaningful
instance-specific control headroom, then whether physical features explain it.
Use `docs/paper_protocol.md` to freeze that experiment before evaluating its test set.
