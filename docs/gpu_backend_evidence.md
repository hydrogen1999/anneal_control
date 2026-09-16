# GPU backend evidence and engineering decision

Verification date: 2026-09-16. This note separates an uncertain name in the meeting recording from independently verified software capabilities. It reports no GPU performance measurements.

## What did the professor name?

**The exact simulator name is not established by the available recording/transcription.** The supplied `main(2).tex` specifies Hamiltonian construction, small-system propagation, and spectral teachers, but does not name Q-GPU, CUDA-Q, QuTiP, CuPy, or cuQuantum. It cannot disambiguate the audio.

Two local ASR passes were inspected. The earlier enhanced transcription produced an uncertain phrase resembling “Open GPU” at approximately 351–382 seconds. A targeted pass on the original audio at 330–570 seconds, with VAD disabled and no software names in the prompt, produced an uncertain phrase resembling “Open và GPU” at 351–355 seconds. Its surrounding discussion concerns learning from a closed/formal model, then adapting toward hardware. **“Open [system] và QPU” is consequently another plausible interpretation**, not a verified quotation. A targeted late-meeting pass also checked the discussion near 2090–2170 seconds. No trustworthy software-name identification resulted.

No audio was sent to a new external transcription service. Local ASR is noisy enough that phonetic strings must not be used as evidence that one specific product was recommended. Ask the professor for a link or spelling if preserving his exact software choice is important; backend-independent implementation can proceed in the meantime.

### Q-GPU is a real, distinct name

Q-GPU is the framework associated with **“Q-GPU: A Recipe of Optimizations for Quantum Circuit Simulation Using GPUs”**, by Yilun Zhao and coauthors, HPCA 2022. This is verified on the author's [publication page](https://zhaoyilun.org/publication/zhao-2022-qgpu/); its [DOI is 10.1109/HPCA53966.2022.00059](https://doi.org/10.1109/HPCA53966.2022.00059). The publisher's full page required browser verification, so this note does not claim to have read its full text. The named work concerns circuit simulation. Its existence alone does not establish that it was the professor's intended package, nor that it offers the analog annealing and spectral-label APIs this project needs.

| Statement | Confidence and basis |
|---|---|
| Q-GPU exists as an HPCA 2022 framework | High; author's publication page and DOI |
| Q-GPU is the software named in the meeting | Unresolved; ASR is inconclusive |
| Q-GPU, CUDA-Q, and cuQuantum are interchangeable names | False; distinct projects/layers |
| CUDA-Q supports GPU quantum dynamics | High; official dynamics documentation |
| A GPU simulator automatically supplies trustworthy training labels | False; construction and numerical validation are separate obligations |

## Recommended backend separation

The decision below is an engineering recommendation for this project, not a claim about what the professor said. The input is a sparse Pauli-term specification of the **physical** Hamiltonian. The backend produces labels; it does not choose the training distribution.

| Workload | First implementation | Optional acceleration/integration | Required validation |
|---|---|---|---|
| Generate logical graphs, embeddings, coefficients, and term provenance | NumPy/Python; deterministic parent seeds | Parallel CPU workers | Unbroken-chain energy equivalence; endpoint checks; split integrity |
| Full small-system reference spectrum | NumPy/SciPy dense Hermitian solve | Batched GPU dense solve only if measured worthwhile | Residuals, orthogonality, symmetry/degeneracy handling |
| Low-energy spectral teacher | Matrix-free SciPy `eigsh` | CuPy `eigsh`; optionally cuDensityMat eigensolvers | Residuals, increased-k convergence, response sum-rule/truncation checks |
| Closed-system final outcomes | Matrix-free CPU propagation | Same term application on CuPy; CUDA-Q Dynamics as an independent adapter | Time-step refinement and comparison against an independent reference |
| Small open-system reference | QuTiP `mesolve` with an explicitly stated bath model | `qutip-cuquantum` or CUDA-Q Dynamics | Trace, Hermiticity, positivity within tolerance, convergence, CPU/GPU agreement |
| Digitized circuit cross-check | A declared Trotter/product-formula construction | qsim/cuStateVec or circuit mode of CUDA-Q | Gate convention checks and Trotter convergence; not an analog label oracle |

CuPy's [`eigsh`](https://docs.cupy.dev/en/stable/reference/generated/cupyx.scipy.sparse.linalg.eigsh.html) accepts a GPU `LinearOperator`, supports smallest-algebraic eigenpairs with `which='SA'`, and uses thick-restart Lanczos. This provides a direct route from a matrix-free Ising operator to low-energy teachers. Do not assume every SciPy option exists in CuPy; the documented signatures differ.

[CUDA-Q Dynamics](https://nvidia.github.io/cuda-quantum/latest/using/dynamics.html) exposes `evolve`, time-dependent coefficients through `ScalarOperator`, quantum operators, and collapse operators. Its `dynamics` target uses NVIDIA cuQuantum. This is a suitable optional analog-dynamics adapter. Intermediate-state storage can dominate memory: retain requested observables or the final state, not every state by default.

[cuDensityMat](https://docs.nvidia.com/cuda/cuquantum/latest/cudensitymat/index.html) is the analog-dynamics component of cuQuantum and supports structured operator action, pure/mixed states, and time propagation. The inspected documentation also includes extreme eigenspectrum and selected-eigenpair functionality; its stated eigensolver restrictions include non-batched Hermitian operators and pure non-batched states. Therefore, saying “cuQuantum has no eigensolver” would be incorrect. These features should be capability-checked against the installed release before writing an adapter.

[QuTiP's master-equation documentation](https://qutip.readthedocs.io/en/stable/guide/dynamics/dynamics-master.html) provides the CPU reference route. The official [`qutip-cuquantum` repository](https://github.com/qutip/qutip-cuquantum) documents GPU support for common `mesolve`/`sesolve` workflows, but explicitly excludes advanced solvers such as `brmesolve` and HEOM from compatibility. The separate [`qutip-cupy` project](https://github.com/qutip/qutip-cupy) is a CuPy data-layer backend; it must not be confused with the cuDensityMat integration.

[cuStateVec](https://docs.nvidia.com/cuda/cuquantum/latest/custatevec/index.html) targets state-vector quantum simulation; [qsim's official GPU guide](https://quantumai.google/qsim/choose_hw) distinguishes its native GPU backend from cuQuantum-backed operation. For this project's analog Hamiltonian labels, a circuit simulator requires a separately verified discretization. A fast circuit benchmark is not evidence of fast spectral-label generation.

## Hamiltonian terms: avoid constructing the full matrix

Use one authoritative term store and expose `matvec`, `derivative_matvec`, and small-system `dense_reference` interfaces. Keep the driver, programmed problem, and optional catalyst separate:

\[
H(s)=a(s)H_X+b(s)H_Z+c(s)H_{XX},\qquad
\partial_sH=a'H_X+b'H_Z+c'H_{XX}.
\]

For computational index `b` with qubit `i` stored as bit `i`, define `z_i(b)=1-2*((b>>i)&1)`. Then

\[
E_Z(b)=\sum_i h_i z_i(b)+\sum_{(i,j)}J_{ij}z_i(b)z_j(b),
\]

\[
(H\psi)_b=b(s)E_Z(b)\psi_b
-a(s)\sum_i\psi_{b\oplus 2^i}
+c(s)\sum_{(i,j)}K_{ij}\psi_{b\oplus2^i\oplus2^j}.
\]

Here the scalar coefficient `b(s)` and integer computational index `b` are mathematically different; use distinct names in source code. Each backend must obey the same little-endian bit convention. Validate against explicit Kronecker products before batching. A matrix-free application needs vector-scale storage, although precomputing every permutation index or storing a large Krylov basis introduces additional factors.

This representation is compositional but does **not** make eigenvalues compositional: noncommuting parts cannot be diagonalized independently and have their spectra added. Likewise, block-diagonal toy Hamiltonians are useful controls, not a substitute for interacting local Hamiltonians.

## Memory arithmetic, not advertised qubit ceilings

For complex128, one amplitude occupies 16 bytes. A pure state uses `16 * 2**N` bytes; a dense Hamiltonian or dense density matrix uses `16 * 4**N` bytes. The entries below are exact storage arithmetic, not benchmarked feasible problem sizes. GiB denotes `2**30` bytes.

| Physical qubits N | One complex128 state | One complex128 dense Hamiltonian/density matrix |
|---:|---:|---:|
| 12 | 64 KiB | 256 MiB |
| 16 | 1 MiB | 64 GiB |
| 20 | 16 MiB | 16 TiB |
| 24 | 256 MiB | 4 PiB |
| 28 | 4 GiB | 1 EiB |
| 30 | 16 GiB | 16 EiB |
| 32 | 64 GiB | 256 EiB |

Complex64 halves these sizes, but its label precision must be audited. With `k` retained eigenvectors, vector storage alone is at least `k * 16 * 2**N`; the Lanczos basis typically has more than `k` vectors. An eight-vector collection at N=28 already occupies 32 GiB before Krylov workspace, Hamiltonian diagonals, allocator overhead, or batches. A full dense Liouvillian has `16**N` entries and should not be materialized for scale-up. Quantum trajectories trade full-density storage for repeated stochastic state-vector solves and require sampling-error accounting.

Thus “30 qubits fit as one state vector” does not imply “30-qubit full spectra, density matrices, and thousands of schedules are affordable.” Start with the document's 6–12 physical-qubit pilot and choose the next size by measured cost per accepted label.

## Acceptance gate for any GPU adapter

1. Run identical seeded term lists on independent CPU and GPU references at N=2–8, including signed fields, chain penalties, and nonzero XX terms.
2. Compare eigenpair residuals, subspace projectors at degeneracies, spectral response, final success probability, and energy. Do not compare arbitrary eigenvector phases.
3. Halve integration steps or tighten solver tolerance; record whether the change is smaller than the declared label-error budget. Small norm error alone does not establish accurate dynamics.
4. Benchmark **accepted labels/second**, including validation failures, transfer, compilation, and synchronization. Report spectral teachers and trajectory outcomes separately.
5. Record backend/package versions, device model, driver/runtime versions, precision, solver tolerances, wall time, and peak memory in every shard manifest. Pin a tested environment only after the target GPU is known.
6. Keep CPU-only tests runnable and GPU tests conditional. Never label an unexecuted CUDA adapter “verified”; never report GPU speedups from a CPU-only run.

The noise model is part of the scientific hypothesis. A generic Lindblad simulation is not a calibrated D-Wave simulator, and neither a GPU nor a more sophisticated integrator removes that modeling gap.
