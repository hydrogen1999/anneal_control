# Reference physics contract and algorithms

## Composition is explicit

`HamiltonianTerms` contains the actual programmed physical terms:

\[
H_X=-\sum_iX_i,\quad H_Z=\sum_i h_iZ_i+\sum_{ij}J_{ij}Z_iZ_j,
\quad H_{XX}=\sum_{ij}K_{ij}X_iX_j.
\]

`AnnealPath` specifies
\[
H(s)=\gamma[(1-s)H_X+sH_Z+4\lambda s(1-s)H_{XX}].
\]

The catalyst amplitude `lambda` is explicit and defaults to zero. Its envelope
vanishes at both endpoints, preserving the initial transverse ground state and
the final classical cost. This is a simulation model: no native programmable XX
couplers on D-Wave hardware are assumed. Device-specific calibrated A/B curves
require a separate tested path implementation, not reinterpreting these units.

Physical graph edges are unique unordered pairs. If an edge carries problem
and chain contributions, sum them once before constructing the terms and retain
the decomposition only as metadata. Bit i is LSB-indexed and bit 0 means spin +1.

The schedule is a callable `s(tau)` for normalized execution time. `runtime`
means the dimensionless quantity E_ref*T/hbar, never an undocumented microsecond
duration. Whole-path multiplication by gamma is equivalent to multiplying
runtime by gamma. Scaling only h and J is not equivalent.

## Algorithm 1: matrix-free action

For computational basis index b, precompute only the diagonal
\[
E_Z(b)=\sum_i h_i(1-2b_i)+\sum_{ij}J_{ij}(1-2b_i)(1-2b_j).
\]

Then evaluate `H psi` by diagonal multiplication and index-XOR gathers:

1. Start with `out = b(s) * E_Z * psi`.
2. For every qubit i, subtract `a(s) * psi[index XOR (1<<i)]`.
3. For every XX pair (i,j), add
   `c(s) * K_ij * psi[index XOR ((1<<i)|(1<<j))]`.

The action costs O((N+|E_XX|)2^N) after diagonal construction, with O(2^N)
storage. The implementation does not cache N full bit-flip index arrays.

## Algorithm 2: closed-system propagation

At each normalized-time midpoint, freeze the path coefficients. Apply a
half Z step, a full X/XX step, and another half Z step:
\[
\psi\leftarrow e^{-i\Delta t bH_Z/2}
 e^{-i\Delta t(aH_X+cH_{XX})}
 e^{-i\Delta t bH_Z/2}\psi.
\]

All X_i and X_i X_j operators commute, so the middle exponential factors
exactly into rotations `cos(theta)*psi - i*sin(theta)*flip(psi)`.
Only splitting the diagonal and transverse parts, and midpoint time sampling,
introduce discretization error. Smooth schedules have second-order global
convergence. Abrupt switches require grid alignment or convergence checks.

`step_doubling=True` returns the 2M-step state and reports
`||psi_2M-psi_M||/3` as an asymptotic fine-grid error estimate. This is not a
rigorous bound. Norm preservation alone does not establish trajectory accuracy.
The implementation never renormalizes to conceal a numerical error.

`dense_reference_propagate` independently integrates the Schrödinger equation
with dense matrices and SciPy DOP853 on at most eight physical qubits by default.
It is a QA oracle, not the production propagation engine.

## Algorithm 3: full-spectrum response teacher

The exact pilot diagonalizes capped small physical systems (default N<=10).
It computes V=dH/ds analytically, all eigenpair residuals, orthogonality, and
cross-band transition strengths. For a ground band P of rank r,
\[
w_{ba}=|\langle b|V|a\rangle|^2/r,\quad
\delta_{ba}=E_b-E_a,\qquad a\in P,b\notin P.
\]

The returned moments are
\[
\mu_0=\sum w_{ba},\quad g_{ss}=\sum w_{ba}/\delta_{ba}^2,
\quad D_2=\sqrt{\sum w_{ba}/\delta_{ba}^4}.
\]

The weight is a uniform-band convention, not a model of evolving populations.
Squared block sums are invariant to basis changes within exactly degenerate
blocks. Endpoints with several accepted ground states use the full band; rank
changes along the path must still be audited, not interpolated as if smooth.

The full-spectrum sum is checked against
\[
\mu_0=(\|VP\|_F^2-\|PVP\|_F^2)/r.
\]

The dataset frequency edges must be fixed in the same energy units. Default
eight log bins use edges from 1e-4 to 16, plus separately recorded low-frequency,
high-frequency, and numerically unresolved masses. `normalized_bin_masses`
divides in-range mass by total response mass, so it need not sum to one if mass
falls outside the bins. Raw mass and all remainder categories are preserved.

Near-degenerate automatic grouping is explicitly flagged. If a denominator is
below the reported numerical resolution, exact inverse moments are NaN and
`moments_resolved=False`. A separate regularized D2 value records its epsilon;
it must not be substituted silently for a resolved physical label.

`spectral_profile` currently evaluates a fixed grid (33 points by default).
Adaptive refinement and independent random-point interpolation auditing remain
required extensions before claiming that narrow bright peaks are resolved.

## Algorithm 4: bounded sparse low-energy prototype

`sparse_low_energy` wraps the same matrix-free action in a SciPy or CuPy
LinearOperator, requests smallest-algebraic eigenvalues (`which='SA'`), and
records residuals, orthogonality, tolerance, and backend. It bounds estimated
Lanczos work and basis memory before allocation. The API always returns
`truncated=True`, `full_response_resolved=False`, and
`ground_band_certified=False`.

This prototype is not a replacement spectral teacher: low-energy truncation
can omit the only bright mode in the two-qubit dark-gap case. A future teacher
must adapt k, demonstrate moment convergence, preserve ground blocks, and bound
omitted response with a justified omitted-gap lower bound.

## GPU boundary and verified scope

CuPy is lazy-loaded only when requested. State propagation and Lanczos work can
remain on the selected GPU, with result arrays copied to the host at return.
The current implementation is single-instance, single-GPU; it does not claim
batch fusion, distributed state vectors, open-system dynamics, or measured GPU
speedup. CuPy parity tests skip explicitly when no usable GPU is available.

Exact state vectors remain exponential in the number of physical qubits.
Matrix-free storage does not make full quantum simulation scalable in N.
Dense spectral teachers additionally need O(4^N) memory and O(8^N) work.

Tests cover independent Pauli construction; bit order; path derivatives; X and
XX propagation; DOP853 agreement; second-order convergence and step doubling;
whole-Hamiltonian/runtime covariance; gauge symmetry; dark first and second
excited states with a bright higher state; degenerate ground bands; sum rules;
histogram mass conservation; near-degeneracy censoring; sparse/dense agreement;
and allocation/initialization checks.
