# Numerical teacher and batched simulator

## What is implemented

The training-data path has three distinct computational operations:

1. Assemble programmed Hamiltonian terms from the logical instance and its embedding.
2. Query complete response labels at selected path coordinates, for systems small enough for dense diagonalization.
3. Propagate a state under each candidate control and evaluate its actual final outcome.

The spectral teacher is privileged supervision. Its cost is charged to label generation; spectra are not inference features. Choosing fewer path queries does not make a full-spectrum eigenproblem scalable. The default complete teacher remains capped at **10 physical qubits**, regardless of how few logical variables produced them.

## Adaptive query API

```python
from annealctrl.spectral import adaptive_spectral_profile

profile = adaptive_spectral_profile(
    terms,
    initial_grid=[0.0, 0.25, 0.5, 0.75, 1.0],
    max_queries=65,             # Includes the 8 held-out audit queries.
    audit_points=8,
    seed=17,
    relative_tolerance=0.05,
    absolute_tolerance=1e-4,
    min_interval_width=1e-4,
    residual_tolerance=1e-9,
    orthogonality_tolerance=1e-9,
    max_qubits=10,
)
labels = profile.points          # Sorted adaptive queries; not interpolated labels.
audit = profile.audit_points     # Independent exact random queries; held out.
diagnostics = profile.diagnostics
strict_json_record = profile.to_dict()
```

All arguments accepted by `spectral_teacher`, including the physical path,
frequency-bin edges, and ground-band tolerances, can be passed through keyword
arguments. The adaptive return object has three fields:

| Field | Meaning |
|---|---|
| `points` | Complete `SpectralPoint` objects for every adaptive query, sorted by `s` |
| `audit_points` | Random exact queries that never influence refinement or enter `points` |
| `diagnostics` | Strict-JSON-safe budget, convergence, rank, numerical and interpolation diagnostics |

`to_dict()` converts arrays to lists and nonfinite numbers to JSON `null`.
Unresolved inverse moments therefore remain missing values, not zero-valued
targets. Preserve `moments_resolved` alongside the converted fields.

### Refinement rule

For each interval, query its midpoint and compare the exact vector

\[
v(s)=\big(\Delta_{\rm raw},\mu_0,g_{ss},D_2,
  m_{\rm low},m_{\rm high},m_{\rm unresolved},m_1,\ldots,m_B\big)
\]

with endpoint linear interpolation. Its largest finite normalized discrepancy is

\[
e=\max_j\frac{|v_j(s)-\widehat v_j(s)|}
 {\epsilon_{\rm abs}+\epsilon_{\rm rel}
   \max(|v_j(s)|,|\widehat v_j(s)|)}.
\]

Refine if `e > 1`, the ground-band rank changes across the three points, or any
point has unresolved moments or a near-degenerate band. Unchecked intervals
are visited widest first. Among checked failures, unresolved/rank-changing
intervals take priority, then the largest interpolation discrepancy. Refinement
stops at the budget or the minimum interval width even when the criterion is
not met; the corresponding failure remains explicit in the result.

Residual acceptance uses
`max_eigenpair_residual <= residual_tolerance * max(1, max(abs(energies)))`.
Orthogonality uses the full Frobenius error against the identity and the stated
absolute tolerance. A failed gate sets `moments_resolved=False` and masks
`g_ss`/`d2` as NaN. It does not silently regularize a ground gap or certify an
approximate eigenbasis. The regularized `D2` field remains a separately named
diagnostic, not a replacement target.

### Independent audit and honest stopping

Only after adaptation ends, generate independent uniform random coordinates
using `seed`. At each coordinate, compare a new exact query against interpolation
of the final adaptive labels. Audit points are never added back to the adaptive
labels or used to decide new queries. A failed audit is reported rather than
repaired using the held-out sample.

Important diagnostics are:

| Diagnostic | Interpretation |
|---|---|
| `query_count` | All full diagonalizations, including held-out audit queries |
| `budget_exhausted` | More requested refinement could not fit the budget |
| `sampled_refinement_converged` | All assessed leaf intervals passed the sampled criterion; no unchecked leaf remains |
| `audit_passed` | All random audit checks passed; `null` when no audit was requested |
| `sampled_checks_passed` | Both preceding checks passed; not a certificate |
| `stop_reason` | `sampled_tolerance`, `query_budget`, or `minimum_interval_width` |
| `intervals` | Every leaf's bounds, query, discrepancy and unresolved reasons |
| `numerical_failures` | Locations that failed residual or orthogonality gates |
| `uniform_certificate` | Always `false` |

Even a passing audit can miss a narrow response peak. Midpoint refinement also
has aliasing failure modes. The test suite deliberately constructs one such
curve: midpoint checks pass, the independent audit fails, and no uniform
accuracy claim is produced. Increasing the query budget is a practical
diagnostic, not a mathematical replacement for a uniform regularity bound.

A rank change invalidates a smooth fixed-rank geometric interpretation across
that interval. Individually queried endpoint labels may still be well-defined;
the interpolation diagnostics must not be discarded when deriving a schedule
from that profile.

## Batched matrix-free propagation

```python
from annealctrl.physics import propagate_batch

result = propagate_batch(
    terms,
    schedules=[linear_schedule, proposed_schedule],
    runtimes=[2.0, 2.0],
    steps=256,
    backend="numpy",           # Or explicitly "cupy" on a CUDA device.
    step_doubling=True,
    state_tolerance=5e-4,
    norm_tolerance=1e-9,
    max_state_bytes=512 * 2**20,
)
if not result.accepted.all():
    failed_rows = (~result.accepted).nonzero()[0]
    # Re-run only those rows at a finer step count, within a declared budget.
    # Do not put failed rows into the accepted-label training set.
```

Batch rows share `HamiltonianTerms` and `AnnealPath`, but may use distinct
waveforms, runtimes and initial states. The result stores host arrays
`states[B, 2**N]`, `norm_errors[B]`, `step_doubling_errors[B]`, and `accepted[B]`.
The returned `steps` is the finer `2 * steps` when step doubling is enabled.

The implementation combines diagonal half rotations and the commuting X/XX
rotations across every batch row. Schedule callables are sampled on the host;
the coefficient table transfers once, and no scalar CPU/GPU synchronization
occurs inside the integration loop. Final states transfer once. No dense
Hamiltonian or bank of all bit-flipped indices is materialized.

The row acceptance gate is finite norm error below `norm_tolerance` and finite
`||psi_(2M)-psi_M|| / 3` below `state_tolerance`. This second-order asymptotic
diagnostic is not a rigorous error bound. Waveform discontinuities still need
grid alignment or additional convergence checks. Without step doubling,
`accepted` is `None`; those results have not passed the state-error gate.
Set `require_accepted=True` to raise an error if any row fails.

This is a reusable simulator API. Whether a particular generation configuration
uses batching is determined by its caller; the presence of this function alone
does not prove that every CLI command batches its workloads.

## Memory and backend checks

`estimate_state_workspace(n_qubits, batch_size=..., step_doubling=...)` returns
the complex128 output-state size and a conservative explicit-array workspace
estimate. `propagate_batch` adds its coefficient-table estimate and checks the
configured byte budget before allocation. Backend memory pools, Python objects
and external eigensolver workspaces are not a measured peak; profile the actual
worker on the target device before setting a production batch size.

`backend_device_info("numpy" | "cupy")` reports backend availability and, for
CUDA, device name and current free/total memory. CUDA allocations also require
the estimate to fit within 80% of currently free device memory. Missing CuPy,
an unavailable CUDA runtime, and insufficient device memory are explicit
failures. There is **no implicit CPU fallback**.

CPU worker count and GPU batch size are different knobs. Do not start multiple
workers against one GPU and assume that each owns all reported free memory.
State vectors remain exponential in physical qubit count, and batch size
multiplies that storage.

## Sparse eigenpairs are not complete spectral labels

`sparse_low_energy` remains a matrix-free Lanczos diagnostic on NumPy or CuPy.
Its return value explicitly has `truncated=True`,
`full_response_resolved=False`, and `ground_band_certified=False`. Small
eigenpair residuals do not establish that omitted bright transitions have
negligible inverse-response contribution. Neither this API nor the adaptive
teacher promotes a truncated spectrum into complete `D2` supervision.

For larger data generation, outcome-only supervision can use matrix-free
dynamics while complete spectral supervision is omitted. That is a different
training regime and must be identified in the dataset/model comparison.

## Verification coverage

`tests/test_adaptive_teacher.py` checks the analytic one-qubit response,
curvature-driven refinement, total budget including the audit, rank changes,
minimum-width stopping, audit independence, missed-peak detection, numerical
masking and strict JSON serialization. Batched propagation is compared with
both single-row propagation and an independent dense DOP853 solver, including
XX catalysts, varied runtimes, custom initial states, explicit failed rows,
input validation and memory guards. Optional CuPy parity tests skip when no
CUDA device is available; a skip is not GPU validation.
