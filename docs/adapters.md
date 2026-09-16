# Calibration, open-system checks, and offline hardware handoff

These pathways are implemented, but their inputs do not invent a calibration,
device permission, or QPU result. The regular data-generation/training pipeline
remains a closed-system simulator unless the caller explicitly uses the
calibrated/open-system APIs. This separation prevents phenomenological noisy
labels from silently replacing the established training target.

## 1. Calibrated Hamiltonian paths

`annealctrl.adapters.CalibratedPath.from_mapping(data)` and
`CalibratedPath.from_json(filename)` accept a supplied table:

```json
{
  "calibration_id": "analytic-example-not-a-device",
  "source": "User-supplied analytical unit-test table",
  "s": [0.0, 0.5, 1.0],
  "a": [1.0, 0.5, 0.0],
  "b": [0.0, 0.5, 1.0],
  "c": [0.0, 0.0, 0.0],
  "coefficient_unit": "dimensionless",
  "time_unit": "dimensionless",
  "component_factors": {"a": 1.0, "b": 1.0, "c": 1.0}
}
```

The table interpolates linearly without extrapolation and must cover `[0,1]`
strictly. Its convention is

\[
H(s)=a(s)H_X+b(s)H_Z+c(s)H_{XX},\quad H_X=-\sum_iX_i,
\quad H_Z=\sum_i h_iZ_i+\sum_{ij}J_{ij}Z_iZ_j.
\]

`component_factors` is mandatory. For a source equation with `A/2` and `B/2`,
explicitly put `a: 0.5` and `b: 0.5`; the code does **not** guess a vendor's
convention. Values in GHz are interpreted as `H/h`, not `H/hbar`:

| Input coefficient unit | Required runtime unit | Angular conversion |
|---|---|---|
| `dimensionless` | `dimensionless` | 1 |
| `GHz` | `us` | \(2\pi\times1000\) |
| `rad/us` | `us` | 1 |

The output of `.coefficients(s)` is ready for `dpsi/dt=-i H psi` in the stated
time unit. `.derivative(s)` uses the right derivative at interior table knots
and the left derivative at the endpoint. These are piecewise derivatives,
not a smooth interpolation or a spectral-label certification.

The path protocol is compatible with existing `propagate(..., path=path)` and
`dense_reference_propagate(..., path=path)`. Those state-vector APIs still
require the caller to supply the runtime in the path's declared unit. For a
non-pure-X starting Hamiltonian, explicitly provide an initial state. The
density solver below checks this initialization condition itself.

## 2. Independent density-matrix Lindblad solver

```python
from annealctrl.adapters import simulate_lindblad
from annealctrl.physics import HamiltonianTerms
from annealctrl.schedules import Schedule

terms = HamiltonianTerms(2, [0.3, -0.1], [[0, 1]], [-0.7])
result = simulate_lindblad(
    terms, Schedule.linear(), runtime=2.0,
    dephasing_rates=[0.01, 0.02],
    relaxation_rates=0.0,
    rates_unit="1/dimensionless",
)
probabilities = result.probabilities
diagnostics = result.diagnostics
```

The solver constructs Hamiltonian components with independent Kronecker
products and integrates

\[
\dot\rho=-i[H,\rho]+\sum_k\left(L_k\rho L_k^\dagger
-\tfrac12\{L_k^\dagger L_k,\rho\}\right)
\]

using installed SciPy's DOP853. It does not reuse the matrix-free Strang
propagator. Qubit `i` is bit `i`, with computational bit zero equal to spin
`+1`, consistent with the rest of the project.

Available local channels:

- Dephasing: `L_i=sqrt(gamma_phi_i) Z_i`. An isolated off-diagonal element
  decays at rate `2*gamma_phi_i` under this definition.
- Relaxation: `L_i=sqrt(gamma_1_i) |0><1|_i`. This is **computational-basis**
  relaxation, not an interacting-Hamiltonian thermal bath or instantaneous
  ground-state relaxation. There is no fitted temperature, detailed-balance
  claim, or inferred device noise in this implementation.

Rates are nonnegative scalars or vectors of length `N`. A calibrated path
whose time unit is `us` requires `rates_unit="1/us"`, including when rates
are zero. Units are not inferred from their numerical magnitude.

`initial_state` and `initial_density` are mutually exclusive. Otherwise the
solver initializes `|+><+|`, but only for a pure positive-X-driver start.
It refuses unnormalized states, non-Hermitian density matrices, invalid trace,
and non-positive density matrices. No state is silently normalized, projected,
or clipped. Thus probabilities can retain harmless negative roundoff within
the explicitly reported physicality tolerance.

The returned `DensityResult` contains the unmodified final density matrix,
its diagonal probabilities, integration tolerances, rate conventions, calibration
provenance, maximum trace/Hermiticity errors, and minimum checked eigenvalue.
Checks run at `check_points` times plus waveform knots, **not continuously**;
passing them is a numerical diagnostic rather than a rigorous solver error
bound. Memory is `O(4**N)`, with a default cap of 6 qubits and an absolute cap
of 8. Tighten tolerances and repeat labels when comparing small effect sizes.

Tests include zero-noise parity with the separate dense state-vector solver,
analytic dephasing decay, analytic relaxation and LSB ordering, calibration
factors/units, invalid initial states, and memory guards. QuTiP and CUDA-Q
are not required or claimed as implemented adapters.

## 3. Explicit device constraints and program export

The offline hardware pathway works independently of the simulation backend.
Supply a real active hardware graph and real constraints before using it for
hardware preparation; the following is deliberately a **fake test device**:

```json
{
  "device_id": "fake-device-not-for-submission",
  "hardware_nodes": [10, 20, 30],
  "hardware_edges": [[10, 20], [20, 30]],
  "h_range": [-2.0, 2.0],
  "J_range": [-2.0, 2.0],
  "runtime_range": [1.0, 100.0],
  "time_unit": "us",
  "max_schedule_points": 5,
  "max_slope": 1.0,
  "coefficient_unit": "dimensionless_ising"
}
```

Optional `time_resolution` and `s_resolution` enforce exact waveform
quantization within a small numerical tolerance. No schedule rounding occurs.
`max_slope` is `ds/dt` in inverse microseconds, not normalized `ds/dtau`.

```python
from annealctrl.hardware import DeviceConstraints, export_program

device = DeviceConstraints.from_mapping(device_json)
program = export_program(
    record, schedule, device,
    physical_ids=[10, 20, 30],  # record qubit i -> device ID
    runtime=10.0,
    time_unit="us",
    gauge="identity",         # or an explicit physical +/-1 vector
    tie_policy="reject",      # or explicitly "plus" / "minus"
    coefficient_scale=1.0,
)
```

Export validates chain membership and connectivity, dimension consistency,
active qubit IDs, every programmed edge, coefficient ranges **after gauge**,
runtime, schedule slope/point count, optional quantization, and unsupported XX
catalysts. It does not autoscale coefficients. `coefficient_scale` is explicit
and must be positive; it is additional to any scaling already present in the
record's `physical_h/physical_J`.

The caller explicitly chooses runtime in microseconds. The source simulation
runtime is retained only as provenance: there is no assumption that one
dimensionless simulation unit equals one microsecond. Similarly, the schedule
does not claim to compensate for a device's unknown A/B calibration.

The JSON-safe return value includes programmed h/J, the full declaration of
constraints, exact physical-ID map, waveform, gauge/decoder conventions,
source coefficients, a source-parameter hash, and a hash of the complete
program. The gauge is `z_programmed=gauge*z_original`. Logical ground energy
is independently enumerated for at most 20 logical variables; for larger
problems it is either unavailable or explicitly marked user-supplied/unverified.

The result is **an offline plan, not a vendor API request or a validated QPU
execution**. Device-specific solver parameters, calibration dates, fast-anneal
restrictions, per-qubit coupling budgets, job authorization, and provider
acceptance remain outside this generic contract. No network or credential
functions exist here.

## 4. Sample ingestion and decoded outcomes

Convert user-provided results into this explicit interchange schema:

```json
{
  "program_hash": "COPY_FROM_EXPORTED_PROGRAM",
  "vartype": "SPIN",
  "variable_order": [30, 10, 20],
  "energy_convention": "programmed_ising",
  "rows": [
    {"values": [-1, -1, -1], "count": 8, "energy": -3.0}
  ]
}
```

The example values and energy are only illustrative. `energy` must equal the
exported dimensionless Ising expression `sum(h*z)+sum(J*z*z)` with no offset.
Do not insert logical decoded energies or device A/B-scaled physical energies
under this convention. If a vendor reports a QUBO energy with an offset,
explicitly convert it first and preserve that conversion in experiment metadata.

`vartype="BINARY"` additionally requires `binary_zero_spin=1` or `-1`, so
the meaning of bit zero is never inferred. `variable_order` must identify each
programmed physical qubit exactly once, in the order of the row values.
Each count must be a positive integer. Repeated samples are merged by summing
their counts; they are neither silently discarded nor treated as new shots
beyond the supplied counts.

```python
from annealctrl.hardware import ingest_samples

report = ingest_samples(program, supplied_samples, confidence=0.95)
```

Ingestion verifies program/source/sample hashes, checks every reported energy,
restores the original spin gauge, and then performs majority-vote chain decoding
with the declared tie rule. It reports counts, rejected ties, average programmed
energy, average decoded energy over accepted shots, chain-break rates, success
probability when ground energy is known, and exact Clopper-Pearson intervals.
Rejected ties count as failures in success probability; decoded-energy means
exclude rejected shots. `plus`/`minus` tie choices refer to the original gauge,
not the submitted gauge.

Binomial confidence intervals assume IID shots and do not account for drift,
autocorrelation, or shared programming errors. Use programming-cycle/device
blocks for paper-level uncertainty if such structure exists. A matching hash
is a consistency check, **not an authenticity signature**: the module explicitly
labels results as user-supplied, not authenticated QPU observations.
