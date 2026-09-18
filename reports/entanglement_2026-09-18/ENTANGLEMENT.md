# How far a tensor network could reach: measured, not assumed

## The question

Exact state-vector simulation stops near 30–40 qubits; the spectral teacher near
16; SVMC reaches thousands but was measured to rank these controls at ρ ≈ 0 and
is ruled out. A matrix product state represents a pure state exactly with bond
dimension χ = exp(S), so whether MPS reaches device scale is a **measurable
property of these trajectories**.

Measured along the real propagated trajectory — the loop mirrors
`physics._split_evolve` and a test pins the final state against
`physics.propagate` to 1e-10, because an instrumented rewrite that drifts
measures something other than what it reports.

## Result: sub-volume-law, but not area-law

Peak entropy across a balanced cut, best-in-bank control, 8 records per size:

| physical qubits | peak S (nats) | volume law | fraction | bond dimension |
|---:|---:|---:|---:|---:|
| 5 | 0.482 | 1.386 | 34.8% | 2 |
| 6 | 0.868 | 2.079 | 41.8% | 3 |
| 7 | 1.135 | 2.079 | 54.6% | 4 |
| 8 | 0.789 | 2.773 | 28.5% | 3 |
| 9 | 0.720 | 2.773 | 26.0% | 3 |
| 10 | 1.151 | 3.466 | 33.2% | 4 |

Linear fit: **0.0730 nats per qubit**, against the volume-law rate of
ln2/2 = 0.3466 — a ratio of **0.21**.

**The slope is not zero, and the fraction of the volume law does not fall with
size** (26–55%, no trend). That is volume-law scaling with a reduced
coefficient, not an area law. χ therefore still grows exponentially, with a
smaller exponent: χ ≈ exp(0.073 N).

## What that buys, and what it does not

| target | extrapolated S | bond dimension | verdict |
|---:|---:|---:|---|
| 50 qubits | 3.96 | 52 | comfortable |
| 100 qubits | 7.61 | 2.0 × 10³ | heavy but standard |
| 500 qubits | 36.8 | 9.7 × 10¹⁵ | impossible |

**MPS plausibly reaches 50–100 qubits — five to ten times beyond exact
simulation, and still five to fifty times short of a deployed device.**

## The extrapolation is weak, and this project has already criticised weaker ones

Six size points spanning a factor of two, extrapolated to 500 qubits, is a 50×
reach. This report earlier rejected reading a trend from three runtime points
over one decade, and the same standard applies here: **the numbers above are
what a linear fit says, not what the system is known to do.** The per-size
fractions are non-monotonic, 8 records per size is small, and nothing rules out
a change of regime above 10 qubits.

What the measurement *does* establish, without extrapolation, is narrow and
solid: in the range actually simulated, these trajectories need χ between 2 and
4. Whatever else is true, the entanglement here is far below maximal.

## Consequence for the scale problem

| path | reach | status |
|---|---:|---|
| exact state vector | 30–40 qubits | feasible, far below device scale |
| spectral teacher (oracle) | ~16 qubits | hard cap |
| SVMC | 5000 qubits | **ruled out**: ranks controls at ρ ≈ 0 |
| **MPS** | **~50–100 qubits, extrapolated** | **plausible, unproven** |
| QPU | device scale | the only path to 500+ |

The honest summary is that simulation can probably be pushed roughly an order of
magnitude past where this project currently sits, and cannot be pushed to where
the devices are.
