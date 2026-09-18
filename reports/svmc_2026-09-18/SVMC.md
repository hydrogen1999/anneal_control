# The semiclassical surrogate does not rank these controls. The scale path is closed.

## Why this was tried

Exact state-vector dynamics costs O(2ᴺ) and stops near 30–40 qubits on any
machine; the spectral teacher stops near 16. A deployed annealer has thousands.
Spin-vector Monte Carlo (Shin, Smith, Smolin, Vazirani 2014) replaces each qubit
with a classical O(2) rotor, costs **O(N × sweeps)**, and reaches 5000 spins in
seconds. If it ranked controls the way exact dynamics does, the project's claims
could be tested at device scale.

The protocol was fixed before running: **validate where truth is known** — the
10–14 qubit records that already have exact outcomes — and only then run where it
is not.

## It does not rank them

40 held-out records, the same 64-candidate bank, the same decoder (the exact
pipeline's own `output_observables` success indicator, not a second
implementation):

| quantity | value |
|---|---:|
| mean Spearman against exact loss | **−0.091** |
| records with positive correlation | 14 / 37 |
| picks the exact best candidate | **0 / 40** |
| regret of the surrogate's pick, in exact loss | 0.0956 |
| regret of the linear schedule, same units | **0.0572** |
| surrogate's pick beats linear under exact dynamics | 10 / 40 |

**Choosing a control by SVMC is worse than not choosing at all.** The linear
ramp, which requires no surrogate, has lower exact regret than the candidate the
surrogate recommends.

## It is not an under-resourced run

A Monte Carlo negative without a convergence check is not a result. Resolution
was raised by 6.7× and temperature swept by an order of magnitude:

| steps | sweeps | temperature | mean Spearman | picks best |
|---:|---:|---:|---:|---:|
| 60 | 2 | 0.20 | −0.071 | 0/6 |
| 200 | 6 | 0.20 | +0.039 | 0/6 |
| 400 | 10 | 0.20 | +0.022 | 0/6 |
| 200 | 6 | 0.05 | −0.009 | 0/6 |
| 200 | 6 | 0.50 | +0.111 | 0/6 |

Every configuration sits at zero correlation and picks the best candidate in
none of six records. More compute does not move it.

## What this closes, and what it opens

**Closed.** SVMC cannot serve as a scale surrogate for control selection. The
plan to test the project's claims at 500–5000 qubits through a semiclassical
model is ruled out **by measurement**, in the overlap region built for exactly
this decision, rather than by argument.

**Opened, tentatively.** The standard semiclassical model of quantum annealing —
the one used in the literature to argue that D-Wave behaviour is classically
explicable — does **not** reproduce the control-preference structure this project
measures. A reading consistent with that is that the structure is a genuinely
quantum effect rather than a classical relaxation artefact.

That reading is **not established here**. This is one surrogate, one update rule,
one parameter family, on records of at most 10 physical qubits, and a failure to
reproduce is not a proof of quantum origin. Establishing it would need the
comparison run deliberately as a physics question — several semiclassical
variants, a size ladder, and a pre-registered criterion — rather than as a
by-product of a feasibility check.

## Consequence for scale

With SVMC ruled out, the honest position is that the **quantum** claims cannot be
scaled by simulation at all:

| path | reach | status |
|---|---:|---|
| exact state vector | 30–40 qubits | feasible, far below device scale |
| spectral teacher (oracle) | ~16 qubits | hard cap, exponential |
| SVMC surrogate | 5000 qubits | **ranks controls at ρ ≈ 0; unusable here** |
| tensor networks (MPS) | 100–1000 qubits | untested |
| **QPU** | device scale | the remaining path |

MPS is the one untried simulation route and it is a different bet: it
approximates the *quantum* state rather than replacing it with a classical one,
so it fails differently. Whether the entanglement in these trajectories is small
enough is measurable on the existing records and has not been measured.
