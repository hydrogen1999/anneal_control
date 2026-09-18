# What actually caps the scale, and what does not

A claim made repeatedly in this session — that the project's ≤14-qubit ceiling
cannot be lifted by more compute — is **wrong as stated**. It is true of exactly
one result and was generalised to all of them.

## Two costs, two very different curves

| physical qubits | state vector | basis observables | **full eigendecomposition** |
|---:|---:|---:|---:|
| 10 | 0.00002 GB | 0.00004 GB | 0.02 GB |
| 14 | 0.0003 GB | 0.0007 GB | 4.3 GB |
| 16 | 0.001 GB | 0.003 GB | **68.7 GB** |
| 20 | 0.017 GB | 0.042 GB | 1.8 × 10⁴ GB |
| 24 | 0.268 GB | 0.671 GB | 4.5 × 10⁶ GB |
| 26 | 1.07 GB | 2.68 GB | 7.2 × 10⁷ GB |

A state vector is 2ᴺ amplitudes. An eigendecomposition is a 2ᴺ × 2ᴺ matrix. On a
96 GB accelerator the first is comfortable at 26 qubits and the second is
impossible past 16.

## What that means per claim

**Capped, genuinely.** The privileged spectral baselines need the Hamiltonian
diagonalised at every point on the path. `pipeline` enforces
`teacher.mode != "none"` only up to 10 physical qubits, and the table above shows
why. **The oracle comparison cannot be scaled, by any amount of compute.** So can
the physics-auxiliary ablation, which needs the same spectral targets.

**Not capped.** Headroom, the embedding interventions, bank selection, direct
generation, the search strategies, warm start, and transfer all need only
propagation and basis enumeration. Nothing in their cost forces 14 qubits.

The present ceiling is a **configuration guard**, `max_physical_qubits` in
`2..20`, not a physical limit. At 20 qubits the state vector is 17 MB.

## The gap between 14 and what is reachable

14 → 20 qubits is a **64×** larger state space. 14 → 24 is **1024×**. Both are
memory-feasible; whether they are *time*-feasible is a separate question, being
measured rather than assumed (`configs/backend_profile_{16,18,20}q.json`, and
the crossover driver which repeats each size and refuses a contended GPU).

The GPU ratio is the thing to watch: measured 1.13× at 10 qubits and 2.83× at 12,
so the regime where a GPU pays is exactly the one this ladder enters.

## Why the error mattered

The reasoning was: the oracle is the headline, the oracle is capped at 10
qubits, therefore the paper is capped. The first two are true and the conclusion
does not follow — most of the evidence does not involve the oracle at all.

Stating a limit correctly is not pedantry here. "≤14 qubits because exact
simulation is exponential" invites a reviewer to dismiss the whole paper on a
constraint that binds one row of the evidence table.
