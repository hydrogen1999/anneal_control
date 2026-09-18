# The control advantage survives decoherence

Every loss elsewhere in this project comes from unitary evolution of a pure
state. The obvious objection is that the ordering of controls might be an
artefact of simulating no environment at all — and it is not an idle objection,
because a schedule that lingers near a small gap buys adiabaticity in a closed
system and buys the bath more time in an open one. The two effects pull opposite
ways.

This measures it with the independent Lindblad solver in `adapters`, on the same
held-out records, with each record's own best-found waveform against the linear
ramp, decoded by the same `output_observables` success indicator. Reproduce with
`annealctrl.open_system.robustness_sweep`.

## Result

Headroom (linear loss − best-found loss) against local dephasing rate:

| record | N | 0 | 0.02 | 0.05 | 0.1 | ordering |
|---|---:|---:|---:|---:|---:|---|
| parent_0013_e0_k2_t1 | 4 | 0.0535 | 0.0496 | 0.0443 | 0.0367 | kept |
| parent_0014_e0_k0_t1 | 4 | 0.2616 | 0.2252 | 0.1822 | 0.1326 | kept |
| parent_0014_e1_k1_t1 | 4 | 0.2000 | 0.1844 | 0.1630 | 0.1330 | kept |
| parent_0016_e0_k2_t1 | 6 | 0.0837 | 0.0686 | 0.0516 | 0.0335 | kept |
| parent_0048_e0_k0_t1 | 5 | 0.0856 | 0.0676 | 0.0484 | 0.0293 | kept |
| parent_0048_e1_k1_t1 | 5 | 0.0609 | 0.0511 | 0.0400 | 0.0283 | kept |
| parent_0063_e0_k2_t1 | 4 | 0.1714 | 0.1446 | 0.1129 | 0.0762 | kept |
| parent_0077_e0_k0_t1 | 4 | 0.1443 | 0.1243 | 0.1020 | 0.0781 | kept |
| parent_0077_e1_k1_t1 | 4 | 0.1443 | 0.1243 | 0.1020 | 0.0781 | kept |
| parent_0079_e0_k2_t1 | 5 | 0.1839 | 0.1536 | 0.1193 | 0.0818 | kept |
| **mean** | | **0.1389** | **0.1193** | **0.0966** | **0.0708** |  |
| **best-found wins** | | 10/10 | 10/10 | 10/10 | 10/10 |  |
| **retained** | | 100% | 86% | 69% | 51% |  |

**The searched control beats the linear ramp on every record at every rate
tested, and the ordering never flips.** At a dephasing rate strong enough to
halve the advantage, the advantage is still there.

## Cross-check: two solvers, same answer

At rate zero the open-system pipeline reproduces the closed-system headroom to
within 2 × 10⁻⁵ on every record, from a solver that shares no integration code
with the one that produced the campaign:

| record | closed | open at rate 0 | difference |
|---|---:|---:|---:|
| parent_0013_e0_k2_t1 | 0.0535 | 0.0535 | +0.00000 |
| parent_0014_e0_k0_t1 | 0.2617 | 0.2616 | −0.00002 |
| parent_0014_e1_k1_t1 | 0.2000 | 0.2000 | −0.00001 |
| parent_0016_e0_k2_t1 | 0.0837 | 0.0837 | +0.00000 |
| parent_0048_e0_k0_t1 | 0.0856 | 0.0856 | −0.00001 |

That agreement is an independent check on the decoder and the physics
reconstruction, not only on the noise model.

## What this is not

**Not a device model.** Local dephasing at a rate chosen by hand. No thermal
bath, no measured T1 or T2, no per-qubit calibration, no working-graph
exclusions, no readout error. It says the ordering is not fragile under *this*
perturbation. It does not say what a real annealer would do, and it is not a
substitute for running on one.

**Small, and selected for smallness.** A density matrix costs O(4ᴺ), so these
are 4–6 physical qubits out of a campaign that reaches 10 and a Pegasus ladder
that reaches 14. Two further records at N = 7 were attempted and **refused by the
solver's own cap** — `simulate_lindblad` defaults to 6 qubits and hard-caps at 8
— rather than abandoned on wall time. They are reported absent instead of
silently excluded, and the mean above is over the 10 that completed. Raising the
cap is a deliberate act, not a default, and was not taken here.

**One noise axis.** Relaxation was held at zero. Dephasing and relaxation need
not degrade a schedule the same way, and only one was varied.

**Not a claim about the learned model.** This compares the *searched* control
against linear. Whether an amortised selector's advantage degrades at the same
rate is untested.
