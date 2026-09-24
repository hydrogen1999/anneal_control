# Title and abstract

## Title

**Control-Relevant Representations for Embedded Ising Annealing**

Alternates, in descending preference:

- *What the Annealer Actually Runs: Control-Relevant Representations for Embedded Ising Problems*
- *The Spectral Schedule Is Not the Target: Learning to Rank Annealing Controls from the Embedded Problem*

The first is the safe choice. The second states the paper's most contrarian
finding in the title, which is attractive but commits the whole paper to
defending a negative; the evidence supports it (§7) but it is not the largest
result.

## Abstract (submission, ~250 words)

A quantum annealer does not execute the logical Ising problem it was given. Minor
embedding replaces each logical spin with a chain, adds penalty couplings, and
rescales the Hamiltonian into the device's range, so the schedule that is best for
the logical problem need not be best for the one that runs. Standard practice picks
that schedule from a spectral rule — blind to the embedding, and requiring at
deployment a spectrum no real budget affords.

We ask whether a learned representation of the *embedded* problem suffices to choose
a control with no spectral oracle. Across five encoders differing only in what they
may see, held-out instances on two topologies, and pairwise contrasts corrected over
the full family of ten, an encoder blind to the embedding separates from **every**
embedding-aware encoder (Holm *p* ≤ 0.0014 on real device connectivity), while **no
two aware encoders separate from each other**: the information is decision-relevant,
the architecture consuming it is not. On Pegasus the resulting selector beats a
matched linear ramp on 48 of 48 held-out instances in every seed (*d* = 1.68) and
cuts reads to 99 % confidence by 37 %.

We report the effect in a unit independent of any chosen search budget: one forward
pass is worth **≈ 24 instance-specific simulator calls**. Selection scales
monotonically with the pool offered, and a critic able to rank a library is able to
overrule its own generator.

All results are exact closed-system propagation at up to 14 physical qubits. No
hardware was used.

## Abstract (extended, 306 words)

Use where the venue allows more room; adds the fallback-rate detail and the
"at least 15 calls" conservative bound.


A quantum annealer does not execute the logical Ising problem it was given. Minor
embedding replaces each logical spin with a chain, adds penalty couplings, and
rescales the Hamiltonian into the device's range — and the annealing schedule that
is best for the logical problem need not be best for the embedded one. Standard
practice chooses that schedule from a spectral rule, which is blind to the
embedding and, worse, needs a spectrum at deployment that a real budget does not
have.

We ask whether a learned representation of the *embedded* problem carries enough
information to choose a control without any spectral oracle. Across five encoders
that differ only in what they are permitted to see, held-out logical instances on
two topologies, and pairwise contrasts corrected over the full family of ten, an
encoder blind to the embedding separates from **every** embedding-aware encoder
(Holm *p* ≤ 0.0014 on real device connectivity), while **no two aware encoders
separate from each other**. The information is decision-relevant; the architecture
consuming it is not. On the Pegasus topology the resulting selector beats a matched
linear ramp on 48 of 48 held-out instances in every training seed (*d* = 1.68) and
cuts the reads needed for 99 % confidence by 37 %.

We also report the effect in a unit that does not depend on a chosen search budget:
one forward pass is worth **≈ 24 instance-specific simulator calls**, and at least
15 under the least favourable of three search strategies. Selection scales with the
pool it is offered, monotonically across four measured points; a critic able to rank
a library is also able to overrule its own generator, with a fallback rate that
self-adjusts from 16 % to 55 % as that generator degrades.

All results are exact closed-system propagation at up to 14 physical qubits. No
hardware was used.
