# 1. Introduction

*Paragraph plan: setting → the gap that matters → why it is not obvious →
what we do → what we find → what we do not claim.*

---

**¶1 — The object that runs is not the object you posed.**
A quantum annealer is given a logical Ising problem and executes a different
one. Minor embedding replaces each logical spin with a chain of physical
qubits held together by penalty couplings, and the whole Hamiltonian is
rescaled into the device's programmable range. Chain strength, chain length,
and the programmed scale are not incidental bookkeeping: they change the
spectrum that the annealing path actually traverses. The control problem —
choosing how fast to move along that path — is therefore posed on the embedded
problem, whatever it was posed on in the user's head.

**¶2 — The standard control rule cannot see that.**
Schedules are conventionally derived from a spectral quantity: allocate time in
inverse proportion to the square of the instantaneous gap, or to a
transition-element-weighted refinement of it. Two things follow. First, such
rules are usually computed from the *logical* spectrum, which is not the one
being traversed. Second, and more limiting, they need a spectrum at all: an
eigendecomposition per instance, per path point. That is privileged
information. A protocol honest about cost must either forbid recomputing it at
deployment or charge for it, and either way a rule that depends on it is not a
deployable control policy.

**¶3 — Why "just learn it" is not obviously the answer.**
The natural move is to learn a policy that maps problem to schedule. Three
things make that harder than it sounds, and each shapes what we do. *(i)* Any
encoder that sees the embedding is also a different network, so showing that
the **information** helps requires separating it from architecture. *(ii)*
Reporting the benefit requires a denominator, and the obvious ones — a share of
what some search found — move with the budget that search was given; the same
result reads as 60 % or 83 % depending on a choice made elsewhere. *(iii)* A
network that proposes a schedule directly is unreliable in a specific and
awkward way: on real device connectivity we find it changes sign across
training seeds and fails to beat a linear ramp at all.

**¶4 — What we do.**
We hold the question to the information. Five encoders differ *only* in which
of the physical graph, the chain membership, and the logical graph they are
permitted to use; everything else — width, depth, training, data, the
candidate set — is identical. We contrast all ten pairs with Holm correction,
on a synthetic family and on real Pegasus connectivity, with the logical
instance as the unit of independence, and we replicate the device arm on an
independent draw at four times the scale. We then price the result in a unit
that carries no reference budget: the number of instance-specific simulator
calls a search needs to match what one forward pass delivers.

**¶5 — What we find.**
An encoder blind to the embedding separates from **every** embedding-aware
encoder, on both topologies (Holm *p* ≤ 0.0014 on Pegasus). No two aware
encoders separate from each other. The information is decision-relevant; the
architecture consuming it is not — a negative result that sharpens the positive
one. The resulting selector beats a matched linear ramp on 48 of 48 held-out
instances in every training seed at *d* = 1.68, cuts reads to 99 % confidence
by 37 %, and is worth roughly 24 instance-specific simulator calls. Its value
scales monotonically with the pool it is allowed to rank, across four measured
points spanning a factor of thirty; and the same critic that ranks a library
turns out to know when to overrule its own generator, with a fallback rate that
self-adjusts from 16 % to 55 % as that generator degrades.

**¶6 — What we do not claim.**
Every number here comes from exact closed-system propagation at up to 14
physical qubits. We use real device *topologies*, not a device. We do not show
that the advantage grows with problem size — we tested, and it does not. We do
not claim that encoders which fail to separate are equivalent. Where a
comparison would cross cost classes — a spectral teacher at deployment against
a single forward pass — we report the classes separately rather than rank them.

---

## Contributions (as they will appear)

1. **Embedding information is decision-relevant, and architecture is not.** On
   two topologies, Holm-corrected over the full family, with an independent
   replication at 4× scale. (§5)
2. **A unit for the effect that survives its denominator.** One forward pass ≈
   24 instance-specific simulator calls on Pegasus, ≥ 15 under the least
   favourable of three search strategies. (§5)
3. **Selection scales with the pool, and the critic polices its own
   generator.** Four measured points; a two-element pool that cannot harm by
   construction. (§6)
4. Supporting negatives that constrain the design space: spectral resolution
   helps as a *control rule* and not as a *supervision target*; a scalar
   bottleneck costs accuracy while adding parameters. (§7)

## Drafting notes

- ¶3 is doing the heavy lifting and must not be cut for space. It is what turns
  three methodological choices from arbitrary into forced.
- The negative in ¶5 ("no two aware encoders separate") is the sentence most
  likely to be trimmed by a co-author and the one that most sharpens the
  claim. Keep it adjacent to the positive.
- Resist promising contribution 4 in the abstract; §7 supports it, but the
  logic chain check flags it as supporting evidence rather than a pillar.
