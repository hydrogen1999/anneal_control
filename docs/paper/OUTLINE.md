# Logic chain

## Paper type

**Technique.** The contribution is a representation claim plus the method that
makes it actionable, not a new problem setting. The Key Idea carries the
narrative; the goal is a short bridge.

## Thinking template

| stage | content |
|---|---|
| **Research background** | A logical Ising problem is not what a quantum annealer executes. Minor embedding replaces each logical spin with a chain, adds penalty couplings, and rescales the whole Hamiltonian to the device's range. The annealing schedule — how fast to traverse *s* — materially changes the outcome, and the standard way to choose one is a spectral rule: allocate time in proportion to the inverse square of the instantaneous gap, or a related quantity. |
| **Limitation 1** | Those rules are computed from the **logical** spectrum, or from a raw first gap. The object actually evolved is the **embedded** Hamiltonian, whose chains, penalties and programmed scale change which control is best. A rule blind to the embedding is solving a different instance than the device runs. |
| **Limitation 2** | Spectral rules need the spectrum. That is privileged information: it costs eigendecompositions the deployment budget does not have, and the design protocol explicitly forbids recomputing it at test time unless the cost is charged. A method that needs a teacher at deployment is not a deployable method. |
| **Limitation 3** | The evaluations that support such rules are reported in units that flatter them — a share of headroom against whichever reference was to hand, with cost classes mixed. The same result can read as 60 % or 83 % depending on the denominator chosen. |
| **Key Idea** | A representation that **sees the embedding** carries enough information to rank candidate controls per instance in a single forward pass, with no spectral oracle at deployment — and *which architecture consumes that information does not measurably matter.* |
| **Challenge 1** | Showing that the **information** helps, not the architecture. Any encoder that sees more is also a different network; a naive comparison confounds the two. |
| **Challenge 2** | Pricing the result without an arbitrary denominator. A "share of achievable" depends on a search budget someone picked, and that choice moves the headline. |
| **Challenge 3** | Keeping a generator honest. A policy that proposes a schedule directly is unstable — on real device connectivity it changes sign across training seeds and loses to a linear ramp outright. |
| **Module A** (Ch. 1) | Five encoders spanning blind → embedding-aware, **identical elsewhere**, contrasted pairwise with Holm correction over the full family of ten, on two topologies, with an independent replication at 4× scale. Separation of `logical` from *all four* aware encoders, and of no two aware encoders from each other, separates information from architecture. |
| **Module B** (Ch. 2) | Price one forward pass in **instance-specific simulator calls** by walking the search's own incumbent curve. A call count needs no reference budget to be meaningful. Report its dependence on search strategy as a range rather than hiding it. |
| **Module C** (Ch. 3) | Let the critic rank a **pool**: its own proposals alongside simple baselines. Measure the pool-size axis end to end, from proposals-only to the full library, and report where graceful degradation is structural and where it is not. |
| **Contribution 1** | Embedding information is decision-relevant on **two topologies**, Holm-corrected, and **architecture-independent**. (§5) |
| **Contribution 2** | A unit for the effect that survives its denominator: one forward pass ≈ **18 (synthetic) / 24 (Pegasus)** instance-specific calls, ≥ 15 under the least favourable of three search strategies. (§5) |
| **Contribution 3** | A critic good enough to rank a library is good enough to **overrule its own generator**; the fallback rate self-adjusts 16–55 % with generator quality, and the two-element pool cannot harm by construction. (§6) |
| **Contribution 4** | Controlled negatives that constrain the design space: spectral **resolution as a supervision target** buys nothing (+0.00001), while the same resolution **as a control rule** matters (−0.01608); a scalar bottleneck costs accuracy with *more* parameters. (§7) |

## Methodology outline

**Topic sentence.** The method is a learned, instance-conditioned critic that
ranks a pool of candidate schedules from a representation of the *embedded*
problem, and its value is set jointly by what it can see and what it may
choose between.

- **§4.1 Representation.** Physical, chain and logical tokens with explicit
  membership; signed edges; scale channels. Encoder variants differ only in
  which of these they are allowed to use.
- **§4.2 Critic and pool.** Predicted loss per candidate; the pool is the
  policy's proposals plus admitted baselines. Selection sees no outcome.
- **§4.3 What is charged.** Cost classes, and why the frontier search is a
  denominator rather than a competitor.

## Self-consistency checks

| check | verdict |
|---|---|
| **Limitations → Key Idea** | *pass.* L1 (blind to embedding) → the representation sees it. L2 (teacher at deployment) → one forward pass, no oracle. L3 (flattering units) → the call-equivalent unit and stated cost classes. |
| **Key Idea → Challenges** | *pass.* Each challenge is a consequence of the idea, not invented for a module: if information is the claim you must separate it from architecture (C1); if "one forward pass" is the claim you must price it (C2); if the representation feeds a generator you must handle its failures (C3). |
| **Challenges → Modules** | *pass.* A↔1, B↔2, C↔3, one-to-one. |
| **Modules → Contributions** | *pass*, with one asymmetry to watch: Contribution 4 draws on ablations reported in §7 rather than on Modules A–C. It is a genuine contribution but should not be promised in the abstract as a fourth pillar; it is supporting evidence that constrains the design space. |

## What the paper is *not* claiming

No hardware. Nothing above 14 physical qubits. Not that non-separation means
equivalence. Not that the advantage grows with problem size — tested, and it
does not (ρ = −0.25 at parent level).
