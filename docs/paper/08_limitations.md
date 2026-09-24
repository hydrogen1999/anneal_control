# 8. Limitations

Stated as constraints on what may be concluded, not as future work.

## 8.1 No hardware

Every number is exact closed-system propagation. We use real device
*topologies* — Pegasus connectivity, real embeddings, real chain strengths and
programmed scales — but no device. Three things follow.

The loss is a simulated success probability, so the reads-to-confidence figures
are a **projection onto independent reads**, not a measured count; the binomial
interval that would apply to hardware counts does not apply and is not used.
Open-system effects appear only through the dephasing and amplitude-relaxation
studies, which show the ordering of searched controls survives small dephasing
and that the advantage decays more slowly than the exact noiseless gain — not
that it survives a real device's noise. And the design protocol's device-adapter
arm, which is the part that would close this, requires a QPU and is not
attempted.

## 8.2 Fourteen physical qubits

Exact propagation costs $O(2^N)$, which caps the study at 14 physical qubits
and 5–7 logical spins. We therefore cannot speak to the regime where annealing
is interesting, and we do not: §7.4 reports that the advantage does **not**
grow with size over the range we can reach, with the correlation more negative
at the parent level than the record level. Nothing here should be read as a
scaling claim in either direction.

An approximate simulator would buy size at the cost of the ground truth every
number rests on. We judged that the wrong trade for this paper.

## 8.3 The ceiling is the library, not the representation

The factorisation of §5.2 is uncomfortable and we state it plainly: the critic
extracts **~85 %** of what its library contains, while the library holds only
**~74 %** of what an instance-specific search finds. The binding constraint is
the action space, which is a dataset design choice, not the representation the
paper is about.

The pool-size axis (§6.4) quantifies what moving that constraint costs: two
thirds of the library's advantage comes from eight designed waveforms, and the
last doubling of a fixed menu returns little. A fixed menu cannot adapt, so
enlarging it is the expensive fix rather than the cheap one.

## 8.4 Scope of the no-harm property

The two-element pool harmed zero parents in 384 parent-seed cells. This is
**structural** — when the only alternative is the reference, choosing it
returns the reference — and does not transfer: the designed library harmed 3
parents in 240 synthetic cells and 1 in 144 on Pegasus. Graceful degradation is
a property of that specific pool.

## 8.5 Arms that were not run, and one that could not be

- **Factor 5 (auxiliary physics) on real connectivity**: impossible on these
  datasets, `resolved_response_fraction = 0.0`. Tested on synthetic only.
- **Rung 4 of the spectral ladder** (response plus intervention labels): never
  run. It needs intervention outcomes attached to these records, and the
  intervention sweep covers different parents — new data generation, not
  analysis.
- **A covariant (gauge-equivariant) model**: not implemented. §7.5 reports the
  augmentation arm only, and its net effect crosses zero.
- **Sample-assisted selection at 64 candidates**: the gate that closed this
  question used an 8-candidate bank, where the selector is near-saturated. On
  the 64-candidate banks used throughout this paper the residual is 18–23×
  larger and the selector is exact-best in under half of records, so the
  question is open on the population we actually report.

## 8.6 Statistical scope

Failure to separate is not equivalence; non-rejection sets are sets. Intervals
condition on the observed training seeds and are not joint over
parent-and-seed. The Holm family on Pegasus contains one degenerate pair
(§5.1), which makes the correction conservative rather than lenient. Where a
reference is a test-split search it is a **denominator**, carries
`online_adaptation: true`, and is never ranked against an amortised method.
