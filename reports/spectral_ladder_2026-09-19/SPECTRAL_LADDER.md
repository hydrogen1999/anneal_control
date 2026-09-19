# The spectral-target ladder: each rung of resolution helps, and the naive rule hurts

## What the document asks

> Spectral target | First gap; all resolved gaps; response bins; response plus
> intervention labels | Whether couplings, frequency resolution, or finite-time
> response add decision value.

Two of those rungs already exist as privileged teachers: `gap_inverse_square`
is the raw first-gap rule, and `d2` weights by drive-induced transition
elements. Both are computed for the same records, so the comparison needs no
new computation — only the common audited population, which had never been
taken.

## Result

303 records, 22 parents, both teachers passing `sampled_point_audit_passed` on
the same record, runtimes exactly balanced at 101 each. Paired at parent level.

| stratum | d2 − gap_inverse_square | d2 − linear | gap_inverse_square − linear |
|---|---|---|---|
| all | **−0.01608** [−0.02293, −0.00962] | +0.01633 [−0.01518, +0.04769] | **+0.03242** [+0.00112, +0.06408] |
| T=1 | **−0.01310** [−0.01858, −0.00816] | **−0.01258** [−0.01805, −0.00749] | +0.00052 [−0.00256, +0.00351] |
| T=4 | **−0.02070** [−0.02851, −0.01296] | +0.03404 [−0.00995, +0.07927] | **+0.05474** [+0.01147, +0.09925] |
| T=12 | **−0.01444** [−0.02742, −0.00271] | +0.02754 [−0.02593, +0.08014] | +0.04198 [−0.00770, +0.09215] |

Bold intervals exclude zero. Lower loss is better, so a negative difference
favours the first named method.

## Reading

**Transition-element resolution adds decision value, at every runtime.**
`d2` beats the raw first-gap rule by 0.013 to 0.021, with all four intervals
excluding zero and the same sign throughout. That is the document's factor-2
question answered for the first step of its ladder, and it is stable under the
runtime stratification that broke three other subgroup claims this session.

**The naive first-gap rule is worse than a linear ramp**, by +0.032 pooled and
+0.055 at runtime 4, both excluding zero. Following the minimum gap is not
merely uninformative here; it actively costs.

Placing it beside the other audited contrasts gives a monotone ladder, each
step measured on its own audited population and paired at parent level:

    gap_inverse_square   +0.032 worse than linear
    linear               reference
    d2                   0.016 better than gap_inverse_square
    learned bank         0.050 better than d2, at every runtime separately

## What this is not

- Two rungs of four. **Response bins** and **response plus intervention
  labels** as learning targets are not tested; `docs/paper_protocol.md` records
  response-bin prediction as an optional extension and it remains one.
- A cost-class ranking. Both teachers consume a privileged exact spectrum; the
  learned selector does not. The ladder above compares *information paths*, not
  deployment costs.
- `d2 − linear` is significant only at runtime 1, and positive elsewhere. The
  step that is robust is **d2 over gap**, not d2 over linear — consistent with
  [the coherent scan](../coherent_2026-09-19/COHERENT_SCAN.md), where the
  teacher's advantage over linear proved unstable across runtime and sampling.

## Reproduction

Computed entirely from committed inputs — `reports/heldout_2026-09-17/heldout_records.json`
and `reports/comparison_2026-09-17/testref_rows.json` — so it reruns without
the cluster.
