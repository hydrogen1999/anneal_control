# Rung 3 of the spectral ladder: finer resolution in the target changes nothing

## The question

The design document's spectral-target ladder is *first gap → all resolved gaps
→ **response bins** → response plus intervention labels*. The first two rungs
were answered by comparing privileged **teachers** — which spectral summary
predicts a better control. Rung 3 asks something different: which summary is
worth **supervising a network with**.

It had never been run, for a mundane reason: the training loop hard-coded
`response_moments`. The datasets carry both — three moments per s-point, and
eight frequency-band masses.

## Design

Three arms on one freshly generated synthetic dataset (240 parents, 4320
records, 48 held-out parents, `resolved_response_fraction = 0.799`), five
training seeds each, differing in exactly one thing:

| arm | auxiliary weight | target | head width |
|---|---:|---|---:|
| `hierarchy_outcome` | 0.00 | — | 3 |
| `hierarchy_moments` | 0.05 | 3 moments | 3 |
| `hierarchy_bins` | 0.05 | 8 bins | 8 |

## Result: a tight null, twice

Holm-corrected over the three pairs, bank selection, 48 held-out parents.

| contrast | difference | 95 % CI | Holm *p* |
|---|---:|---|---:|
| outcome − bins | +0.00055 | [−0.00049, +0.00165] | 0.9253 |
| outcome − moments | +0.00055 | [−0.00049, +0.00166] | 0.9253 |
| **moments − bins** | **+0.00001** | [−0.00104, +0.00118] | 0.9937 |

**Zero of three separate.** In direct mode likewise zero of three, with wider
intervals.

Two readings, and the second is the new one.

**The auxiliary spectral loss does nothing**, replicating the earlier Factor 5
result on a fresh dataset: +0.00055 here against −0.00055 on `research_v1`,
the same magnitude under the opposite sign convention.

**Raising the target's spectral resolution does nothing either.** Eight
frequency bins against three moments differ by **+0.00001**, bounded within
±0.0012 — under 2 % of the learned selector's own effect on this set
(+0.05627). A non-separation is not equivalence, but the bound is tight enough
that a useful effect is excluded at this resolution step.

## Why this matters to the thesis

The ladder does not behave the same way at every rung. As a **teacher** —
where the spectral object *is* the control rule — resolution mattered:
`d2` beat `gap_inverse_square` by −0.01608 [−0.02293, −0.00962], and the raw
first-gap rule was *worse than linear*. As a **supervision target**, the same
extra resolution is worth nothing measurable.

That asymmetry supports the paper's central claim rather than complicating it.
Spectral structure is informative about the control; supervising a network to
reproduce more of it is not how that information gets used.

## Scope

Five seeds, 48 held-out parents, one auxiliary weight (0.05), one resolution
step (3 → 8). This does not test a weight sweep, and it does not test rung 4:
response plus intervention labels would require intervention outcomes attached
to these records, and the G3 intervention sweep covers different parents. That
rung remains open and needs new data, not new analysis.

`hierarchy_bins` is also untestable on either Pegasus dataset, where
`resolved_response_fraction` is 0.0 — see
[the 48-parent Pegasus report](../pegasus240_2026-09-20/PEGASUS240.md).
