# Symmetry: the inductive bias buys robustness and charges for it

## The question, in full

> Symmetry | Signed baseline; gauge augmentation; specialized covariant model |
> **Whether stronger inductive bias improves robustness without deleting
> frustration.**

Three clauses, and the in-distribution ablation alone answers none of them
cleanly: the test split stores one particular gauge, so a model specialised to
it is never charged for the specialisation.

## The transformation

A spin reversal chosen on logical spins and induced on the physical graph
through `membership`. Chains see `s_i s_j = σ² = +1`, so intra-chain couplings
are untouched and chain strength keeps its meaning; fields and inter-chain
couplings transform together.

Two properties are asserted by tests against numerics, not by comment:

- **the spectrum is unchanged** at every `s`, checked against
  `spectral_profile` — which is what makes every stored candidate loss remain
  a correct label, and the augmentation legitimate;
- **the loop product is invariant** over 20 random gauges — which is the
  document's "without deleting frustration" clause, and the reason this is an
  augmentation rather than the sign-stripping the document refuses.

## Result 1: it costs accuracy in the stored gauge

Two arms differing only in `gauge_augment`, three seeds, one experiment.
Holm-corrected:

| mode | signed_baseline − gauge_augmented | Holm p | |
|---|---|---:|---|
| bank | **−0.00683** [−0.01329, −0.00115] | 0.0282 | separated |
| direct | **−0.00750** [−0.01418, −0.00083] | 0.0279 | separated |

The signed baseline is better by ~0.007 on the gauge the dataset stores.

## Result 2: it buys robustness, and the baseline turns out to be fragile

Each test record placed in **8 fresh random gauges**, every checkpoint seeing
the same ones. No simulation: the spectrum is invariant, so the stored labels
still apply.

| arm | stored gauge | over random gauges | **penalty** | selection unchanged |
|---|---:|---:|---:|---:|
| signed_baseline | 0.5453 | 0.5562 | **+0.0109** | **~60 %** |
| gauge_augmented | 0.5522 | 0.5520 | **−0.0001** | **~92 %** |

**The signed baseline is not gauge-invariant.** It changes its selection in
about 40 % of gauges and pays 0.011 for it. The augmented model pays nothing
and changes its selection in about 8 %.

Paired, at parent level:

    in the stored gauge                 -0.00683 [-0.01329, -0.00115]  *
    averaged over random gauges         +0.00417 [-0.00093, +0.00955]
    gauge penalty, baseline - augmented +0.01100 [+0.00598, +0.01667]  *

## The answer

All three clauses, separately:

1. **Does the inductive bias improve robustness?** **Yes, and significantly.**
   The gauge penalty falls from +0.0109 to zero, a difference of
   +0.01100 [+0.00598, +0.01667].
2. **Does it delete frustration?** **No** — the loop product is invariant, by
   test.
3. **Is it worth it?** **Not established.** It costs 0.0068 in the stored gauge
   (significant) and gains 0.0042 under a random one (**interval crosses
   zero**). The crossover is suggested at 48 parents and not demonstrated.

That is the shape the document's central contribution promises — *"the benefit
depends on symmetry … measured rather than assumed"* — and here the dependence
is on **whether the deployment gauge is the training gauge**. On a device with
gauge averaging, it would be; on one without, it would not.

## Limits

- Three seeds, one dataset, 48 held-out parents, bank selection and direct
  modes. 8 gauges per record.
- The **specialized covariant model**, the third arm of the row, is not
  implemented and remains open.
- Robustness is measured against an exact symmetry of the simulated
  Hamiltonian. A device deviates from it, and that deviation is not modelled
  here — the document separates ideal invariance from device asymmetry, and
  only the first is tested.
