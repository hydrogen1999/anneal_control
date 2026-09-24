# 7. What does not work, and why that matters

Four controlled negatives. Each removes a design option that a reader would
otherwise reasonably propose, and together they say where the remaining gain is
not.

## 7.1 The spectral schedule is not the target

The design protocol proposes a ladder of spectral targets of increasing
resolution: first gap → all resolved gaps → response bins → response plus
intervention labels. We find the ladder behaves **oppositely at its two ends**,
depending on what the spectral object is *for*.

**As a control rule**, resolution matters a great deal. On a common audited
population, a transition-element-weighted allocation beats a raw first-gap rule
by **−0.01608 [−0.02293, −0.00962]**, at every runtime separately — and the raw
first-gap rule is *worse than a linear ramp* (+0.03242 [+0.00112, +0.06408]).

**As a supervision target**, the same extra resolution is worth nothing. Three
arms identical but for the auxiliary target — none, three moments, eight
frequency bins — over 48 held-out parents and five seeds:

| contrast | difference | 95 % CI | Holm *p* |
|---|---:|---|---:|
| no auxiliary − 8 bins | +0.00055 | [−0.00049, +0.00165] | 0.9253 |
| no auxiliary − 3 moments | +0.00055 | [−0.00049, +0.00166] | 0.9253 |
| **3 moments − 8 bins** | **+0.00001** | [−0.00104, +0.00118] | 0.9937 |

Zero of three separate. Eight bins against three moments differ by one
hundred-thousandth, bounded within ±0.0012 — **under 2 % of the method's own
effect**. The auxiliary loss itself is equally null, replicating an earlier
measurement on a dataset that did not exist when it was made.

Spectral structure is informative *about* the control; training a network to
reproduce more of it is not how that information gets used. This is the sharpest
form of the paper's title claim.

## 7.2 Compression is not the problem; scalar compression is

The protocol proposes a mandatory bottleneck — graph → a scalar density → a
schedule. Forcing the learned model through a scalar chain costs
**+0.00566 [+0.00230, +0.00907]** (Holm *p* = 0.0029) **with more parameters
than the unconstrained model**. An eight-number profile in the same position is
not separated (Holm *p* = 0.45).

So the bottleneck's width is what matters, not the presence of a bottleneck. A
control-relevant representation need not be large, but it cannot be one number.

## 7.3 An arm that could not be tested, and why we say so

`resolved_response_fraction` is **0.0** on both Pegasus datasets. The auxiliary
physics arm is therefore *vacuous there* — the two hierarchy variants are the
same model, identical on every record of every seed. The design protocol's
Factor 5 has been tested only on the synthetic population, and reporting the
resulting `Holm p = 1.0000` as a tight null on real connectivity would be
wrong. We report it as an identity.

## 7.4 The advantage does not grow with problem size

It is tempting to add that the benefit increases with instance size. It does
not, and we tested rather than assumed:

| unit | ρ(physical qubits, gain) | ρ(physical qubits, headroom) |
|---|---:|---:|
| record (n = 864) | −0.1231 | −0.1812 |
| **parent (n = 48)** | **−0.2493** | **−0.3597** |

Both units are shown because the parent is this paper's unit of independence
and a record-level correlation quietly claims 864 observations where there are
48. Both are negative, and the stricter unit is the more negative, so the
scaling hypothesis fares worse under it rather than better. Across 3 – 9
physical qubits the per-size gain is flat within noise.

The larger Pegasus effect is therefore attributable to **the setting** — real
connectivity, a harder instance family, a different difficulty scale — and not
to size. At 14 physical qubits we cannot speak to scaling at all, and do not.

## 7.5 Robustness is bought, not free

Gauge augmentation — training on random spin-reversal gauges, which leave the
spectrum invariant — removes the model's gauge sensitivity and charges for it.
In the stored gauge the signed baseline wins by **−0.00683 [−0.01329,
−0.00115]** (Holm 0.028). Under random gauges the baseline pays a penalty of
+0.0109 and changes its selection in ~40 % of gauges, while the augmented model
pays ≈ 0 and changes in ~8 %; the penalty difference is **+0.01100 [+0.00598,
+0.01667]**.

Net under a random gauge, +0.00417 [−0.00093, +0.00955] — **crosses zero**. So
which arm wins off the stored gauge is *not* established at this sample size,
and we report the trade rather than a winner. This measures robustness to an
exact symmetry of the simulated Hamiltonian, not to device asymmetry.
