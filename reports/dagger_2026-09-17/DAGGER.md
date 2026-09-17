# One round of dataset aggregation, against a matched control

Three seeds, four arms each, all evaluated on the 864 held-out test records over
48 parents. Reproduced from `dagger_arms.json`.

| arm | bank | what it isolates |
|---|---|---|
| `BEFORE` | 64 | the shipped campaign checkpoint |
| `CONTROL` | 64 | retrain, same seed and recipe — measures retrain noise |
| `BANKEXT` | 67 | +3 candidates from the **same Sobol sequence** — measures bank size |
| `DAGGER` | 67 | +3 of the policy's **own labelled proposals** |

The attributable effect of aggregation is `DAGGER − BANKEXT`, not
`DAGGER − CONTROL`. Aggregation changes the bank's size and its source at once,
and only `BANKEXT` holds the size fixed while varying the source.

## Result

| metric | BEFORE | BANKEXT | DAGGER | D−B | per-seed D−B |
|---|---:|---:|---:|---:|---|
| selected loss | 0.593482 | 0.587285 | 0.564730 | **−0.022555** | −0.0195, −0.0238, −0.0243 |
| best proposal | 0.574516 | 0.570724 | 0.551315 | **−0.019409** | −0.0134, −0.0236, −0.0212 |
| generation gap | 0.030178 | 0.025886 | 0.007587 | **−0.018299** | −0.0120, −0.0222, −0.0207 |
| ranking regret | 0.018965 | 0.016561 | 0.013416 | −0.003146 | −0.0062, −0.0002, −0.0030 |
| critic rho | 0.575 | 0.630 | 0.633 | **+0.003** | +0.034, +0.062, **−0.085** |

**Retrain noise is exactly zero.** `CONTROL` reproduced `BEFORE` to every printed
digit on every metric and every seed, so the comparison is not absorbing
optimiser variance.

**Aggregation is worth −0.0226 in selected loss**, the same sign and roughly the
same size on all three seeds. Of the −0.0288 total improvement over the shipped
checkpoint, **78% is aggregation and 22% is the bank merely growing** from 64
candidates to 67. Without the `BANKEXT` arm the whole −0.0288 would have been
credited to the method.

## The mechanism is not the one we predicted

`dagger.py` was written to fix **critic distribution shift**: the critic is
trained on bank waveforms and must rank waveforms its own policy emits, and 92%
of policy waveforms sit further from the bank than a typical bank waveform sits
from its nearest neighbour. The expected signature of that fix is a rise in rank
correlation.

**That is not what happened.** Against the control, rho moves by +0.003 and is
*negative* on one seed of three. Nearly all of the apparent rho gain — 0.575 to
0.630 — is reproduced by the control arm, so it is the bigger bank, not the
aggregation.

What aggregation moves is **generation**. The policy's own best proposal improves
by −0.0194 and the generation gap closes by −0.0183 (consistent across seeds),
while ranking regret improves by only −0.0031.

A plausible reading: the policy and critic share an encoder and are trained
jointly, so appending truthfully-labelled proposals changes the policy head's
targets more than it recalibrates the critic. **That is a hypothesis about a
measurement, not a second measurement**, and it is not claimed as established.
Testing it would need an arm that freezes the policy head and aggregates for the
critic alone.

## Cost, and what this is not

Each aggregation arm spends **10,368 true propagations** labelling proposals on
train and validation. This is charged, not free, and it must enter any
amortisation claim.

Proposals are collected on train and validation only. `collect_labelled_proposals`
refuses the test split outright, and `collect_bank_extension` refuses it too:
labelling a held-out record's proposals and training on them is leakage whatever
it is called afterwards.

The augmented bank is a **different bank** from the one the frontier instrument
uses, so a headroom number computed against it is not comparable to one computed
against the original.

This is one round on one encoder (`summary`). Whether a second round continues to
help, and whether the effect survives on the hierarchical encoders, is untested.
