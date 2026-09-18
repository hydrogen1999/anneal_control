# One round of dataset aggregation, against a matched control

Three seeds, four arms each, all evaluated on the 864 held-out test records over
48 parents. Reproduced from `dagger_arms.json`.

| arm | bank | what it isolates |
|---|---|---|
| `BEFORE` | 64 | the shipped campaign checkpoint |
| `CONTROL` | 64 | retrain, same seed and recipe — measures retrain noise |
| `BANKEXT` | 67 | +3 candidates from the **same Sobol sequence** — measures bank size |
| `DAGGER` | 67 | +3 of the policy's **own labelled proposals** |

The observed contrast associated with proposal-source replacement is `DAGGER − BANKEXT`, not
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

`CONTROL` reproduced `BEFORE` for the same deterministic seed and recipe.
This checks replay consistency; it does not imply zero uncertainty across
training seeds, initializations, data draws, or hyperparameter choices.

**DAGGER − BANKEXT is −0.0226 in selected loss**, the same sign and roughly the
same size on all three seeds. Of the −0.0288 total improvement over the shipped
checkpoint, the arithmetic decomposition attributes **78% to the DAGGER–BANKEXT contrast and 22% to BANKEXT–CONTROL** from 64
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

## An independent pilot does not reproduce this, and the two are compatible

A separate controlled-acquisition pilot (`acquisition_pilot_2026-09-17/`) ran the
same idea on 60 fresh parents with three matched controls and settings frozen
before the test outcomes. Its headline is negative.

| contrast | pilot | this study |
|---|---|---|
| **matched** (POLICY − BANKEXT) | +0.004107, CI [−0.0500, +0.0517] | **−0.022555** |
| unmatched (POLICY − CONTROL) | +0.027589, CI [+0.0039, +0.0487] | −0.028752 |

**On the matched contrast the two studies agree, because the pilot cannot tell
them apart.** Its interval is 0.102 wide over 12 held-out parents and three
seeds, and −0.0226 sits inside it. The pilot does not refute this result; it
lacks the power to confirm or deny it.

On the **unmatched** contrast the signs are opposite, and that is the part worth
taking seriously rather than explaining away.

## The confound both studies share

The pilot names it: its POLICY checkpoints were selected at epochs **3, 2, 2**
against CONTROL's 43, 20, 20 — effectively untrained. Early stopping uses
validation bank regret, and both studies augment the **validation** bank as well
as the training bank. Regret measured against a bank the policy's own proposals
were just added to is easier to reduce, so patience can fire before the model is
trained.

This study's epochs do not show that pattern as severely:

| seed | CONTROL | BANKEXT | DAGGER |
|---:|---:|---:|---:|
| 0 | 55 | 20 | 20 |
| 1 | 33 | 63 | 55 |
| 2 | 52 | 29 | 26 |

Two seeds stop earlier than control, one stops later. That is not the pilot's
collapse to epoch 2, and the matched DAGGER−BANKEXT comparison holds both
augmented arms to the same selection rule. But "less severe" is not "absent",
and no study here has run the ablation the pilot asks for.

**It is running now.** Every arm early-stops on the **un-augmented** validation
records, so the selection criterion is one object across arms while the training
banks still differ. If −0.0226 survives, selection was not driving it. If it
disappears, the number was selection rather than aggregation and is withdrawn.
Until that lands, this section is the correct reading of the result, not the
headline above it.

## Cost, and what this is not

Each aggregation arm spends **10,368 objective evaluations** labelling proposals on
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

## Correction to causal and reproducibility scope

This historical campaign augmented both training and validation banks, so it
changed checkpoint selection as well as the training labels. The source contrast
is not an isolated train-only acquisition effect. `dagger_arms.json` retains
aggregate metrics, not the paired record-by-parent-by-seed outcomes needed to
recompute uncertainty or tied Spearman statistics. All three seed differences
have the same sign; this alone is not a paired parent significance test.

The current acquisition runner uses train-only labels and frozen validation/test
banks, includes decoder-random controls, and archives raw evaluations. Its
independent small fixed-validation pilot is negative; see
`reports/acquisition_pilot_2026-09-17/README.md`. Different recipes prevent treating
that pilot as a direct replication or refutation of this historical campaign.
The revised collector now refuses validation acquisition; this report describes
the earlier collector and must not be presented as a run of the current code.
