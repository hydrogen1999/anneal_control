# More proposals is the wrong lever

The direct policy emits three waveforms and its critic picks one. Two fixes were
proposed for its weakness: **emit more waveforms**, and **train on the waveforms
it actually emits** (dataset aggregation). This measures the first.

`configs/experiment_proposals16.json` raises the policy head from 3 proposals to
16 — a 5.3× larger generation budget — and changes nothing else. Compared against
the 3-proposal campaign on matched training seeds, paired on the logical parent,
over the same 48 held-out test parents.

## Result

| | value |
|---|---|
| pooled difference (16 − 3) | **−0.00380** |
| 95% parent bootstrap CI | **[−0.00835, +0.00032]** |
| parents favouring 16 | 26 / 48 (54%) |

**The interval crosses zero.** Raising the proposal count 5.3× does not produce a
separated improvement, and barely more than half the parents move the right way.

| method | 3 proposals | 16 proposals | difference | seeds |
|---|---:|---:|---:|---|
| `physical` | 0.5991 | 0.5866 | −0.01254 | 0,1,2 |
| `hierarchy_physics` | 0.5959 | 0.5899 | −0.00597 | 0 |
| `hierarchy_outcome` | 0.5902 | 0.5883 | −0.00184 | 0,1,2 |
| `logical` | 0.5873 | 0.5870 | −0.00030 | 0,1,2 |
| `summary` | 0.5935 | 0.5951 | **+0.00165** | 0,1,2 |

`summary` — the encoder that ranks first everywhere else — gets slightly *worse*
with more proposals.

## Against the other lever

Dataset aggregation, measured against its own matched control on the same
records, is worth **−0.0226** in selected loss and is consistent on all three
seeds (−0.0195, −0.0238, −0.0243). See `reports/dagger_2026-09-17/`.

| lever | cost | effect | separated |
|---|---|---:|---|
| 3 → 16 proposals | 5.3× generation, 16 critic scores instead of 3 | −0.0038 | **no** |
| one DAgger round | 10,368 true propagations, offline | −0.0226 | yes, all seeds |

Aggregation is roughly six times the effect, and it is the one that survives its
control. That is a useful pairing to report: the obvious lever, applied
generously, does almost nothing, while the one that changes *what the model is
trained on* moves the number.

## What this does not say

Bank-mode selection is unaffected by proposal count and is not compared here —
the bank is fixed at 64 candidates regardless.

`hierarchy_physics` contributes one matched seed, not three, so its −0.00597 is
the least supported row in the table and is not evidence on its own.

This tests one specific increase, 3 to 16, on one dataset. It does not show that
proposal count can never matter, only that at this scale the lever is close to
exhausted while the aggregation lever is not.
