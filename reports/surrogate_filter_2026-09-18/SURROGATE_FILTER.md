# The critic ranks waveforms it did not generate — better than it ranks its own bank

## The question

The learned model beats a fixed global schedule by ~0.021, and equal-budget
search beats the spectral oracles. The obvious next question — the one the user
posed as *"model + search"* — is whether the two compose: can the critic score
candidates for free and let the simulator pay only for the promising ones?

A first attempt at that idea, warm-starting search from the model's proposal,
separated only at budget 8 and bought no budget. This is the stronger reading:
the model as a **ranker over a search's own candidate pool**, not as a single
hint.

Everything hinges on one prerequisite. The critic is trained on an 8-entry bank
of curated waveforms and reaches rank correlation 0.834 there. Search-generated
waveforms are a different distribution. **If the ranking does not transfer, the
idea is dead.**

## Two hazards, handled rather than assumed away

**The grid.** `predict_losses` requires `schedule_points` values on a common
uniform tau grid; no search family puts its knots there (`eight_bin`'s nine
knots are uniform in *s*, not tau). Re-expressing a search waveform on the bank
grid moves it by a sup norm of **0.0538 mean, 0.1222 max** over 6144 archived
trials, against a bank spread of ~0.117. That is large enough to dominate a
ranking signal, so **the archived search loss is not a valid label for the
regridded waveform.** Every candidate was therefore re-scored with the true
simulator after regridding, so prediction and label describe the same object.
`regrid()` returns the representation error beside the waveform so it cannot be
ignored silently.

**The split.** Record identities come from the dataset's own validation and
test splits; the sweeps supply only waveforms. 50 held-out records, 25 parents,
97 candidates each (4850 fresh simulations), 96.9% of them distinct waveforms.

## Result: the ranking transfers, and improves

| checkpoint | rank ρ on search candidates | keep top 25%: shortfall | finds the true best |
|---|---:|---|---:|
| summary/seed_0 | **+0.9269** | +0.00022 [+0.00000, +0.00061] | 96.0% |
| summary/seed_1 | +0.9276 | +0.00084 [+0.00000, +0.00253] | 96.0% |
| summary/seed_2 | +0.9243 | +0.00042 [+0.00000, +0.00111] | 96.0% |
| hierarchy_physics/seed_0 | +0.9151 | +0.00217 [+0.00048, +0.00417] | 90.0% |
| hierarchy_physics/seed_1 | +0.9211 | +0.00076 [+0.00003, +0.00162] | 92.0% |
| hierarchy_physics/seed_2 | +0.9289 | +0.00054 [+0.00000, +0.00125] | 94.0% |

*In-distribution baseline: ρ = 0.834 on its own bank.* The critic ranks
search-generated waveforms **better** than the ones it was trained on, across
two architectures and three seeds each.

Budget curve, pooled over 50 records (summary/seed_0):

| simulate | shortfall against the full 97 | finds the true best |
|---|---|---:|
| top 12.5% (12 of 97) | +0.00172 [+0.00022, +0.00409] | 90.0% |
| top 25% (24 of 97) | +0.00022 [+0.00000, +0.00061] | 96.0% |
| top 50% (48 of 97) | +0.00016 [+0.00000, +0.00048] | 98.0% |

For scale: the search headroom this project reports is **0.1061** and the whole
learned-versus-linear advantage is **~0.021**. A shortfall of 0.00022 is **one
percent of that advantage** for **a quarter of the simulator calls**.

## Why ρ = 0.927 is not an easy-ranking artefact

The search pool does span a wider range than the bank (0.2098 against 0.1166),
which makes ranking easier, so the headline was stress-tested three ways.

**Restricting to the good candidates**, where the spread collapses and ranking
is genuinely hard:

| restricted to | mean ρ | positive in |
|---|---:|---:|
| all 97 | +0.9269 | 50/50 |
| best 50% | +0.8212 | 49/50 |
| best 25% (spread 0.048) | +0.7287 | 47/50 |
| best 10% | +0.6272 | 48/50 |

**Within a single family**, where every candidate has the same shape and only
the parameters differ — no shape cue to exploit:

    eight_bin   +0.9279      one_window  +0.9264      two_window  +0.9073

**Duplicates** are not the explanation: 96.9% of the 97 trials are distinct
waveforms.

The skill is real and it is not confined to separating good from obviously bad.

## The limitation that matters

**This is an offline re-ranking of a completed search trace, not a closed-loop
budget saving, and the difference is not cosmetic.** The sweeps use "Sobol plus
incumbent-local search": the incumbent updated 3–4 times per 31 evaluations in
each family, so roughly 87% of proposals would have been drawn identically
regardless of feedback — but not all of them. Had the search actually simulated
only 24 of the 97, its incumbent updates would have differed and the later
proposals with them. The counterfactual is therefore not exactly the trace
measured here.

What this establishes is the **prerequisite**, which was the open question: the
critic's ranking survives the distribution shift from curated bank to
search-generated candidates, at ρ ≈ 0.93 overall and ρ ≈ 0.63 even among the
best 10%. The budget numbers should be read as what a filter would have
achieved on a realistic candidate pool, not as a measured end-to-end saving.
Confirming the saving requires running the search with the filter inside the
loop and reporting simulator calls as the budget.

Other limits: 50 records and 25 parents is small; one dataset and one
connectivity; closed-system losses.
