# Model + search, closed loop: the same simulator budget, spent better

## What this is, and what the earlier attempts were not

Two earlier readings of "combine the model with search" were measured and are
retained:

- **Warm start** — the model's proposal as the initial incumbent. Separated only
  at budget 8 and bought no budget: warm@8 = 0.5616 against cold@64 = 0.5473.
- **Offline re-ranking** — the critic scoring a completed search trace. Ranked
  search-generated waveforms at ρ = 0.927, better than its own bank's 0.834,
  but re-ranking a finished trace cannot claim a saving.

This is the third and the only one that is a budget claim. The critic sits
**inside** the loop: each simulator call goes to the best of `oversample`
proposals the critic scored, instead of to the next proposal in sequence. The
budget is simulator calls and does not move — a test asserts the call count is
exactly `budget` at oversample 1, 4 and 16, and the run reports
`budget_mismatches: 0`.

Setup: 44 held-out validation parents (one record each, ≤ 8 physical qubits),
budget 32 true calls per family, oversample 8, families `one_window`,
`two_window`, `eight_bin`. 249 proposals considered per family for 32 true
calls.

## Headline, and the control that halves it

| | effect | positive in |
|---|---|---:|
| total gain from filtering | **+0.00729** [+0.00601, +0.00872] | 44/44 parents |
| the wider proposal stream alone (random pick of 8) | +0.00284 [+0.00169, +0.00400] | 37/44 parents |
| **the critic's own choice** | **+0.00445** [+0.00327, +0.00565] | **42/44 parents** |

**The control was necessary and it changed the answer.** At oversample 8 the
filtered arm walks eight times as much of the Sobol sequence as the baseline
does, so part of any gain is wider coverage rather than the model. Replacing
the critic with a seeded random chooser on the identical proposal stream still
beats the baseline by +0.00284. Reporting +0.00729 as the model's contribution
would have over-credited it by 39%.

The paired contrast is exact, not merely matched: the baseline arm is re-run in
both files and verified identical to 1e-12 on every record, so critic and
random are differenced parent by parent against the same reference.

Per family, critic filter against baseline:

    one_window   +0.01285 [+0.01055, +0.01521]
    two_window   +0.01385 [+0.01010, +0.01787]
    eight_bin    +0.00380 [+0.00256, +0.00512]

Mean best-found loss over the 44 parents: **0.71113 → 0.70384**.

## What it costs

Measured, not asserted:

    surrogate   0.019 ms per proposal
    simulator   14.1  ms per true call
    -> one simulator call buys 742 surrogate scores

The 217 extra scored proposals per family cost 4.1 ms against 450 ms of
simulation: **0.91% overhead on the search's wall clock**.

That is the *online* cost. The offline cost is the model's training, and under
this project's cost-class discipline the filtered search is therefore
**amortised**, not `fixed` — it must not be ranked in one column against an
unamortised search. What the table above compares is like with like: both arms
pay the same 32 simulator calls, and the critic arm additionally carries a
training cost already accounted for elsewhere.

## Why the critic is credited at all

The prerequisite was measured separately and independently
([offline study](../surrogate_filter_2026-09-18/SURROGATE_FILTER.md)): the
critic ranks search-generated waveforms at ρ = +0.9269, against +0.834 on the
bank it was trained on, holding across two architectures and three seeds each.
Restricted to the best 10% of candidates, where the spread collapses, ρ is
still +0.6272 (positive in 48/50 records); within a single family, where every
candidate has the same shape, ρ is +0.907 to +0.928.

Note the handicap: the critic scores a nine-point regridding of each proposal,
which differs from the real waveform by a sup norm of ~0.05, while the true loss
is always the simulator on the real waveform. It chooses from a blurred view and
still wins.

## Limits

- **One search seed** (0) and **one checkpoint** (summary/seed_0). The offline
  study covers six checkpoints; this closed-loop run does not.
- Validation split, ≤ 8 physical qubits, 44 parents, one dataset, one
  connectivity, closed-system losses.
- `sobol_local` only. Filtering is refused for the other strategies rather than
  silently accepted, because a filtered budget that never filtered would be a
  false claim.
- The gain is measured against the *best-found* loss at budget 32, not against a
  global optimum. Nothing here says the filtered search found one.
