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

## Headline: the control is where the story is

Every configuration was run twice — once with the critic choosing, once with a
seeded random chooser on the **identical** oversampled proposal stream. That
control is not decoration. At oversample 8 the filtered arm walks eight times
as much of the Sobol sequence as the baseline, so part of any gain is wider
coverage rather than the model.

| checkpoint | search seed | filtering total | random control | **the critic alone** |
|---|---:|---|---|---|
| seed_0 | 0 | +0.00729 [+0.00601, +0.00872] 44/44 | +0.00284 [+0.00169, +0.00400] 37/44 | **+0.00445** [+0.00327, +0.00565] 42/44 |
| seed_0 | 1 | +0.00829 [+0.00673, +0.00997] 44/44 | +0.00214 [+0.00012, +0.00411] 30/44 | **+0.00615** [+0.00441, +0.00822] 43/44 |
| seed_0 | 2 | +0.00087 [+0.00052, +0.00128] 33/44 | **−0.00397** [−0.00563, −0.00245] 8/44 | **+0.00484** [+0.00326, +0.00660] 40/44 |
| seed_1 | 0 | +0.00708 [+0.00591, +0.00838] 44/44 | +0.00348 [+0.00211, +0.00486] 36/44 | **+0.00360** [+0.00242, +0.00495] 36/44 |
| seed_2 | 0 | +0.00757 [+0.00628, +0.00903] 44/44 | +0.00235 [+0.00089, +0.00376] 33/44 | **+0.00522** [+0.00374, +0.00685] 44/44 |

| across 5 configurations | mean | spread |
|---|---:|---|
| filtering total | +0.00622 | +0.00087 … +0.00829 — **9.5×** |
| random control | +0.00137 | **−0.00397 … +0.00348 — changes sign** |
| **the critic alone** | **+0.00485** | +0.00360 … +0.00615 — 1.7× |

**Every configuration's critic-alone interval excludes zero.**

Read the rows, not the first one. Had this been reported from search seed 0
alone it would have claimed +0.00729 for the model — 50% too much. Had it been
reported from search seed 2 alone it would have claimed +0.00087 and looked
like a failure. The raw "filtering helps" number swings by a factor of 9.5 and
is not a property of the method; it is a property of which slice of the Sobol
sequence the wider stream happened to land in, and at seed 2 that slice is
actively **worse** than the baseline's. The critic's own contribution, measured
against that same stream, sits at +0.005 and barely moves.

This is the same composition error that turned a claimed 2.6× scale-arm effect
into a matched 1.52×: an effect measured without holding the confound fixed
reports the confound.

The pairing is exact rather than matched. The baseline arm is re-run inside both
files of every pair, and `closed_loop_summary.py` **verifies** the two baselines
agree to 1e-12 on every record before differencing — if they did not, the
decomposition would be meaningless and the script says so instead of printing a
number. It also refused the first critic run outright, because that run predates
the `--chooser` flag and so does not record which chooser it used; rather than
patch a field into an archived artifact, the configuration was simply re-run,
and it reproduced +0.00729 [+0.00599, +0.00874] 44/44 exactly.

Per family at checkpoint seed_0 / search seed 0, critic filter against baseline:

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

- Three search seeds and three checkpoints, but not crossed: seeds 1 and 2 were
  run only at checkpoint seed_0, and checkpoints seed_1/seed_2 only at search
  seed 0. A full 3x3 grid would separate the two sources of variation cleanly;
  this does not.
- Validation split, ≤ 8 physical qubits, 44 parents, one dataset, one
  connectivity, closed-system losses.
- `sobol_local` only. Filtering is refused for the other strategies rather than
  silently accepted, because a filtered budget that never filtered would be a
  false claim.
- The gain is measured against the *best-found* loss at budget 32, not against a
  global optimum. Nothing here says the filtered search found one.
