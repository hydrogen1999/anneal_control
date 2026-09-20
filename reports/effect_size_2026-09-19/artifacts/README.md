# Artifacts behind EFFECT_SIZE.md

This directory exists because the report it supports was, until 2026-09-20,
the one paper-facing artifact with **no generator**. It was assembled by hand,
which is how a 257-call frontier headroom and a 64-candidate bank-oracle
headroom came to sit in one column under a single header. Every number in the
report now comes from one of these files, and every file from one command.

Analysis code: `src/annealctrl/effect_size.py` (unit tests in
`tests/test_effect_size.py`, budget-axis tests in
`tests/test_search_equivalence.py`). Runs are on the remote hosts; paths below
are as invoked there.

| file | produced by |
|---|---|
| `synth_both.json` | `effect_size_report.py --evaluations research_v1/evaluations --method summary --reference both --frontier-sweep g2_frontier_test` |
| `synth_equiv_bank.json` | `search_equivalence.py ... --mode bank --target learned_gain` |
| `synth_equiv_direct.json` | `search_equivalence.py ... --mode direct --target learned_gain` |
| `synth_equiv_bank_ceiling.json` | `search_equivalence.py ... --target bank_ceiling` |
| `equiv_g2_frontier_test_bayes.json` | as above, `--frontier-sweep g2_frontier_test_bayes` |
| `equiv_g2_frontier_test_pg.json` | as above, `--frontier-sweep g2_frontier_test_pg` |
| `synth_menu.json` | `menu_size_curve.py --evaluations research_v1/evaluations --data research_v1/data --frontier-sweep g2_frontier_test` |
| `tts_synth.json` | `time_to_solution_report.py --evaluations research_v1/evaluations` |
| `effect_size_bank.json` | `effect_size_report.py --evaluations pegasus240_exp/evaluations --method summary --reference bank_oracle` |
| `time_to_solution.json` | `time_to_solution_report.py --evaluations pegasus240_exp/evaluations` |
| `peg240_policy_seed_stability.json` | `policy_seed_stability.py --evaluations pegasus240_exp/evaluations` |

## Reading them

Every share carries its `reference` block, and that block carries
`objective_calls_per_instance` **read from the artifacts** rather than
asserted — from `total_objective_calls` on frontier rows, `candidate_count` on
evaluation rows. A share against one reference is not comparable to a share
against another; `scope` on each file says so.

`search_call_equivalent` outputs separate three outcomes that must not be
pooled: `n_resolved` (the search reached the gain), `n_censored` (it never
did, within the budget — the method beat the search), and `n_no_gain` (the
method did not beat the linear reference at all, so no call equivalent
exists). Only the first contributes to `mean_calls`.

## Not yet here

The Pegasus 257-call frontier sweep over the same 48 test parents was still
running when these were collected, so there is no `peg240` frontier,
equivalence or menu artifact. Section 2b of the report is synthetic-only until
there is.
