# ADR-0008: Contrast the scale arms on matched pairs, never on marginal means

## Status
Accepted — amends ADR-0003, which stays in force.

## Date
2026-09-17

## Context

ADR-0003 requires two arms for any intervention that moves the programmed scale
α: `total_compiled_effect` (what a user actually gets) and `scale_controlled`
(both sides compiled at one conservative common α, so the declared factor moves
and the global scale does not). It also forbids pooling them. All of that holds.

What ADR-0003 did not say is how to *compare* them, and the campaign report
compared them the obvious wrong way: it divided one arm's marginal mean by the
other's and reported

> Holding the global `H_Z` scale fixed makes the effect **2.6× larger**.

That number is a composition artefact. A `scale_controlled` arm is only built
when the intervention actually moves α, so the controlled population contains
only `chain_strength` (189 pairs) and `ports` (48). The total-effect population
also contains `geometry` (231) and `field_allocation` (18), which have no
controlled counterpart at all and carry much smaller penalties. Dividing the two
marginal means therefore charges the difference *between factors* to the scale
control.

Recomputed on the 214 interventions observed under both arms, weighted by
parent:

| quantity | marginal (wrong) | matched (right) |
|---|---:|---:|
| ratio | 2.56× | **1.52×** |
| within-pair difference | — | +0.0199, 95% CI [0.0132, 0.0267] |
| pairs moving that way | — | 130/214 (60.7%) |

The finding survives — the interval excludes zero — at a little over half the
claimed size, and it stops being unanimous.

A second defect was found while fixing the first: the point estimate was
pair-weighted while its bootstrap interval was parent-weighted, so with parents
contributing unequal pair counts the estimate could fall outside its own
interval, and on the real data it did (0.0100 against [0.0132, 0.0267]).

## Decision

1. `scale_arm_matched_contrast` pairs each base intervention with its own
   controlled twin and averages the **within-pair** difference, so factor
   composition cancels by construction.
2. Every number it reports is weighted by the logical parent, the unit of
   independence the bootstrap resamples. The point estimate and its interval
   describe the same estimand. A pair-weighted mean is still reported, under a
   name that says so.
3. The confounded marginal ratio is reported beside the matched one, labelled,
   together with `unmatched_is_composition_confounded`, which is set whenever
   the two arms' factor compositions differ. Deleting the superseded number
   would make the correction unauditable.
4. `aggregate_interventions` always emits the block. When no intervention has
   both arms it emits `{"status": "unavailable", "reason": ...}` rather than
   omitting the key, so a missing contrast is visible rather than inferred.

## Alternatives Considered

### Restrict the total-effect arm to the factors that have a controlled arm
- Pros: one-line change; removes the grossest part of the confound.
- Cons: still unmatched within those factors, and the real data show it lands at
  1.79×, between the wrong answer and the right one. Better than nothing is not
  a reason to ship a third number.
- Rejected.

### Drop the comparison and report the two arms separately
- Pros: nothing left to get wrong.
- Cons: "does controlling scale change the answer?" is a question ADR-0003
  raises on purpose, and refusing to answer it leaves the reader to divide the
  two printed means themselves — which is precisely the error being fixed.
- Rejected.

## Consequences

- The headline number in `reports/CAMPAIGN_G2_G3_RESULTS.md` changes from 2.6×
  to 1.52×, with the old value retained and marked superseded.
- Any future arm that exists for only a subset of factors inherits the guard:
  the composition flag fires without anyone having to remember this ADR.
