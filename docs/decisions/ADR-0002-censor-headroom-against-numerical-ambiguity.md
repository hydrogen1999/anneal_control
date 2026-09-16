# ADR-0002: Censor control headroom against the integrator's own loss ambiguity

## Status
Accepted

## Date
2026-09-16

## Context

Headroom is a difference of two simulated losses: the matched-duration linear
schedule and the best control found under a declared budget. Each loss comes
from `score_schedule`, which accepts a trajectory only once its step-doubling
diagnostic falls under `tolerance`, and reports
`loss_ambiguity_indicator = 2e + e²` for state error `e`.

A difference of two quantities each uncertain at that scale is not evidence when
the difference is itself of that scale. `IMPLEMENTATION_PLAN_VI.md` §G2 states
the requirement directly: if the gain is smaller than numerical or statistical
resolution, learning a policy on that distribution cannot produce a
contribution. Nothing in v0.2 enforced this — `benchmark_record_controls`
reports the ambiguity per trial but never compares it to the effect being claimed.

## Decision

`record_headroom` computes

```
combined_ambiguity = ambiguity(linear) + ambiguity(best_found)
resolution_status  = "resolved" if headroom > margin * combined_ambiguity
                     else "censored_numerical"
```

with `margin` defaulting to 1.0 and recorded in every row. Censored rows are
**retained in full** and counted, but excluded from headroom statistics by
`aggregate_frontier`, which reports the censored fraction alongside every
quantile.

## Alternatives considered

**Report headroom unconditionally and let the reader judge.** Rejected: the mean
over a population where most rows are numerical noise is a positive number, and
that number would end up in a table.

**Tighten `tolerance` until ambiguity is negligible.** Partly done — the caller
controls it — but tolerance is bounded by `max_steps`, and tightening it is an
exponential cost on larger instances. The gate is still needed at whatever
tolerance is affordable.

**Use the per-trial ambiguity of only the best-found control.** Rejected: the
linear reference carries its own error and the two are independent.

## Consequences

- The instrument can return "no measurable signal", which is a valid and
  expected outcome, and the correct one to publish if it occurs.
- `margin` is a declared knob; raising it is conservative, lowering it below 1
  must be justified in the run's configuration and is visible in every row.
- Censoring is per record, so a distribution can be partly resolved; the
  aggregate reports both strata rather than a single pooled number.
