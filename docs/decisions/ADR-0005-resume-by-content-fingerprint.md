# ADR-0005: Sweeps resume by content fingerprint, never by row count

## Status
Accepted

## Date
2026-09-16

## Context

`pipeline.generate_dataset` already resumes by verifying a stored record
fingerprint and a payload fingerprint, and `experiments.run_experiment` refuses
to resume when the config or source hash drifted. A sweep over records has the
same hazard with a sharper edge: the expensive object is a search trajectory
whose value depends on the record, the budget, the seed, the families, the
numerical tolerances and the source that produced it. Two of those (budget and
tolerance) are easy to edit between invocations without noticing.

Resuming by counting completed rows would silently splice results computed under
different settings into one file.

## Decision

Each unit of sweep work is keyed by

```
unit_key = sha256(record_fingerprint ‖ settings_hash ‖ source_hash)
```

where `settings_hash` covers every argument that can change a result. On resume
the driver reads the existing JSONL, indexes completed units by `unit_key`, and:

- skips a unit whose key is present;
- recomputes a unit whose key is absent;
- **refuses the whole run** if the output contains a unit for the same
  `record_id` under a different `settings_hash`.

The settings hash and source hash are written in a header line and in every row.

## Alternatives considered

**Row count / index-based resume.** Cheapest, and wrong for exactly the reason
above.

**Delete and recompute on any settings change.** Safe but discards hours of
valid work when, say, one family is added to the list. The refusal instead tells
the operator to choose a new output path, which keeps both result sets.

**Per-unit files instead of one JSONL.** Simpler resume, but produces tens of
thousands of small files on a shared cluster filesystem. Rejected on operational
grounds; the JSONL is appended atomically per line.

## Consequences

- Adding a family or changing a budget requires a new output directory. This is
  intended: those runs are not comparable.
- A partially written final line (hard kill mid-write) is detected by JSON parse
  failure and truncated on resume, with the event logged.
- Aggregation reads the JSONL streaming and asserts a single settings hash
  before producing any statistic.
