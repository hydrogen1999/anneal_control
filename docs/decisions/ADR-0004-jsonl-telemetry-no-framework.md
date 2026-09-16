# ADR-0004: Append-only JSONL telemetry with a run correlation id, no logging framework

## Status
Accepted

## Date
2026-09-16

## Context

A frontier or intervention sweep is a long batch job on `apollo` or `goose`. The
questions an operator will ask mid-run or after a failure are:

1. How far has the sweep got, and which unit is it on right now?
2. What is the accepted-label / objective-call throughput, and is it degrading?
3. What was the peak resident memory, and did any unit approach the state budget?
4. Which units failed a numerical gate, and with what diagnostic?

v0.2 answers none of these during a run: `print()` lines go to a log with no
structure, and `profiling.py` computes throughput only after a dataset is
complete. A sweep that dies at hour six leaves nothing queryable.

## Decision

A `telemetry.RunLog` writes one JSON object per line to a `.jsonl` file:

```json
{"run_id":"…","seq":12,"t":41.9,"event":"unit_end","unit":"parent_0003_e0_k0_t1",
 "objective_calls":129,"wall_seconds":3.7,"peak_rss_bytes":210763776}
```

`run_id` correlates every line of one invocation. `seq` is monotone so
interleaved writes from a resumed run stay orderable. Events are opened with
`run_start` and closed with `run_end` carrying totals. Field names are
allowlisted per event and a name matching a credential pattern is refused at
write time.

No logging framework is introduced: `pyproject.toml` declares three runtime
dependencies and the project treats added dependencies as an "ask first" change.
`json` plus an open file handle is sufficient and keeps the artifact readable by
`jq` on a cluster login node with nothing installed.

## Alternatives considered

**`logging` with a JSON formatter.** Global configuration state leaks between
tests and into library consumers, and handler setup order becomes a source of
lost lines. Rejected for a batch tool that owns its own output file.

**OpenTelemetry / Prometheus.** The canonical answer for a service; wrong for a
detached batch job on a cluster with no collector, no network egress policy in
our favour, and no operator dashboard.

**Reuse `profiling.py`.** It measures a completed `generate_dataset` call. It is
a summary, not a stream, and extending it to stream would change its contract.

## Consequences

- Progress, throughput and memory are queryable with `jq` during a run.
- The log is append-only, so a resumed sweep's lines follow the original run's
  and both `run_id`s are visible — the resume boundary is legible.
- Peak RSS is sampled via `resource.getrusage`, which is coarse and
  platform-dependent (bytes on Linux, bytes on macOS after unit correction);
  the unit is recorded and the value is labelled an estimate.
- Nothing in the telemetry participates in a scientific claim. It is operational
  only, and the reported costs in results JSON remain the authoritative numbers.
