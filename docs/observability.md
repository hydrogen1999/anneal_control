# Observability: what a running campaign tells you

A G2 or G3 campaign is a long detached batch job on `apollo` or `goose`. Before
this, a sweep that died at hour six left nothing queryable: `print()` lines with
no structure, and throughput computed only after a dataset completed.

## The four questions

Every signal below exists to answer one of these. If a proposed addition answers
none of them, it does not belong in the telemetry.

1. How far has the sweep got, and which unit is executing now?
2. What is the objective-call throughput, and is it degrading?
3. What was peak resident memory, and did any unit approach the state budget?
4. Which units failed a numerical gate, and with what diagnostic?

## Signals

| Signal | Where | Answers |
|---|---|---|
| `telemetry.jsonl` | every sweep output directory | 1, 2, 3, 4 during the run |
| `manifest.json` | every sweep output directory | 1, 4 after the run, including an aborted one |
| `rows.jsonl` | every sweep output directory | the scientific record, including failure rows |
| `doctor.json` | campaign root | what the host actually had (CUDA, CuPy, matplotlib) |

**Telemetry is operational, not scientific.** Every `run_start` line carries
`telemetry_is_operational_not_scientific_evidence: true`. The authoritative
costs, losses and diagnostics are the fields of `rows.jsonl` and the report JSON.
Never quote a telemetry timing in a paper.

## Reading a live run

```bash
# Which unit is running, and how long each one took.
jq -c 'select(.event=="unit_end") | {unit, wall_seconds}' runs/frontier_val/telemetry.jsonl | tail

# Throughput drift: seconds per unit over the run.
jq -r 'select(.event=="unit_end") | [.t, .wall_seconds] | @tsv' runs/frontier_val/telemetry.jsonl

# Failures and their diagnostics.
jq -c 'select(.event=="unit_failed")' runs/*/telemetry.jsonl

# Peak resident memory of the process at the end of each run.
jq -c 'select(.event=="run_end") | {run_id, status, units, failed, skipped, peak_rss_bytes}' \
   runs/frontier_val/telemetry.jsonl

# Progress against the plan.
jq '{planned, completed, skipped, failed, status}' runs/frontier_val/manifest.json
```

## Structure

One JSON object per line. Reserved fields are owned by the log and cannot be
overwritten by a caller:

| field | meaning |
|---|---|
| `run_id` | one invocation. A resumed sweep appends under a **new** id with `seq` restarting at 0, so the resume boundary is visible instead of disguised as one continuous run. |
| `seq` | monotone within a run, so interleaved writes stay orderable |
| `t` | seconds on a monotonic clock since `run_start` |
| `event` | `run_start`, `unit_start`, `unit_end`, `unit_failed`, `run_end` |

`run_end` always exists, including on an abort, and carries `status`
(`ok` / `failed` / `aborted`), totals, and `peak_rss_bytes`.

### Two details that are easy to get wrong

- **`ru_maxrss` units differ by platform** — kilobytes on Linux, bytes on macOS.
  `peak_rss_bytes` normalises to bytes; reading it raw would misreport memory by
  1024×. It is a coarse process-wide high-water mark, flagged
  `peak_rss_is_estimate: true`, not an allocation to one unit.
- **JSON has no NaN** — RFC 8259 does not permit `NaN` or `Infinity`, and a file
  containing them is not parseable by `jq` or by a strict reader. Nonfinite
  values are written as `null`.

### Credentials

Field names are matched against a credential pattern (`secret`, `token`,
`password`, `api_key`, `credential`, `bearer`, `private_key`) and **refused at
write time**, in both event fields and the settings block. Telemetry pipelines
are a classic accidental-exfiltration path; refusing at the sink is cheaper than
auditing every call site.

## No logging framework, no collector

`pyproject.toml` declares three runtime dependencies and this project treats an
added dependency as an "ask first" change. `json` plus an open file handle is
enough, and it keeps the artifact readable with `jq` on a cluster login node with
nothing installed. OpenTelemetry is the right answer for a service; it is the
wrong answer for a detached batch job with no collector and no dashboard.

See `docs/decisions/ADR-0004-jsonl-telemetry-no-framework.md`.

## Alerting

There is none, deliberately. A batch campaign has no on-call rotation and no
user-facing error rate to page on. The operational equivalents are:

- the campaign script is **resumable**, so the response to any failure is to
  re-launch the same output directory;
- `on_error: "record"` in `frontier_research.json` and
  `intervention_research.json` lets a handful of hard instances exhaust their
  step budget without killing an overnight run, while still recording each one
  as a row;
- `frontier_report` and `intervention_report` count failed units in the summary,
  so an exclusion cannot silently make a conditional result look unconditional.
