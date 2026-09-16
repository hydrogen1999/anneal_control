# ADR-0001: Keep the G2/G3 measurement instruments inside `annealctrl`

## Status
Accepted

## Date
2026-09-16

## Context

G2 (control headroom) and G3 (paired embedding interventions) need code that
reads generated records, calls the simulator, and writes provenance-stamped
results. Three homes were possible: a sibling package, a `scripts/` directory of
analysis programs, or new modules inside `src/annealctrl/`.

Two existing facts constrain the choice. First, `experiments.py` freezes a
source hash over the installed package and refuses to resume a run whose source
drifted; measurement code that lives outside that hash would be silently
unversioned relative to the data it interprets. Second, `benchmark_record_controls`,
`score_schedule`, `record_physics` and `optimize_control_family` are already the
correct primitives and are internal APIs, not a published interface.

## Decision

New modules `telemetry.py`, `sweeps.py`, `headroom.py`, `screening.py`,
`interventions.py`, `figures.py` live in `src/annealctrl/`, and their CLI entry
points are added to `workflow_cli.py`.

## Alternatives considered

**Sibling analysis package (`annealctrl_analysis`).** Cleaner dependency story,
but it falls outside `source_hash()`, so a change to the headroom rule would not
invalidate a resumed run that depends on it. Rejected: silent analysis drift is
exactly the failure mode the repository is built to prevent.

**Loose scripts under `scripts/`.** No import surface to test, no resume
contract, and the existing `scripts/` content is deliberately unimported
templates. Rejected: these instruments need unit tests more than the pipeline does.

## Consequences

- The source fingerprint now covers measurement logic, so changing a censoring
  rule correctly invalidates dependent resumes. This is intended.
- The package gains six modules; `pyproject.toml` is unchanged (no new runtime
  dependency — `matplotlib` stays optional and is imported lazily).
- `physics.py`, `spectral.py`, `learning.py`, `models.py` and the record schema
  in `pipeline.py` are untouched, so every existing dataset stays valid.
