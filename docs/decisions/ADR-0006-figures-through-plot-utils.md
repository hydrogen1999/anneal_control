# ADR-0006: Render figures through `research-os/scripts/plot_utils.py`

## Status
Accepted

## Date
2026-09-16

## Context

Matplotlib's default `pdf.fonttype` is 3. Type 3 fonts are rejected outright by
IEEE PDF eXpress and flagged by NeurIPS, ICML and CVPR submission systems. No
skill or tool on this machine sets `pdf.fonttype = 42` — including the ones that
claim to audit figure quality — so a figure can pass every review pass and still
be stopped by the submission system.

`~/research-os/scripts/plot_utils.py` exists precisely for this: it sets
`fonttype = 42`, provides per-venue column widths so a figure dropped in at
`scale=1.0` keeps the font size it was given, uses an Okabe–Ito colourblind-safe
palette, and refuses to save a figure that still embeds a Type 3 font.

## Decision

`figures.py` imports `plot_utils` from `~/research-os/scripts` and renders every
figure through `use_venue(...)` / `save(...)`. Venue defaults to `neurips` and is
configurable. If the helper is absent, `figures.py` raises an explicit error
naming the path — it never falls back to raw matplotlib defaults.

## Alternatives considered

**Set `rcParams['pdf.fonttype'] = 42` locally.** One line, and it solves the font
problem alone. Rejected: it duplicates a maintained helper, skips the post-save
verification, and leaves column widths and palette to be re-derived per figure.

**Vendor a copy of `plot_utils.py` into the repo.** Would make the repository
self-contained for a collaborator without `research-os`. Rejected for now
because two copies drift; revisit if the repository is shared externally, at
which point vendoring with a recorded upstream revision is the right move.

**Silent fallback to matplotlib defaults when the helper is missing.** Rejected:
it produces a figure that looks fine and fails at submission, which is the exact
failure this ADR exists to prevent.

## Consequences

- Figure generation depends on a path outside the repository. This is recorded
  in `docs/g2_headroom.md` and the failure mode is loud.
- Figures are venue-parameterised, so a venue change is a flag, not a rewrite.
- `matplotlib` stays an optional extra; `figures.py` is imported lazily by the
  CLI so the core package still installs with three dependencies.
