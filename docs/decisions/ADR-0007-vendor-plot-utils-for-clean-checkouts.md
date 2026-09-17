# ADR-0007: Vendor `plot_utils.py` into the package, with upstream kept canonical

## Status
Accepted — amends ADR-0006, which stays in force.

## Date
2026-09-17

## Context

ADR-0006 requires every figure to render through
`~/research-os/scripts/plot_utils.py`, because matplotlib's default
`pdf.fonttype` is 3 and Type 3 fonts are rejected by IEEE PDF eXpress and
flagged by NeurIPS/ICML/CVPR. That decision is correct and is not revisited
here.

It had a consequence nobody measured until CI ran on a machine that is not this
laptop: `~/research-os` is outside the repository, so a clean checkout cannot
import it. The public CI run reported **605 passed, 10 failed, 3 skipped**, with
all ten failures in `tests/test_figures.py` raising `RuntimeError: plot_utils.py
not found`. Two things were wrong with that state at once:

1. **The repository advertised a red build.** The physics and training tests
   passed, but a reader cannot tell that from a red badge, and a reviewer
   checking the artefact sees failures.
2. **Figures were not reproducible by anyone else.** A paper whose figures can
   only be regenerated on one particular laptop has a reproducibility gap, and
   the gap was undeclared.

The error message written for ADR-0006 already named the remedy: *"Set
ANNEALCTRL_PLOT_UTILS to the directory containing it, or vendor a copy with its
upstream revision recorded."*

## Decision

Vendor `plot_utils.py` to `src/annealctrl/_vendor/`, inside the package so it
ships with a non-editable install, and resolve it in this order:

1. `ANNEALCTRL_PLOT_UTILS`, if set — **strictly**. If that directory has no
   `plot_utils.py`, raise. An explicit pointer that gets silently replaced by a
   fallback is how figures end up rendered by something other than what the
   author asked for.
2. `~/research-os/scripts`, if present. Upstream stays canonical.
3. The vendored copy.

`load_plot_utils()` sets `__annealctrl_vendored__` on the module so any caller
can record which copy rendered a figure. The vendored file carries its upstream
path, revision and sha256 in its header, and a test asserts that provenance
header exists — a vendored file with no recorded origin cannot be audited or
re-synced.

## Alternatives Considered

### Skip the figure tests when `plot_utils` is absent
- Pros: one line, CI turns green immediately.
- Cons: turns green by testing less. It fixes the badge and leaves the actual
  problem — that nobody else can regenerate the figures — in place, now hidden
  behind a skip rather than visible as a failure.
- Rejected.

### Reimplement an equivalent inside the package
- Pros: no vendored third-party file.
- Cons: two implementations of the same Type-3 guarantee that can drift apart,
  so a figure rendered in CI would not be the figure rendered for the paper.
- Rejected.

### Install `research-os` as a CI dependency
- Pros: keeps exactly one copy.
- Cons: it is a private path, not a published package; a public CI job cannot
  fetch it, and an external reader still cannot reproduce a figure.
- Rejected.

## Consequences

- CI covers figure rendering on a clean machine, and the Type-3 guarantee is
  tested rather than assumed.
- The vendored copy can drift from upstream. The recorded sha256 makes drift
  detectable, and precedence means upstream always wins where it exists, so
  drift cannot silently affect figures rendered on this laptop.
- `third_party/` at the repository root was considered and rejected: it is not
  importable from an installed package.
