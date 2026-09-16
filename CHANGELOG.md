# Changelog

## [0.3.0] - 2026-09-16 — G2/G3 measurement instruments

v0.2 could generate data, train five declared ablations and evaluate on held-out
parents. It could not answer the two questions that decide whether the paper has
a contribution. This release adds those instruments. It adds **no** results: the
numbers quoted below are software verification on 4–6 physical qubits.

### Added

- **`headroom.py` (G2)** — control-complexity frontier per record, with headroom
  censored against the integrator's own loss ambiguity; parent-level aggregation
  reporting quantiles and tails, paired-parent bootstrap, and a
  `no_resolved_headroom` verdict when nothing survives censoring.
- **`screening.py` (G2b)** — hardness qualification by *measured* control gain.
  Refuses to fit a threshold on test rows; restricts the screening quantity to
  headroom, its normalised form, or a family restriction loss.
- **`interventions.py` (G3)** — single-factor paired embedding interventions,
  cross-control 2×2 loss matrices, signed transfer penalties, and a tie-aware
  preferred-control swap test requiring both directions to be decisive.
- **`sweeps.py`** — resumable driver keyed by
  `sha256(record fingerprint, settings hash, source hash)`.
- **`telemetry.py`** — append-only JSONL run log with correlation id and
  normalised peak RSS.
- **`figures.py`** — Figure 3 and Figure 4 rendered through
  `research-os/scripts/plot_utils.py`, verified free of Type 3 fonts.
- CLI: `control-sweep`, `frontier-report`, `screen`, `intervention-sweep`,
  `intervention-report`.
- Configs: `frontier_{smoke,research}.json`, `intervention_{smoke,research}.json`.
- Launchers: `scripts/run_campaign.sh`, `scripts/launch_apollo.sh` (no
  scheduler), `scripts/launch_goose.slurm` (SLURM). No stage requires a GPU.
- Docs: `SPEC.md`, `PLAN.md`, `docs/g2_headroom.md`, `docs/g3_interventions.md`,
  `docs/observability.md`, `docs/decisions/ADR-0001..0006`.

### Changed

- `generation.compile_embedding` accepts an optional `scale_override` that may
  only **tighten** the declared h/J caps. It scales `H_Z` only; the driver and
  the runtime are untouched. Added `conservative_common_scale`.
- `generation.synthetic_lift` accepts an optional `port_rng`.
- `generation.compile_embedding` accepts an optional `coupling_rng`.

### Fixed

Two random-stream confounds of the same shape, both found by building the
intervention audit rather than by reading the code:

- **`synthetic_lift` confounded chain shape with port placement.** Chain edges
  and boundary ports drew from one random stream, so a `random_tree` shape
  consumed draws a `path` did not and silently relocated the ports. Any
  "geometry" intervention was therefore also a port intervention.
- **`compile_embedding` confounded field allocation with coupler allocation.**
  `field_distribution="concentrated"` consumes draws `"uniform"` does not, so
  with `coupling_distribution="random"` a field intervention also re-allocated
  the inter-chain couplers. Invisible at `ports: 1`, where a Dirichlet over one
  element is always `[1.0]`.

Tests pin both the fixes and the original confounds, so neither parameter can be
removed as redundant.

The coefficient audit also surfaced a scientific issue: `weighted_maxcut` and
`planted_loops` have zero logical fields, so a `field_allocation` intervention on
them is empty. Those pairs previously ran and contributed meaningless zero
penalties to the aggregate; they are now skipped and counted (148 of 1208
candidate pairs in `intervention_research.json`).

### Not in this release

No results, no hardness claim, no speedup, no hardware or QPU work, no changes to
the training contract (`learning.py` and `models.py` are untouched), and no
paper-scale campaign. Existing datasets remain valid.

### Verification

Full suite **497 passed, 2 skipped**. Baseline v0.2 (340 passed, 2 skipped)
reproduced on this machine before any change. Full campaign verified end to end
on smoke configs — dataset → G2 sweeps → reports + figures → screening → G3 sweep
→ report + figure in 19 s, and a second invocation skipped every completed unit.
