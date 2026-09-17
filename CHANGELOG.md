## 2026-09-17 — Baselines, corrections, and one held-out table

### The comparison table

Every method on the **same 864 held-out records over 48 parents**, grouped by
what it consumes and ordered only within a group. `reports/comparison_2026-09-17/`.

| cost class | method | loss | vs linear |
|---|---|---:|---:|
| fixed | linear / global | 0.6009 / 0.5654 | — / −0.0355 |
| privileged spectrum | `d2` (n=846) / `gap_inverse_square` (n=432) | 0.5913 / 0.7750 | −0.0041 / +0.0328 |
| amortised | summary bank / best direct | **0.5447** / 0.5898 | −0.0563 / −0.0111 |
| online adaptation | search, 257 calls per instance | 0.5074 | −0.0936 |

The amortised learned selector beats both privileged spectral oracles while
needing nothing at deployment beyond a forward pass, and beats the best single
fixed schedule by 0.021. Search still leads by 0.037; that gap is printed, not
hidden. There is no global rank in the artifact, the report or the figure.

### The privileged baselines are not the ceiling

`gap_inverse_square` is the local-adiabatic rule and it had been computed for
every G2 record since 2026-09-16 without ever being aggregated. Over 3447 units
it **loses to a 64-candidate search by 0.110 and to a linear ramp by 0.0230**,
and the runtime stratification (+0.0063, +0.0625, −0.0001 at runtimes 1, 4, 12)
shows the adiabatic theorem behaving as advertised rather than a broken
implementation. This reframes the motivation: the spectral schedule is not a
ceiling the learner chases, it is a reference the search already passes.

### Architecture is not supported; information is

Holm-corrected parent-paired contrasts over all ten comparisons: every
separation in bank mode is the embedding-blind encoder losing to an
embedding-aware one, and **no aware encoder separates from any other**.
Embedding-aware against blind is −0.00782 [−0.01109, −0.00447]. `summary`, the
cheapest aware encoder, ranks first. The paper claims the information, not the
architecture.

### Dataset aggregation, against a matched control

One DAgger round moves the direct policy from 0.588815 to 0.554681 on held-out
parents. Aggregation changes the bank's size *and* its source at once, so a
`BANKEXT` arm grows the bank identically using the next points of the same Sobol
sequence: **43% of the gain is bank size**. Attributable to aggregation:
−0.0195 (seed 0) and −0.0238 (seed 1). A `CONTROL` arm retrained on the original
bank reproduced the shipped checkpoint to every printed digit, so retrain noise
is zero.

### Corrections

- The G3 **2.6x** scale-arm ratio was a composition artefact; the matched value
  is **1.52x** (ADR-0008). Its point estimate had also been pair-weighted while
  its interval was parent-weighted, putting the estimate outside its own CI.
- The **12/14/16-qubit GPU ratios** were withheld: they existed only in a commit
  message, with no artifact and no record of host load. A loaded host starves the
  NumPy arm and inflates the ratio toward "the GPU wins", so
  `scripts/backend_crossover.py` censuses the machine, refuses to run at nonzero
  nice, and repeats each size to bound timing noise.
- **CI was red** (605 passed, 10 failed) because `plot_utils.py` lives outside the
  repository, so nobody else could regenerate the figures either. Vendored to
  `src/annealctrl/_vendor/` with upstream path and sha256 recorded (ADR-0007).
  Skipping the tests was rejected: it greens the badge by testing less.

### Verification

Full suite **692 passed, 2 skipped**; CI green on a clean checkout.

---

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
  scheduler), `scripts/launch_goose.slurm` (SLURM, with job-array sharding).
  No stage requires a GPU.
- Sweep sharding (`--shard`, `--shard-count`) and multi-directory reporting, so
  the ~411k-control research intervention plan can run as a SLURM array and be
  merged afterwards. Merging shards computed under different settings is refused.
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

### Campaign

Executed on `apollo` (32 cores, RTX PRO 6000 Blackwell, CUDA 13.0), 12 concurrent
shards, **zero failed units**. Reports and figures in
`reports/campaign_2026-09-16/`, numbers in `reports/CAMPAIGN_G2_G3_RESULTS.md`.

- **G2**: headroom mean 0.106 (validation, 48 parents, CI [0.095, 0.117]) and
  0.102 (train, 144 parents). Linear wins 0 of 3456 records. Restricting to
  `two_window` costs 0.003; `eight_bin` is worse at equal budget.
- **G3**: 610 of 1060 pairs (57.5%) show a decisive preference reversal. The
  scale-controlled arm gives a larger transfer penalty than the total compiled
  effect. *(Corrected 2026-09-17: this originally read "2.6x larger", computed by
  dividing the two arms' marginal means. Those arms have different factor
  compositions and the ratio is a composition artefact. On 214 matched pairs the
  effect is **1.52x**, +0.0199 [0.0132, 0.0267]. See ADR-0008.)*
- **GPU**: 1.08x at 10 physical qubits with 1.22e-15 outcome parity; CuPy is
  capped at one worker, so CPU shards win. The host suite is 512 passed, 0
  skipped - the CuPy/CUDA parity tests recorded as never executed in
  `reports/V02_VERIFICATION.md` now run and pass.

### Not in this release

No hardness claim, no speedup claim, no hardware or QPU work, no changes to the
training contract (`learning.py` and `models.py` are untouched), and no learned
model evaluated against these measurements. Existing datasets remain valid.

### Verification

Full suite **497 passed, 2 skipped**. Baseline v0.2 (340 passed, 2 skipped)
reproduced on this machine before any change. Full campaign verified end to end
on smoke configs — dataset → G2 sweeps → reports + figures → screening → G3 sweep
→ report + figure in 19 s, and a second invocation skipped every completed unit.
