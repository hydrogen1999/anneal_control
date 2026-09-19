## 2026-09-18 — What noise does to optimisation, and a filter that survives it

### The result the controls did not kill

**The control that is best in a noiseless simulator is systematically the one an
environment erodes most.** Within a record, closed-system quality predicts
degradation at ρ = −0.9081 raw. Two artefacts could have manufactured that and
both were removed:

| | ρ | negative in |
|---|---:|---:|
| raw | −0.9081 | 200/200 |
| ÷ headroom — removes the [0,1] ceiling | −0.7925 | 198/200 |
| candidate-demeaned — removes the shared bank's waveform identity | −0.9320 | 200/200 |
| **both** | **−0.6312** | **192/200** |

The second control matters more than it sounds. All 558 test records at ≤6
qubits draw from **one** bank, and two candidates win 80% of them, so two
fragile waveforms could have carried the whole correlation. Removing each
candidate's own mean strengthens the effect instead of dissolving it. The
reports now quote −0.63 and not the flattering −0.91.

Not dephasing-specific: under amplitude relaxation ρ ≈ −0.39 in 78/100 records,
though half as strong and, unlike dephasing, not growing with rate. Not a
small-system artefact either: the doubly-controlled figure is **−0.63, −0.60,
−0.58** at 6, 7 and 8 physical qubits, and 8 is the last size the density
solver accepts. `reports/erosion_2026-09-18/`.

This is erosion, not inversion — ρ(loss at 0, loss at 0.1) = +0.83 and the
noiseless favourite still wins 78.5% of the time. It simply gives up the most.

**And the dramatic reading of it is false.** "Past some point, optimising
harder against a noiseless simulator is self-defeating" was tested directly:
searching to budget 64 and re-evaluating every budget's incumbent under
dephasing 0.1, the noisy curve is **monotone**, the best budget is the largest
one tested, and spending it costs 0.00000 in 0 of 32 parents. Repeated at rates
0.2, 0.3 and 0.5: monotone every time.

Erosion buys a **discount, not a reversal**, and the discount is a
dose–response curve. The retained fraction of the noiseless gain falls
**86.5% → 75.1% → 65.5% → 50.4%** across those four rates — at 0.5 half of
what the search buys is erased, and the other half is still a gain.
`reports/overoptimisation_2026-09-18/`.

### The learned selector is more robust than exact noiseless optimisation

Retention of the noiseless advantage at dephasing 0.1: noise-aware oracle
82.5%, learned 69.5–74.5%, **exact noiseless argmax 61.5%**. The learned rule
sits between them having never seen noise. Paired retention contrast excludes
zero for all three seeds (+8.1% to +13.6%). Every advantage-vs-linear interval
excludes zero at every rate. `reports/learned_robustness_2026-09-18/`.

### Model + search, closed loop — and the control that halves the credit

A critic inside the search loop, budget measured in simulator calls and
verified unchanged (`budget_mismatches: 0`):

| across 5 configurations | mean | spread |
|---|---:|---|
| filtering total | +0.00622 | +0.00087 … +0.00829 — **9.5×** |
| random chooser on the identical stream | +0.00137 | **−0.00397 … +0.00348 — changes sign** |
| **the critic alone** | **+0.00485** | +0.00360 … +0.00615, every interval excludes zero |

Reporting search seed 0 alone would have over-credited the model by 50%;
seed 2 alone would have looked like failure. A third arm splits what the critic
buys: **tail-avoidance 75%** (+0.00334), **fine ranking 25%** (+0.00111), both
excluding zero. Replicated on real Pegasus connectivity (+0.00530, 10/12
parents). Online overhead 0.91% — one simulator call buys 742 surrogate scores.
`reports/closed_loop_2026-09-18/`, `reports/surrogate_filter_2026-09-18/`.

The prerequisite behind it: the critic ranks **search-generated** waveforms at
ρ = +0.9269, better than the +0.834 it manages on its own bank. On Pegasus the
coarse ranking transfers (+0.8163, 44/44 records) while the fine ranking does
not (+0.1613 among the best 10%) — which is exactly why 75/25 matters.

### Scale, measured rather than argued

- **GPU generation reaches 20 physical qubits**: 0.3277 labels/s, numerical
  gate passed, norm error 2.9e-14. Per-two-qubit slowdown 1.60×, 1.36×, 3.74×.
  Memory is not the limit (16 MB against 97 GB); time is, near 24–26 qubits.
- **Entanglement**: peak S grows at 0.073 nats/qubit, 21% of the volume-law
  rate but not an area law, so χ ≈ 52 at 50 qubits and ≈2×10³ at 100. **MPS
  plausibly reaches 50–100 qubits, not device scale** — and six points over
  5–10 qubits is a weak basis for that extrapolation, which the report says.
- **The Lindblad wall is 8 qubits**, timed: 1.16 s/solve at 6, 5.12 at 7, 30.4
  at 8, refused at 9. The erosion result therefore **cannot** be replicated on
  Pegasus, whose smallest records are 10 qubits. Stated as a gap.

### Corrections

- `docs/scale_ceiling.md` claimed "1.13× at 10 qubits". No artifact produces
  1.13 and the sign was backwards: the archived 10-qubit run has the **GPU
  1.43× slower**. Corrected.
- Every GPU ratio in the ladder is an **upper bound**. Two 14-qubit replicates
  disagree by 13%, and the variance is entirely CPU-side (numpy walls differ
  11%, cupy walls 1.4%) on a host at load 34–55 with 32 cores. A load
  correction puts the 16-qubit 89.5× nearer 57×.
- The first robustness report said the mechanism was **not** established, on
  the strength of one proxy (dwell time, pooled ρ = +0.0097). The direct test
  establishes it. Corrected in a separate commit.
- Analyses that produced published numbers have moved out of `/tmp` into
  `scripts/erosion_channels.py` and `scripts/critic_ranking_diagnostics.py`,
  and were re-run from the committed code to confirm they reproduce.

### Also

`docs/paper_skeleton.md` — the logic chain, every cell naming its artifact,
four consistency checks passing, and a severity list that currently reads
0 critical / 2 major / 3 minor.

1007 tests passing.

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
