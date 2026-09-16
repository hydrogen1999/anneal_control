# Implementation plan: G2/G3 measurement instruments

Derived from `SPEC.md`. Task IDs are stable; check them off in place.
Dependency order is bottom-up; each task leaves the suite green.

## Architecture decisions

Recorded as ADRs in `docs/decisions/`:

- **ADR-0001** New measurement modules live inside `annealctrl`, not a sibling package.
- **ADR-0002** Headroom is censored against the integrator's own loss ambiguity.
- **ADR-0003** Embedding interventions change exactly one declared factor, and
  chain-strength interventions always report a scale-controlled arm.
- **ADR-0004** Telemetry is append-only JSONL with a run correlation id; no logging framework dependency.
- **ADR-0005** Sweeps resume by content fingerprint, never by row count.
- **ADR-0006** Figures render through `research-os/scripts/plot_utils.py` (fonttype 42).

## Dependency graph

```
telemetry.py ─┬─► sweeps.py ─┬─► headroom.py ──► screening.py ──► figures.py (Fig 3)
              │              └─► interventions.py ──────────────► figures.py (Fig 4)
              └─► (CLI subcommands, launchers)
generation.py scale_override ──► interventions.py
```

---

## Phase 0 — Foundation

- [x] **T0.1** `git init`, extend `.gitignore`, commit v0.2 as received.
  - Acceptance: baseline commit exists; no source file modified.
  - Verify: `git log --oneline`; suite green before the commit (340 passed, 2 skipped).
- [x] **T0.2** Write `SPEC.md`.
- [x] **T0.3** Write `PLAN.md` and ADR-0001..0006.
  - Files: `SPEC.md`, `PLAN.md`, `docs/decisions/*.md`.

### Checkpoint 0
- [x] Repository under version control, spec and plan reviewable, suite green.

---

## Phase 1 — Observability substrate

- [x] **T1.1** `src/annealctrl/telemetry.py`: `RunLog` writing one JSON object per
  line with `run_id`, monotonic `t`, `event`, and typed fields; peak-RSS sampler;
  refusal to log a field whose name looks like a credential.
  - Acceptance: events are valid JSON lines; `run_id` on every line; re-opening
    an existing log appends rather than truncates; a `close()`d log writes a
    terminal `run_end` with totals.
  - Verify: `pytest -q tests/test_telemetry.py`
  - Files: `src/annealctrl/telemetry.py`, `tests/test_telemetry.py`
  - Scope: S

- [x] **T1.2** `src/annealctrl/sweeps.py`: resumable record-level driver —
  fingerprint each unit of work from (record fingerprint, settings hash, source
  hash), skip units already present in the output JSONL, append new ones
  atomically, emit telemetry, and return a manifest.
  - Acceptance: interrupted sweep resumes without recomputation; a changed
    setting hash is refused rather than silently mixed.
  - Verify: `pytest -q tests/test_sweeps.py`
  - Depends: T1.1
  - Scope: M

### Checkpoint 1
- [x] Telemetry + resume are testable independently of any science.

---

## Phase 2 — G2: control-complexity frontier and headroom

- [x] **T2.1** `headroom.py::record_headroom` — consume one
  `benchmark_record_controls` result and produce: per-family best-found loss,
  linear reference, nested-incumbent audit, `headroom`, `family_restriction_loss`,
  combined loss ambiguity, and `resolution_status ∈ {resolved, censored_numerical}`.
  - Acceptance: censoring triggers when `headroom ≤ margin × combined ambiguity`;
    nested-incumbent violation is reported, not silently repaired.
  - Verify: `pytest -q tests/test_headroom.py -k "censor or nested or restriction"`
  - Scope: S

- [x] **T2.2** `headroom.py::sweep_control_frontier` — run
  `benchmark_record_controls` + `record_headroom` over a record subset through
  `sweeps.py`; charge every objective call; refuse `test` without the adaptation flag.
  - Acceptance: `total_objective_calls` equals the declared budget arithmetic;
    test split refused without `--allow-test-adaptation`.
  - Verify: `pytest -q tests/test_headroom.py -k sweep`
  - Depends: T1.2, T2.1
  - Scope: M

- [x] **T2.3** `headroom.py::aggregate_frontier` — parent-level means, then
  quantiles (10/25/50/75/90) and tail counts over parents, censored fraction,
  per-family restriction distribution, and paired-parent bootstrap of
  (linear − best found).
  - Acceptance: censored rows are excluded from headroom statistics but counted
    and reported; output has no mean-only summary.
  - Verify: `pytest -q tests/test_headroom.py -k aggregate`
  - Scope: M

- [x] **T2.4** CLI `control-sweep`, `frontier-report`; configs
  `frontier_smoke.json`, `frontier_research.json`; `docs/g2_headroom.md`.
  - Verify: `python -m annealctrl control-sweep --help`; end-to-end smoke on a
    generated 3-qubit dataset.
  - Scope: M

### Checkpoint 2
- [x] `control-sweep` runs end to end on a smoke dataset, resumes, and its
      aggregate distinguishes resolved from censored headroom.

---

## Phase 3 — G2b: measured hardness qualification

- [x] **T3.1** `screening.py` — fit a headroom threshold **only** on train and
  validation parents, apply it to any split, and record unscreened population,
  rule, selected fraction, and screening cost in objective calls.
  - Acceptance: fitting on test raises; applying to test is allowed and labelled
    conditional; selection never references a model's advantage.
  - Verify: `pytest -q tests/test_screening.py`
  - Depends: T2.3
  - Scope: S

### Checkpoint 3
- [x] A stress subset can be declared from train/validation evidence alone.

---

## Phase 4 — G3: paired embedding interventions

- [x] **T4.1** `generation.py` additive `scale_override` on `compile_embedding`,
  plus `conservative_common_scale(...)` helper.
  - Acceptance: with an override, `E_phys(z∘π) = α E_logical(z) + C_chain` still
    holds exhaustively on small instances; an override that breaches the declared
    caps is refused; driver and runtime untouched.
  - Verify: `pytest -q tests/test_generation.py -k scale_override`
  - Scope: S

- [x] **T4.2** `interventions.py::build_pair` — construct an (A, B) pair from one
  logical parent with exactly one declared factor changed
  (`geometry | ports | field_allocation | chain_strength`), carrying an explicit
  record of what was held fixed and whether physical size is matched.
  - Acceptance: a spec that changes two factors is refused; a size-changing
    factor must be declared as such; both arms share logical coefficients,
    decoder, driver, runtime and candidate budget.
  - Verify: `pytest -q tests/test_interventions.py -k pair`
  - Depends: T4.1
  - Scope: M

- [x] **T4.3** `interventions.py::cross_control_matrix` — equal-budget family
  search on A and on B, then execute each arm's selected control on the other;
  emit the 2×2 loss matrix, within-arm advantage, transfer penalties, tie-aware
  preferred-control swap flag, near-optimal set overlap, and a resolution status.
  - Acceptance: matrix diagonal equals each arm's own best-found loss; transfer
    penalty is signed and never clipped; swap flag is false when the two
    near-optimal sets overlap within ambiguity.
  - Verify: `pytest -q tests/test_interventions.py -k cross`
  - Depends: T4.2
  - Scope: M

- [x] **T4.4** `interventions.py::sweep_interventions` + `aggregate_interventions`
  through `sweeps.py`; CLI `intervention-sweep`, `intervention-report`; configs;
  `docs/g3_interventions.md`.
  - Acceptance: parent-level aggregation with tails and paired bootstrap of the
    transfer penalty; swap rate reported with its censored fraction.
  - Verify: end-to-end smoke; `pytest -q tests/test_interventions.py`
  - Scope: M

### Checkpoint 4
- [x] A paired-intervention sweep produces Figure-4-shaped evidence, including
      the case where nothing swaps.

---

## Phase 5 — Figures

- [x] **T5.1** `figures.py` — Figure 3 (control complexity vs best-found loss,
  with budget axis and censored cases visible) and Figure 4 (cross-control
  matrices and transfer-penalty distribution), rendered through
  `research-os/scripts/plot_utils.py`; graceful, explicit failure if that helper
  is absent.
  - Acceptance: saved PDFs contain no Type 3 font; censored/unresolved cases are
    drawn, not dropped.
  - Verify: `pytest -q tests/test_figures.py`
  - Scope: M

### Checkpoint 5
- [x] Both figures render from real sweep output with camera-ready fonts.

---

## Phase 6 — Campaign runner and launchers

- [x] **T6.1** `scripts/launch_apollo.sh` (no scheduler; `nohup`, pinned threads,
  `doctor` precheck, resumable stages) and `scripts/launch_goose.slurm` (SLURM,
  partition/account left as required edits).
  - Acceptance: both refuse to start without an explicit output directory; both
    run `doctor` first; neither assumes CUDA.
  - Verify: `bash -n` syntax check; dry-run path documented.
  - Scope: S

- [x] **T6.2** `docs/observability.md`, RUNBOOK/README updates, `CHANGELOG.md`.
  - Scope: S

### Checkpoint 6
- [x] Full suite green; campaign is launchable on `apollo` and `goose` by
      editing configuration only.

---

## Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Headroom on the current distribution is below numerical resolution | High — invalidates the learned-policy story | This is exactly what T2.1's censoring is built to detect and report. The instrument must be able to return "no signal". |
| Equal-budget search is too weak, so "best found" is a poor reference | Medium | Budget is an explicit axis; frontier output retains every trial so a budget-sensitivity curve is derivable without re-running |
| Paired interventions change physical size as a side effect | Medium | `build_pair` refuses undeclared size changes; κ pairs always carry a scale-controlled arm |
| Laptop CPU cannot reach useful scale | Certain | By design: laptop runs software verification only; campaign runs on apollo/goose |
| Sweep output grows large (every trial retained) | Low | JSONL append + per-record rows; aggregation reads streaming |

## Open questions

Tracked in `SPEC.md` §8.
