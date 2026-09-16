# Spec: G2/G3 measurement instruments for `annealctrl`

Status: **active**. Created 2026-09-16. Owner: project researcher (VCU).
Supersedes nothing. Implements the "highest-value next code work" list in
`IMPLEMENTATION_PLAN_VI.md` §7 items 3–5 and the gate definitions in
`docs/paper_protocol.md` §12 (G2, G3).

---

## 1. Objective

`annealctrl` v0.2 is a working pipeline: it generates compositional Hamiltonians,
audits labels, trains five declared ablations, evaluates on held-out parents and
emits tables. What it cannot do is answer the two questions that decide whether
the paper has a contribution at all:

> **G2.** Is there instance-specific control headroom that exceeds the numerical
> and statistical resolution of our own measurement? Where is the boundary at
> which a richer control family stops paying for itself?
>
> **G3.** Does embedding information *change the preferred control*, not merely
> change a predicted parameter — and does transferring the wrong embedding's
> control cost measurable loss?

If G2's answer is "headroom is below our resolution", no learned policy on that
distribution can produce a contribution, and the distribution must change before
any more model work. If G3's answer is "the preferred control never swaps", the
embedding-essential claim must be dropped. **Both must be allowed to fail.**

This spec covers the measurement instruments for those two questions, and
nothing else. It deliberately does not add architectures, losses, or datasets.

### Users

1. The researcher, running campaigns on `apollo` (lab box, no scheduler) and
   `goose` (SLURM). Nothing numerical runs on the laptop.
2. A reviewer reading the appendix, who must be able to see the budget, the
   seed, the waveform, the censoring rule and the failures.

### Success criteria

| # | Criterion | How it is checked |
|---|---|---|
| S1 | One command measures the control-complexity frontier over a declared record subset with equal objective-call budgets and exact switching waveforms | `annealctrl control-sweep` produces a JSONL row per record; `tests/test_headroom.py` asserts equal budgets and nested incumbents |
| S2 | Headroom smaller than the integrator's own loss ambiguity is **censored, not reported as signal** | `resolution_status` field on every headroom row; unit test drives ambiguity above headroom and asserts `censored` |
| S3 | One command measures paired embedding interventions in which **exactly one** declared factor differs | `annealctrl intervention-sweep`; unit test asserts a two-factor pair spec is rejected |
| S4 | Chain-strength interventions report both the total compiled effect and a scale-controlled arm | Both arms present in every κ pair row; test asserts `E_phys(z∘π) = α E_logical(z) + C_chain` holds under the common-scale override |
| S5 | Every pair yields a 2×2 cross-control loss matrix, a transfer penalty, and a tie-aware preferred-control swap flag | Schema test on `cross_control_matrix` output |
| S6 | Aggregates are parent-level, report tails (not only means), and carry paired-parent bootstrap CIs | `aggregate_*` tests; no mean-only summary is emitted |
| S7 | Figures embed TrueType (fonttype 42), never Type 3 | `figures.py` renders through `research-os/scripts/plot_utils.py`, which refuses to save a Type-3 figure |
| S8 | Both sweeps are resumable, provenance-stamped and emit structured telemetry | Resume test re-runs a partial sweep and asserts no record is recomputed; telemetry test asserts one JSON object per line with a run-correlation id |
| S9 | Test-split adaptation is impossible without an explicit flag, and screening thresholds cannot be fitted on test | Tests assert `ValueError` on both paths |

### Explicit non-goals

- No hardness, speedup, novelty or acceptance claim.
- No QPU submission, no vendor topology, no calibrated open-system model
  (that is G5, and it needs authorization the project does not yet have).
- No new encoder, loss, or training contract. `learning.py` and `models.py`
  are **not modified**.
- Not running the paper-scale campaign. This laptop produces software
  verification only; every number it emits is labelled as such.

---

## 2. Assumptions

Stated so they can be corrected rather than silently absorbed:

1. **Venue is not yet fixed** ("cứ làm tốt nhất"). Figure widths are therefore
   parameterised by venue (`neurips|icml|iclr|acl|cvpr|aaai`) and default to
   `neurips`. Nothing else depends on the venue.
2. **Compute**: `apollo` = lab machine, no scheduler → `nohup`/`tmux` launcher.
   `goose` = SLURM → batch script. Other VCU hosts are congested and are not
   targeted. GPU availability on each is **unknown**; launchers therefore call
   `annealctrl doctor` first and never assume CUDA.
3. The closed-system simulator remains the ground truth for this stage. Every
   causal statement is "inside the declared simulator", per protocol §7.
4. Full-spectrum teachers stay capped at 10 physical qubits. The new instruments
   must work with `teacher.mode=none` datasets too, since headroom measurement
   does not need a spectrum.
5. Primary loss remains `1 − decoded_logical_success`, as defined by
   `output_observables`. Nothing here redefines it.

---

## 3. Commands

Environment (once, on the target host):

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -e '.[test,plots]'
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu   # or CUDA build
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```

| Purpose | Command |
|---|---|
| Test | `python -m pytest -q` |
| Focused test | `python -m pytest -q tests/test_headroom.py tests/test_interventions.py` |
| Capability check | `python -m annealctrl doctor` (add `--require-gpu` on a GPU host) |
| **G2 frontier sweep** | `python -m annealctrl control-sweep --data RUN/data --split validation --config configs/frontier_smoke.json --output runs/frontier` |
| **G2 aggregate** | `python -m annealctrl frontier-report --sweep runs/frontier --output runs/frontier/report` |
| **G2 screening** | `python -m annealctrl screen --sweep runs/frontier --output runs/screen.json` |
| **G3 intervention sweep** | `python -m annealctrl intervention-sweep --config configs/intervention_smoke.json --output runs/interventions` |
| **G3 aggregate** | `python -m annealctrl intervention-report --sweep runs/interventions --output runs/interventions/report` |
| Dry plan (budget only) | any sweep command with `--dry-run` |

All sweep commands accept `--resume`; none silently overwrite an output.

---

## 4. Project structure

Existing layout is kept. New files only:

```
src/annealctrl/
  telemetry.py        NEW  structured JSONL events, run correlation id, peak RSS
  headroom.py         NEW  G2: per-record headroom + frontier aggregation
  screening.py        NEW  G2: train/validation-only hardness qualification
  interventions.py    NEW  G3: paired embeddings + cross-control matrices
  figures.py          NEW  Type-3-free Figure 3 / Figure 4 rendering
  sweeps.py           NEW  resumable record-level sweep driver shared by G2/G3
  generation.py       EDIT one optional `scale_override` argument, default None
  workflow_cli.py     EDIT five new subcommands
tests/
  test_telemetry.py test_headroom.py test_screening.py
  test_interventions.py test_sweeps.py test_figures.py           NEW
configs/
  frontier_smoke.json frontier_research.json
  intervention_smoke.json intervention_research.json             NEW
docs/
  g2_headroom.md g3_interventions.md observability.md            NEW
  decisions/ADR-0001..0006.md                                    NEW
scripts/
  launch_apollo.sh launch_goose.slurm                            NEW
```

`generation.py` is the only pre-existing science module touched, and only by an
additive optional keyword. `physics.py`, `spectral.py`, `learning.py`,
`models.py`, `pipeline.py` are **not** modified.

---

## 5. Code style

Match the existing house style exactly: module docstring states the scientific
boundary; dense validation at the top of each public function; `ValueError` for
contract violations and `ArithmeticError` for numerical-gate failures; results
returned as JSON-safe dicts with non-finite values mapped to `null`; comments
explain *why* a guard exists, never restate the code.

```python
def record_headroom(benchmark: Mapping[str, Any], *, ambiguity_margin: float = 1.0) -> dict:
    """Headroom of one record, censored when it cannot exceed its own numerics.

    ``headroom`` is linear loss minus the best found across the evaluated
    families. It is a *finite-budget* quantity: a larger budget can only lower
    the reference, so this is a lower bound on any global control gain and must
    never be called an optimum. When the differenced losses carry a combined
    step-doubling ambiguity of the same order, the row is marked ``censored``
    and MUST be excluded from headroom claims rather than reported as a small
    positive effect.
    """
```

Naming follows the repo: `snake_case`, explicit `*` keyword-only boundaries,
no abbreviations that are not already used in the codebase (`kappa` is spelled
`chain_strength`, `alpha` is `programmed_scale`).

---

## 6. Testing strategy

`pytest`, tests in `tests/`, no network, no GPU required. The suite must stay
under ~60 s on a laptop CPU, so every new test uses 3–6 physical qubits.

| Level | Share | What it covers here |
|---|---|---|
| Small (pure, no I/O) | ~80% | headroom arithmetic, censoring rule, pair-spec validation, transfer-penalty algebra, telemetry record shape |
| Medium (tmp dirs, real simulator on ≤6 qubits) | ~20% | sweep resume, cross-control matrix against the real propagator, figure files written and font-checked |
| Large | 0 | none — the campaign is a remote experiment, not a test |

Rules specific to this project:

- Every new numerical claim gets a test that would fail if the guard were
  removed. Censoring, equal budgets, single-factor pairs and the compilation
  identity under scale override are all such guards.
- Prefer the real propagator over a stub wherever a 3-qubit instance runs in
  milliseconds; use a synthetic `loss_fn` only when testing search bookkeeping.
- No test may write outside `tmp_path`.
- A bug fix lands with a reproduction test first (Prove-It).

---

## 7. Boundaries

**Always**

- Run `python -m pytest -q` before every commit.
- Keep the parent as the unit of independence in any aggregate.
- Record budget, seed, family, waveform and split beside every loss.
- Label anything produced on this laptop as software verification.
- State the scientific boundary in each new module's docstring.

**Ask first**

- Changing `physics.py`, `spectral.py`, `learning.py`, `models.py`, or the
  record schema in `pipeline.py`.
- Adding a runtime dependency.
- Anything that would invalidate an already-generated dataset fingerprint.
- Running anything on a remote host.

**Never**

- Fit a screening threshold, tune a hyperparameter, or choose a tolerance using
  test-split information.
- Clip a negative teacher-relative regret to zero.
- Substitute a nearest bank label for an un-evaluated proposal's loss.
- Report a headroom that is below its own numerical ambiguity as a positive
  result.
- Call a best-found control a global optimum, or a toy graph a hardware graph.
- Submit a QPU job or contact any external service.

---

## 8. Open questions

| # | Question | Blocking? | Default taken meanwhile |
|---|---|---|---|
| Q1 | Target venue and deadline | No | `neurips` figure geometry; all gates built |
| Q2 | Do `apollo`/`goose` have CUDA GPUs? | No | Launchers probe with `doctor`; CPU path always valid |
| Q3 | Is there authorized D-Wave topology/calibration data? | No (G5, out of scope) | Toy graphs only, explicitly labelled |
| Q4 | Paper-scale parent count | No | Diagnostic configs sized for profiling; research configs left as budget *requests* |
