# G2: control headroom and the control-complexity frontier

This gate answers the project's go/no-go question:

> On the generated distribution, does instance-specific control choice buy
> anything that exceeds the resolution of the measurement itself — and where does
> a richer control family stop paying for itself?

If the answer is no, no learned policy on that distribution can produce a
contribution, and the distribution must change before any further model work.
The instrument is built so that "no" is a reachable, reportable outcome.

---

## 1. What is measured

For each record, `benchmark_record_controls` runs an equal-budget, exact-waveform
search in each declared control family. `record_headroom` then reduces that
trajectory to one row.

| Quantity | Definition | Sign |
|---|---|---|
| `linear_loss` | Loss of the matched-duration linear schedule. Every family search evaluates it as trial 0. | — |
| `best_found_loss` | Minimum loss found across all evaluated families under the declared budget. | — |
| `headroom` | `linear_loss − best_found_loss` | ≥ 0 by construction |
| `family_restriction_loss[f]` | `best_loss[f] − best_found_loss` — what restricting to family `f` costs | ≥ 0 |
| `relative_headroom` | `headroom / linear_loss`, **withheld** when the reference is near zero | ≥ 0 or `null` |
| `combined_loss_ambiguity` | Step-doubling ambiguity of the linear trial plus that of the best-found trial | ≥ 0 |
| `resolution_status` | `resolved` if `headroom > margin × combined_loss_ambiguity`, else `censored_numerical` | — |

### Headroom is a lower bound, never an optimum

A larger budget can only lower the reference, so `headroom` under budget *B* is a
lower bound on any global control gain. The row records
`headroom_is_global_optimum: false` and `reference_status:
best_found_within_evaluated_candidates`. Do not write "optimal schedule" anywhere
downstream of this file.

### Families are nested only through linear

Each family search evaluates `Schedule.linear()` as its first trial, so
`best_loss[f] ≤ linear_loss` holds for every family, and that is the *only*
nesting relation asserted. A one-window waveform is not exactly representable in
the eight-bin duration parameterisation, so one-window and eight-bin sit side by
side at equal budget, not on a ladder. `families_are_nested_only_through_linear`
records this in every row.

---

## 2. The censoring rule

See `docs/decisions/ADR-0002-censor-headroom-against-numerical-ambiguity.md`.

`score_schedule` accepts a trajectory when its step-doubling diagnostic falls
below `tolerance`, and reports `loss_ambiguity_indicator = 2e + e²`. Headroom is
a difference of two such losses. When that difference is of the same order as the
summed ambiguity, it is not evidence — but averaged over a population it is still
a positive number, and that number would otherwise reach a table.

```
combined = ambiguity(linear trial) + ambiguity(best-found trial)
resolved = headroom > ambiguity_margin * combined
```

Censored rows are kept in full and counted. `aggregate_frontier` excludes them
from every headroom statistic and reports `censored_fraction` beside each
quantile. A fully censored population yields `verdict: no_resolved_headroom`.

**The correct response to a censored population is to change the distribution or
tighten the integrator, not to lower `ambiguity_margin`.** The margin is recorded
in every row precisely so that lowering it is visible.

---

## 3. Three audits v0.2 had no place to report

| Audit | Field | Why it matters |
|---|---|---|
| Linear reference consistency | `linear_reference_consistent`, `linear_reference_spread` | The same waveform on the same Hamiltonian at the same runtime is scored once per family. Disagreement beyond ambiguity means the scoring path is not reproducible, and every downstream difference inherits that. |
| Linear incumbent respected | `linear_incumbent_respected[f]` | A family whose reported best is worse than the linear control it already evaluated discarded a feasible solution. That is an optimisation failure, and attributing it to control complexity would be wrong. |
| Low-headroom reference | `low_headroom_reference` | Normalised improvement ratios explode when `linear_loss → 0`. The ratio is withheld and the prevalence of such tasks is reported instead. |

Anything that fails lands in `audit_failures`, which `aggregate_frontier`
surfaces as `records_with_audit_failures` and `audit_failure_reasons`. Nothing is
repaired silently.

---

## 4. Running it

```bash
# Plan the budget before spending it.
python -m annealctrl control-sweep --data runs/research/data \
  --config configs/frontier_research.json --output runs/frontier_val --dry-run

# Validation split (no adaptation flag needed).
python -m annealctrl control-sweep --data runs/research/data \
  --config configs/frontier_research.json --output runs/frontier_val --report

# Resume after an interruption. Changing the budget or tolerance is refused.
python -m annealctrl control-sweep --data runs/research/data \
  --config configs/frontier_research.json --output runs/frontier_val --resume

# Aggregate separately (e.g. with a different bootstrap count).
python -m annealctrl frontier-report --sweep runs/frontier_val --bootstrap-resamples 10000
```

### Test split

```bash
python -m annealctrl control-sweep ... --split test --allow-test-adaptation
```

The flag is **command-line only**; `load_frontier_config` rejects
`allow_test_adaptation` as a configuration key, so a copied config file cannot
enable it. Results produced this way are online adaptation and must be reported
as adapted, never as zero-shot.

### Output layout

```
runs/frontier_val/
  rows.jsonl          one headroom row per record, appended; failures included
  benchmarks/*.json   every evaluated control of every family, per record
  telemetry.jsonl     run_start / unit_start / unit_end / run_end events
  manifest.json       settings, settings hash, source hash, counts, status
  report/summary.json aggregate statistics
  report/FRONTIER.md  human-readable table with the caveats attached
```

`rows.jsonl` stays streamable because full trajectories live beside it in
`benchmarks/`. Set `retain_full_trials: false` to skip those files; the per-family
`incumbent_trace` in each row is enough to draw a budget-sensitivity curve.

### Budget arithmetic

Objective calls per record are `1 + budget × (number of tunable families)` —
linear is parameter-free and is evaluated once, never padded. With
`configs/frontier_research.json` that is `1 + 64×4 = 257` propagation-scored
controls per record. Multiply by the record count before launching.

---

## 5. Screening: hardness as a measured attribute

`docs/paper_protocol.md` §3.3 requires the screening rule to be fitted on
training/validation parents only, and forbids selecting on the proposed method's
advantage. Both are enforced:

- `fit_threshold` raises on any row whose split is `test`.
- The screening quantity is restricted to `headroom`, `relative_headroom`, or
  `family_restriction_loss:<family>`. Any other name is rejected.

```bash
python -m annealctrl screen \
  --fit-sweep runs/frontier_train runs/frontier_val \
  --apply-sweep runs/frontier_test \
  --quantile 0.75 --min-headroom 0.02 \
  --output runs/screen.json
```

The result reports the unscreened population, the rule, the selected fraction,
the parent overlap (`in_sample`), and the objective-call cost of screening on
both sides. The selected subset carries
`is_unbiased_deployment_sample: false` and is described as a **conditional stress
benchmark**: it says how the method behaves given measured headroom above the
threshold, and nothing about how often such instances occur.

---

## 6. What this gate cannot tell you

- Nothing about hardness in the quantum-annealing sense. Headroom is a property
  of the declared control families, the runtime, and the simulator.
- Nothing about a global control optimum. The reference is what a Sobol +
  incumbent-local search found under a stated budget with a stated seed.
- Nothing about hardware. These are closed-system dimensionless runtimes.
- Nothing about whether a *model* can exploit the headroom. That is G3 and the
  learned-policy evaluation, and a large headroom with no learnable structure is
  a perfectly possible outcome.
