# G3: paired embedding interventions

This gate tests the representation claim, and the claim is causal:

> Does changing embedding information change **which control is preferred** —
> not merely change a predicted schedule parameter — and does importing the wrong
> embedding's control cost measurable loss?

A model whose predicted knots move when the embedding moves has shown nothing:
both controls may be equally good. The evidence that counts is a preference
reversal with a measurable transfer penalty. This instrument is built so that
"the preference never changes" is a reachable, reportable outcome.

---

## 1. One factor, and only one

`build_pairs` takes a declared `factor` and a `change` dictionary with exactly
one key, and that key must be the one the factor implies:

| factor | changes | physical size |
|---|---|---|
| `geometry` | `shape` (`path` / `star` / `random_tree`) | unchanged |
| `ports` | `ports` (boundary couplers per logical edge) | unchanged |
| `field_allocation` | `field_distribution` (`uniform` / `concentrated`) | unchanged |
| `chain_strength` | `chain_strength` (κ) | unchanged |
| `chain_length` | `chain_lengths` | **changes — must be declared** |

Held fixed on both arms, and recorded in `held_fixed` on every pair: logical
coefficients, logical graph, decoder, driver family, runtime, control families,
objective budget, search seed, numerical tolerance.

### The declaration is verified, not trusted

After construction, `_audit_single_factor` inspects what actually moved in the
physical graph and raises if it does not match the declaration:

- a `geometry` pair whose **boundary ports** moved is not a geometry intervention;
- a `ports` pair whose **chain shape** changed is not a port intervention;
- a `field_allocation` or `chain_strength` pair whose graph changed at all is not
  a coefficient intervention.

This audit caught a real confound during development. `synthetic_lift` drew the
chain edges and the boundary ports from one random stream, so a `random_tree`
shape consumed draws that a `path` did not — and changing the shape silently
relocated the ports. `synthetic_lift` now accepts an independent `port_rng`, and
`tests/test_generation.py` keeps both the fix and the original confound visible.

### Coefficient interventions are audited on the coefficients, not just the graph

`field_allocation` and `chain_strength` leave the physical graph untouched, so a
graph check alone proves nothing about them. The audit therefore un-scales both
arms' compiled coefficients by their programmed scale and checks that **only** the
intended raw coefficient moved:

| factor | must move | must not move |
|---|---|---|
| `field_allocation` | `h` | `problem_J`, `chain_J` |
| `chain_strength` | `chain_J` | `h`, `problem_J` |

This caught the second confound of the same shape as the port one:
`field_distribution="concentrated"` consumes random draws that `"uniform"` does
not, so with `coupling_distribution="random"` a field intervention also
re-allocated the inter-chain couplers. `compile_embedding` now accepts an
independent `coupling_rng`, and `build_pairs` passes one. The confound is
invisible at `ports: 1` — a Dirichlet over one element is always `[1.0]` — which
is exactly why the audit exists rather than a one-off test.

It also caught something scientifically larger: `weighted_maxcut` and
`planted_loops` generate **zero logical fields**. Redistributing zero changes
nothing, so a `field_allocation` intervention on those families is empty. Before
the audit those pairs ran anyway and contributed meaningless zero penalties to
the aggregate; now they raise `VacuousIntervention`, are skipped, and are counted
in `plan["vacuous_skipped"]`. On `configs/intervention_research.json` that is 148
of 1208 candidate pairs.

### Vacuous versus confounded

A `random_tree` can happen to reproduce the path on a short chain, and on a
two-member chain `path` and `star` give the identical single edge. That pair is
**empty**, not wrong, so it raises `VacuousIntervention`; the planner skips and
counts it in `plan["vacuous_skipped"]`. A *confounded* pair raises a plain
`ValueError` and always aborts. The two are different exception types precisely
so a planner cannot skip the dangerous one.

---

## 2. The scale-controlled arm

`compile_embedding` derives a common coefficient scale
`α = 1 / max(1, max|h|/h_limit, max|J_total|/j_limit)`. Chain strength enters
`J_total` through the intra-chain penalty, so raising κ generally lowers α: a κ
intervention moves the penalty **and** the global scale of `H_Z`.

Therefore, whenever α differs between the arms — always for `chain_strength`,
and whenever it happens for any other factor — `build_pairs` emits a second pair:

| scale arm | meaning |
|---|---|
| `total_compiled_effect` | each arm uses its own α, as a device would |
| `scale_controlled` | both arms compiled at one conservative common α, so the factor moves and the global scale does not |

Both are reported. Neither is called "the" effect, and `aggregate_interventions`
refuses to pool them — it emits `by_scale_arm` and no pooled penalty at all.

`scale_override` may only *tighten*: an override that would push a coefficient
past the declared caps is refused, because the caps are the compiled program's
feasibility contract. It scales `H_Z` only; the driver and the runtime are
untouched (`scale_applies_to: H_Z_only`).

---

## 3. The cross-control matrix

Each arm gets an equal-budget, exact-waveform search over the same families with
the same seed. Then each arm's selected control is executed on the other:

```
              executed on A     executed on B
control A       A_on_A            A_on_B
control B       B_on_A            B_on_B
```

```
transfer_penalty_on_A = B_on_A − A_on_A     (importing B's control into A)
transfer_penalty_on_B = A_on_B − B_on_B
```

**Penalties are signed and never clipped.** A negative value means the imported
control beat this arm's own best found, which says the equal-budget search on
this arm was the weaker of the two — information about the search that clipping
to zero would hide.

### When is it a swap?

A swap requires **both** directions to be decisive against their own numerical
ambiguity, using the same censoring rule as G2 headroom:

| `resolution_status` | condition | counts as a swap? |
|---|---|---|
| `resolved` | both penalties exceed `margin × combined ambiguity` | yes, unless the two selected waveforms are identical |
| `one_sided` | exactly one direction is decisive | no — reported separately |
| `censored_numerical` | neither is | no |

This is tie-aware without needing near-optimal set machinery: if importing the
other arm's control costs nothing measurable on one side, the two controls are
effectively tied there, and no clean preference reversal exists.

### Three different populations

The report keeps them apart on purpose, because dividing one by another would be
wrong:

| column | population |
|---|---|
| `pairs` | every pair |
| `incl.` | decisive in at least one direction — the penalty statistics |
| `swaps / resolved` | decisive in **both** directions — the swap fraction |

---

## 4. Running it

```bash
# Plan only: parents, pairs, factors, vacuous skips, objective-call budget.
python -m annealctrl intervention-sweep --config configs/intervention_research.json \
  --output runs/interventions --dry-run

python -m annealctrl intervention-sweep --config configs/intervention_research.json \
  --output runs/interventions --report

# Resume; changing the search budget or tolerance is refused.
python -m annealctrl intervention-sweep --config configs/intervention_research.json \
  --output runs/interventions --resume

python -m annealctrl intervention-report --sweep runs/interventions --bootstrap-resamples 10000
```

`configs/intervention_research.json` plans **96 parents, 1060 pairs, ~411k
propagation-scored controls** (148 further candidate pairs are skipped as
vacuous). Profile before launching; this is a cluster job, not a laptop job.

The driver is single-process, so split that plan across a job array and merge
afterwards:

```bash
sbatch --array=0-15 --export=ALL,OUTPUT=$HOME/runs/campaign_v1 scripts/launch_goose.slurm

python -m annealctrl intervention-report \
  --sweep $HOME/runs/campaign_v1/interventions_shard_* --figures
```

Shards are assigned by position in the id-sorted plan, so they are disjoint and
cover everything regardless of the order each task enumerates the plan in.
Merging shards computed under different settings is refused.

### Parents and splits

Parents are split with the same `parent_splits` rule the training pipeline uses,
and only `train`/`validation` parents are used by default. Intervention pairs on
test parents share logical objectives with the held-out evaluation, so
`--allow-test-parents` is required to build them and the choice is recorded in
`plan.json` as `includes_test_parents`.

### Output layout

```
runs/interventions/
  rows.jsonl      one cross-control matrix per pair, failures included
  plan.json       parents, splits, factors, vacuous skips, budget
  telemetry.jsonl run/unit events with peak RSS
  manifest.json   settings, settings hash, source hash, counts, status
  report/summary.json
  report/INTERVENTIONS.md
```

---

## 5. What this gate cannot tell you

- Nothing about real hardware. Every causal statement is **inside the declared
  closed-system simulator** (`causal_scope` on every row). A toy lifted graph is
  not a commercial topology, and `is_commercial_hardware` is `false` throughout.
- Nothing about a global control optimum: both arms' references are best-found
  under a declared budget and seed.
- Nothing about whether a *learned model* responds to the intervention. That is a
  separate comparison — run the same pairs through logical-only and
  physical-aware checkpoints and compare their responses to the same
  intervention. The pairs built here are the instrument for it.
- Nothing about frequency in deployment: the factors, their magnitudes, and the
  base configuration are all declared choices in the config.
