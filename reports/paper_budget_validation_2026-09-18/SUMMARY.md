# Adaptive-budget diagnostic follow-up — 18 September 2026

The optimizers now execute adaptive GP steps, but this small follow-up does **not**
establish a learned warm-start advantage. The original pilot's horizon 9 was below
the initial-design length of both GP optimizers in the eight-dimensional budget
study. This follow-up freezes horizon 25: Finzgar-inspired GP-UCB executes 15 adaptive
queries after 10 initial queries; generic GP-EI executes 8 after its charged linear
reference plus 16 design points. Sobol/local search uses the same 25-query budget.

The completed six-step campaign uses the **same** CONTROL/POLICY checkpoints from
three training seeds, six held-out records belonging to **three logical parents**,
and two optimizer seeds. It is a diagnostic decision made after the pilot, not an
independent confirmatory experiment. Budgets are trajectory prefixes
`[0,1,5,9,17,25]`. Source fingerprint:
`1ccf566c336be9e7208cafee210ecb75908347299245fff6a67f8f4a0f5427ef`.

## Completed measurements

Loss is one minus decoded logical success; lower is better. Means weight logical
parents and training seeds equally. The fixed-source schedule and selected bank
control coincide on this particular small subset; their warm-start outcomes differ
only by negligible waveform floating-point construction effects.

| Optimizer, 25 online queries | Cold | Source-global hint | CONTROL direct hint | POLICY direct hint |
|---|---:|---:|---:|---:|
| Sobol/local |0.591751|0.591382|0.590085|0.590316|
| GP expected improvement |0.588522|0.587641|0.588376|0.588703|
| Finzgar-inspired GP-UCB |0.590638|0.588912|0.591307|0.593982|

All six direct-hint-minus-cold crossed parent/training-seed descriptive intervals
contain zero. For example, CONTROL Sobol/direct minus cold is
−0.001666, CI [−0.008044,+0.002619]; POLICY GP-UCB/direct minus cold is
+0.003343, CI [−0.001606,+0.008771]. These unadjusted intervals with only three parents
and three training seeds have weak uncertainty resolution. They are not equivalence
tests. Every final-budget contrast and every curve point is in `summary.json`.

Original **zero-online-query** losses are 0.643136 for bank/source-global, 0.642097
for CONTROL direct, 0.652998 for POLICY direct and 0.659946 for linear. Projected
hints are distinct controls whose simulator evaluation is charged; these original
zero-query outcomes are never supplied to search for free.

There were **21,600 online queries and 144 separate offline diagnostic queries**,
with zero failed or interrupted queries in the completed campaign. The six steps
took 189.106 seconds in this CPU environment. Timing is descriptive and does not
establish hardware-independent speedup. Full source-study preparation cost is
reported automatically by each budget run; any emitted break-even arithmetic is
conditional on a fixed common mean-quality target and the measured grid, not an
independently validated deployment rule.

An earlier horizon 17 request was interrupted when the horizon was extended to25.
Its separate archive retains 1,622 requested calls, 1,621 completed calls and 8.719
seconds of known objective time. **One in-flight call has unknown cost.** Those
partial quality results are not used here, and the discarded diagnostic cost is a
lower bound. It is not hidden inside the completed25-query budget comparisons.

## Reproduction and audit

- `budget_validation_evidence.json.gz`: standard campaign JSON/JSONL/Markdown
  archive, preserving 2,071 files byte-for-byte with individual hashes.
- `budget_interrupted_evidence.json.gz`: separately labelled incomplete archive;
 228 files from the interrupted horizon 17 request.
- `reproduce.py --check`: checks archived file hashes and exactly reconstructs
 `summary.json`, including parent-by-training-seed contrasts and discarded work.
- `resume_verification.json`: exact resume left all 2,071 scientific artifacts
  unchanged. An owned stale lock was removed after confirming the execution
  sessions had ended; no scientific receipt was altered.
- `configs/paper_budget_validation.json`: frozen execution recipe. It requires
  the intact sibling `runs/paper_pilot_20260918` data/checkpoints. Those binary
  training inputs are referenced by hashes, not included in these compact archives.

```bash
PYTHONPATH=src python reports/paper_budget_validation_2026-09-18/reproduce.py --check
```

The follow-up validates the adaptive optimization path and its accounting. It
retains a negative scientific result: learned hints have not demonstrated a robust
advantage over cold or fixed-source initialization on this subset.
