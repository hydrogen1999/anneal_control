# Frozen transfer evaluation

The transfer evaluator asks whether a **frozen finite-bank critic** selects useful controls on unseen logical problems. The normalizer, network, candidate contract and source-global waveform are fixed before target evaluation. It does not retrain or select models using target outcomes. A held-out source split is an in-distribution reference, not evidence of distribution shift.

## Leakage repair and legacy checkpoints

Compiled record hashes change with runtime, embedding, generator settings and source revision. They cannot identify a logical parent across datasets. Dataset-local `parent_id` values are also insufficient. The new guard recomputes `generation.logical_fingerprint(IsingProblem(h, edges, J))` on every target and compares with **both source training and validation parents**. The fingerprint normalizes edge ordering/orientation and signed zero; it is an exact labelled-coefficient identity, not a graph/gauge-isomorphism certificate. Family metadata cannot hide duplicates.

New training checkpoints include:

- per-split logical fingerprint sets and counts;
- the complete parent-ID-to-logical-fingerprint mapping;
- record IDs, full training/validation content digests and the earlier provenance fields.

Training refuses an identical logical problem under different train/validation IDs. Transfer refuses incomplete modern provenance; it never downgrades to names or compiled hashes.

Older checkpoints need the independently archived source dataset. The study loads that dataset using `pipeline.load_records`, which verifies manifest consistency, record file hashes and logical coefficients. Migration then verifies full checkpoint training/validation content digests. For older artifacts without content digests it requires complete matching compiled-record fingerprint sets **and** both source split memberships. Only after this source-to-checkpoint association is proven are logical identities recovered from the actual source coefficients. IDs alone are refused. Legacy augmented checkpoints require exact augmented source records for migration. Modern augmented checkpoints can use the original source dataset only after the complete logical maps, record IDs and compiled fingerprints match and the original validation content matches exactly. The result explicitly records that augmented training label content was not reconstructed or verified; it does not pretend that the base dataset was the complete training input.

Known inference checkpoint schemas 1 and 2 are supported with explicit source-normalizer validation. Unknown versions, missing normalizers, nonfinite predictions and incomplete provenance fail closed. Target normalization is never fitted.

## Baselines and outcomes

The target bank must include a genuine linear ramp on the shared nine-knot grid. The nearest available waveform is not a linear baseline. Candidate shapes, finiteness, endpoints, monotonicity and success-loss ranges are checked.

The source-global baseline minimizes the **equal-logical-parent average loss on source validation only**. Its exact waveform must be present in every target bank. If it is missing, generation must label that waveform or rebuild a matching frozen bank; the evaluator does not choose another control using target outcomes. The target-bank oracle is a diagnostic upper reference, never an online selection method.

Use the same `candidate_seed`, candidate count and control contract across source and target generation, while using distinct dataset `seed` values to obtain independent logical problems. This separates the control-bank definition from the logical-instance random seed. Different source and target banks otherwise confound model transfer with a different decision set.

Every row preserves record ID, recomputed logical fingerprint, method, training seed, selected candidate index, selected loss, exact linear loss, source-global loss and bank-best loss. No target loss is used as an input to the selector. Reading stored labels for retrospective scoring does not turn those labels into free deployment-time information.

## Study and declared axes

`transfer-study` evaluates all predeclared checkpoints on every target. Checkpoints must have the same declared seeds across methods, and each seed must match its saved training seed. All checkpoint provenance checks happen before inference. A source-data association is verified even for modern checkpoints so the global baseline cannot silently come from an unrelated dataset. Modern augmented training labels may differ, but original validation content must match exactly; this distinction is saved in the manifest.

Supported axes:

| Axis | Check |
| --- | --- |
| `topology` | Observed source and target topology descriptors differ |
| `family` | No target family occurs in source fitting/selection records |
| `physical_size` | No target physical qubit count occurs in source fitting/selection records |
| `runtime` | No target runtime occurs in source fitting/selection records |
| `heldout_parents` | Same observed topology and target family/size/runtime support contained in source; otherwise refused |

Optional target filters `families`, `physical_sizes` and `runtimes` are frozen in the configuration and applied before evaluation. A config with `physical_size` and overlapping source/target sizes fails, rather than turning a partly seen size range into an unseen-size claim. All targets use their test split. Several axes may change together; topology transfer alone does not isolate the causal effect of topology.

The output contains `manifest.json`, one raw `rows/*.json` per checkpoint/target, immutable request/result receipts in `attempts/`, and `summary.json`. Configuration, source code, input checkpoint and dataset-manifest hashes are frozen. Resume refuses altered inputs, completed raw files or cost receipts. Both request and result receipt hashes are bound in the manifest. An orphan receipt from a crash before registration fails closed for manual audit instead of silently entering the cost summary. Failed/retried attempts remain in the cost ledger; missing result receipts mark interrupted attempts and make known elapsed cost an explicit lower bound.

## Estimand and uncertainty

Within each training seed, records are averaged within logical parent, then parents receive equal weight. This applies to **all** reported means, including linear and source-global means. A single checkpoint's paired parent bootstrap is conditional on that checkpoint. The `beats_linear` verdict requires a strictly negative upper CI endpoint and at least two parents. A negative sample mean or majority of record wins alone is insufficient. Non-rejection is reported as `inconclusive`, never equivalence.

For multiple seeds, the study reports each seed separately and a crossed parent-by-training-seed bootstrap for selector minus linear and selector minus source-global. The parent-by-seed panel must be complete. Seeds are not treated as independent new test datasets; few seeds yield poorly resolved optimization uncertainty. These intervals are **descriptive and unadjusted**, conditional on the fixed source corpus and recipe. The study deliberately emits no global positive verdict across many methods/targets. The paper must predeclare its primary contrast and multiplicity family, or present the full family with the project's corrected inference tools. Source-global is the stronger fixed-control comparator; a win only over linear does not establish instance-specific learning benefits over source selection.

## Cost interpretation

The study uses zero online simulator calls and performs no incremental training. Its measured elapsed time includes checkpoint loading, provenance checks, graph construction and inference; it is not a clean deployment latency benchmark. The ledger reports original source and target stored-label counts separately, without equating them to original integration/propagation work. Training time and source-generation time remain unknown unless explicitly supplied with an evidence note under `offline_costs`; absence never means zero. The matched budget study provides the quality-versus-online-query comparison and amortization calculation.

## Commands

An executable smoke reference, using an existing source experiment configuration:

```bash
annealctrl run --config configs/experiment_smoke.json --output runs/transfer_source_smoke
annealctrl transfer-study --config configs/transfer_smoke.json --output runs/transfer_smoke
annealctrl transfer-study --config configs/transfer_smoke.json --output runs/transfer_smoke --resume
```

The first command produces the two summary checkpoints and original dataset expected by `transfer_smoke.json`. The transfer smoke evaluates held-out source parents and is explicitly not OOD evidence.

For a research run, `configs/transfer_research.json` names five seeds for summary, logical, physical and hierarchy-outcome encoders, plus held-out-source, Pegasus and Zephyr targets. Prepare these datasets/checkpoints with a common candidate bank before invoking:

```bash
annealctrl transfer-study --config configs/transfer_research.json --output runs/transfer_research
```

Adjust paths before the first run; JSON paths resolve relative to the config file. The integrated paper campaign can generate sources/targets and fill checkpoint paths automatically. Freeze model choices and target definitions before opening results. Do not tune on a transfer test and keep calling the same test unseen.

Unit/integration tests cover changed runtime/source/config on the same logical parent, independent parents with colliding names, validation overlap, legacy source verification, forged/incomplete provenance, source-only baseline selection, frozen normalization, invalid banks, parent weighting, CI verdicts, complete multi-seed panels, resume corruption, source drift and failed-attempt accounting. Test success validates these contracts; it is not an empirical generalization result.
