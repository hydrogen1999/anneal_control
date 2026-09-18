# Upstream transfer aggregate audit — 2026-09-18

Upstream commit `974207aa24cb8ea02cf800649e0fa354a43f7587` adds `reports/pegasus_learned_2026-09-18/transfer.json`. This is new measured aggregate evidence, not a completed independently auditable transfer campaign. Its commit message describes synthetic 3–10-qubit training and Pegasus 10–14-qubit evaluation, five encoders and three seeds. The file contains five method lists of three aggregate entries, without seed identities or checkpoint hashes.

Input SHA-256: `6a0c022627a0491f5de0051f7aab7a5690773ce4589b306fabcd638947e456b8`. The original JSON is preserved unchanged. This audit checks its arithmetic and interval/verdict consistency; it does not reconstruct raw outcomes or bootstrap new intervals.

## Descriptive method averages

Every entry reports 144 target records and 12 logical parents, linear loss 0.7070519901, and best-in-bank loss 0.6115838701. Matching counts and means do not independently prove identical record identities. Best-in-bank is an outcome-informed finite-bank diagnostic, not a global control optimum.

| Method | Mean selected loss across three entries | Mean selected − linear | Entries whose archived CI excludes zero in favour of selection |
|---|---:|---:|---:|
| hierarchy_physics | 0.658554 | -0.048498 | 3/3 |
| hierarchy_outcome | 0.663326 | -0.043726 | 3/3 |
| physical | 0.670822 | -0.036230 | 2/3 |
| summary | 0.674128 | -0.032924 | 2/3 |
| logical | 0.705839 | -0.001213 | 1/3 |

The hierarchical methods have favourable point estimates, and each of their three archived entry-level parent intervals favours selection over linear. Summary and physical each have one interval crossing zero. Logical has mixed entry-level effects. These are descriptive observations; the arithmetic mean does not have a recoverable pooled confidence interval from this file. Entries are not presumed to be independent test populations, and all 15 intervals are pointwise rather than corrected as one family.

## All archived entry-level intervals

Entry numbers are one-based array positions, **not seed IDs**. Differences are selected loss minus linear loss; negative is favourable. Each archived interval reports 8,000 parent resamples and explicitly excludes training-seed uncertainty.

| Method | Array entry | Difference | Archived 95% parent CI | Interpretation |
|---|---:|---:|---|---|
| summary | 1 | -0.004714 | [-0.024154, +0.015091] | not_separated |
| summary | 2 | -0.044478 | [-0.073708, -0.016327] | CI_favours_selector |
| summary | 3 | -0.049581 | [-0.078691, -0.021616] | CI_favours_selector |
| logical | 1 | +0.012609 | [-0.012853, +0.039781] | not_separated |
| logical | 2 | -0.039801 | [-0.055049, -0.025689] | CI_favours_selector |
| logical | 3 | +0.023552 | [-0.019801, +0.073548] | not_separated |
| physical | 1 | -0.036811 | [-0.053418, -0.020601] | CI_favours_selector |
| physical | 2 | -0.024041 | [-0.047005, +0.001171] | not_separated |
| physical | 3 | -0.047838 | [-0.069618, -0.028589] | CI_favours_selector |
| hierarchy_outcome | 1 | -0.025901 | [-0.044360, -0.008167] | CI_favours_selector |
| hierarchy_outcome | 2 | -0.048880 | [-0.068456, -0.029683] | CI_favours_selector |
| hierarchy_outcome | 3 | -0.056397 | [-0.080406, -0.035507] | CI_favours_selector |
| hierarchy_physics | 1 | -0.032121 | [-0.048986, -0.014265] | CI_favours_selector |
| hierarchy_physics | 2 | -0.051501 | [-0.074024, -0.030861] | CI_favours_selector |
| hierarchy_physics | 3 | -0.061872 | [-0.086731, -0.039853] | CI_favours_selector |

## Two legacy verdict errors

The summary entry 1 interval is [−0.024154, +0.015091]; physical entry 2 is [−0.047005, +0.001171]. Both store `separated: false` but `verdict: beats_linear`. Their intervals do not establish improvement. This is the legacy verdict bug, not evidence of a successful corrected-code run. A non-separated result also does not establish equivalence to linear.

## What is still required

The aggregate JSON contains no raw paired record/parent outcomes, training-seed IDs, checkpoint hashes, source/target dataset manifests, logical fingerprints, or guard receipts. The commit message reports zero overlap under a content fingerprint, but the historical guard used compiled-record identity rather than logical-problem identity. This artifact cannot independently establish logical-parent disjointness. No leakage is demonstrated by the absence of provenance; the required validation remains missing.

There is no target evaluation of the fixed global baseline learned on the source, and no archived paired hierarchy-versus-summary contrast. The apparent ranking reversal is a hypothesis worth testing; it does not establish hierarchy superiority or an architecture contribution. Separate per-method intervals against linear cannot answer the between-method question. Do not assign seed IDs from array positions, bootstrap the three means as if they were independent parents, average CI endpoints into a pooled CI, or infer Holm decisions from these summaries.

Rerun the corrected transfer protocol using explicit source checkpoints, frozen source normalizers, disjoint logical fingerprints, shared target rows, the source-selected global baseline, and retained per-record/per-parent/per-seed outcomes. Then compute parent and crossed parent-by-seed contrasts with a declared multiplicity family. Until those artifacts are available, the new result is **promising aggregate transfer evidence with incomplete provenance**.

After this upstream input is present in the working tree, reproduce this audit with `python scripts/rebuild_evidence.py`; `--check` verifies it alongside the other evidence outputs. `UPSTREAM_TRANSFER.json` contains the machine-readable arithmetic audit and all 15 archived interval interpretations.
