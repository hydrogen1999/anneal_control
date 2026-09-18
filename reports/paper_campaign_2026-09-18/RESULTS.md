# Bounded paper campaign: measured pilot results

All **11 configured stages completed** on CPU. The controlled measurements are mostly negative or inconclusive: policy acquisition does not improve direct selection reliably, family transfer gains are matched by a fixed source-selected schedule, and the pilot establishes no amortization break-even. Successful execution validates the experimental workflow; it does not establish an A* main-track contribution or guarantee acceptance.

Frozen source fingerprint: `1ccf566c336be9e7208cafee210ecb75908347299245fff6a67f8f4a0f5427ef`. Campaign archive: `pilot_evidence.json.gz`; SHA-256 `300feca0b50a26e7dce79b750e53cf7533f9a9f8448fa8635965b45380548d3d`. Observed campaign execution time was 212.734 seconds; this is one CPU run, not a scaling benchmark.

## Design and independent units

Source: **48 logical parents**, split 32 train / 8 validation / 8 test; 96 records, 16 candidate schedules each. Logical sizes 3–4, physical sizes 3–6, synthetic spin-glass/weighted-MaxCut families, runtime 1 or 3. The summary encoder has width 32 and four proposals, at most 30 epochs with patience 8. Training seeds are 0, 1, 2. There are **33 fits**: three frozen baselines, twelve primary-arm fits, and eighteen mechanism continuations. The embedded historical `_scope` string incorrectly says 60 parents; the actual manifest and configuration specify 48.

Each primary acquisition arm adds four schedules to each of 64 training records: 256 objective labels per arm/seed, 2,304 labels across BANKEXT, decoder-random, and POLICY. Validation/test banks stay frozen. Mechanism arms reuse acquired labels and hold the encoder fixed. Source evaluation adds 3,840 offline diagnostic objective calls; those are not deployment queries.

Intervals below are descriptive crossed parent×training-seed bootstraps with 2,000 resamples, conditional on this dataset and recipe. They are unadjusted and not equivalence tests; three training seeds and eight test parents provide limited precision.

## Four acquisition arms

| Arm | Bank loss | Direct loss |
|---|---:|---:|
| control | 0.709660 | 0.720478 |
| bankext | 0.710495 | 0.724739 |
| decoder_random | 0.709660 | 0.729601 |
| policy | 0.710977 | 0.725893 |

Global fixed schedule: 0.709660; linear: 0.735915. Lower loss is better.

| Direct-mode contrast | Difference and 95% crossed CI |
|---|---|
| policy_minus_control | +0.005416 [+0.000014, +0.017507] |
| policy_minus_bankext | +0.001154 [-0.004890, +0.008823] |
| policy_minus_decoder_random | -0.003708 [-0.010236, +0.000610] |

POLICY is worse than CONTROL on all three seed averages. Its descriptive interval only narrowly excludes zero; this is not a multiplicity-adjusted finding. Neither POLICY−BANKEXT nor POLICY−decoder-random excludes zero. These outcomes must remain visible rather than selecting the favourable historical aggregation campaign.

| Frozen-backbone mechanism: acquired − original labels | Direct difference and 95% crossed CI |
|---|---|
| critic | -0.000888 [-0.005584, +0.001781] |
| policy | -0.001927 [-0.008058, +0.002066] |
| heads | -0.004339 [-0.011919, +0.000727] |

All direct mechanism intervals contain zero. Critic/head bank-mode intervals reach zero; policy-only continuation leaves the bank predictions unchanged as expected. No causal repair of critic distribution shift or proposal generation is established by this pilot.

## Transfer with source-selected global control

CONTROL and POLICY checkpoints from all three training seeds were evaluated with frozen source normalizers. The global waveform was selected from eight source-validation parents, without target labels. Every target evaluation has eight logical parents: 16 source-held-out records, 16 unseen-family (`weak_field`) records, or eight unseen-runtime (`T=8`) records. Eighteen checkpoint-target evaluations completed with zero online simulator queries. This is synthetic family/runtime transfer, not topology or larger-qubit transfer.

| Target | Checkpoint arm | Bank loss | Selector − source-global: difference and 95% crossed CI |
|---|---|---:|---|
| source_heldout | control | 0.709660 | +0.000000 [+0.000000, +0.000000] |
| source_heldout | policy | 0.710977 | +0.001317 [+0.000000, +0.006274] |
| unseen_family | control | 0.685273 | +0.000000 [+0.000000, +0.000000] |
| unseen_family | policy | 0.685273 | +0.000000 [+0.000000, +0.000000] |
| unseen_runtime | control | 0.500023 | -0.050679 [-0.145888, +0.000000] |
| unseen_runtime | policy | 0.557616 | +0.006913 [+0.000000, +0.027653] |

On unseen family, both learned selectors and source-global obtain 0.685273; the −0.050568 improvement over linear is therefore not evidence of added learned per-instance value. At runtime 8, CONTROL−linear is −0.022316 [−0.108558, +0.056430], and POLICY−linear is +0.035277 [−0.007527, +0.063986]. CONTROL’s favourable mean relative to global comes entirely from training seed 2 and its interval reaches zero. Transfer is not established as robust across training seeds.

The separate upstream synthetic→Pegasus aggregates in commit `974207a` concern different checkpoints, instances, and sizes. They are audited in [UPSTREAM_TRANSFER.md](../evidence_audit_2026-09-18/UPSTREAM_TRANSFER.md) and are not pooled with this pilot.

## Six budget studies

One report per CONTROL/POLICY checkpoint × three training seeds. Each uses the **same six source-test records from three parents**, optimizer seeds 0 and 1, budgets 0/2/5/9, and one eight-bin family. Search seeds are not extra training seeds. Each search trajectory supplies nested budget prefixes; budget points and repeated cold-start runs across checkpoints are not independent replications.

| Checkpoint | Direct at 0 calls | Sobol direct-warm at 9 | GP-EI label, direct-warm at 9 | GP-UCB label, direct-warm at 9 |
|---|---:|---:|---:|---:|
| budget_control_0 | 0.653149 | 0.596719 | 0.603131 | 0.617629 |
| budget_control_1 | 0.657495 | 0.597299 | 0.603131 | 0.618104 |
| budget_control_2 | 0.615646 | 0.592222 | 0.603023 | 0.603894 |
| budget_policy_0 | 0.653548 | 0.596856 | 0.603131 | 0.617724 |
| budget_policy_1 | 0.657286 | 0.597271 | 0.603131 | 0.618174 |
| budget_policy_2 | 0.648160 | 0.599522 | 0.603131 | 0.618242 |

The cold-start losses at nine queries are Sobol 0.609837, GP-EI-labeled 0.602952, and GP-UCB-labeled 0.618101. Bank warm starts yield 0.597805, 0.601833, and 0.614906, respectively—but source-global warm starts reproduce those values within 4.3×10⁻⁹. Thus these bank-warm gains do not isolate an ML advantage. The direct results vary with checkpoint; no pooled superiority over source-global is established.

**Initialization coverage matters.** In these budget studies, Finžgar GP-UCB uses ten initial points and never reaches an adaptive query at maximum budget nine. The generic eight-dimensional GP-EI implementation also stays within its initial-design phase. Their algorithm labels must not be interpreted as evidence about Bayesian adaptation. The separate [25-query diagnostic follow-up](../paper_budget_validation_2026-09-18/SUMMARY.md) exercises both adaptive optimizers using the same checkpoints; it is not part of this frozen pilot.

Each budget report records 1,296 online search calls plus 24 offline diagnostics, with no failed calls. Across six reports: 7,776 online calls and 144 diagnostics. Whole-study measured offline preparation is 76.964 seconds, shared across these reports and not six separate costs; it includes all acquisition/mechanism fits, not a minimal single-method training cost. No fully observed cold/warm pair attains a common requested threshold of 0.4, 0.5, or 0.6, so **all break-even estimates are unavailable**, not zero or infinite measured advantages.

## Literature-oriented adapted baseline

The separate literature campaign compares the Finžgar-inspired GP-UCB adaptation against uniform random on **four parents/eight records**, optimizer seeds 0/1/2, independent budgets four and eight: 96 optimizer runs, 576 completed objective calls, no failed queries. It is explicitly an embedded-task adaptation with reduced settings, not reproduction of the published numerical study. Here `n_initial=4`, so budget four uses only initialization while budget eight exercises four adaptive queries.

| Budget | GP-UCB − random | 95% crossed parent×optimizer-seed CI |
|---|---:|---|
| 4 | +0.000000 | [+0.000000, +0.000000] |
| 8 | -0.000292 | [-0.006267, +0.004761] |

The four-query tie follows from shared initialization. The eight-query interval includes zero; only 200 bootstrap resamples were configured. No literature-baseline efficacy claim is supported by this pilot.

## Reproduce and inspect

```bash
python reports/paper_campaign_2026-09-18/reproduce.py
python reports/paper_campaign_2026-09-18/reproduce.py --check
```

The compact `results.json` includes all six budget curves and contrasts, source/transfer contrasts, input hashes, counts, and cost scopes. `pilot_evidence.json.gz` contains raw text reports, trajectories, receipts, manifests, and logs; `smoke_evidence.json.gz` is a separate integration run. The archive omits trained-weight binaries, dataset NPZ files, and figures. Hashes identify those omitted files but are not substitutes for them: table reconstruction works from the archive, while exact model replay requires regenerating or retrieving the binaries. No completed QPU, 16–20-qubit, or confirmatory efficacy result is claimed.
