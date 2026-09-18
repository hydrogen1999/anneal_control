# Evidence ledger

This is the current claim index. New code, passing tests, and a planned campaign
are not completed experimental evidence. The authoritative numerical reanalysis
is [evidence_audit_2026-09-18/RESULTS.md](evidence_audit_2026-09-18/RESULTS.md).
Rebuild or verify it with `python scripts/rebuild_evidence.py [--check]`.
Historical JSON outcomes are preserved; superseded analyses and figures must not
be copied into a paper without the corrections below.

| Claim | Archived evidence | Supported scope and limitation |
|---|---|---|
| Instance-specific schedule search helps | Validation headroom 0.1061, parent CI [0.0953, 0.1172], 48 parents. [Campaign](CAMPAIGN_G2_G3_RESULTS.md). | Finite-budget simulator search at 3–10 physical qubits. Best-found control is not a global optimum. |
| Embedding changes control preference | 610/1060 pairs show decisive reversal in the small campaign; 123/228 (53.9%) at 14 qubits on 20 parents. [14-qubit report](interventions14_2026-09-17/SCALE.md). | Report the denominator and censoring. Independent generated populations support replication, not a matched size trend or hardware claim. |
| Scale preservation affects the embedding intervention | Matched small-system effect ratio 1.52, difference +0.0199 [0.0132, 0.0267], 214 pairs. | The old 2.6× ratio mixed factor populations and is withdrawn. |
| Learned bank selection improves on a fixed global schedule | Synthetic summary/bank − global = −0.020775, CI [−0.027734, −0.014167], 48 parents; Pegasus = −0.022905 [−0.038362, −0.010572], 12 parents. [Reanalysis](evidence_audit_2026-09-18/RESULTS.md). | Closed-system held-out evaluation. Intervals condition on archived training-seed averages. Pegasus training and test both use Pegasus; this is not cross-topology transfer. |
| Teacher comparison remains positive after excluding audit failures | Summary/bank − d2 = −0.050399 [−0.069051, −0.032866], 711 records/45 parents; minus gap schedule = −0.083240 [−0.114283, −0.052837], 339 records/23 parents. [Paired tables](evidence_audit_2026-09-18/RESULTS.md). | Each teacher uses its own audited population, paired against the learned method on exactly those records. Different cost classes; no general spectral-method dominance. The old 846/432 eligibility counts are invalid. |
| Embedding information is useful on the synthetic bank task | Aware − blind = −0.007817 [−0.011093, −0.004471], 48 parents. [Reanalysis](evidence_audit_2026-09-18/RESULTS.md). | Exploratory pooled contrast, not Holm-adjusted. On Pegasus the interval [−0.020195, +0.001062] includes zero. Architecture superiority and equivalence are both unestablished. |
| Aggregate transfer measurements are now archived | Commit 974207a reports synthetic→Pegasus bank selection: mean of three hierarchy_physics entries 0.658554 versus linear 0.707052; hierarchy_outcome 0.663326, physical 0.670822, summary 0.674128, logical 0.705839. [Aggregate audit](evidence_audit_2026-09-18/UPSTREAM_TRANSFER.md). | Provisional aggregate evidence. Both hierarchy methods have three favourable entry-level intervals; summary and physical have only two each. No raw paired rows, checkpoint/seed identities, logical-disjointness receipts, or global baseline are archived. No between-method architecture claim or pooled CI is established. |
| Direct policy remains weak | Synthetic summary/direct − global = +0.025096; Pegasus +0.055564. [Reanalysis](evidence_audit_2026-09-18/RESULTS.md). | Lower loss is better. Bank selection currently provides stronger evidence than direct generation. |
| Historical aggregation is promising but not an isolated train-only effect | DAGGER − BANKEXT = −0.022555, negative on all three seeds. [Historical campaign](dagger_2026-09-17/DAGGER.md). | Both training and validation banks changed. Raw parent-by-seed panels are absent; a new paired CI cannot be reconstructed. Deterministic CONTROL replay does not imply zero optimizer uncertainty. The [independent fixed-validation pilot](acquisition_pilot_2026-09-17/README.md) is negative and retained. |
| Increasing proposal count has mixed evidence | 16 − 3 proposals: −0.00380 [−0.00835, +0.00032], 26/48 parents. [Report](proposals16_2026-09-17/PROPOSALS16.md). | Mixed encoder pool, unequal seed counts. No pooled separation; no equivalence or general saturation conclusion. Not directly comparable to the summary-only aggregation effect. |
| Schedule-search headroom exists on hardware connectivity up to 14 qubits | Pegasus validation 0.1441 [0.1047, 0.1891]; Zephyr 0.1479 [0.1150, 0.1867]. [Pegasus](pegasus_2026-09-17/PEGASUS.md), [Zephyr](zephyr_2026-09-18/ZEPHYR.md). | Small subgraphs, no QPU. Per-size uncertainty does not establish no decay or constancy. Pegasus family winner counts are per record: two_window wins 12/24 records, not 12/12 parents. |
| Three search algorithms provide stronger internal baselines | Same 864 records and 257 calls: Bayesian 0.505460, Sobol-local 0.507352, corrected policy gradient 0.511392. [Reanalysis](evidence_audit_2026-09-18/RESULTS.md). | Generic GP-EI and REINFORCE, not faithful reproductions of a specific paper. Only one archived search outcome per record/strategy; search-seed variability is unmeasured. The buggy PG 0.5130 result is withdrawn. |
| Literature-oriented implementation is available | [Finžgar adaptation/reference smoke](literature_baseline_smoke_2026-09-17/SUMMARY.md). | Embedded-task GP-UCB is an explicitly adapted baseline. The original-system reduced-space reference wins over random in only 1/3 seeds; this is not reproduction of the paper's full numerical study. |
| GPU throughput has a limited observed crossover | Archived 12-qubit ratio 2.83×; [crossover.json](backend_2026-09-17/crossover.json). | Raw timing profiles are not archived and parity deltas are null. Machine-idleness statements are author reports, not independently verifiable from this artifact. No verified 16–20-qubit throughput result. |
| Small dephasing perturbations preserve searched-control ordering | Ten 4–6-qubit records; searched control wins all ten at rates 0, .02, .05, .1. [Report](open_system_2026-09-17/OPEN_SYSTEM.md). | One hand-chosen noise axis. No measured device calibration and no learned-policy robustness evaluation. |

## Completed controlled CPU pilot (18 September)

The [new bounded campaign](paper_campaign_2026-09-18/RESULTS.md) completed all
11 stages, 33 fits and three training seeds with fixed validation/test data.
POLICY−BANKEXT direct loss is +0.001154, descriptive crossed CI
[−0.004890, +0.008823]; all three direct mechanism intervals contain zero.
Unseen-family bank selection matches the source-global schedule. The six
nine-query budget studies establish no break-even at their requested thresholds
and stay inside Bayesian initialization. These are completed measurements with
negative/inconclusive conclusions, not missing experiments. The separate
[25-query diagnostic extension](paper_budget_validation_2026-09-18/SUMMARY.md) reuses the same data and checkpoints, so it is
not an independent confirmatory study.

Raw pilot reports and receipts, with per-file hashes, are archived in
`paper_campaign_2026-09-18/pilot_evidence.json.gz`; the reproduction script
rebuilds its tables without model binaries. The archive explicitly omits model
weights and dataset NPZ files. The larger research configuration has passed
[geometry/split preflight](paper_campaign_2026-09-18/RESEARCH_PREFLIGHT.md),
but its GPU simulation and training have not been executed.

## What remains unestablished

A strong method claim needs reproducible positive evidence from controlled
acquisition or warm-start experiments; the bounded completed campaign does not
supply that evidence. Generalization beyond the small synthetic pilot requires
the larger held-out distributions. Deployment value requires successful
quality–budget comparisons that include offline costs. The new transfer aggregates need provenance and paired contrasts before a
confirmatory cross-distribution claim. Full-paper-scale literature reproduction,
learned robustness, 16–20-qubit learned evaluation, and QPU outcomes are not established
by current archives. Code implementing these protocols does not change that
status until raw completed outcomes are archived and audited.

All intervals must name the unit being resampled. The fresh reanalysis resamples
logical parents, conditional on archived seed averages; it cannot recreate
unarchived training uncertainty. Failure to reject a contrast is not evidence
of equality, and a sample-size estimate made after seeing test effects is not a
guarantee of future significance.
