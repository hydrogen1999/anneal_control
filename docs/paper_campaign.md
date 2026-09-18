# Controlled annealing: execution and evidence protocol

This campaign tests whether labelled policy proposals improve amortized control,
whether frozen selectors transfer to unseen problems, and when learned initial
controls reduce online search expenditure. Conference acceptance, architecture
superiority and favorable experimental results are not software guarantees.

## Start from one frozen checkout

```bash
python -m pip install -e '.[ml,test,plots,hardware]'
python -m annealctrl doctor
python -m annealctrl paper-campaign --config configs/paper_campaign_smoke.json --output runs/paper_smoke --dry-run
python -m annealctrl paper-campaign --config configs/paper_campaign_smoke.json --output runs/paper_smoke
```

`smoke` exercises generation, 33 training fits, transfer, all three search
strategies with cold/learned/fixed starts, the literature adaptation, and learned
noise robustness. Its tiny models, few parents and four training epochs are
integration checks, not efficacy evidence.

```bash
python -m annealctrl paper-campaign --config configs/paper_campaign_pilot.json --output runs/paper_pilot
```

`pilot` is a bounded CPU experiment with three training seeds, 48 source parents,
fixed validation, independent unseen-family/runtime datasets and all six
mechanism continuations. Budget searches cover a predeclared small subset and
short horizons; these choices are in the frozen configuration. They must be
reported as exploratory.

```bash
python -m annealctrl doctor --require-gpu
python -m annealctrl paper-campaign --config configs/paper_campaign_research.json --output runs/paper_research --dry-run
python -m annealctrl paper-campaign --config configs/paper_campaign_research.json --output runs/paper_research --through source
python -m annealctrl paper-campaign --config configs/paper_campaign_research.json --output runs/paper_research --resume
```

The research configuration requests five training seeds, a larger independent
parent population, Pegasus and Zephyr targets, held-out family/runtime targets,
16/18/20-qubit targets, and budget curves for CONTROL and POLICY checkpoints.
It requires a working CUDA PyTorch/CuPy environment installed for the machine's
CUDA version. `--dry-run` validates orchestration; it does not certify memory or
runtime feasibility. Run the existing backend profiling protocol before a large
generation job. The density-matrix noise step deliberately limits itself to
six physical qubits and records every cap exclusion.

Configuration files contain explicit seeds, simulator tolerances, training
epochs, label counts and solver limits. Change them **before** starting a fresh
campaign. A completed stage is reused only after its artifacts are hash checked.
An interrupted stage retains its attempts and delegates resumption to its own
ledger. Source changes require a new output directory; do not edit a checkout
while it is running. Run on a separate checkout if development continues.

## What each experiment identifies

| Experiment | Controlled comparison | What it can establish |
|---|---|---|
| Primary acquisition | CONTROL, BANKEXT, DECODER_RANDOM, POLICY, same initialization/recipe, original validation/test banks | Effect of query source at a fixed added-label count |
| Mechanisms | Original vs POLICY labels for critic-only, policy-only and joint-head continuations | Effects within a fixed representation; shared encoder/attention frozen |
| Transfer | Frozen checkpoints and source normalization/global schedule, unseen logical coefficient fingerprints | Performance under the declared target shift |
| Budget curves | Common eight-bin search family, same query horizons, cold/bank/direct/source-global starts | Finite-budget quality and online time; projected controls are explicitly different from original learned waveforms |
| Literature | Finzgar-inspired GP-UCB vs uniform, identical initial points, charged evaluations | A declared adaptation to this task; the independent original p-spin reference is separate |
| Noise | Decisions frozen before outcomes, linear/source-global/bank/direct under dephasing and computational-basis relaxation | Small-system perturbation sensitivity; no thermal or device claim |

The mechanism experiment holds the representation fixed. It does not identify
the effect of training a shared encoder on acquired labels, and its fixed-epoch
continuations are compared to matching original-label continuations rather than
the primary from-scratch arms. Raw best-proposal and ranking-regret diagnostics
separate proposal quality from selection quality.

`candidate_seed` fixes the common waveform bank independently of the logical
problem seed. This lets independent source and target data contain the same
source-selected global waveform without reselecting it on target labels.
The default falls back to `seed`, preserving historical generation recipes.

Modern checkpoints record logical parent identities derived from coefficients.
Source association requires exact validation content. Acquired training labels
may differ from the original source bank; this is recorded explicitly rather
than claiming the original dataset reconstructs augmented training. Legacy
checkpoints require independently verified complete source provenance and may
be refused. Exact coefficient fingerprints do not prove gauge/isomorphism
disjointness.

## Costs and inference

Zero online outcome calls means the deployed selector does not query the
simulator. Offline dataset generation, acquired labels, training, tuning and
evaluation still cost work. Study-backed cost receipts conservatively charge
the full predeclared acquisition campaign, including control/ablation fits.
This is not the minimum cost to train one method. An interrupted or unrecorded
stage makes complete amortization claims unavailable; unknown costs never
become zero. Break-even points are descriptive comparisons at preregistered
quality thresholds and are absent when no common qualified operating point
exists.

Logical parents are the sampling units; record variants are paired within
parents. Training seeds and optimizer seeds are distinct. Parent/seed intervals
are descriptive and retain negative results. A CI crossing zero is inconclusive;
failure to reject is not equivalence. Family, size and runtime targets are
filtered using declared metadata before scoring, not observed difficulty.

Start the paper tables from `reports/evidence_audit_2026-09-18/RESULTS.md`.
Historical figures with unmatched teacher populations are superseded. Rebuild:

```bash
python scripts/rebuild_evidence.py
python scripts/rebuild_evidence.py --check
```

Use `campaign.json` to locate the actual completed studies and their own
`summary.json` or `report.json`, plus raw rows. A plan file or a stage marked complete certifies
execution, not a positive hypothesis. Report direct-versus-bank failures,
inconclusive transfer and unsuccessful warm starts with the same visibility as
positive outcomes. Select the paper's primary claim before inspecting final
test results; subsequent recipe changes need a fresh holdout campaign.

## Remaining empirical work

Full research training, larger GPU trajectories and calibrated QPU measurements
must be executed on the corresponding resources. This repository provides
explicit local experiments and offline hardware adapters; it contains no QPU
submission step. A budget-matched literature adaptation does not reproduce a
published result on its original distribution. Run the documented independent
p-spin reference when reporting original-system reproduction.

## Executed integration and pilot evidence (18 September 2026)

The complete smoke and exploratory pilot campaigns were executed on CPU with
package source hash `1ccf566c336be9e7208cafee210ecb75908347299245fff6a67f8f4a0f5427ef`.
See [the measured results](../reports/paper_campaign_2026-09-18/RESULTS.md).
The pilot retained negative and inconclusive acquisition findings; it does not
establish superiority of POLICY or a new architecture. The newly supplied
Pegasus transfer aggregates are audited separately in
[UPSTREAM_TRANSFER.md](../reports/evidence_audit_2026-09-18/UPSTREAM_TRANSFER.md).

Compact archives preserve JSON/JSONL evidence byte for byte, including raw
outcomes, frozen configurations, query receipts and artifact hashes. They omit
binary weights, dataset NPZ files and figures; hashes do not replace those
artifacts. Full runs remain reproducible through the campaign command above.
To archive a completed run:

```bash
python scripts/archive_paper_campaign.py --run runs/paper_pilot --output pilot_evidence.json.gz
```

The original pilot's nine-query budget curves stay inside the initial design for
the two Bayesian methods. The separate diagnostic extension uses 25 queries,
so both GP-EI and GP-UCB reach adaptive queries. It reuses the same checkpoints
and test subset, and is not independent confirmatory evidence. To reproduce it,
first run the pilot into the exact sibling directory `runs/paper_pilot_20260918`:

```bash
python -m annealctrl paper-campaign --config configs/paper_campaign_pilot.json --output runs/paper_pilot_20260918
python -m annealctrl paper-campaign --config configs/paper_budget_validation.json --output runs/paper_budget_validation_20260918
```

The research geometry and split checks are recorded in
[RESEARCH_PREFLIGHT.md](../reports/paper_campaign_2026-09-18/RESEARCH_PREFLIGHT.md).
These do not measure GPU memory, throughput or outcome quality.

Measured extended-budget results and its interrupted-attempt accounting are in
[the diagnostic report](../reports/paper_budget_validation_2026-09-18/SUMMARY.md).
Both archive-based reproduction commands are also run in CI.
