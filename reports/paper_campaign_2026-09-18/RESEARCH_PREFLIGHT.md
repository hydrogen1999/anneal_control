# Research campaign configuration and geometry preflight

**Status: passed configuration/planning checks; research performance remains unmeasured.**

Date: 2026-09-18. Campaign: `configs/paper_campaign_research.json`. No GPU jobs, quantum-state propagation, outcome labelling or training were run in this preflight. Source/config files were not edited.

The campaign was resolved with `load_campaign`. Every generation config was passed through the actual `_validate_config`, `_plan_parents` and `_split_parents`. Every planned parent and embedding variant was constructed using the production generator and compiled at the first declared chain strength. The production `_endpoint_budget` estimator was evaluated on each constructed Hamiltonian. All declared runtime-specific candidate banks were rebuilt, and planned test-parent logical fingerprints and transfer axes were checked against source train/validation. Planning was bounded to 60 seconds per dataset; every dataset finished inside that limit.

## Planned populations, not completed datasets

| Dataset | Parents | Train/validation/test parents | Physical qubits | Planned records | Planned bank labels |
| --- | ---: | ---: | --- | ---: | ---: |
| source | 480 | 300/90/90 | 3–10 | 8,640 | 552,960 |
| pegasus | 480 | 288/96/96 | 10/12/14 | 5,760 | 368,640 |
| zephyr | 480 | 288/96/96 | 10/12/14 | 5,760 | 368,640 |
| unseen_family | 240 | 144/48/48 | 3–10 | 4,320 | 276,480 |
| unseen_runtime | 240 | 150/45/45 | 3–10 | 1,440 | 92,160 |
| unseen_size | 240 | 150/45/45 | 16/18/20 | 4,320 | 276,480 |

All **2,160 parents and 4,320 parent×variant geometry constructions** passed. The source has 390 fitting/selection parents and 90 held-out test parents. The size-transfer target has **15 test parents at each of 16, 18 and 20 physical qubits**; these correspond to 8, 9 and 10 logical variables. The two topology targets each have 32 test parents at each of 10, 12 and 14 physical qubits. Counts describe the deterministic plan; no corresponding statevector outcomes were generated here.

The full dataset plan contains **30,240 records and 1,935,360 bank labels** before acquisition, direct-policy diagnostics, optimizer baselines, numerical refinement retries or noise evaluation. This is a substantial proposed compute workload, not a measured throughput or completion claim.

## Transfer contracts

All six target definitions, including source held-out, passed the production `validate_axes` checks. Every target test logical fingerprint is disjoint from all 390 source training/validation fingerprints. This establishes exact labelled-coefficient disjointness for these planned parents, not graph/gauge-isomorphism separation.

All six datasets use **64 candidates and `candidate_seed=1801`**, with distinct logical-instance dataset seeds `2026091801` through `2026091806`. All sampled nine-knot candidate waveforms match the source bank to absolute tolerance `1e-12`, including across runtime changes.

- Source support: synthetic embeddings, physical sizes 3–10, `spin_glass`, `weighted_maxcut`, `planted_loops`, runtime 1/4/12.
- Unseen family: `weak_field`, absent from source fitting/selection support.
- Unseen runtime: 24, absent from source runtime support.
- Unseen size: 16/18/20, disjoint from source physical sizes.
- Pegasus/Zephyr: different topology from synthetic source. They also change size support and include unseen `weak_field`; this is joint distribution shift and does **not** isolate topology as a causal factor.
- Source held-out: same observed topology and family/size/runtime support as source fitting/selection records.

Hardware metadata is consistent with Pegasus P16 (5,640 source qubits) and Zephyr Z15 (7,440 source qubits), with 96-site patches used to build the small active subgraphs. Both are marked `is_full_device=false` and `is_calibrated_device=false`. These are connectivity models, not QPU experiments.

## Memory and untested gates

The production analytical estimator passes every configuration. Its largest state estimate is **1,280 MiB per worker** for the 20-qubit target; its largest endpoint estimate is **94.5 MiB**. For the 14-qubit topology targets, the corresponding maxima are 40 and 21 MiB. These are analytical bounds under the configured batching/chunk settings, **not observed GPU peak allocations**, and they do not establish wall-clock feasibility.

Remaining execution gates are CUDA/CuPy availability, CPU/GPU numerical parity, idle-machine repeated timing, convergence at the prescribed tolerances, full acquisition/training completion, and the predeclared statistical comparisons. No accuracy, transfer benefit, architecture advantage, GPU speedup or conference-readiness result follows from this preflight. A passed plan can still produce negative empirical results.

## Snapshot

- Source fingerprint: `1ccf566c336be9e7208cafee210ecb75908347299245fff6a67f8f4a0f5427ef`.
- Campaign file SHA-256: `e33b36b1ed2e054649812c63663245bfb2e869070d7a1c3899f8a960c78ee56e`.
- Source remained unchanged throughout preflight: `True`.

Resolved dataset config hashes:

- `source`: `0277612a7a55b30eb7920b980aa1aaa29bfefd7c7ac26bd4c9f3468d2f56ca3a`.
- `pegasus`: `c39e93452d5fcceed0d71f8aa23d4a2fbba0a67404be99e9b09c8b2575ff8bec`.
- `zephyr`: `6ae60b87be50a52b1ba5f2a463e5d662af367d2eb43a7d574c420c071e0dac0a`.
- `unseen_family`: `e10f3a44ec679622e0a6f7b060547b2d515644999edc3d12e4f557aba43999a0`.
- `unseen_runtime`: `49605f37d36431356d13cff5c6f0c7c08f7e457840e2a6fd4eb6ef2be566963e`.
- `unseen_size`: `73f6bdc5b4279ef59611cd069958e2ba81a75171cf6710ebc593159049aed51a`.
