# G2/G3 campaign — apollo, 2026-09-16

First execution of the two measurement gates on a real host. This records what
was run, on what, and what the numbers are allowed to mean. It is **not** a
paper results table: there is no learned model in it, no hardware, and no
comparison against any published method.

## 1. Host and environment

| | |
|---|---|
| Host | `apollo` (`egr-w-cs-td-10.rams.adp.vcu.edu`), VCU lab machine, no scheduler |
| CPU | 32 cores; campaign used 12 concurrent shards at `nice 10` alongside ~20 other users |
| GPU | NVIDIA RTX PRO 6000 Blackwell, 96 GiB, CUDA 13.0, driver 590.48.01 |
| Python | 3.12.3 · NumPy 2.5.3 · SciPy 1.18.1 · NetworkX 3.6.1 · PyTorch 2.14.0+cpu · CuPy 14.2.0 (`cupy-cuda13x`) |
| Test suite on host | 512 passed, **0 skipped** — the two CuPy/CUDA parity tests that `reports/V02_VERIFICATION.md` records as unrun now execute and pass |

Threads pinned to one per process (`OMP/OPENBLAS/MKL_NUM_THREADS=1`) so wall
times are comparable between runs.

## 2. GPU: measured, not assumed

`IMPLEMENTATION_PLAN_VI.md` §7 item 5 asks for GPU throughput measured as
**accepted labels/s and peak memory**, not matvec/s. `profile-generation` was run
on a 10-physical-qubit distribution, 4608 accepted labels, one worker per backend:

| backend | wall | accepted labels/s |
|---|---:|---:|
| `numpy` | 243.48 s | 18.93 |
| `cupy` | 225.30 s | 20.45 |

**Ratio 1.08.** CPU/GPU outcome parity: max absolute difference in candidate
losses **1.22e-15**, i.e. machine precision — so the GPU path is numerically
sound, and it is simply not faster here.

The reason is size: 2^10 complex amplitudes do not fill a Blackwell GPU, and
kernel-launch overhead dominates. CuPy generation is additionally restricted to
**one** parent worker while NumPy runs many, so in deployment terms *N* CPU
shards beat one GPU process by roughly *N*. The campaign therefore ran on CPU.

This says nothing about larger systems, and the crossover where the GPU wins is
**not yet admissible evidence**. Figures for 12, 14 and 16 physical qubits
(ratios 3.22x, 10.67x, 103.57x) appear in the message of commit `eb9de9f` and
nowhere else: there is no committed artifact, no config hash and no record of
what else the host was running. They are withheld here rather than quoted.

The measurement is also easy to get wrong in a self-serving direction. A loaded
host starves the NumPy arm while CuPy, which is launch-bound, barely notices, so
timing under load inflates the ratio in favour of the GPU. The 10-qubit number
above was taken with `teacher.mode=none` on a quiet host and stands;
`scripts/backend_crossover.sh` reproduces the full curve and **refuses to run**
above a load threshold, writing one `profile-generation` artifact per size to
`configs/backend_profile_{10,12,14,16}q.json`. Until those artifacts exist, the
only backend claim this project makes is the 1.08x at 10 qubits.

## 3. What was run

| Stage | Config | Workload |
|---|---|---|
| G2 dataset | `configs/data_g2_research.json` | 240 parents, 4320 records, 4 families, logical sizes 3/4/5, physical 3–10, splits 144/48/48 parents. 105.6 s on 10 workers. |
| G2 frontier | `configs/frontier_research.json` | 5 families, budget 64 per tunable family → 257 propagation-scored controls per record, on `train` and `validation` |
| G3 interventions | `configs/intervention_research.json` | 96 parents, 1060 pairs after 148 vacuous skips, 388 objective calls per pair → 411,280 scored controls |

Test-split control search was **not** run: it is online adaptation and requires a
deliberate `--allow-test-adaptation`.

### Why a separate G2 dataset

The frontier instrument re-searches controls from scratch with exact waveforms,
so it needs neither the adaptive spectral teacher nor a 64-candidate stored bank
— and those are where `data_research.json` spends nearly all its time. Measured
on 48 parents / 864 records at 8 workers: adaptive teacher **508.4 s**,
`teacher.mode=none` **37.0 s**. Same distribution, 13.7× cheaper, nothing a
headroom measurement uses is lost. `data_g2_research.json` therefore cannot train
the physics-auxiliary ablations; `data_research.json` remains the dataset for that.

## 4. Defects found by running at scale

Three, all invisible to the smoke configs:

1. **Pair-id collision.** `intervention_research.json` declares two `geometry`
   changes (`star` and `random_tree`); `pair_id` was keyed on the factor alone,
   so they collided. Caught by the sweep's uniqueness guard on the first real run.
2. **A tautological metric.** The first G3 report said "100% swap rate over 610
   resolved pairs". `resolved` already means both transfer penalties exceed their
   ambiguity, which can only happen when each arm's own control wins — so
   P(swap | resolved) ≡ 1. The reportable quantity is the fraction of **all**
   pairs reaching a decisive reversal.
3. **An impossible numerical tolerance.** 9 of 3456 frontier units aborted with
   `schedule violates maximum ds/dt`. The waveforms were feasible: `window_schedule`
   builds τ by cumulative summation, so a narrow final segment (captured case
   Δτ = 1.36e-7) inherits a relative slope error of ~8e-9, and the fixed 1e-12
   gate rejected a 1.22e-9 overshoot. `validate_slope` now allows
   `1e-12 + 8·eps/Δτ` per segment; a 10% violation at the same width still fails.

After (3) the source hash changed, and `run_sweep` refused to merge results
across source revisions — the ADR-0005 guard doing its job. The whole campaign
was therefore re-run so that every row carries one source hash.

## 5. Results

See `results/` alongside this file, or on the host:

```
~/runs/v2/g3/report/INTERVENTIONS.md            + figure4_interventions.pdf
~/runs/v2/g2_validation/report/FRONTIER.md      + figure3_frontier.pdf
~/runs/v2/g2_train/report/FRONTIER.md           + figure3_frontier.pdf
~/runs/v2/screen.json
```

Numbers are recorded in `CAMPAIGN_G2_G3_RESULTS.md`.

## 6. What these numbers may not be used for

- No hardness claim. Headroom is a property of the declared control families,
  the runtimes, and the closed-system simulator.
- No global optimum. Every reference is best-found under a stated budget and seed.
- No hardware claim. Every causal statement is inside the declared simulator;
  toy lifted graphs are not a commercial topology.
- No statement about a learned model. Headroom is the *opportunity*; whether a
  model can exploit it is a separate experiment.
- No deployment frequency. Factors, magnitudes and the base configuration are
  declared choices in the config files.
