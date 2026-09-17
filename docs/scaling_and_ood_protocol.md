# Scaling evidence and out-of-distribution protocol

This document specifies executable timing controls and the next scientific
experiments. Adding a configuration does not establish a speedup or solve the
generalization question. The withdrawn 12/14/16-qubit ratios remain withdrawn.
No new GPU result is reported by this change.

## Repeated CPU/GPU measurements

The old one-shot script guarded host load only once and the 12/14/16-qubit
configurations omitted the mandatory `endpoint_max_qubits` field. Those
configurations now pass validation. The replacement driver:

1. Keeps one complete warm-up per backend separate from four measured runs.
2. Alternates backend order from a declared seeded first order; an even number
   of repetitions balances the first-running backend.
3. Generates fresh datasets for every run, synchronizes the CUDA device before
   and after the timed work, and rejects reuse of an existing output directory.
4. Records host load before and after each backend, CPU affinity, cgroup CPU
   quota where available, thread settings, package/device versions, CPU time,
   wall time, candidate counts and numerical diagnostics.
5. Refuses a declared load threshold violation and retains the partial report.
   CPU/GPU record IDs, Hamiltonians, executed waveforms and outcomes are checked.
   Every outcome must meet the explicit absolute parity tolerance before a
   repeated ratio summary is emitted.
6. Reports each paired throughput ratio, its median, geometric mean and range.
   Warm-ups never enter that aggregate. Repetitions are timing replicates of
   the same workload, not additional logical parents for scientific inference.

Run from an installed checkout on a reserved CPU/GPU machine:

```bash
ANNEAL_PYTHON="$PWD/.venv/bin/python" OUT="$PWD/runs/crossover_v2" \
  MAX_LOAD=2 REPEATS=4 WARMUPS=1 SIZES="10 12 14 16" \
  bash scripts/backend_crossover.sh
```

`MAX_LOAD` is a host-wide one-minute load-average threshold, not a percentage.
Choose it before running using the host's CPU allocation and cgroup quota;
record the choice. Do not raise it after seeing an attractive speedup. Reserve
the GPU and record other GPU processes with the machine's normal monitoring
tools. Endpoint load samples can miss interference during an arm and do not
prove that the GPU is idle. Keep the CPU thread allocation fixed and publish
its settings. Retain all failures, warm-ups and repetitions alongside the
summary artifact. A source edit during the benchmark invalidates the run.

The module also supports a CPU-only diagnostic run:

```bash
python -m annealctrl.profiling --config configs/backend_profile_10q.json \
  --output runs/cpu_profile --backends numpy --repeats 4 --warmups 1
```

That command emits no CPU/GPU ratio. Omitting the load guard marks a two-backend
aggregate ineligible for a load-guarded timing claim. There is no fallback from
a requested unavailable CuPy backend to NumPy.

The output directory contains `repeated_profile.json` and the individual
`warmups_*/profile.json` / `repetitions_*/profile.json` reports and datasets.
Timings include assembly, endpoint observables, numerical refinement, writes
and lightweight RSS monitoring. Runtime initialization probes precede timing;
warm-ups and measured calls share runtime and allocator caches. This is a
steady-runtime full-generation measurement, not cold process startup latency.

Memory reporting intentionally distinguishes three quantities:

| Quantity | Meaning | Limitation |
| --- | --- | --- |
| Estimated numeric workspace | Existing allocation guard / array arithmetic | Excludes allocator and process overhead |
| Sampled process RSS | Observed host process memory, every 10 ms | Can miss short peaks; excludes children and GPU |
| CuPy allocator endpoint snapshot | Used and reserved bytes in this process's default pool | Not the GPU peak; other allocations can exist |

The benchmark does **not** claim to measure GPU peak memory. Capture independent
device monitoring for a paper memory claim. All current dynamics use
complex128; a future complex64 benchmark needs its own accuracy gate and must
not be mixed silently with this protocol.

## Bounded 18/20-qubit feasibility pilots

`configs/backend_profile_18q.json` and `backend_profile_20q.json` use 9 and 10
logical variables respectively, each with two physical qubits per chain. They
request three parents, two embedding variants, eight controls, batch size two,
one chain strength and runtime, and at most 4096 integration steps. Explicit
caps are 256 MiB for endpoint work and 1024 MiB for state work; the existing GPU
guard additionally checks currently free device memory. These are conservative
allocation requests, not measured peak budgets or guaranteed convergence.

Run these sizes only after the smaller crossover measurements have completed:

```bash
ANNEAL_PYTHON="$PWD/.venv/bin/python" OUT="$PWD/runs/crossover_large_pilot" \
  MAX_LOAD=2 SIZES="18 20" bash scripts/backend_crossover.sh
```

These pilots deliberately use a smaller workload and batch size than the
10–16-qubit profiles. They can establish per-size feasibility and CPU/GPU
agreement; do not combine them into a common-workload scaling curve without
rerunning all sizes under a matched protocol. Numerical nonconvergence is a
failed workload, never a silently relaxed accuracy target. Estimate total
campaign cost from observed accepted labels per second before increasing
parents, controls, runtimes or steps. Stop after an allocation or convergence
failure, inspect the artifact and revise the registered pilot separately.

At 20 physical qubits the state has 1,048,576 amplitudes, but these configurations
still encode only 10 logical variables. Exact logical endpoint enumeration is
therefore affordable. Exhaustive endpoint search cannot optimize a continuous
annealing waveform, yet it is a necessary classical reference for endpoint
quality. Neither a larger state vector nor a GPU crossover establishes
computational hardness, quantum advantage, hardware robustness or QPU results.

## Scientific scale and distribution shifts

The timing configurations are not scientific test sets. A full campaign must
vary logical size as well as physical overhead and retain the same parent
across all matched embedding/control comparisons.

| Question | Training and validation | Untouched evaluation | Required controls |
| --- | --- | --- | --- |
| Size extrapolation | Logical 3–6 with validation parents at those sizes | Logical 7–10 under a separately fixed physical cap | Same family, coefficient/scale protocol, control budget and runtime grid |
| Chain overhead | Fixed logical sizes and parent problems | Longer chains for those held-out logical parents | Match logical instance and programmed scale; report physical size separately |
| Family transfer | Train on spin glass, validate on new spin-glass parents | Weighted Max-Cut parents from disjoint seeds | Report within-family and cross-family; do not select checkpoint on target labels |
| Embedding/topology transfer | Freeze source topology and port/chain distribution | Unseen topology or chain/port distribution | Match feasible logical support where possible; separate support shift from embedding shift |
| Runtime/scale transfer | Predeclare bounded source intervals | Disjoint target intervals | Keep runtime and programmed scale visible to all eligible models |

Before training, construct one immutable split manifest across configurations.
Namespace record/parent IDs by dataset identity, compare logical fingerprints
for duplicates and graph isomorphism where relevant, and group every embedding,
runtime and acquisition round from a parent in the same split. Independent
generator seeds alone do not prove no overlap. Archive source hashes, generator
configuration, parent manifest and all rejected/infeasible counts.

For each shift evaluate the same frozen source-trained summary encoder and
hierarchical encoder, bank selector, direct policy, linear/global controls and
the literature-derived optimizer with explicit simulator-call budgets. Both
bank selection and direct policy have zero online simulator calls. Report their
actual latency, label-generation cost, tuning cost and regret–cost frontier;
catching up to the bank is not an automatic deployment advantage. Use at least
three predeclared training seeds, parent-clustered contrasts, seed sensitivity,
numerical uncertainty and multiplicity correction for the declared contrast
family. Select hyperparameters using source validation only. Any target-domain
fine-tuning becomes a separate labeled adaptation experiment with its own cost.

Without these completed experiments the valid claim is that the infrastructure
supports a controlled scale-up study. The negative hierarchy result remains
part of the paper until independent evidence changes it.
