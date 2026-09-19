# The GPU ladder, and why none of its ratios is a clean number

## What the archive actually contains

`backend_2026-09-17/crossover.json` declares exactly one stable size. Reading
every run out of it, including the failures:

| qubits | repeat | return code | numpy labels/s | cupy labels/s | cupy/numpy |
|---:|---:|---:|---:|---:|---:|
| 10 | 0 | 0 | 14.493 | 10.102 | **0.70** |
| 10 | 1 | 1 | — | — | — |
| 12 | 0 | 0 | 2.665 | 7.486 | 2.81 |
| 12 | 1 | 0 | 2.568 | 7.329 | 2.85 |
| 14 | 0, 1 | 1, 1 | — | — | — |
| 16 | 0, 1 | 1, 1 | — | — | — |

`stable_sizes` is `["12"]`, and that is correct: 12 qubits is the only size
where both repeats returned. **At 10 qubits the GPU was 1.43× slower than the
CPU**, on the single repeat that ran.

### A number in this repository had no source

`docs/scale_ceiling.md` stated "measured 1.13× at 10 qubits". No artifact
produces 1.13, and the sign is wrong: the archived 10-qubit run has the GPU
losing. The line is corrected in this commit. The 12-qubit 2.83× is real and
reproduced across both repeats.

## New: 14 qubits now returns, and it disagrees with itself

The old ladder failed at 14 and 16. Two fresh 14-qubit replicates completed:

| replicate | numpy wall | cupy wall | numpy labels/s | cupy labels/s | cupy/numpy |
|---|---:|---:|---:|---:|---:|
| r0 | 1735 s | 216 s | 0.3320 | 2.6641 | **8.03** |
| r1 | 1928 s | 213 s | 0.2987 | 2.7075 | **9.06** |

The ratio moves 13% between two runs of identical code over identical records.

**The variance is entirely on one side.** The cupy wall times agree to 1.4%
(216 s against 213 s); the numpy wall times differ by 11% (1735 s against
1928 s). Both replicates ran on a host at load ≈ 34 with 32 cores. A GPU kernel
does not compete with other users' CPU jobs; the numpy arm does. That is a
measured attribution from matched replicates, not an assertion about the
machine.

The direction of the bias is therefore known: **a contended host starves the
NumPy arm, so every ratio in this ladder is an upper bound on the true
speed-up.** The archived artifact says the same thing in its own scope field,
and its census records `other_cpu_percent` of 3166–3534 — thirty-one to
thirty-five cores of competing work — at every point in the ladder.

## 16 qubits: the point the old ladder never produced

The 16-qubit profile completed, single-worker, both backends on the same 36
records:

| backend | wall | rate | numerical gate |
|---|---:|---:|---|
| numpy | 15 489 s (4.3 h) | 0.0186 labels/s | passed |
| cupy | 173 s | 1.6647 labels/s | passed |

Ratio **89.5×**, parity again 2.22e-16.

The ladder, with every point read out of its artifact:

| qubits | numpy | cupy | cupy/numpy |
|---:|---:|---:|---:|
| 10 | 14.493 | 10.102 | **0.70** (GPU slower) |
| 12 | 2.62 | 7.41 | 2.83 |
| 14 | 0.332 / 0.299 | 2.664 / 2.708 | 8.03 / 9.06 |
| 16 | 0.0186 | 1.665 | **89.5** |

The shape is what matters more than any single ratio. Per two qubits the CPU
rate falls by 5.5×, 8.2×, then 17.2×; the GPU rate falls by 1.36×, 2.76×, then
1.62×. **The GPU degrades far more gently**, which is the whole reason the ratio
opens up.

### The 16-qubit ratio is inflated, and by roughly how much

`contention_16q.csv` samples the host from outside the job: over 83 samples
covering the tail of the NumPy phase and all of the CuPy phase, load1 ran
40.1–55.6 with a **median of 50.0 on 32 cores**. The profile used **one
worker**, so the NumPy arm was a single-threaded process on a box
oversubscribed by ~1.56×. A first-order correction — a runnable thread receives
about `cores / load` of a core — puts the uncontended NumPy rate near
0.0186 × 1.56 ≈ 0.029 labels/s and the ratio near **57×** rather than 89.5×.

That arithmetic is a bound, not a measurement: load-average scaling is
approximate, the sampler missed the first 3.7 hours of the NumPy phase, and
this is **one replicate** at a size where the two 14-qubit replicates already
disagreed by 13%. The defensible statement at 16 qubits is "somewhere around
60–90×, measured once, under contention".

## How far the GPU actually reaches: 18 and 20 qubits, measured

The NumPy arm at 18 qubits would take roughly 74 hours by extrapolation from the
measured per-two-qubit CPU slowdown, which answers nothing worth 74 hours. So
these two points are **GPU-only** — no parity check is possible without the CPU
arm, and parity was already verified at 10, 12, 14 and 16 qubits at 2.22e-16.

| physical qubits | labels | wall | labels/s | numerical gate | max norm error |
|---:|---:|---:|---:|---|---:|
| 14 | 576 | 216 s | 2.6641 | passed | 8.8e-14 |
| 16 | 288 | 173 s | 1.6647 | passed | 1.0e-13 |
| 18 | 144 | 118 s | 1.2250 | passed | 3.8e-14 |
| 20 | 72 | 220 s | **0.3277** | passed | 2.9e-14 |

*(Label counts halve because each config labels fewer candidates per record —
8 at 16 qubits, 4 at 18. Rate per label is the comparable quantity and is what
is tabulated.)*

Slowdown per two qubits on the GPU: 1.60×, 1.36×, then 3.74×. **20 physical
qubits generates, and it generates in under four minutes for 72 labels.**

### What that means for scale

The configs record the reason this ladder exists: at 18 qubits a full
eigendecomposition needs **1.1 × 10³ GB** while the state vector needs
**0.004 GB**. Only the spectral teacher is exponentially capped — the
propagation path that every non-oracle claim in this project uses is not.

Extrapolating the measured GPU rate, a 2880-record dataset at 20 qubits is
roughly 39 GPU-hours: large but ordinary. Memory is nowhere near binding — a
20-qubit state vector is 16 MB against 97 GB of device memory, and 30 qubits
would still fit at 16 GB. Time binds first: at 3.74× per two qubits, 24 qubits
is ~0.023 labels/s and 26 is ~0.006, where a single label takes minutes.

So the honest scale statement for exact simulation on this hardware is
**dataset generation to about 20 physical qubits, individual evaluations to
about 24–26**, with the spectral oracle stuck near 16 for a completely
different reason.

## The GPU computes the same science

Worth stating separately, because it is the part that is clean. On the same 36
records, NumPy against CuPy:

    candidate_losses   max absolute difference   2.22e-16
                       mean absolute difference  4.16e-17

That is machine epsilon in double precision. The backends agree exactly; only
their speed is in question.

## Contention trace for the 16-qubit run

The deployed profiling code predates the in-artifact host-load capture, so the
running 16-qubit profile would have produced timings with nothing to check them
against. `contention_16q.csv` samples `/proc/loadavg` from outside the job every
30 s for its lifetime. It is deliberately **not** part of the signed artifact:
it records what the machine was doing, not what the job measured. Opening
samples show load 40–55 on 32 cores, so the 16-qubit timing will be contaminated
in the same direction and by more than the 14-qubit points were.

## What would make this a publishable claim

The code already has the mechanism: `profiling._check_load` rejects a timing
whose host load exceeds a declared maximum, and `_host_status` records the load
inside the artifact. Neither is in the deployed tree, which is why the existing
points are unguarded. The fix is deployment discipline, not new code:

1. Redeploy, so `host_before`/`host_after` land inside each profile.
2. Re-run the ladder with `--max-load` set, on a quiet host, so a contended
   timing is refused rather than caveated.

Until then, the defensible statement is narrow: **the GPU is slower than the
CPU at 10 qubits, reproducibly ~2.83× faster at 12, and somewhere in 8–9× at
14, with every figure an upper bound measured under heavy contention.** The
trend across those three sizes is real in direction and unreliable in
magnitude; it is not a scaling law.
