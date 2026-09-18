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
