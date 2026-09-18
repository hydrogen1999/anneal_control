# Does the local-adiabatic oracle catch up given more time? No.

## Why this exists

The campaign measured the privileged spectral baselines at runtimes 1, 4 and 12
and found them losing to an equal-budget search at every point. The strongest
objection to that finding — **which I raised against my own result** — is that
the adiabatic rule is asymptotically correct, so the whole comparison might only
reflect being far from the adiabatic regime. Three runtimes over one decade
cannot settle it, and the gap to search is non-monotonic there, so it extrapolates
in neither direction.

So the axis was extended instead of argued about: runtimes **12, 36 and 108**,
with the same families, the same budget of 64 per tunable family, the same two
privileged teachers and the same censoring rules. Physical size is fixed (≤10, as
a full spectral teacher requires) and logical size is pinned at 4. **Only the
runtime moves.**

355 rows, `max_steps` raised to 262144 because a waveform at runtime 108 is nine
times flatter than at 12 and the integrator has to follow it nine times as long.

## Result: the gap to search does not close

`gap_inverse_square` minus equal-budget search, parent bootstrap:

| runtime | teacher | linear | search | vs linear | **vs search** |
|---:|---:|---:|---:|---:|---|
| 12 | 0.5891 | 0.5600 | 0.4508 | +0.0364 | **+0.1410** [+0.0929, +0.2012] |
| 36 | 0.3921 | 0.4399 | 0.3070 | −0.0450 | **+0.0840** [+0.0344, +0.1454] |
| 108 | 0.3530 | 0.4093 | 0.2542 | −0.0475 | **+0.1004** [+0.0303, +0.1986] |

`d2`:

| runtime | vs linear | **vs search** |
|---:|---:|---|
| 12 | −0.0109 | **+0.1182** [+0.0692, +0.1739] |
| 36 | −0.0522 | **+0.0598** [+0.0167, +0.1145] |
| 108 | −0.0274 | **+0.0859** [+0.0150, +0.1874] |

**Every interval excludes zero at every runtime, out to 108 — nine times the
longest runtime in the main campaign.** `gap_inverse_square` beats the search on
7.0% of records; `d2` on 24.8%.

## Both things are true at once, and only one is the claim

**Against a linear ramp the oracle does win at long runtime.** It is behind at
runtime 12 (+0.0364) and ahead at 36 and 108 (−0.0450, −0.0475). The adiabatic
rule behaves exactly as theory says it should: given enough time, slowing down
where the gap is small beats a uniform sweep.

**Against the search it still loses, by about the same margin as at runtime 1.**
Extending the axis ninefold did not close it.

The paper's claim is the second. An earlier revision of the teacher report quoted
the first and described it as evidence the finding would reverse — reading the
trend of one quantity to undermine a claim resting on another. This ladder is
what settles it.

## What this does not cover

**One decade further, not asymptotically.** Runtime 108 is not infinity. The
adiabatic theorem is a statement about a limit, and no finite ladder refutes a
limit. What this shows is that across the runtimes anyone would actually deploy —
and a factor of nine beyond the campaign's longest — the rule is not the target.

**Non-monotonic, still.** The gap dips at 36 and rises at 108, and the interval
at 108 is the widest of the three. The right reading is "does not close", not
"grows".

**A different dataset.** 32 parents at logical size 4, not the campaign's 240
across sizes 3–5. This replicates the finding on a comparable distribution while
varying runtime; it is not the same instances at three runtimes.

**`gap_inverse_square` resolves on 46.2%** of rows here, the same conditioning as
in the main campaign: a degenerate first gap is both why the teacher fails and a
property of the instance. `d2` resolves on 96.3% and tells the same story.

**Still a closed system**, still ≤10 physical qubits, still no hardware.
