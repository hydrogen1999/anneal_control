# The coherent-failure test: it largely did not apply, and it withdrew a claim

## What was asked

> For coherent failure modes, use paths with two or more transition regions and
> scan runtime densely enough to detect interference. A scalar profile might
> work after averaging across noise or runtime uncertainty while failing for a
> sharply specified coherent experiment; report that distinction rather than
> declaring it universally sufficient or useless.

16 records — the 8 multi-crossing instances the
[crossing census](../crossings_2026-09-19/CROSSINGS.md) found across two
datasets, and 8 monotone-gap records as a control group — scanned at **64
runtimes** over [1, 17]. Without the control group, finding no interference
would not distinguish smooth instances from too coarse a grid.

Three schedules per record: linear; the `d2` teacher **rebuilt at each
runtime**, which is how a runtime-aware construction is deployed; and a control
searched at the record's own runtime and then asked about the others. No
runtime was censored for infeasibility.

## Result 1: there was little interference to fail at

| group | records with a non-monotone loss-versus-runtime curve | mean turning points |
|---|---:|---:|
| multi-crossing | **2/8** | 0.50 |
| monotone control | **1/8** | 0.12 |

Multi-crossing instances show four times as many turning points, which is the
expected direction, but from a base so small that the comparison rests on
three records in total.

**The honest report is that the test did not apply**, not that the scalar
profile passed it. A dense scan over 64 runtimes on the instances that qualify
found essentially smooth curves. Detecting coherent structure here would need
either instances with much smaller gaps — the census found one Pegasus record
at 0.0020 that this scan's 10-qubit cap excluded — or a finer grid than 64
points over a factor of 17.

## Result 2: a claim withdrawn, for the third time in one day

The teacher's advantage over linear, by subgroup:

| group | d2 advantage | better than linear at |
|---|---:|---:|
| multi-crossing | +0.01250 | 54 % of runtimes |
| monotone control | **−0.05788** | 35 % of runtimes |

Here the teacher is **worse** on monotone-gap instances. Pooled over runtimes
1, 4 and 12 on a larger sample earlier the same day, it was **better** on
exactly that subgroup (−0.02519 [−0.04990, −0.00060] in the teacher-minus-linear
convention). At runtime 1 alone it was better on the *other* subgroup.

Three analyses of the same quantity, three different answers. They differ in
runtime grid (three points against 64), in sample (23 monotone parents against
8), and in selection (one record per parent against the first four per census)
— and any of those could explain it. What none of them supports is a stable
subgroup effect.

**All subgroup claims about the teacher's own advantage are therefore
withdrawn.** What can be said is narrower: *the spectral teacher's benefit is
unstable across runtime and sampling, and this project has not identified the
conditions under which it helps.*

## What survives

The contrast between **which spectrum builds the schedule** has been stable
wherever it was measured: on the interior-minimum subgroup the physical
embedded spectrum beats the logical one by **+0.01137 [+0.00320, +0.01975]**
pooled over three runtimes, with the same sign at each
([report](../logical_spectrum_2026-09-19/LOGICAL_SPECTRUM.md)). That quantity
is a *difference between two teachers built the same way*, so the instability
above — which is about the teacher against linear — cancels out of it. That is
probably why it is the one that held.

## An aside worth keeping

The searched control, tuned at each record's own runtime and then evaluated
across the whole range, beats linear at **88 %** of runtimes on multi-crossing
instances and only **46 %** on monotone ones, changing sign in 8 of 8 monotone
records. A control optimised at one runtime does not transfer reliably to
others. That is a practical point about runtime transfer, on 8 records per
group, and is offered as an observation rather than a result.

## Limits

- 8 records per group; the multi-crossing pool is everything the census found.
- ≤10 physical qubits, so the smallest-gap Pegasus instance is excluded.
- `--allow-test-adaptation` was passed deliberately: both censuses cover the
  test split, and the searched arm is a within-record reference control with no
  learned model and no held-out claim. The artifact records the flag and the
  reason.
