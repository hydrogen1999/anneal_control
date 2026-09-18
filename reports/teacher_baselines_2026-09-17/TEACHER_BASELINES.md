# The privileged spectral baselines are not the target

Source: the G2 control-sweep rows, `train` and `validation`, 3447 successful
units over 192 logical parents. Every number is a true simulator outcome of the
waveform each rule specifies. Intervals are parent bootstraps.

Reproduce with:

```
annealctrl teacher-baseline-report --sweep <g2_frontier_train> <g2_frontier_validation> \
    --output reports/teacher_baselines_2026-09-17/teacher_baselines.json
```

## Why this measurement exists

`gap_inverse_square` is the local-adiabatic rule — ds/dt proportional to the
square of the instantaneous gap. It is the textbook answer to how one should
anneal, and computing it requires the exact spectrum at every point on the path,
which is exponentially expensive and which no deployed method has. `d2` is the
band-aware second-moment variant.

Learned control selection is usually motivated as *a cheap approximation to the
spectral schedule*. Under that framing the spectral schedule is the ceiling and
the learner is chasing it. **That framing is an assumption, and it is wrong
here.**

## Result

| baseline | resolved | teacher | linear | search | vs linear | vs search | verdict |
|---|---:|---:|---:|---:|---|---|---|
| `gap_inverse_square` | 44.7% | 0.7159 | 0.7013 | 0.6116 | **+0.0230** [+0.0049, +0.0413] | **+0.1103** [+0.0973, +0.1238] | loses to linear |
| `d2` | 98.1% | 0.5569 | 0.5667 | 0.4641 | −0.0017 [−0.0137, +0.0105] | **+0.0982** [+0.0888, +0.1079] | beats linear, loses to search |

Both privileged baselines lose to a 64-candidate search over five simple control
families by **0.098–0.110** in logical success probability. The local-adiabatic
rule, given the exact spectrum, is also **worse than a linear ramp**
(+0.0230, interval excludes zero) and beats it on only 44.6% of instances.
`d2` is statistically indistinguishable from linear.

## The runtime trend is exactly what adiabatic theory predicts

| runtime | `gap_inverse_square` vs linear | beats linear | `d2` vs linear | beats linear |
|---:|---|---:|---|---:|
| 1.0 | +0.0063 | 37.1% | +0.0057 | 43.1% |
| 4.0 | +0.0625 | 45.1% | +0.0232 | 56.7% |
| 12.0 | −0.0001 | 51.6% | **−0.0341** | 66.0% |

Against a **linear ramp** the rules do catch up: the fraction of instances where
`gap_inverse_square` beats linear rises monotonically (37.1%, 45.1%, 51.6%) and
the mean difference reaches parity at runtime 12. That is the adiabatic theorem
behaving as advertised.

**Against the search it does not.** That is the comparison the finding rests on,
and it is the one to read:

| runtime | `gap_inverse_square` − search | `d2` − search |
|---:|---|---|
| 1 | +0.0540 [+0.0462, +0.0623] | +0.0677 [+0.0598, +0.0758] |
| 4 | +0.1671 [+0.1428, +0.1926] | +0.1393 [+0.1229, +0.1561] |
| 12 | +0.1105 [+0.0955, +0.1259] | +0.0879 [+0.0748, +0.1016] |

Every interval excludes zero, and the gap at runtime 12 is roughly **twice** the
gap at runtime 1. `gap_inverse_square` beats the search on 0.7% of records and
`d2` on 3.8%.

An earlier revision of this report quoted only the vs-linear column and described
it as evidence that the finding would reverse at longer runtime. That was reading
the trend of one quantity to undermine a claim that rests on a different one. The
gap to search is what the claim is about, and across the runtimes tested it does
not close.

It is also **non-monotonic**, peaking at runtime 4, so it licenses no
extrapolation in either direction — including the favourable one. A runtime
ladder at 12, 36 and 108 (`configs/data_runtime_ladder.json`) is running to test
that directly rather than argue about it.

## What this licenses, and what it does not

**Licensed.** Learned control selection is not an approximation to the spectral
schedule. At every runtime measured, simple parameterized search beats the exact
gap rule by roughly 0.10, so the spectral rule is not a ceiling the learner is
chasing — it is a reference the search already passes. A paper that motivates
learned control as "cheaper than computing the gap" is motivated wrongly; the
correct motivation is that the gap rule does not win at deployable runtimes.

**Not licensed — conditioning.** `gap_inverse_square` resolves on only 44.7% of
instances, and the population it resolves on is harder than the one it does not
(mean linear loss 0.7085 against 0.4565). A vanishing or degenerate first gap is
both the reason the teacher fails and a property of the instance, so this subset
is not missing at random and the comparison is conditional on it. `d2` resolves
at 98.1% and tells the same story about search, which corroborates but does not
remove the conditioning.

**Not licensed — audit fragility.** For `d2`, including the 435
interpolation-audit failures flips the sign of the comparison against linear.
The headline uses audited waveforms only; the flag `audit_failures_change_the_sign`
is set in the artifact and must be quoted with the number.

**Not licensed — scale.** At most 10 physical qubits, closed system, no noise.
Whether the adiabatic rule recovers its advantage at larger sizes or longer
runtimes is untested here, and the runtime trend suggests it plausibly would.
