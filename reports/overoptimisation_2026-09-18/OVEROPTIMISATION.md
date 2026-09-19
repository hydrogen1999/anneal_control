# Does searching longer against a noiseless simulator return a worse control?

## The claim being tested, and it is mine

The erosion measurement shows that within a record, the better a control is in
a noiseless simulator, the more an environment takes back from it. I wrote the
obvious practical reading into the erosion report as if it followed:

> Past some point, optimising harder against a noiseless simulator is
> self-defeating.

It does not follow, and this tests it directly rather than restating the
correlation. A search runs to budget 64 against the noiseless simulator; the
incumbent at each budget — the best candidate seen so far, monotone by
construction — is re-evaluated under a Lindblad channel. 32 held-out parents at
≤6 physical qubits, `eight_bin`, one record per parent.

## Result: the curve never turns up, at any rate tested

| budget | 1 | 2 | 4 | 8 | 16 | 32 | 64 | best budget | cost of the largest |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| noiseless | 0.74020 | 0.74020 | 0.71473 | 0.70661 | 0.68082 | 0.67879 | 0.67876 | — | — |
| rate 0.1 | 0.74815 | 0.74815 | 0.72539 | 0.71936 | 0.69677 | 0.69506 | 0.69503 | **64** | +0.00000, 0/32 |
| rate 0.2 | 0.75504 | 0.75504 | 0.73463 | 0.73027 | 0.71039 | 0.70894 | 0.70891 | **64** | +0.00000, 0/32 |
| rate 0.3 | 0.76103 | 0.76103 | 0.74265 | 0.73964 | 0.72207 | 0.72084 | 0.72081 | **64** | +0.00000, 0/32 |
| rate 0.5 | 0.77083 | 0.77083 | 0.75577 | 0.75471 | 0.74081 | 0.73991 | 0.73989 | **64** | +0.00000, 0/32 |

**Monotone at every rate.** The best budget under noise is always the largest
one tested, and spending it instead of any smaller budget costs exactly zero in
zero of 32 parents. **More noiseless search never hurt.** The slogan is wrong
and is retracted in the erosion report.

## What is there instead: a dose–response discount

The effect is real and it behaves exactly as the mechanism predicts — it just
never crosses zero. The fraction of the noiseless gain that survives the
environment falls steadily with the noise:

| dephasing rate | noiseless gain | gain under noise | **retained** |
|---:|---:|---:|---:|
| 0.1 | 0.06144 | 0.05312 | **86.5%** |
| 0.2 | 0.06144 | 0.04613 | **75.1%** |
| 0.3 | 0.06144 | 0.04022 | **65.5%** |
| 0.5 | 0.06144 | 0.03094 | **50.4%** |

At the strongest rate tested, **half of what a noiseless search buys is erased
by the environment** — and the other half is still a gain. That is the honest
shape of the result: a search curve that overstates its purchase by a factor
that grows with the noise, not a search that should be stopped early.

Extrapolating past rate 0.5 would be meaningless here: at that rate the
budget-1 loss is 0.77083 and the system is close to destroyed, so a crossing
inferred from this trend would be a crossing between two useless numbers.

## Why the ladder is long enough to answer the question

The noiseless curve is flat from budget 32 to 64 (0.67879 → 0.67876). The
search has converged, so "run it even longer" is not an untested option hiding
a turn-up — there is nothing left for a longer budget to find.

## Limits

- One family (`eight_bin`), one search seed, 32 parents, ≤6 physical qubits,
  one channel (dephasing, uniform, declared rate). Not a calibrated device
  model.
- Budgets 1–64. The closed-system search has converged by 32, but a different
  family or a genuinely larger space could behave differently.
- The incumbent ladder comes from a single search trace per record, so
  search-seed variability is not measured here.
