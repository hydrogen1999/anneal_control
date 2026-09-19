# The prior method's own claim, at its own run count

## Why this had to be run

Our comparisons lean on an independent implementation of the GP-UCB schedule
search of Finžgar et al. (Phys. Rev. Research 6, 023063, 2024). A reviewer's
first question about that is whether the implementation is faithful, and until
now the only evidence was a **three-seed smoke test** in which GP-UCB beat
uniform random search in **one seed of three** — which the smoke report
correctly declined to interpret.

Three seeds is not the paper's run count. The config for eighty has existed
since 2026-09-17 and had never been executed.

## Result: the prior method's claim holds, and the smoke test was noise

Original-system p-spin setting, `N=15`, `p=3`, `Γ=5`, `T=3`, four real-space
parameters, `ζ=2` — the Fig. 4(b) configuration. **80 paired optimizer seeds,
9600 charged queries, 0 failures, 80/80 runs resolved.** Within each seed,
GP-UCB and uniform search are *required* to share their first ten parameter
points; the module raises rather than proceeding if they diverge.

| method | q25 | median | q75 |
|---|---:|---:|---:|
| GP-UCB | 0.20022 | **0.33316** | 0.34414 |
| uniform random | 0.14081 | **0.17188** | 0.21356 |

    GP-UCB − uniform random:  +0.106837  [+0.080929, +0.132300]
    GP-UCB wins 64 of 80 seeds

The interval excludes zero by a wide margin and the median fidelity is close to
double. **The three-seed 1-of-3 result was sample noise**, and the smoke
report's refusal to read anything into it was the right call.

## The artifact

`reference_80.json.gz` — all 80 seeds, both methods, every query with its
parameters, fidelity and solver diagnostics, plus the paired summary. Gzipped
to follow this repository's convention for large artifacts; it was first
committed uncompressed by mistake, and because this repository is shared with
another agent no history was rewritten to remove that blob.

## What this does and does not establish

**Does:** the baseline our method is compared against reproduces the prior
work's qualitative claim — Bayesian optimisation beats uniform search on the
authors' own system, at their own run count, in an independent implementation
with a different GP library, different acquisition optimiser and different
integrator. That is the property a comparison needs.

**Does not:** `figure_reproduction` stays **False** in the artifact. Nothing
here has been compared numerically against the published distribution; showing
the same qualitative ordering is weaker than reproducing the reported values,
and the flag should not be flipped on the strength of a matching sign. Every
disclosed deviation in `docs/literature_baseline.md` still applies.

This is also one fixed Hamiltonian. The bootstrap resamples optimizer seeds, so
the interval describes seed variability on that Hamiltonian and is not a
population claim over instances.
