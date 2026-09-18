# Warm-starting a search from the learned selection: a small effect that does not buy budget

## Why this was run

The evidence ledger names warm-start experiments twice — as one of two routes to
a method claim, and under deployment value. The comparison table has two isolated
points, an amortised selector at zero simulator calls and a search at 257, and
reading that as "the learned method loses by 0.039" answers the wrong question.
The deployment question is how many calls a search needs when it starts from the
model's answer.

## Design

Both arms search the same five control families with the same per-family budget
and the same seed. The only difference is the incumbent:

- **cold** starts from the linear schedule
- **warm** spends one of its own budget calls evaluating the model's
  highest-ranked bank candidate of that family, then refines from there

The hint is **charged**, not free. Trial 0 stays linear in both arms, so the
reference every headroom number is defined against is untouched. The model
consults no outcome — it ranks the stored bank and the hint is its top pick.
432 records, 24 parents.

## Result

| budget | cold | warm | difference | 95% parent CI | separated |
|---:|---:|---:|---:|---|:--:|
| 8 | 0.5688 | 0.5616 | **−0.00719** | [−0.01106, −0.00336] | **yes** |
| 16 | 0.5599 | 0.5583 | −0.00159 | [−0.00389, +0.00099] | no |
| 32 | 0.5514 | 0.5512 | −0.00014 | [−0.00127, +0.00094] | no |
| 64 | 0.5473 | 0.5463 | −0.00104 | [−0.00211, −0.00014] | marginal |

**The effect is real at the smallest budget and washes out.** At 8 calls per
family the hint is worth −0.0072 and 18 of 24 parents favour it. By 16 calls the
interval crosses zero. The separation at 64 is marginal, its upper bound almost
touching zero, and the pattern across budgets is non-monotonic — so it should not
be read as the effect returning.

## The headline this does not support

The hoped-for result was a budget saving: *warm start reaches cold-search quality
using far fewer calls*. It does not.

**warm at budget 8 is 0.5616. Cold at budget 64 is 0.5473.** The warm arm never
gets far enough ahead to let a smaller budget stand in for a larger one. There is
no crossing point in this range, and the honest summary is that a learned
selection is worth about seven thousandths of a loss unit when a search is
starved, and nothing once it is not.

## Why this is the expected shape, in hindsight

Warm starting trades exploration for exploitation. A cold search has no incumbent
until an exploratory trial produces one, so its early calls all explore; a hint
fills the incumbent immediately and half the remaining budget becomes local
refinement. That is a good trade when there are few calls to spend and a bad one
when there are many, because a cold search with 64 calls finds a comparable point
on its own.

A unit test in `tests/test_search.py` pins both directions of that trade,
including the case where a bad hint loses to a cold start.

## What this does not cover

**One hint per family, from one checkpoint.** `summary`, seed 0. A stronger or
differently-trained selector might give a better hint; that is untested.

**Budgets 8 to 64 per family.** The campaign's own budget is 64, and 257 total
objective calls per record. Behaviour below 8 is unmeasured.

**Refinement is the search's existing local move.** No attempt was made to design
a refinement operator suited to a learned hint, which is the obvious next thing
to try and is not tried here.

**Bank selection only.** The hint is a bank candidate; warm starting from the
policy's *direct* proposal is a different experiment, and given that those
waveforms sit off the bank manifold it would likely behave differently.
