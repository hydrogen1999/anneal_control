# Embedding information on real device connectivity, at 48 held-out parents

## What this run was for

The embedding-information claim — that an encoder which can see the minor
embedding beats one that cannot — was **synthetic-only**. On Pegasus it had
been tested at 12 held-out parents, where the Holm-corrected matrix separated
**nothing**: all five encoders sat in one non-rejection set, and the pooled
aware-minus-blind interval was [−0.020100, +0.001087], straddling zero.

Twelve parents is not much to conclude from. This run trains the same five
encoders on a 240-parent Pegasus dataset with **48 held-out parents**.

It is not an extension of the earlier arm but an **independent replication**.
The two dataset configurations are identical in every substantive parameter —
64-candidate bank, four balanced instance families, logical sizes 5/6/7, a
14-qubit cap, runtimes 4 and 12, `hardware_growth` on Pegasus, the same
tolerances, no teacher — and differ only in the generator seed (20260918
against 20260919) and the parent count (96 against 240). The held-out sets
share **no parent**.

## Result: the claim now holds on Pegasus

Holm-corrected over all ten pairwise comparisons, bank selection, 48 parents,
three training seeds averaged per record.

| comparison | difference | 95 % CI | Holm *p* | |
|---|---:|---|---:|---|
| logical − hierarchy_outcome | +0.01401 | [+0.00861, +0.02017] | **0.0005** | separated |
| logical − summary | +0.01559 | [+0.01011, +0.02179] | **0.0005** | separated |
| logical − hierarchy_physics | +0.01401 | [+0.00858, +0.02021] | **0.0008** | separated |
| logical − physical | +0.01252 | [+0.00678, +0.01906] | **0.0014** | separated |
| summary − physical | −0.00307 | [−0.00672, +0.00058] | 0.5991 | — |
| physical − hierarchy_* | +0.00149 | [−0.00040, +0.00352] | 0.6562 | — |
| summary − hierarchy_* | −0.00159 | [−0.00463, +0.00162] | 0.9489 | — |
| hierarchy_physics − hierarchy_outcome | +0.00000 | [0, 0] | 1.0000 | *see caveat* |

Maximal non-rejection sets: **{hierarchy_outcome, hierarchy_physics, physical,
summary}** and **{logical}**, alone.

Pooled aware against blind: **−0.01403 [−0.01993, −0.00877]**, *p* ≈ 1e-4,
favouring the aware encoders in **43 of 48** parents.

**The four separated pairs are exactly the four logical-versus-aware
contrasts, and every aware-versus-aware pair is non-separated.** That is the
same structure the synthetic set produced (Holm 0.0005–0.0049 there, 0.0005–
0.0014 here): seeing the embedding helps, and which architecture consumes it
does not measurably matter. A non-rejection set is not an equivalence class.

## Caveat that changes what one arm means

`resolved_response_fraction` on this dataset is **0.0**.

`hierarchy_physics` and `hierarchy_outcome` differ in exactly one training
key, `response_weight` 0.05 against 0.0, and that weight multiplies a loss
term supervised by resolved spectral response. With no resolved response in
the data there is nothing for it to weight, and the two arms are **identical
on every record of every seed** — mean loss agreeing to eight decimals at
0.60147424, 0.60079890, 0.60295049. Their checkpoints hash differently only
because the stored training config differs.

So the `+0.00000 [0, 0]` entry is an **identity, not a measurement**. This run
cannot speak to the design document's Factor 5 (learning objective, ±
auxiliary physics) at all; the manipulation did not happen. The synthetic
arm, where `resolved_response_fraction` is 0.799, remains the only place that
factor has been tested, and its tight null stands there. The same applies to
the 12-parent Pegasus run, whose response fraction is also 0.0.

Two consequences for reading the table above:

- The Holm family nominally has ten members but **one pair is degenerate by
  construction**, so the correction is applied over more comparisons than
  carry information. That is conservative — it makes separation harder, not
  easier — so the four separated pairs survive the stricter treatment.
- The pooled aware mean averages four encoders of which two are the same
  model, giving the hierarchy architecture **half the weight**. All four beat
  `logical` individually and each separates from it after correction, so the
  direction does not depend on the weighting, but the pooled point estimate
  is not a clean four-way average.

## Direct generation: no encoder effect, and no reliable effect at all

In direct mode **zero** of ten pairs separate after correction and all five
encoders sit in one set; pooled aware-minus-blind is −0.00759 [−0.01724,
+0.00203], crossing zero.

That is consistent with the direct policy being weak here in a stronger sense:
it changes sign across training seeds, −0.00890 / +0.00230 / +0.00302 against
a matched linear ramp, pooling to −0.00119 [−0.01380, +0.01202] and beating
the ramp on 26 of 48 parents. Bank selection on the identical records and
seeds is −0.08761 / −0.08541 / −0.08590, winning **48 of 48 in every seed**.

## Provenance

Dataset reused by verification rather than regeneration: the probe tree that
produced it still hashes to the `source_fingerprint` its manifest records, so
`generate_dataset(resume=True)` validated `config_hash`, `schema_version` and
that fingerprint before accepting it. Zero generation lines in the run log.

Contrasts computed by `scripts/reanalyze_heldout_statistics.py` over the
runner's own `paper/results.json`, 20 000 parent bootstrap resamples, seed 0.
