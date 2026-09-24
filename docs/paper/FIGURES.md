# Figure plan

Four figures. Every plotted quantity names the committed artifact it comes
from; nothing is drawn from a number that is not in `CLAIMS.md`. Rendering goes
through `scripts/plot_utils.py` (`pdf.fonttype = 42`, venue column widths, and
a refusal to save a figure embedding Type 3).

**Not yet rendered.** These are briefs. Legibility, clipping and caption
overflow must be checked on the real render at final column width before any
of them is called done.

---

## Figure 1 — Why the posed problem is not the executed one

```
Claim/evidence ID:      motivating; supports §2.1, no statistical claim
Reader question:        What does the annealer actually run, and why would the
                        best schedule differ from the logical problem's?
What varies/fixed:      One real logical instance, held fixed; what varies is
                        the representation of it (logical vs embedded).
Panels/reading order:   (a) logical graph G_L with h, J.
                        (b) the same instance after minor embedding: branch
                            sets, penalty couplings at strength kappa, and the
                            programmed scale alpha annotated.
                        (c) two schedules and their losses on THIS instance —
                            the one a logical-spectrum rule picks, and the one
                            the embedded problem prefers.
Data/formal object:     A real record: `membership`, `logical_edges`,
                        `physical_edges`, `chain_strength`, `programmed_scale`
                        from the Pegasus-240 test split. Panel (c) losses from
                        that record's stored candidate losses.
Mapping to symbols:     G_L, C_v, kappa, alpha, s(tau) exactly as in §2.
Main contrast:          Same instance, two problems.
Necessary exception:    Panel (c) is ONE record chosen for legibility. Label it
                        as an illustration, not as the effect size — the effect
                        is Figure 2.
Caption takeaway:       The device executes a larger, differently connected,
                        rescaled Hamiltonian; the control problem is posed on
                        that one.
Output:                 single column, vector
```

**Build notes.** Verify from the graph data that each branch set is nonempty,
connected and disjoint, and that every logical edge is realised by a physical
edge — do not trust the drawing. Distinguish branch sets by **shape or label as
well as colour**. Do not describe a branch set as a "path" unless it is one.
Mark unused hardware qubits if they help the working graph read correctly.

---

## Figure 2 — The central claim

```
Claim/evidence ID:      C1 (§5.1)
Reader question:        Does seeing the embedding matter, and does it matter
                        which architecture sees it?
What varies/fixed:      Encoder variant varies; width, depth, training, data,
                        candidate set identical.
Panels/reading order:   (a) Pegasus, 48 parents: all ten pairwise differences
                            as a forest plot, ordered by Holm p.
                        (b) synthetic, 48 parents: the same ten.
                        Reading line: the four logical-vs-aware rows sit clear
                        of zero; every aware-vs-aware row straddles it.
Data:                   `reports/evidence_audit_2026-09-18/contrasts.json`,
                        datasets `pegasus240` and `synthetic`, `bank_pairs`.
Main contrast:          logical vs {physical, summary, hierarchy x2}.
Necessary exception:    The hierarchy_physics-hierarchy_outcome row is a
                        DEGENERACY on Pegasus (resolved response = 0), not a
                        tight null. Draw it in a distinct style and say so in
                        the caption; do not let it read as evidence.
Uncertainty:            95% parent bootstrap; Holm-corrected p annotated.
                        Non-rejection sets shown as a bracket, labelled as
                        sets, not as equivalence.
Caption takeaway:       An embedding-blind encoder separates from every aware
                        encoder on both topologies; no two aware encoders
                        separate from each other.
Output:                 double column, vector
```

**Build notes.** Do not sort by effect size — sort by Holm p so the corrected
decision is the reading order. Zero line must be visible and unclipped. If
space forces one panel, keep Pegasus: it is the arm that was previously null.

---

## Figure 3 — What the critic is worth, in two honest units

```
Claim/evidence ID:      C2, C3 (§5.2, §6.4)
Reader question:        How much is one forward pass worth, and what sets it?
Panels/reading order:   (a) Quality-budget curve: mean gain over a linear ramp
                            against POOL SIZE, four measured points, both
                            topologies on one axis.
                        (b) The same result in simulator calls: forward pass
                            vs a perfect ranker on the same library, both
                            topologies, with the strategy range as a band.
                        (c) Fallback rate against that seed's generator
                            quality, one point per training seed, both
                            topologies.
Data:                   (a) `pool_selection_2026-09-20/artifacts/*` plus the
                            effect-size artifacts for the 64-point.
                        (b) `effect_size_2026-09-19/artifacts/*equiv*` and
                            `pegasus240_2026-09-20/artifacts/equiv_*`.
                        (c) the same pool artifacts; x = proposal-only gain.
Main contrast:          (a) monotone in pool size on both topologies.
Necessary exception:    (b) the call equivalent MOVES with search strategy
                        (15.5 / 17.9 / 23.3 on synthetic). Show the range as a
                        band, not the Sobol number alone.
Uncertainty:            95% parent bootstrap on (a) and (b).
Caption takeaway:       Selection scales with the pool offered; one forward
                        pass is worth about 24 instance-specific calls; and the
                        fallback rate rises as the generator degrades.
Output:                 double column, vector; (c) may be dropped to a margin
                        panel if space is tight
```

**Build notes.** Panel (a) is a line chart whose y-range need not include zero,
but the range must be stated and must not hide the flattening between 8 and 64.
Panel (c) is three or five points per topology — plot them as points with seed
labels, never as a fitted line; eight points do not support a slope.

---

## Figure 4 — The ladder cuts both ways

```
Claim/evidence ID:      C4 (§7.1, §7.2)
Reader question:        Does more spectral resolution help?
Panels/reading order:   (a) resolution as a CONTROL RULE: d2 vs gap-only vs
                            linear, paired differences with CIs.
                        (b) resolution as a SUPERVISION TARGET: none vs 3
                            moments vs 8 bins, the same style, same x-range.
                        Reading line: (a) clears zero, (b) sits on it.
Data:                   (a) `spectral_ladder_2026-09-19/`;
                        (b) `spectral_rung3_2026-09-20/rung3_contrasts.json`.
Main contrast:          Same axis, same units, opposite outcome.
Necessary exception:    (b) is a non-separation, not a demonstration of
                        equality. Caption must say the interval bounds any
                        effect under 2% of the method's own gain rather than
                        claiming none exists.
Uncertainty:            95% parent bootstrap; Holm p annotated on both.
Caption takeaway:       Spectral resolution is worth a lot as a control rule
                        and nothing as a supervision target.
Output:                 single column, vector
```

**Build notes.** The two panels **must share an x-range**, or the contrast that
is the entire point disappears. The scalar-bottleneck result (§7.2) is a
candidate third panel; drop it before shrinking the shared range.

---

## Deferred

- A failure-distribution heatmap for the parents the selector does not win.
  Currently uninformative on Pegasus (it wins all 48); revisit if a harder
  population is added.
- An open-system panel. The dephasing and relaxation studies exist but are a
  robustness check, not a claim of the paper, and a panel would overstate them.
