# The policy already emits two-window controls, so the family cannot explain its gap

## The hypothesis this kills — mine, from earlier today

The Stage A gate showed `eight_bin` is the worst tunable family at a fixed
search budget (34.7–58.3 % of records exceed a two-percentage-point tolerance,
against 5.6–20.8 % for `two_window`), and `models.monotone_samples` decodes the
policy head to eight equal-time increments. I proposed:

> a design-level explanation for the gap this project has reported all along
> between direct generation (0.590) and bank selection (0.545) — and it is a
> hypothesis rather than a measurement

It is now a measurement, and it is **false**.

The reasoning had a hole worth naming: Stage A's family-restriction loss is
about how hard a family is to *search* at 32 objective calls. A trained network
does not search; it predicts. There is no reason the difficulty of searching a
family should transfer to a network that emits a point in it.

## Measurement

For 20 held-out parents, the trained policy's first proposal was fitted with
the best two-window control (six parameters, six restarts of Nelder–Mead on the
sup norm), then both were scored by the true simulator.

| quantity | value |
|---|---:|
| sup-norm residual of the two-window fit | mean **0.0022**, median 0.0021, max 0.0058 |
| widest gap between two bank candidates, for scale | **0.3956** |
| loss of the two-window projection minus the policy's own proposal | mean **+0.00020**, median +0.00014 |

**The policy's output is a two-window control to within 0.0022**, a residual
**180 times smaller** than the spacing between the candidates it is choosing
among. Projecting it exactly onto the two-window family costs 0.0002 — worse in
20 of 20 records, and by an amount two orders of magnitude below the effect
sizes this project reports.

## Consequences

1. **The family cannot explain the direct-versus-bank gap.** The head's
   effective family is already `two_window`, which is the family Stage A
   favours. Whatever makes direct generation weaker than bank selection, it is
   not that the head emits eight-bin shapes.
2. **A restricted two-window head is not worth building.** It would constrain
   the policy to a family it already occupies. That experiment is cancelled on
   the strength of this rather than run to find the same thing expensively.
3. **Stage A's finding stands, and its scope narrows.** It is a statement about
   search at a fixed budget — where it selected the bank's composition
   correctly, the bank containing `linear`, `one_window` ×3, `two_window` ×2
   and `pause` ×2 and no eight-bin candidate at all. It is not a statement
   about a learned policy's output.

The remaining explanation for the direct-versus-bank gap is the one the
existing proposal decomposition already gives: both ranking regret and a
generation gap against the bank, neither dominant.

## Limits

- 20 parents, ≤8 physical qubits, one checkpoint (`summary/seed_0`), the first
  of three proposals per record.
- The fit is a numerical optimum from six restarts, so the residual is an upper
  bound on the true distance to the family; a better fit would only strengthen
  the conclusion.
