# Paper documents

Draft sections for *Control-Relevant Representations for Embedded Ising
Annealing*. Markdown, to be transcribed to LaTeX once the section order is
settled.

| file | role |
|---|---|
| `CLAIMS.md` | **every** quantity the draft may use, with its artifact |
| `OUTLINE.md` | logic chain, paper type, four self-consistency checks |
| `00_abstract.md` | title options and two abstract lengths |
| `01_introduction.md` | six-paragraph plan and the contribution list |
| `02_setup.md` | embedded Ising, the control, cost classes |
| `03_data.md` | two populations, the candidate library, protocol |
| `04_method.md` | representation, heads, pool, training |
| `05_results.md` | the encoder contrast; effect size; replication |
| `06_pool.md` | pool-size axis; the critic overruling its generator |
| `07_negatives.md` | five controlled negatives |
| `08_limitations.md` | constraints on what may be concluded |
| `09_related.md` | **unverified** groupings — see the banner in that file |
| `FIGURES.md` | four figure briefs, each naming its artifact |
| `04a_architecture.md` | the model as implemented: features, blocks, variants, heads |
| `results/` | one JSON per table, exported mechanically; **derived, never hand-edit** |

## Before touching a number

```bash
python scripts/rebuild_evidence.py --check   # artifacts vs analysis code
python scripts/export_paper_results.py       # regenerate docs/paper/results/
python scripts/check_paper_numbers.py        # draft vs artifacts
python scripts/check_paper_numbers.py --self-test
```

The second refuses any four- or five-decimal quantity in these sections that
does not appear **verbatim** in a committed report. Run
`--self-test` for its measured power rather than trusting the description:

| fabricated value | caught |
|---|---:|
| 4 decimals | **96 %** |
| 5 decimals | **64 %** |

The 5-decimal figure is a hard limit. The curated corpus writes ~38.5k distinct
4–5 decimal tokens against 100k possible 5-decimal values in [0, 1), so a third
of invented numbers land on one by coincidence. Two weaker designs measured far
worse and both *looked* fine: substring matching against full-precision
artifact floats accepted everything, and matching rounded values accepted 97 %
of fabricated 4-decimal quantities.

So this is a **lint, not a guarantee**. It catches stale numbers, typos and
inventions; it cannot tell you a number means what its sentence claims. That
still needs a reader.

## Order of work

1. ~~logic chain and claim map~~ done
2. ~~section drafts~~ done
3. **verify `09_related.md` against the actual papers**, then run
   `scripts/verify_bib.py` — `MISMATCH` is the verdict that matters
4. render the four figures through `scripts/plot_utils.py`, check legibility at
   final column width
5. transcribe to LaTeX in the venue template
6. `pre-submission-reviewer` pass

## Known gaps carried into the draft

- No hardware; nothing above 14 physical qubits (§8.1–8.2).
- The ceiling is the candidate library, not the representation (§8.3).
- Factor 5 untestable on Pegasus; rung 4 never run; no covariant arm (§8.5).
- Related-work characterisations are unverified.
