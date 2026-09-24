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

## Before touching a number

```bash
python scripts/rebuild_evidence.py --check   # artifacts vs analysis code
python scripts/check_paper_numbers.py        # draft vs artifacts
```

The second refuses any four- or five-decimal quantity in these sections that
appears in no committed report. It compares **values at the draft's own
precision**, not substrings, and excludes record-level dumps from the corpus —
the full corpus holds ~838k numeric tokens, dense enough that a fabricated
`0.98765` matched two per-record losses. It is mutation-tested: injecting that
number fails the check.

What it cannot do is tell you a number means what the sentence claims. That
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
