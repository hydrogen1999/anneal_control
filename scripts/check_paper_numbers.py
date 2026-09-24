"""Refuse a paper number that no committed artifact contains.

The session that produced this draft spent a morning removing a published
figure that was computed against one denominator and described as another. It
survived because prose was the only place it existed: nothing to diff, no
command to re-run.

So every four- or five-decimal quantity in docs/paper/ must appear **verbatim**
in a committed report or curated artifact.

How much this is worth, measured rather than assumed (`--self-test`):

    a fabricated 4-decimal value is caught ~96% of the time
    a fabricated 5-decimal value is caught ~64% of the time

The 5-decimal figure is a hard limit, not a bug: the curated corpus writes
~38.5k distinct 4-5 decimal tokens, against 100k possible 5-decimal values in
[0,1), so a third of invented values land on one by coincidence. Two weaker
designs were tried and both measured far worse -- substring matching against
full-precision artifact floats accepted everything, and matching rounded values
accepted 97% of fabricated 4-decimal quantities.

This is a lint, not a guarantee, and it cannot tell you a number means what its
sentence claims. It catches stale numbers, typos, and inventions.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

NUMBER = re.compile(r"\d\.\d{4,5}")
def _tokens(text: str) -> set[str]:
    """Four- and five-decimal quantities as the corpus actually writes them.

    Verbatim, deliberately. Substring matching lets a fabricated 0.98765 hide
    inside a stored 0.9876543..., and rounding stored floats to four places
    builds a set dense enough to accept 97% of invented values. Reports quote
    numbers at the precision they are claimed at, so requiring the exact token
    is both tighter and closer to what the check is for.
    """
    return set(NUMBER.findall(text))


# Record-level dumps are excluded on purpose. The full corpus holds ~838k
# numeric tokens, mostly per-record losses and bootstrap draws, and at that
# density every five-decimal value finds a host -- a fabricated 0.98765 matched
# two per-record losses. A paper number should appear in a REPORTED finding,
# not merely somewhere inside a data file, so the corpus is the curated reports
# plus the small committed analysis artifacts.
DUMP_KEYS = ('"records"', '"record_means"', '"rows"')


def _is_dump(text: str) -> bool:
    return any(key in text[:4000] for key in DUMP_KEYS)


def corpus(root: pathlib.Path) -> str:
    parts = []
    for path in (root / "reports").rglob("*"):
        if path.is_file() and path.suffix in (".md", ".json"):
            try:
                text = path.read_text()
            except (UnicodeDecodeError, OSError):
                continue
            if path.suffix == ".json" and _is_dump(text):
                continue
            parts.append(text)
    claims = root / "docs/paper/CLAIMS.md"
    if claims.exists():
        parts.append(claims.read_text())
    if not parts:
        raise SystemExit("no reports found; run from the repository root")
    return "\n".join(parts)


def _self_test(known: set[str]) -> int:
    """Report the false-accept rate instead of asserting the check is sound."""
    import random
    rng = random.Random(0)
    print(f"corpus holds {len(known)} distinct 4-5 decimal tokens")
    worst = 0.0
    for places in (4, 5):
        trials = 20000
        accepted = sum(1 for _ in range(trials)
                       if f"{rng.random():.{places}f}" in known)
        rate = accepted / trials
        worst = max(worst, rate)
        print(f"  fabricated {places}-decimal value caught "
              f"{100 * (1 - rate):.1f}% of the time")
    # A check that accepts most inventions is worse than none, because it
    # reassures. Fail loudly if it ever degrades that far.
    if worst > 0.5:
        print("\nFAIL: this check now accepts the majority of fabricated values.")
        return 1
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--paper", default="docs/paper")
    parser.add_argument("--self-test", action="store_true",
                        help="measure how often a fabricated value is wrongly accepted, "
                             "so the docstring's claim about this check stays true")
    args = parser.parse_args(argv)

    root = pathlib.Path(args.root).resolve()
    known = _tokens(corpus(root))
    if args.self_test:
        return _self_test(known)
    failures, total = {}, 0
    for path in sorted((root / args.paper).glob("*.md")):
        # Plans, the claim map, and the index are not draft prose: they quote
        # targets, formats, and -- in the README's description of this script --
        # a deliberately fabricated number as an example.
        if path.name in {"CLAIMS.md", "FIGURES.md", "OUTLINE.md", "README.md"}:
            continue
        found = sorted(set(NUMBER.findall(path.read_text())))
        total += len(found)
        unverified = [n for n in found if n not in known]
        if unverified:
            failures[path.name] = unverified

    for name, bad in failures.items():
        print(f"{name}: {len(bad)} unverified -> {bad}")
    if failures:
        print(f"\nFAIL: {sum(len(v) for v in failures.values())} of {total} "
              "quantities appear in no committed artifact.")
        return 1
    print(f"OK: all {total} quantities in {args.paper} appear in a committed artifact.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
