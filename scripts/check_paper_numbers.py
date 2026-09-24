"""Refuse a paper number that no committed artifact contains.

The session that produced this draft spent a morning removing a published
figure that was computed against one denominator and described as another. It
survived because prose was the only place it existed: nothing to diff, no
command to re-run.

So every four- or five-decimal quantity in docs/paper/ must appear in a
committed report or artifact. This does not check that the number means what
the sentence says it means -- no script can -- but it does make an invented or
stale number fail loudly, which is the failure mode that actually occurred.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

NUMBER = re.compile(r"\d\.\d{4,5}")
# Any number at all, at whatever precision an artifact happens to store it.
ANY_NUMBER = re.compile(r"\d+\.\d+")


def _rounded(text: str) -> dict[int, set[str]]:
    """Every number in `text`, indexed by decimal places.

    Two things this must not do. It must not substring-match: artifacts store
    full-precision floats, so "0.98765" occurs inside 0.9876543... and any
    fabricated quantity would find a host. And it must not accept a match at a
    coarser precision than the draft quotes: rounding thousands of stored
    floats to four places makes that set dense enough to absorb almost
    anything. A draft number is therefore checked only at its own precision.
    """
    out: dict[int, set[str]] = {4: set(), 5: set()}
    for raw in ANY_NUMBER.findall(text):
        try:
            value = float(raw)
        except ValueError:
            continue
        for places in out:
            out[places].add(f"{value:.{places}f}")
    return out


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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--paper", default="docs/paper")
    args = parser.parse_args(argv)

    root = pathlib.Path(args.root).resolve()
    known = _rounded(corpus(root))
    failures, total = {}, 0
    for path in sorted((root / args.paper).glob("*.md")):
        if path.name in {"CLAIMS.md", "FIGURES.md", "OUTLINE.md"}:
            continue  # plans and the map itself may quote targets and formats
        found = sorted(set(NUMBER.findall(path.read_text())))
        total += len(found)
        unverified = []
        for n in found:
            places = len(n.split(".")[1])
            if f"{float(n):.{places}f}" not in known.get(places, set()):
                unverified.append(n)
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
