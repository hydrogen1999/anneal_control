"""Export a completed experiment's held-out rows as a committed audit input.

`rebuild_evidence.py` recomputes every contrast in EVIDENCE.md from committed
record-level outcomes rather than from a stored conclusion, which is what lets
`--check` detect drift. A new experiment only enters that audit once its rows
are in the tree in this shape.

The runner's `paper/results.json` already holds seed-averaged `record_means`
and the per-method/mode `summary`; this wraps them with the provenance the
audit expects and refuses anything the audit would later reject, so the
failure happens here rather than three commits downstream.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


# The audit reads only these. Everything else the evaluator stores -- per-record
# metrics dicts, selected_index, candidate_count -- is provenance for the run,
# not input to a contrast, and carrying it would triple the committed file for
# no auditable gain. The set matches the 2026-09-18 Pegasus export exactly, so
# a new export is diffable against the precedent.
AUDIT_FIELDS = ("bank_best_loss", "bank_regret", "family", "global_loss",
                "inference_seconds", "linear_loss", "logical_n", "loss", "method",
                "mode", "parent_id", "physical_n", "record_id", "runtime", "split")


def project(row):
    missing = [f for f in AUDIT_FIELDS if f not in row]
    if missing:
        raise ValueError(f"row {row.get('record_id')!r} lacks audit fields {missing}")
    return {field: row[field] for field in AUDIT_FIELDS}


def check_balanced(rows):
    """The same guard rebuild_evidence applies, applied early."""
    panels = {}
    for row in rows:
        if row.get("split") != "test":
            raise ValueError(f"row {row.get('record_id')!r} is split "
                             f"{row.get('split')!r}; the audit needs held-out test rows")
        panel = panels.setdefault((row["method"], row["mode"]), {})
        record = str(row["record_id"])
        if record in panel:
            raise ValueError(f"duplicate record {record} within {row['method']}/{row['mode']}")
        panel[record] = str(row["parent_id"])
    if not panels:
        raise ValueError("no rows to export")
    reference_key, reference = next(iter(panels.items()))
    for key, panel in panels.items():
        if panel != reference:
            raise ValueError(
                f"{key} covers a different record-parent population than {reference_key}; "
                "every method and mode must share one population or the contrasts are "
                "not paired")
    return len(panels), len(reference), len(set(reference.values()))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, help="the run's paper/results.json")
    parser.add_argument("--experiment", required=True, help="name recorded in the export")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    source = Path(args.results)
    payload = json.loads(source.read_text())
    for key in ("record_means", "summary"):
        if key not in payload:
            parser.error(f"{source} has no {key!r}; is this a completed report stage?")
    rows = payload["record_means"]
    n_panels, n_records, n_parents = check_balanced(rows)

    export = {"schema_version": 1, "experiment": args.experiment, "n_rows": len(rows),
              "record_means": [project(row) for row in rows], "summary": payload["summary"]}
    Path(args.output).write_text(json.dumps(export, indent=2, allow_nan=False) + "\n")
    print(f"{args.experiment}: {len(rows)} rows, {n_panels} method/mode panels, "
          f"{n_records} records, {n_parents} parents")
    print(f"  source sha256 {hashlib.sha256(source.read_bytes()).hexdigest()}")
    print(f"  wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
