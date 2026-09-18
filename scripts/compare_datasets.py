"""Fail-closed comparison of NPZ dataset records, with explicit exclusions.

Example: python scripts/compare_datasets.py OLD/records NEW/records --output audit.json
Scientific numeric equality is exact by default. Wall clocks and provenance are
reported separately; this comparison never certifies byte equality or integrity
against a manifest. ``metadata_json`` is parsed and compared after recursively
removing only the declared non-scientific fields.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from zipfile import BadZipFile

import numpy as np

TIMING = {"candidate_seconds", "generation_seconds", "wall_seconds",
          "teacher_seconds", "elapsed_seconds", "search_seconds"}
PROVENANCE = {"source_fingerprint", "config_hash", "dataset_fingerprint",
              "fingerprint", "payload_fingerprint"}
EXCLUDED = TIMING | PROVENANCE


def _science_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _science_metadata(item) for key, item in value.items()
                if key not in EXCLUDED}
    if isinstance(value, list):
        return [_science_metadata(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("non-finite scientific metadata")
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value}")


def _compare_array(x: np.ndarray, y: np.ndarray, *, atol: float) -> tuple[str | None, float | None]:
    if x.shape != y.shape:
        return f"shape differs: {x.shape} vs {y.shape}", None
    if x.dtype != y.dtype:
        return f"dtype differs: {x.dtype} vs {y.dtype}", None
    if x.dtype.kind in "fc":
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            return "non-finite scientific values", None
        # Preserve the imaginary component; converting to float would hide it.
        with np.errstate(over="ignore", invalid="ignore"):
            delta = float(np.max(np.abs(x - y))) if x.size else 0.0
            equal = np.allclose(x, y, rtol=0.0, atol=atol)
        return (None if equal else "numeric values differ"), delta if np.isfinite(delta) else None
    return (None if np.array_equal(x, y) else "values differ"), None


def compare_datasets(old: Path, new: Path, *, atol: float = 0.0) -> dict:
    if not np.isfinite(atol) or atol < 0:
        raise ValueError("atol must be finite and nonnegative")
    old, new = Path(old), Path(new)
    for path in (old, new):
        if not path.is_dir():
            raise ValueError(f"records directory does not exist: {path}")
    old_names = {path.name for path in old.glob("*.npz")}
    new_names = {path.name for path in new.glob("*.npz")}
    if not old_names or not new_names:
        raise ValueError("both directories must contain at least one NPZ record")
    missing_old, missing_new = sorted(new_names - old_names), sorted(old_names - new_names)
    mismatches, excluded_changes = [], []
    worst_delta, identical = 0.0, 0
    for name in sorted(old_names & new_names):
        errors = []
        try:
            with np.load(old / name, allow_pickle=False) as a, np.load(new / name, allow_pickle=False) as b:
                if not a.files or not b.files:
                    errors.append("record has no arrays")
                if len(set(a.files)) != len(a.files) or len(set(b.files)) != len(b.files):
                    errors.append("record has duplicate array names")
                if not (set(a.files) - EXCLUDED) or not (set(b.files) - EXCLUDED):
                    errors.append("record has no scientific fields")
                for key in sorted(set(a.files) | set(b.files)):
                    if key not in a.files or key not in b.files:
                        errors.append(f"{key}: key is missing")
                        continue
                    x, y = a[key], b[key]
                    if key in EXCLUDED:
                        if x.shape != y.shape or x.dtype != y.dtype:
                            errors.append(f"{key}: excluded field schema differs")
                        elif not np.array_equal(x, y):
                            excluded_changes.append({"record": name, "key": key})
                        continue
                    if key == "metadata_json":
                        if x.shape != y.shape or x.shape != ():
                            errors.append("metadata_json: expected matching scalar JSON strings")
                            continue
                        left = json.loads(str(x.item()), parse_constant=_reject_constant)
                        right = json.loads(str(y.item()), parse_constant=_reject_constant)
                        if not isinstance(left, dict) or not isinstance(right, dict):
                            raise ValueError("metadata_json must contain an object")
                        if _science_metadata(left) != _science_metadata(right):
                            errors.append("metadata_json: scientific metadata differs")
                        if left != right:
                            excluded_changes.append({"record": name, "key": "metadata_json (includes excluded fields)"})
                        continue
                    error, delta = _compare_array(x, y, atol=atol)
                    if delta is not None:
                        worst_delta = max(worst_delta, delta)
                    if error:
                        errors.append(f"{key}: {error}")
        except (ValueError, TypeError, OSError, KeyError, EOFError, BadZipFile) as error:
            errors.append(f"unreadable record: {type(error).__name__}: {error}")
        if errors:
            mismatches.append({"record": name, "errors": errors})
        else:
            identical += 1
    passed = not (missing_old or missing_new or mismatches)
    return {"schema_version": 2, "verdict": "scientific_content_matches" if passed else "comparison_failed",
            "passed": passed, "old_records": len(old_names), "new_records": len(new_names),
            "compared_records": len(old_names & new_names), "matching_records": identical,
            "missing_from_old": missing_old, "missing_from_new": missing_new,
            "mismatches": mismatches, "excluded_field_changes": excluded_changes,
            "max_scientific_numeric_difference": worst_delta, "atol": atol, "rtol": 0.0,
            "excluded_fields": sorted(EXCLUDED),
            "scope": "scientific values and schema only; no byte equality, source equivalence, or manifest integrity claim"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old", type=Path)
    parser.add_argument("new", type=Path)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--output", type=Path, help="write the same machine-readable JSON emitted to stdout")
    args = parser.parse_args(argv)
    try:
        result = compare_datasets(args.old, args.new, atol=args.atol)
    except ValueError as error:
        result = {"schema_version": 2, "verdict": "invalid_input", "passed": False, "error": str(error)}
    rendered = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
