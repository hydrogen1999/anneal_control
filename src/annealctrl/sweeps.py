"""Resumable record-level sweep driver shared by the G2 and G3 instruments.

A unit of sweep work is expensive (a whole equal-budget search trajectory) and
its value depends on the record, the budget, the seed, the families and the
numerical tolerances. Resuming by counting rows would splice results computed
under different settings into one file, so every unit is keyed by

    unit_key = sha256(fingerprint ‖ settings_hash ‖ source_hash)

and a resume that changes any of those three is refused rather than mixed.
See ``docs/decisions/ADR-0005-resume-by-content-fingerprint.md``.

Nothing here interprets a result: the worker's returned mapping is stored
verbatim under ``result``. Failures are recorded as rows, never swallowed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

from .telemetry import RunLog, _environment, _safe

ROWS = "rows.jsonl"
TELEMETRY = "telemetry.jsonl"
MANIFEST = "manifest.json"


@dataclass(frozen=True)
class SweepUnit:
    """One schedulable piece of work.

    ``fingerprint`` must change whenever the scientific content of the unit
    changes — for records it is the stored record fingerprint, for constructed
    pairs it is a hash of both compiled instances.
    """
    unit_id: str
    fingerprint: str
    payload: Any = field(default=None)

    def __post_init__(self) -> None:
        if not isinstance(self.unit_id, str) or not self.unit_id.strip():
            raise ValueError("unit_id must be a nonempty string")
        if not isinstance(self.fingerprint, str) or not self.fingerprint.strip():
            raise ValueError("fingerprint must be a nonempty string")

    def key(self, settings_digest: str, source_hash: str) -> str:
        payload = "␟".join((self.fingerprint, settings_digest, source_hash))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def settings_hash(settings: Mapping[str, Any]) -> str:
    """Stable digest of everything that can change a unit's result.

    Sorted keys make the digest order independent; ``allow_nan=False`` means a
    NaN threshold cannot silently become a distinct-but-equal setting.
    """
    if not isinstance(settings, Mapping):
        raise ValueError("settings must be a mapping")
    try:
        encoded = json.dumps(settings, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"settings must be JSON-serialisable: {error}") from error
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _read_rows(path: Path) -> tuple[list[dict], int]:
    """Parse the row log, repairing only a truncated **final** line.

    A hard kill can leave the last line half written. Corruption anywhere else
    means the file is not what it claims to be, and continuing would produce a
    result set nobody can reconstruct.
    """
    if not path.exists():
        return [], 0
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    parsed, repaired = [], 0
    for index, line in enumerate(lines):
        try:
            parsed.append(json.loads(line))
        except json.JSONDecodeError as error:
            if index != len(lines) - 1:
                raise ValueError(f"corrupt sweep row at line {index + 1} of {path}: {error}") from error
            repaired = 1
    if repaired:
        path.write_text("".join(json.dumps(row) + "\n" for row in parsed), encoding="utf-8")
    return parsed, repaired


def completed_rows(output: str | Path) -> dict[str, dict]:
    """Last successful row per ``unit_id``; failed and superseded rows dropped."""
    rows, _ = _read_rows(Path(output) / ROWS)
    latest: dict[str, dict] = {}
    for row in rows:
        if row.get("status") == "ok":
            latest[row["unit_id"]] = row
    return latest


def load_rows(output: str | Path) -> list[dict]:
    """Every row in order, including failures. Aggregators must see failures."""
    rows, _ = _read_rows(Path(output) / ROWS)
    return rows


def select_shard(units: Sequence[SweepUnit], index: int, count: int) -> list[SweepUnit]:
    """One shard of a sweep, for a job array. Shards are disjoint and cover everything.

    Assignment is by position in the id-sorted plan, so it does not depend on the
    order the caller happened to build the units in — two array tasks that
    enumerate the plan differently still agree on who owns what.
    """
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("shard count must be a positive integer")
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < count:
        raise ValueError(f"shard index must satisfy 0 <= index < {count}")
    ordered = sorted(units, key=lambda unit: unit.unit_id)
    return ordered[index::count]


def run_sweep(
    units: Sequence[SweepUnit],
    worker: Callable[[SweepUnit], Mapping[str, Any]],
    *,
    output: str | Path,
    settings: Mapping[str, Any],
    command: str,
    source_hash: str,
    resume: bool = False,
    dry_run: bool = False,
    on_error: str = "raise",
    extra_manifest: Mapping[str, Any] | None = None,
) -> dict:
    """Execute ``worker`` over ``units``, appending one JSONL row per unit.

    ``on_error='raise'`` stops the sweep at the first failure after recording it
    (the default: a numerical-gate failure usually means the settings are wrong
    for this distribution). ``on_error='record'`` continues, which is the right
    choice for an overnight campaign where a handful of hard instances are
    expected to exhaust their step budget. Either way the failure is a row, so
    it cannot vanish from the aggregate.
    """
    if on_error not in {"raise", "record"}:
        raise ValueError("on_error must be 'raise' or 'record'")
    if not isinstance(source_hash, str) or not source_hash.strip():
        raise ValueError("source_hash must be a nonempty string")
    units = list(units)
    seen = set()
    for unit in units:
        if not isinstance(unit, SweepUnit):
            raise TypeError("units must be SweepUnit instances")
        if unit.unit_id in seen:
            raise ValueError(f"duplicate unit_id {unit.unit_id!r} in sweep plan")
        seen.add(unit.unit_id)

    digest = settings_hash(settings)
    root = Path(output)

    if dry_run:
        return {"dry_run": True, "planned": len(units), "settings_hash": digest,
                "source_hash": source_hash, "command": command,
                "note": "requested workload only; no unit executed and no output written"}

    if root.exists() and not resume:
        raise FileExistsError(f"{root} exists; pass resume=True or choose a new output directory")
    root.mkdir(parents=True, exist_ok=True)

    previous, repaired = _read_rows(root / ROWS)
    for row in previous:
        if row.get("settings_hash") != digest:
            raise ValueError(
                f"existing rows in {root} used settings_hash {row.get('settings_hash')!r}; "
                f"this invocation uses {digest!r}. Those runs are not comparable — use a new output directory.")
        if row.get("source_hash") != source_hash:
            raise ValueError(
                f"existing rows in {root} used source_hash {row.get('source_hash')!r}; "
                f"this invocation uses {source_hash!r}. Refusing to mix results across source revisions.")
    done = {row["unit_key"] for row in previous if row.get("status") == "ok"}

    began = perf_counter()
    completed = skipped = failed = 0
    failed_ids: list[str] = []
    status = "ok"
    handle = (root / ROWS).open("a", encoding="utf-8")
    log = RunLog(root / TELEMETRY, command=command,
                 settings={"settings_hash": digest, "source_hash": source_hash,
                           "planned_units": len(units), "resume": resume, "on_error": on_error,
                           **_safe(dict(settings))})
    try:
        for unit in units:
            key = unit.key(digest, source_hash)
            if key in done:
                skipped += 1
                continue
            log.event("unit_start", unit=unit.unit_id)
            started = perf_counter()
            base = {"unit_id": unit.unit_id, "unit_key": key, "fingerprint": unit.fingerprint,
                    "settings_hash": digest, "source_hash": source_hash}
            try:
                result = worker(unit)
            except Exception as error:  # recorded, then re-raised unless told otherwise
                elapsed = perf_counter() - started
                _append(handle, {**base, "status": "failed", "error_type": type(error).__name__,
                                 "error": str(error)[:2000], "wall_seconds": elapsed})
                log.event("unit_failed", unit=unit.unit_id, error_type=type(error).__name__,
                          error=str(error)[:500], wall_seconds=elapsed)
                failed += 1
                failed_ids.append(unit.unit_id)
                if on_error == "raise":
                    raise
                continue
            if not isinstance(result, Mapping):
                raise TypeError(f"worker for unit {unit.unit_id!r} must return a mapping, got {type(result).__name__}")
            elapsed = perf_counter() - started
            _append(handle, {**base, "status": "ok", "wall_seconds": elapsed, "result": _safe(dict(result))})
            log.event("unit_end", unit=unit.unit_id, wall_seconds=elapsed)
            completed += 1
    except BaseException:
        # An aborted campaign still needs a readable manifest describing how far
        # it got; the exception propagates untouched after the finally block.
        status = "aborted"
        raise
    finally:
        handle.close()
        log.close(status=status, units=completed, failed=failed, skipped=skipped)
        manifest = {"schema_version": 1, "command": command, "status": status,
                    "settings": _safe(dict(settings)), "settings_hash": digest,
                    "source_hash": source_hash, "planned": len(units), "completed": completed,
                    "skipped": skipped, "failed": failed, "failed_unit_ids": failed_ids,
                    "repaired_truncated_rows": repaired, "wall_seconds": perf_counter() - began,
                    "resumed": resume, "on_error": on_error, "environment": _environment(),
                    "rows": ROWS, "telemetry": TELEMETRY,
                    **(_safe(dict(extra_manifest)) if extra_manifest else {})}
        (root / MANIFEST).write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


def _append(handle, row: Mapping[str, Any]) -> None:
    handle.write(json.dumps(row, allow_nan=False) + "\n")
    handle.flush()
