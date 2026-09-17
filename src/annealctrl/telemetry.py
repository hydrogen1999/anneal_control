"""Append-only JSONL run telemetry for long sweeps on a detached host.

Operational only. Nothing written here participates in a scientific claim: the
authoritative costs, losses and diagnostics remain the fields of the results
JSON produced by ``benchmarking``/``headroom``/``interventions``. This module
exists so an operator on ``apollo`` or ``goose`` can answer, mid-run and with
nothing but ``jq``, which unit is executing, what the throughput is, how much
memory was resident at peak, and which units failed a numerical gate.

See ``docs/decisions/ADR-0004-jsonl-telemetry-no-framework.md``.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import platform
import re
import resource
import sys
from time import perf_counter
from typing import Any, Mapping
import uuid

# Names that must never reach a log file. Matched case-insensitively as a
# substring, so ``AUTH_TOKEN`` and ``s2_api_key`` are both refused. Telemetry is
# the classic accidental exfiltration path; refusing at write time is cheaper
# than reviewing every call site.
_CREDENTIAL = re.compile(r"secret|token|password|passwd|api[_-]?key|credential|bearer|private[_-]?key", re.IGNORECASE)

# Fields the log owns. A caller overwriting ``seq`` or ``run_id`` would make the
# file unorderable and uncorrelatable, which is the whole point of having them.
_RESERVED = frozenset({"run_id", "seq", "t", "event"})

_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def peak_rss_bytes() -> int:
    """Peak resident set size of this process, normalised to bytes.

    ``ru_maxrss`` is kilobytes on Linux and bytes on macOS; getting this wrong
    silently misreports memory by 1024x. The value is a coarse high-water mark
    for the whole process, not an allocation of memory to one unit of work.
    """
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw if sys.platform == "darwin" else raw * 1024


def _safe(value: Any) -> Any:
    """JSON RFC 8259 has no NaN/Infinity; unresolved quantities become null."""
    if isinstance(value, Mapping):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if hasattr(value, "item") and getattr(value, "ndim", None) == 0:
        return _safe(value.item())
    if hasattr(value, "tolist"):
        return _safe(value.tolist())
    return str(value)


def _reject_credentials(names) -> None:
    for name in names:
        if _CREDENTIAL.search(str(name)):
            raise ValueError(f"refusing to log credential-shaped field {name!r}")


class RunLog:
    """One invocation's event stream, appended to ``path``.

    Reopening an existing path appends under a **new** ``run_id`` with ``seq``
    restarting at zero, so a resume boundary stays visible in the file instead of
    being disguised as one continuous run.
    """

    def __init__(self, path: str | Path, *, command: str, settings: Mapping[str, Any],
                 run_id: str | None = None) -> None:
        if not isinstance(command, str) or not command.strip():
            raise ValueError("command must be a nonempty string")
        if not isinstance(settings, Mapping):
            raise ValueError("settings must be a mapping")
        _reject_credentials(settings)
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or uuid.uuid4().hex
        self._seq = 0
        self._started = perf_counter()
        self._closed = False
        self._handle = self.path.open("a", encoding="utf-8")
        self.event("run_start", command=command, settings=_safe(dict(settings)),
                   environment=_environment(), pid=os.getpid(),
                   telemetry_is_operational_not_scientific_evidence=True)

    def event(self, name: str, **fields: Any) -> None:
        if self._closed:
            raise RuntimeError("cannot write to a closed RunLog")
        if not isinstance(name, str) or not _IDENTIFIER.match(name):
            raise ValueError("event name must be a nonempty identifier")
        overwritten = _RESERVED.intersection(fields)
        if overwritten:
            raise ValueError(f"reserved field names cannot be overwritten: {sorted(overwritten)}")
        _reject_credentials(fields)
        line = {"run_id": self.run_id, "seq": self._seq, "t": round(perf_counter() - self._started, 6),
                "event": name, **{key: _safe(value) for key, value in fields.items()}}
        self._seq += 1
        self._handle.write(json.dumps(line, allow_nan=False) + "\n")
        self._handle.flush()

    def close(self, *, status: str = "ok", **totals: Any) -> None:
        """Write the terminal ``run_end``. Idempotent: a second call is a no-op."""
        if self._closed:
            return
        self.event("run_end", status=status, wall_seconds=round(perf_counter() - self._started, 6),
                   peak_rss_bytes=peak_rss_bytes(), peak_rss_is_estimate=True, **totals)
        self._closed = True
        self._handle.close()

    def __enter__(self) -> "RunLog":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if exc_type is None:
            self.close()
        else:
            self.close(status="failed", error_type=exc_type.__name__, error=str(exc)[:500])
        return False


def _environment() -> dict:
    return {"python": platform.python_version(), "platform": platform.platform(),
            "machine": platform.machine(), "hostname": platform.node()}


def unknown_config_keys(config, allowed) -> list[str]:
    """Keys in ``config`` that are neither allowed nor annotations.

    Underscore-prefixed keys are annotations: provenance notes, scope caveats,
    the reason a config file exists. Every config validator in this package has
    to permit them and reject everything else, so that a genuine typo like
    ``budgte`` still fails loudly instead of silently taking a default.

    This exists because that rule was written out separately in three places and
    fixed one place at a time, each time after four sharded jobs had already died
    on ``unknown ... keys: ['_scope_note']``. One implementation, three callers.
    """
    return sorted({str(key) for key in config if not str(key).startswith("_")} - set(allowed))
