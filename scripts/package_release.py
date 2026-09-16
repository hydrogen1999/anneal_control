"""Package source and selected generated evidence without private source attachments."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    destination = Path(args.output).resolve()
    if destination.exists():
        raise FileExistsError("release archive exists; choose a new versioned path")
    allowed_roots = {"src", "tests", "configs", "docs", "scripts", "reports", ".github"}
    allowed_files = {"README.md", "RUNBOOK_VI.md", "IMPLEMENTATION_PLAN_VI.md", "pyproject.toml", ".gitignore"}
    forbidden = {"__pycache__", ".pytest_cache", ".venv"}
    files = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if not path.is_file() or any(part in forbidden or part.endswith(".egg-info") for part in relative.parts):
            continue
        if relative.parts[0] not in allowed_roots and str(relative) not in allowed_files:
            continue
        if path.suffix in {".pyc", ".aux", ".log", ".out", ".lock", ".tmp"}:
            continue
        files.append(path)
    checksum_path = root / "reports" / "release_source_checksums.json"
    source = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in files if p.suffix in {".py", ".json", ".toml", ".tex", ".yml"}
              and p != checksum_path and "smoke" not in p.relative_to(root).parts
              and "v02" not in p.relative_to(root).parts}
    checksum_path.write_text(json.dumps(source, indent=2, sort_keys=True), encoding="utf-8")
    if checksum_path not in files:
        files.append(checksum_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, arcname=str(Path("anneal_control") / path.relative_to(root)))
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise IOError("archive checksum verification failed")
    print(json.dumps({"archive": str(destination), "files": len(files),
                      "bytes": destination.stat().st_size,
                      "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
