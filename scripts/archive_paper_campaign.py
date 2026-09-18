"""Archive completed campaign evidence without silently claiming binary weights.

JSON/JSONL receipts and reports are preserved byte-for-byte with hashes. Dataset
NPZ files and model weights are represented by hashes in manifests and omitted
from this compact Git artifact; regenerate them from the frozen configuration.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path


def archive(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((root / "campaign.json").read_text())
    if manifest.get("status") != "complete":
        raise ValueError("only a completed campaign can receive a complete evidence archive")
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("campaign contains a symlink")
        if path.is_file() and path.suffix in {".json", ".jsonl", ".md"}:
            data = path.read_bytes()
            files[str(path.relative_to(root))] = {"sha256": hashlib.sha256(data).hexdigest(),
                                                "utf8": data.decode("utf-8")}
    artifact = {"schema_version": 1, "source_hash": manifest["source_hash"],
                "config_hash": manifest["config_hash"], "campaign_name": manifest["config"].get("name"),
                "omitted": "binary trained weights, dataset NPZ, figures; recorded hashes are not copies of those artifacts",
                "files": files}
    payload = json.dumps(artifact, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(gzip.compress(payload, mtime=0))
    return {"files": len(files), "compressed_bytes": output.stat().st_size,
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "source_hash": manifest["source_hash"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(archive(args.run, args.output), indent=2))


if __name__ == "__main__":
    main()
