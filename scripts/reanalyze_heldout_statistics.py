#!/usr/bin/env python3
"""Archive corrected contrast semantics without changing historical artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from annealctrl import contrasts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    payload = json.loads(args.records.read_text())
    rows = payload["record_means"]
    result = {
        "schema_version": 2,
        "source": str(args.records),
        "source_sha256": hashlib.sha256(args.records.read_bytes()).hexdigest(),
        "analysis_source_sha256": hashlib.sha256(Path(contrasts.__file__).read_bytes()).hexdigest(),
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_seed": args.seed,
        "scope": "Reanalysis of archived seed-averaged rows; no new training or simulation.",
        "joint_parent_seed_interval": {
            "status": "unavailable",
            "reason": "record_means were averaged over seeds; per-seed records are required",
        },
        "modes": {},
    }
    for mode in ("bank", "direct"):
        result["modes"][mode] = {
            "contrast_matrix": contrasts.contrast_matrix(
                rows, mode=mode, bootstrap_resamples=args.bootstrap_resamples, seed=args.seed),
            "embedding_information": contrasts.encoder_information_contrast(
                rows, mode=mode, bootstrap_resamples=args.bootstrap_resamples, seed=args.seed),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for mode, block in result["modes"].items():
        matrix = block["contrast_matrix"]
        print(mode, "Holm-separated pairs:", matrix["n_separated_after_correction"])
        print("maximal non-rejection sets:", matrix["nonseparated_maximal_sets"])
        effect = block["embedding_information"]
        print("exploratory pooled aware-minus-blind:", effect["mean_difference"],
              [effect["ci_low"], effect["ci_high"]])


if __name__ == "__main__":
    main()
