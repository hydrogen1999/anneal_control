"""Would a bigger menu close the gap to instance-specific search?

The share factorisation names bank_coverage as the binding factor -- the
critic extracts most of its menu, but the menu holds much less than a search
finds. The natural reply is "use a bigger menu". This measures that, using the
candidate losses already stored with every record, so it costs no simulation.

The bank is shared across records, so a menu of size k is one fixed subset
applied everywhere. Reported beside it is the headroom an instance-specific
search reaches, which is what the menu is trying to buy.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from annealctrl.effect_size import menu_size_curve


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluations", required=True)
    parser.add_argument("--data", required=True, help="dataset root holding records/*.npz")
    parser.add_argument("--method", default="summary")
    parser.add_argument("--output", required=True)
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frontier-sweep", nargs="*", default=None,
                        help="optional: report the search headroom beside the curve")
    args = parser.parse_args(argv)

    paths = sorted(glob.glob(str(Path(args.evaluations) / f"{args.method}__seed_*.json")))
    if not paths:
        parser.error(f"no evaluations for {args.method!r}")
    records = json.loads(Path(paths[0]).read_text())["records"]

    rows = []
    for record in records:
        npz = Path(args.data) / "records" / f"{record['record_id']}.npz"
        with np.load(npz, allow_pickle=True) as payload:
            rows.append({"parent_id": record["parent_id"], "record_id": record["record_id"],
                         "linear_loss": float(record["linear_loss"]),
                         "candidate_losses": np.asarray(payload["candidate_losses"],
                                                        dtype=float).tolist()})
    width = len(rows[0]["candidate_losses"])
    sizes = [k for k in (1, 2, 4, 8, 16, 32, 64, 128) if k <= width]
    if width not in sizes:
        sizes.append(width)
    result = menu_size_curve(rows, sizes=sizes, draws=args.draws, seed=args.seed)

    if args.frontier_sweep:
        headrooms = defaultdict(list)
        for root in args.frontier_sweep:
            files = sorted(glob.glob(str(Path(root) / "rows.jsonl"))) or \
                    sorted(glob.glob(str(Path(root) / "shard_*" / "rows.jsonl"))) or \
                    sorted(glob.glob(str(Path(root) / "extra_*" / "rows.jsonl")))
            for path in files:
                for line in open(path):
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    if entry.get("status") == "ok":
                        headrooms[entry["result"]["parent_id"]].append(
                            float(entry["result"]["headroom"]))
        if headrooms:
            result["search_headroom"] = float(np.mean([np.mean(v) for v in headrooms.values()]))
            result["search_parents"] = len(headrooms)

    Path(args.output).write_text(json.dumps(result, indent=2))
    print(f"{args.method}: fixed-menu headroom, {result['n_parents']} parents, "
          f"bank of {result['bank_size']}")
    for k in result["sizes"]:
        value = result["curve"][str(k)]
        print(f"   menu {k:4d}   {value:.5f}")
    if "search_headroom" in result:
        print(f"   search      {result['search_headroom']:.5f}  "
              f"(instance-specific, {result['search_parents']} parents)")
    gain = result["last_doubling_gain"]
    if gain is not None:
        print(f"   last doubling of the menu bought {gain:+.5f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
