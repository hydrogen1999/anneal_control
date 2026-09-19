"""Does the logical spectrum suffice to schedule the embedded system?

The design document names Tx-NQDT as the closest related method and records
that its spectrum is logical rather than physical. This runs the same
spectrum-guided construction from each spectrum in turn and scores both on the
same embedded record, so the only difference is the spectrum's provenance.

A null result here would be bad news for this project's central thesis, which
is why it is worth running rather than assuming.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


def _one(args):
    (data_dir, split, record_id, method, max_ds_dtau, max_qubits, tolerance,
     max_steps, scale) = args
    import numpy as np

    from annealctrl.logical_teacher import compare_spectra
    from annealctrl.pipeline import load_records

    record = next(r for r in load_records(data_dir, split)
                  if str(np.asarray(r["record_id"]).item()) == record_id)
    try:
        return compare_spectra(record, method=method, max_ds_dtau=max_ds_dtau,
                               max_qubits=max_qubits, tolerance=tolerance,
                               max_steps=max_steps, scale=scale)
    except Exception as exc:                       # a refusal is data, not a crash
        return {"record_id": record_id,
                "parent_id": str(np.asarray(record["parent_id"]).item()),
                "error": f"{type(exc).__name__}: {exc}"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--method", default="d2", choices=("d2", "gap_inverse_square"))
    parser.add_argument("--records", type=int, default=48)
    parser.add_argument("--max-qubits", type=int, default=10)
    parser.add_argument("--max-ds-dtau", type=float, default=4.0)
    parser.add_argument("--runtime", type=float, default=None,
                        help="restrict to one runtime; without it the "
                             "one-record-per-parent rule silently picks the "
                             "shortest, which is what the first study did")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--tolerance", type=float, default=5e-4)
    parser.add_argument("--max-steps", type=int, default=8192)
    parser.add_argument("--scale", choices=("programmed", "raw"), default="programmed",
                        help="'programmed' isolates the embedding; 'raw' also carries "
                             "the common rescaling and is a different question")
    args = parser.parse_args(argv)

    from annealctrl.headroom import _bootstrap, _describe, _parent_means
    from annealctrl.pipeline import load_records

    records = [r for r in load_records(args.data, args.split)
               if len(np.asarray(r["physical_h"])) <= args.max_qubits]
    if args.runtime is not None:
        records = [r for r in records
                   if abs(float(np.asarray(r["runtime"]).item()) - args.runtime) < 1e-9]
        if not records:
            raise SystemExit(f"no record at runtime {args.runtime}")
    runtimes = sorted({float(np.asarray(r["runtime"]).item()) for r in records})
    print(f"runtimes present: {runtimes}", flush=True)
    seen, chosen = set(), []
    for r in sorted(records, key=lambda r: str(np.asarray(r["record_id"]).item())):
        pid = str(np.asarray(r["parent_id"]).item())
        if pid in seen:
            continue
        seen.add(pid)
        chosen.append(str(np.asarray(r["record_id"]).item()))
        if len(chosen) >= args.records:
            break
    print(f"{len(chosen)} records, one per parent, <= {args.max_qubits} physical qubits, "
          f"method {args.method}", flush=True)

    jobs = [(args.data, args.split, rid, args.method, args.max_ds_dtau, args.max_qubits,
             args.tolerance, args.max_steps, args.scale) for rid in chosen]
    rows, failed = [], []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
        for row in pool.map(_one, jobs):
            if "error" in row:
                failed.append(row)
                print(f"  {row['record_id']}: {row['error']}", flush=True)
                continue
            rows.append(row)
            print(f"  {row['record_id']}: logical {row['logical_spectrum_loss']} "
                  f"physical {row['physical_spectrum_loss']} "
                  f"delta {row['logical_minus_physical']}", flush=True)

    usable = [r for r in rows if r["logical_minus_physical"] is not None]
    if not usable:
        raise SystemExit("no record produced both schedules")

    blocks = {}
    for name in ("logical_minus_physical", "logical_spectrum_loss",
                 "physical_spectrum_loss", "linear_loss"):
        values, parents = _parent_means(usable, lambda r, n=name: r[n])
        block = _describe(values)
        block["parent_bootstrap_ci"] = _bootstrap(values, n_resamples=8000, seed=0)
        block["n_parents"] = len(parents)
        blocks[name] = block

    payload = {"schema_version": 1, "data": args.data, "split": args.split,
               "method": args.method, "logical_scale": args.scale,
               "max_ds_dtau": args.max_ds_dtau, "runtime_filter": args.runtime,
               "runtimes_covered": runtimes,
               "n_requested": len(chosen),
               "n_usable": len(usable), "n_unresolved": len(rows) - len(usable),
               "n_failed": len(failed), "failures": failed,
               "summary": blocks, "rows": rows,
               "scope": ("both schedules control the same embedded record and differ only "
                         "in which spectrum built them; inspired by the logical-spectrum "
                         "information path of Tx-NQDT, not a reimplementation")}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2, default=float))

    identity = [r for r in usable if r.get("one_to_one_embedding")]
    violations = [r for r in identity if abs(r["logical_minus_physical"]) > 1e-9]
    payload["one_to_one_records"] = len(identity)
    payload["one_to_one_violations"] = len(violations)
    if args.scale == "programmed" and violations:
        print(f"WARNING: {len(violations)}/{len(identity)} one-to-one records differ; "
              "the two Hamiltonians should coincide exactly", flush=True)
    Path(args.output).write_text(json.dumps(payload, indent=2, default=float))

    d = blocks["logical_minus_physical"]
    ci = d["parent_bootstrap_ci"]
    print(f"\nusable {len(usable)} of {len(chosen)} ({len(failed)} failed, "
          f"{len(rows) - len(usable)} unresolved)")
    print(f"logical-spectrum minus physical-spectrum: {d['mean']:+.5f} "
          f"[{ci['low']:+.5f},{ci['high']:+.5f}] over {d['n_parents']} parents")
    print(f"  logical  {blocks['logical_spectrum_loss']['mean']:.5f}"
          f"  physical {blocks['physical_spectrum_loss']['mean']:.5f}"
          f"  linear   {blocks['linear_loss']['mean']:.5f}")
    worse = sum(1 for r in usable if r["logical_minus_physical"] > 0)
    print(f"  logical spectrum is worse in {worse}/{len(usable)} records")
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
