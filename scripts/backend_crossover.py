#!/usr/bin/env python
"""Measure NumPy against CuPy across physical sizes, on a machine that is in use.

The naive version of this measurement is biased in a direction that flatters the
conclusion. CuPy at these sizes is launch-bound and nearly flat; NumPy is
compute-bound. Timing on a busy host starves the NumPy arm and leaves CuPy
alone, so the ratio inflates in favour of "the GPU wins" -- which is exactly the
answer one is hoping for, and therefore exactly the answer to distrust.

Waiting for an empty host is not available here: apollo is the only machine with
a GPU, and it is shared. So the measurement defends itself three ways instead.

**Census.** The competing load is attributed by user and nice level before and
after every repeat, and recorded in the artifact. A reader can see what else was
running rather than taking "the machine was quiet" on trust.

**Priority.** The measurement refuses to start while this user's own processes
are consuming CPU at a nice level that would compete with it. Under Linux CFS a
nice-19 process carries roughly one sixty-eighth the weight of a nice-0 one, so
once our own jobs are reniced out of the way a nice-0 measurement runs
effectively alone -- but that has to be true, not assumed.

**Repetition.** Each size is measured more than once. If the repeats of a single
backend disagree by more than the tolerance, the number is reported as unstable
instead of averaged into something that looks precise. A benchmark that cannot
reproduce itself on the same machine within minutes is not measuring the machine.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from time import perf_counter


def census() -> dict:
    """Who is using this machine right now, by user and nice level."""
    load = Path("/proc/loadavg").read_text().split()[:3] if Path("/proc/loadavg").exists() else []
    # ps truncates the USER column to eight characters and appends '+', so a
    # login name longer than that never matches itself and every one of this
    # user's own processes gets filed under "other". That inverted the whole
    # census: a machine carrying 3168% of my own load was recorded as 3168% of
    # somebody else's. Ask for a wide column instead of guessing at prefixes.
    rows = subprocess.run(["ps", "-eo", "user:32,ni,pcpu,etimes,comm", "--no-headers"],
                          capture_output=True, text=True).stdout.splitlines()
    me = subprocess.run(["id", "-un"], capture_output=True, text=True).stdout.strip()
    buckets: dict[str, float] = {}
    for row in rows:
        parts = row.split()
        if len(parts) != 5:
            continue
        user, nice, cpu, age, comm = parts[0], parts[1], float(parts[2]), int(parts[3]), parts[4]
        # The census must not count itself. `ps` and the ssh session that invoked
        # it show a high instantaneous pcpu over a sub-second lifetime, which is
        # not load anyone else is feeling -- and counting it made the guard refuse
        # on an otherwise idle machine.
        if cpu <= 0.5 or age < 5 or comm in {"ps", "sshd", "bash", "awk", "id"}:
            continue
        who = "self" if user == me else "other"
        buckets[f"{who}_nice_{nice}"] = buckets.get(f"{who}_nice_{nice}", 0.0) + cpu
    # A GPU benchmark whose census only counts CPU is measuring half the machine.
    # A stray sweep of my own held 556 MiB and 18% of the GPU through one run and
    # cut the measured CuPy throughput almost in half, while NumPy barely moved --
    # exactly the kind of contamination that looks like a real size effect.
    gpu = []
    try:
        listing = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20)
        for line in listing.stdout.splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) == 2 and parts[0].isdigit():
                described = subprocess.run(["ps", "-o", "stat=,user=,args=", "-p", parts[0]],
                                           capture_output=True, text=True).stdout.strip()
                state = described.split(maxsplit=1)[0] if described else "?"
                gpu.append({"pid": int(parts[0]), "used": parts[1], "state": state,
                            # A stopped process keeps its CUDA context and its
                            # memory but executes nothing, so it does not compete
                            # for compute. Its reservation is still recorded.
                            "competing": not state.startswith("T"),
                            "process": described[:140]})
    except (OSError, subprocess.SubprocessError):
        gpu = None

    return {"load_average": [float(x) for x in load],
            "gpu_compute_processes": gpu,
            "n_gpu_compute_processes": (len(gpu) if gpu is not None else None),
            "cpu_percent_by_user_and_nice": {k: round(v, 1) for k, v in sorted(buckets.items())},
            "self_competing_cpu_percent": round(
                sum(v for k, v in buckets.items()
                    if k.startswith("self_nice_") and int(k.rsplit("_", 1)[1]) < 10), 1),
            "other_cpu_percent": round(
                sum(v for k, v in buckets.items() if k.startswith("other_")), 1)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path.home() / "annealctrl_deploy" / "anneal_control"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--sizes", nargs="+", type=int, default=[10, 12, 14, 16])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--tolerance", type=float, default=0.15,
                        help="max relative spread between repeats of one backend")
    parser.add_argument("--max-self-competing-cpu", type=float, default=50.0,
                        help="refuse to start above this own-CPU percent below nice 10")
    parser.add_argument("--force", action="store_true",
                        help="measure anyway and record that the guard was overridden")
    args = parser.parse_args(argv)

    root = Path(args.root)
    before = census()
    # A driver running at a high nice value is in the same starved position as the
    # NumPy arm it is timing. Inheriting nice 19 from the shell that reniced the
    # rest of the jobs is an easy way to produce a measurement that looks careful
    # and is not, so it is checked rather than assumed.
    own_nice = __import__("os").nice(0)
    before["driver_nice"] = own_nice
    if own_nice > 0 and not args.force:
        print(f"refusing to measure: this driver is running at nice {own_nice}. It would be "
              "starved exactly like the NumPy arm it is timing.", file=sys.stderr)
        print("Launch it from a shell that was not reniced (a fresh session), or pass --force.",
              file=sys.stderr)
        return 1
    occupants = before.get("gpu_compute_processes") or []
    competing = [entry for entry in occupants
                 if str(os.getpid()) != str(entry["pid"]) and entry.get("competing", True)]
    if competing and not args.force:
        print("refusing to measure: another process is running on the GPU.", file=sys.stderr)
        for entry in competing:
            print(f"  pid {entry['pid']} ({entry['state']}) using {entry['used']}: "
                  f"{entry['process']}", file=sys.stderr)
        print("A contended GPU depresses the CuPy arm while NumPy is untouched, which reads as a "
              "smaller size effect than the truth. Stop or pause that process, or pass --force.",
              file=sys.stderr)
        return 1
    if before["self_competing_cpu_percent"] > args.max_self_competing_cpu and not args.force:
        print(f"refusing to measure: this user's own processes below nice 10 are consuming "
              f"{before['self_competing_cpu_percent']}% CPU, above the "
              f"{args.max_self_competing_cpu}% guard.", file=sys.stderr)
        print("Renice them (renice -n 19 -u $USER) so the measurement runs effectively alone, "
              "or pass --force and accept that the recorded ratio is biased toward the GPU.",
              file=sys.stderr)
        return 1

    # Validate every configuration before measuring any of them. The first run of
    # this script spent forty minutes measuring 10q and then failed 12q, 14q and
    # 16q on config errors that a validator would have named in a second: a
    # missing endpoint_max_qubits above ten qubits, and a parent count below the
    # minimum of three. Preflight turns that into an immediate, complete list.
    sys.path.insert(0, str(root / "src"))
    try:
        from annealctrl.pipeline import _validate_config  # noqa: PLC0415
    except ImportError:
        _validate_config = None
    invalid = {}
    if _validate_config is not None:
        for size in args.sizes:
            config = root / "configs" / f"backend_profile_{size}q.json"
            if not config.exists():
                invalid[size] = "no config file"
                continue
            payload = json.loads(config.read_text())
            try:
                _validate_config(payload)
            except Exception as error:                       # noqa: BLE001
                invalid[size] = str(error)
                continue
            # generation.parent_splits needs three logical parents per family so
            # train/validation/test are each nonempty. _validate_config does not
            # know this -- it only checks the global minimum of three -- so the
            # failure surfaces at generation time, after the measurement has
            # already been paid for. Check it here instead.
            families = len(payload.get("families") or [])
            if families and payload.get("parents", 0) < 3 * families:
                invalid[size] = (f"needs >=3 parents per family: {payload['parents']} parents "
                                 f"over {families} families, minimum {3 * families}")
    if invalid and not args.force:
        print("refusing to measure: these configurations are invalid.", file=sys.stderr)
        for size, reason in sorted(invalid.items()):
            print(f"  {size}q: {reason}", file=sys.stderr)
        print("Fix them, or pass --force to measure only the valid sizes.", file=sys.stderr)
        return 1

    report = {"schema_version": 1, "sizes": {}, "repeats": args.repeats,
              "preflight_invalid": {str(k): v for k, v in invalid.items()},
              "tolerance": args.tolerance, "guard_overridden": bool(args.force),
              "census_before": before,
              "scope": ("NumPy against CuPy on a shared host; repeats bound the timing noise and "
                        "the census records what else was running. A loaded host starves the "
                        "NumPy arm and inflates the ratio, so an unstable size is reported as "
                        "unstable rather than averaged.")}
    workdir = Path(args.output).parent
    workdir.mkdir(parents=True, exist_ok=True)

    for size in args.sizes:
        config = root / "configs" / f"backend_profile_{size}q.json"
        if not config.exists():
            report["sizes"][str(size)] = {"status": "no_config", "config": str(config)}
            continue
        runs = []
        for repeat in range(args.repeats):
            # profile-generation treats --output as a directory root and writes its
            # report to <output>/profile.json beside the generated datasets.
            out = workdir / f"profile_{size}q_r{repeat}"
            report_file = out / "profile.json"
            # profile-generation refuses to write into an existing directory, so a
            # repeat that inherits one from an abandoned attempt dies on
            # FileExistsError and is recorded as a failed measurement. Clear it.
            if out.exists():
                shutil.rmtree(out)
            started = perf_counter()
            done = subprocess.run(
                [str(root / ".venv" / "bin" / "python"), "-m", "annealctrl.workflow_cli",
                 "profile-generation", "--config", str(config), "--output", str(out),
                 "--backends", "numpy", "cupy", "--workers", "1"],
                capture_output=True, text=True)
            entry = {"repeat": repeat, "wall_seconds": perf_counter() - started,
                     "returncode": done.returncode, "artifact": str(report_file),
                     "census": census()}
            if done.returncode != 0:
                entry["stderr_tail"] = done.stderr.strip().splitlines()[-4:]
            elif report_file.exists():
                payload = json.loads(report_file.read_text())
                entry["status"] = payload.get("status")
                for backend in ("numpy", "cupy"):
                    block = payload.get("backends", {}).get(backend, {})
                    entry[backend] = block.get("accepted_labels_per_second")
                comparison = payload.get("cpu_gpu_comparison") or {}
                entry["max_candidate_loss_difference"] = comparison.get(
                    "max_absolute_candidate_loss_difference")
            runs.append(entry)
            print(f"[{size}q r{repeat}] numpy={entry.get('numpy')} cupy={entry.get('cupy')} "
                  f"rc={entry['returncode']}", flush=True)

        block = {"runs": runs}
        good = [r for r in runs if r.get("numpy") and r.get("cupy")]
        if len(good) < 2:
            block["status"] = "insufficient_successful_repeats"
        else:
            block["status"] = "measured"
            for backend in ("numpy", "cupy"):
                values = [float(r[backend]) for r in good]
                low, high = min(values), max(values)
                block[f"{backend}_labels_per_second"] = sum(values) / len(values)
                block[f"{backend}_relative_spread"] = (high - low) / high if high else None
            spread = max(block["numpy_relative_spread"], block["cupy_relative_spread"])
            block["repeats_agree"] = bool(spread <= args.tolerance)
            block["ratio_cupy_over_numpy"] = (block["cupy_labels_per_second"]
                                              / block["numpy_labels_per_second"])
            if not block["repeats_agree"]:
                block["status"] = "unstable_repeats"
                block["warning"] = (f"repeats of one backend differ by {spread:.1%}, above the "
                                    f"{args.tolerance:.0%} tolerance; this ratio is not a benchmark")
        report["sizes"][str(size)] = block

    report["census_after"] = census()
    report["stable_sizes"] = [size for size, block in report["sizes"].items()
                              if block.get("status") == "measured"]
    Path(args.output).write_text(json.dumps(report, indent=1) + "\n")
    print(f"\nwrote {args.output}")
    for size, block in report["sizes"].items():
        if block.get("status") == "measured":
            print("  %3sq  numpy %8.2f  cupy %8.2f  ratio %7.2fx  (spread %.1f%%/%.1f%%)" % (
                size, block["numpy_labels_per_second"], block["cupy_labels_per_second"],
                block["ratio_cupy_over_numpy"], 100 * block["numpy_relative_spread"],
                100 * block["cupy_relative_spread"]))
        else:
            print(f"  {size:>3}q  {block.get('status')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
