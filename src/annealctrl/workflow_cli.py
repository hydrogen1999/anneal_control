"""Research workflow commands; no implicit network/cloud execution."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .pipeline import environment, load_records, write_json


COMMANDS = {"run", "tune", "report", "audit", "doctor", "train-config", "infer",
            "control-benchmark", "export-hardware", "ingest-hardware", "simulate-open", "profile-generation",
            "control-sweep", "frontier-report", "screen",
            "intervention-sweep", "intervention-report"}


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _save(path, result):
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite {target}")
    write_json(target, result)


def _record(args):
    if getattr(args, "record", None):
        with np.load(args.record, allow_pickle=False) as file:
            return {key: file[key] for key in file.files}
    records = load_records(args.data, args.split)
    selected = [r for r in records if str(r["record_id"].item()) == args.record_id]
    if len(selected) != 1:
        raise ValueError("record-id must match exactly one record in requested split")
    return selected[0]


def doctor(require_gpu=False):
    result = {"environment": environment(), "capabilities": {},
        "notes": ["No external services contacted.", "Capability detection does not establish GPU numerical parity/speedup."]}
    for name in ("torch", "cupy", "matplotlib", "qutip"):
        result["capabilities"][name] = importlib.util.find_spec(name) is not None
    if result["capabilities"]["torch"]:
        import torch
        result["torch_cuda"] = torch.cuda.is_available()
        result["torch_cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            result["torch_devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if result["capabilities"]["cupy"]:
        try:
            import cupy as cp
            result["cupy_devices"] = cp.cuda.runtime.getDeviceCount()
            if result["cupy_devices"]:
                result["cupy_free_total_bytes"] = list(cp.cuda.runtime.memGetInfo())
        except Exception as error:
            result["cupy_error"] = f"{type(error).__name__}: {error}"
    result["gpu_ready"] = bool(result.get("torch_cuda") and result.get("cupy_devices", 0) > 0)
    if require_gpu and not result["gpu_ready"]:
        raise RuntimeError("GPU readiness requires working CUDA PyTorch and CuPy; run doctor without --require-gpu for details")
    return result


def main(argv):
    parser = argparse.ArgumentParser(prog="annealctrl")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="one configuration -> generation/train/evaluation/paper files")
    run.add_argument("--config", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--stage", choices=["all", "generate", "train", "evaluate", "report"], default="all")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    tune = sub.add_parser("tune", help="validation-only Cartesian hyperparameter grid")
    tune.add_argument("--config", required=True)
    tune.add_argument("--output", required=True)
    tune.add_argument("--resume", action="store_true")
    tune.add_argument("--dry-run", action="store_true")
    report = sub.add_parser("report")
    report.add_argument("--run", required=True)
    report.add_argument("--no-plots", action="store_true")
    audit = sub.add_parser("audit")
    audit.add_argument("--data", required=True)
    audit.add_argument("--output", required=True)
    doc = sub.add_parser("doctor")
    doc.add_argument("--require-gpu", action="store_true")
    profile = sub.add_parser("profile-generation", help="measure accepted labels/sec including teacher and audits")
    profile.add_argument("--config", required=True)
    profile.add_argument("--output", required=True)
    profile.add_argument("--backends", nargs="+", choices=["numpy", "cupy"], default=["numpy"])
    profile.add_argument("--workers", type=int, default=1)
    train = sub.add_parser("train-config", help="train existing dataset using model/training JSON")
    train.add_argument("--config", required=True)
    train.add_argument("--data", required=True)
    train.add_argument("--output", required=True, help="best checkpoint .pt")
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--resume-from")
    infer = sub.add_parser("infer", help="predict a waveform without consulting true outcomes")
    infer.add_argument("--record", required=True, help="instance specification NPZ; labels unused")
    infer.add_argument("--checkpoint", required=True)
    infer.add_argument("--device", default="cpu")
    infer.add_argument("--output", required=True)
    bench = sub.add_parser("control-benchmark")
    bench.add_argument("--data", required=True)
    bench.add_argument("--split", choices=["train", "validation", "test"], default="validation")
    bench.add_argument("--record-id", required=True)
    bench.add_argument("--budget", type=int, default=32)
    bench.add_argument("--seed", type=int, default=0)
    bench.add_argument("--allow-test-adaptation", action="store_true")
    bench.add_argument("--backend", choices=["numpy", "cupy"], default="numpy")
    bench.add_argument("--output", required=True)
    export = sub.add_parser("export-hardware", help="offline validated program, NEVER submits a QPU job")
    export.add_argument("--record", required=True)
    export.add_argument("--schedule", required=True, help="JSON tau_knots/s_knots")
    export.add_argument("--config", required=True, help="device constraints + explicit mapping/units")
    export.add_argument("--output", required=True)
    ingest = sub.add_parser("ingest-hardware")
    ingest.add_argument("--program", required=True)
    ingest.add_argument("--samples", required=True)
    ingest.add_argument("--output", required=True)
    sweep = sub.add_parser("control-sweep", help="G2: equal-budget control-complexity frontier over a split")
    sweep.add_argument("--data", required=True)
    sweep.add_argument("--config", required=True, help="frontier JSON; see configs/frontier_smoke.json")
    sweep.add_argument("--output", required=True)
    sweep.add_argument("--split", choices=["train", "validation", "test"],
                       help="override the configured split")
    sweep.add_argument("--record-ids", nargs="+", help="restrict the sweep to these record ids")
    sweep.add_argument("--resume", action="store_true")
    sweep.add_argument("--dry-run", action="store_true")
    sweep.add_argument("--allow-test-adaptation", action="store_true",
                       help="test-split search is ONLINE ADAPTATION and must be reported as such")
    sweep.add_argument("--report", action="store_true", help="aggregate immediately after the sweep")
    frontier = sub.add_parser("frontier-report", help="G2: aggregate a completed control-sweep")
    frontier.add_argument("--sweep", required=True)
    frontier.add_argument("--output")
    frontier.add_argument("--bootstrap-resamples", type=int, default=10000)
    frontier.add_argument("--seed", type=int, default=0)
    screen = sub.add_parser("screen", help="G2: qualify a stress subset using train/validation headroom only")
    screen.add_argument("--fit-sweep", nargs="+", required=True,
                        help="train and/or validation control-sweep directories used to FIT the threshold")
    screen.add_argument("--apply-sweep", required=True, help="sweep directory the rule is APPLIED to")
    screen.add_argument("--output", required=True)
    screen.add_argument("--quantile", type=float, default=0.75)
    screen.add_argument("--min-headroom", type=float)
    inter = sub.add_parser("intervention-sweep",
                           help="G3: paired single-factor embedding interventions and cross-control matrices")
    inter.add_argument("--config", required=True, help="plan JSON; see configs/intervention_smoke.json")
    inter.add_argument("--output", required=True)
    inter.add_argument("--resume", action="store_true")
    inter.add_argument("--dry-run", action="store_true")
    inter.add_argument("--allow-test-parents", action="store_true",
                       help="intervention pairs on test parents share logical objectives with evaluation")
    inter.add_argument("--report", action="store_true")
    ireport = sub.add_parser("intervention-report", help="G3: aggregate a completed intervention-sweep")
    ireport.add_argument("--sweep", required=True)
    ireport.add_argument("--output")
    ireport.add_argument("--bootstrap-resamples", type=int, default=10000)
    ireport.add_argument("--seed", type=int, default=0)
    opened = sub.add_parser("simulate-open", help="small independent Lindblad model, not a calibrated QPU")
    opened.add_argument("--record", required=True)
    opened.add_argument("--schedule", required=True)
    opened.add_argument("--config", required=True)
    opened.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        from .experiments import load_experiment, plan_experiment, run_experiment
        cfg = load_experiment(args.config)
        result = plan_experiment(cfg) if args.dry_run else run_experiment(cfg, args.output, stage=args.stage, resume=args.resume)
        print(json.dumps(result if args.dry_run else {"status": result["status"], "output": args.output}, indent=2))
    elif args.command == "tune":
        from .experiments import expand_tuning, load_experiment, tune_validation
        config_path = Path(args.config).resolve()
        cfg = _json(config_path)
        if isinstance(cfg.get("experiment"), str):
            cfg["experiment"] = load_experiment(config_path.parent / cfg["experiment"])
        result = {"trials": len(expand_tuning(cfg)), "test_evaluated": False} if args.dry_run else tune_validation(cfg, args.output, resume=args.resume)
        print(json.dumps(result, indent=2))
    elif args.command == "report":
        from .reporting import build_report
        build_report(args.run, plots=not args.no_plots)
        print(f"Report: {Path(args.run) / 'paper' / 'RESULTS.md'}")
    elif args.command == "audit":
        from .reporting import audit_dataset
        _save(args.output, audit_dataset(args.data))
    elif args.command == "doctor":
        print(json.dumps(doctor(args.require_gpu), indent=2))
    elif args.command == "profile-generation":
        from .profiling import profile_generation
        result = profile_generation(_json(args.config), args.output, backends=tuple(args.backends), workers=args.workers)
        print(json.dumps(result, indent=2))
    elif args.command == "train-config":
        from .experiments import _device
        from .learning import fit_records
        cfg = _json(args.config)
        if set(cfg) - {"model", "training", "execution"}:
            raise ValueError("Training file accepts model, training, execution only")
        output = Path(args.output)
        if output.exists() and not args.resume_from:
            raise FileExistsError("Checkpoint exists; explicitly resume or use new output")
        latest = output.with_name(output.stem + ".latest.pt")
        start = perf_counter()
        fitted = fit_records(load_records(args.data, "train"), load_records(args.data, "validation"),
            model_config=cfg.get("model", {}), checkpoint=output, latest_checkpoint=latest,
            resume_from=args.resume_from, seed=args.seed, device=_device(cfg.get("execution", {})), **cfg.get("training", {}))
        write_json(output.with_suffix(".history.json"), {"best_epoch": fitted.best_epoch, "history": fitted.history,
            "elapsed_seconds": perf_counter() - start, "environment": environment()})
        print(f"Best: {output}; latest resume state: {latest}")
    elif args.command == "infer":
        import torch
        from .learning import load_checkpoint
        from .models import graph_from_record
        from .schedules import Schedule
        record = _record(args)
        model, normalizer = load_checkpoint(args.checkpoint, device=args.device)
        began = perf_counter()
        graph = normalizer.transform(graph_from_record(record, device=args.device))
        with torch.no_grad():
            proposals = model(graph)["proposal_schedules"]
            predicted = model.predict_losses(graph, proposals)
            index = int(predicted.argmin())
        wave = proposals[index].cpu().double().numpy()
        wave[0], wave[-1] = 0., 1.
        tau = np.linspace(0, 1, len(wave))
        schedule = Schedule(tau, wave)
        runtime = float(record["runtime"])
        schedule.validate_slope(runtime=runtime, max_slope=model.max_ds_dtau / runtime + 1e-6)
        _save(args.output, {"tau_knots": tau.tolist(), "s_knots": wave.tolist(), "runtime": runtime,
            "selected_proposal": index, "predicted_losses": predicted.cpu().tolist(),
            "inference_seconds": perf_counter() - began, "true_outcome_observed": False,
            "note": "Predicted loss is not a probability certificate; no simulator/hardware calls"})
    elif args.command == "control-sweep":
        from .headroom import frontier_report, load_frontier_config, sweep_control_frontier
        settings, report_settings = load_frontier_config(args.config)
        if args.split:
            settings["split"] = args.split
        result = sweep_control_frontier(args.data, output=args.output, resume=args.resume,
                                        dry_run=args.dry_run, record_ids=args.record_ids,
                                        allow_test_adaptation=args.allow_test_adaptation, **settings)
        if args.report and not args.dry_run:
            result = {"sweep": result, "report": frontier_report(args.output, **report_settings)}
        print(json.dumps(result, indent=2))
    elif args.command == "frontier-report":
        from .headroom import frontier_report
        summary = frontier_report(args.sweep, output=args.output,
                                  bootstrap_resamples=args.bootstrap_resamples, seed=args.seed)
        print(json.dumps({key: summary[key] for key in
                          ("split", "n_records", "n_parents", "censored_fraction", "verdict")}, indent=2))
        print(f"Report: {Path(args.output or Path(args.sweep) / 'report') / 'FRONTIER.md'}")
    elif args.command == "screen":
        from .screening import screen_records
        _save(args.output, screen_records(args.fit_sweep, args.apply_sweep,
                                          quantile=args.quantile, min_headroom=args.min_headroom))
    elif args.command == "intervention-sweep":
        from .interventions import intervention_report, plan_intervention_pairs, sweep_interventions
        config = _json(args.config)
        pairs, plan = plan_intervention_pairs(config, allow_test_parents=args.allow_test_parents)
        result = sweep_interventions(pairs, output=args.output, resume=args.resume,
                                     dry_run=args.dry_run, **dict(config.get("search") or {}))
        if not args.dry_run:
            write_json(Path(args.output) / "plan.json", plan)
            if args.report:
                result = {"sweep": result, "report": intervention_report(args.output)}
        print(json.dumps({"plan": {k: plan[k] for k in ("n_parents", "n_pairs", "factors",
                                                        "scale_arm_counts", "splits")},
                          "sweep": result}, indent=2))
    elif args.command == "intervention-report":
        from .interventions import intervention_report
        summary = intervention_report(args.sweep, output=args.output,
                                      bootstrap_resamples=args.bootstrap_resamples, seed=args.seed)
        print(json.dumps({key: summary[key] for key in
                          ("n_pairs", "n_parents", "censored_fraction", "swap_rate", "verdict")}, indent=2))
        print(f"Report: {Path(args.output or Path(args.sweep) / 'report') / 'INTERVENTIONS.md'}")
    elif args.command == "control-benchmark":
        from .benchmarking import benchmark_record_controls
        _save(args.output, benchmark_record_controls(_record(args), budget_per_family=args.budget, seed=args.seed,
            backend=args.backend, allow_test_adaptation=args.allow_test_adaptation))
    elif args.command == "export-hardware":
        from .hardware import DeviceConstraints, export_program
        from .schedules import Schedule
        cfg, wave = _json(args.config), _json(args.schedule)
        constraints = DeviceConstraints.from_mapping(cfg.pop("constraints"))
        _save(args.output, export_program(_record(args), Schedule(wave["tau_knots"], wave["s_knots"]), constraints, **cfg))
    elif args.command == "ingest-hardware":
        from .hardware import ingest_samples
        _save(args.output, ingest_samples(_json(args.program), _json(args.samples)))
    elif args.command == "simulate-open":
        from .adapters import CalibratedPath, simulate_lindblad
        from .physics import AnnealPath, HamiltonianTerms
        from .schedules import Schedule
        cfg, wave, record = _json(args.config), _json(args.schedule), _record(args)
        path_data = cfg.pop("path", None)
        catalyst = float(record.get("catalyst_strength", 0.))
        if catalyst and ("xx_edges" not in record or "xx_weights" not in record):
            raise ValueError("Catalyst requires explicit stored XX graph")
        path = AnnealPath(catalyst_strength=catalyst, energy_scale=float(record.get("energy_scale", 1.))) if path_data is None else CalibratedPath.from_mapping(path_data)
        if isinstance(path, CalibratedPath) and path.time_unit != "dimensionless" and "runtime" not in cfg:
            raise ValueError("Calibrated physical time requires explicit config.runtime in path.time_unit; do not reuse dimensionless runtime")
        runtime = cfg.pop("runtime", float(record["runtime"]))
        terms = HamiltonianTerms(len(record["physical_h"]), record["physical_h"], record["physical_edges"], record["physical_J"],
            record.get("xx_edges"), record.get("xx_weights"))
        result = simulate_lindblad(terms, Schedule(wave["tau_knots"], wave["s_knots"]), runtime, path=path, **cfg)
        _save(args.output, {"probabilities": result.probabilities.tolist(), "diagnostics": result.diagnostics})


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
