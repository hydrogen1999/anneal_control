"""Small explicit commands: generation, fit, evaluation, GPU parity and hardware growth."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .pipeline import generate_dataset, load_records, write_json, environment


def _unused(path: Path):
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output {path}")


def train(args):
    import torch
    from .learning import fit_records
    from .models import AnnealController

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    output = Path(args.output)
    _unused(output)
    model = AnnealController(width=args.width)
    began = perf_counter()
    fitted = fit_records(load_records(args.data, "train"), load_records(args.data, "validation"),
                         model=model, epochs=args.epochs, seed=args.seed, device=args.device,
                         checkpoint=output, response_weight=args.response_weight, patience=args.patience)
    write_json(output.with_suffix(".history.json"), {"history": fitted.history, "best_epoch": fitted.best_epoch,
              "training_seconds": perf_counter() - began, "environment": environment(),
              "warning": "research scaffold fit, not an A-star benchmark result"})
    print(f"saved checkpoint {output}; selected validation epoch={fitted.best_epoch}")


def evaluate(args):
    import torch
    from .benchmarking import evaluate_checkpoint

    torch.set_num_threads(args.threads)
    output = Path(args.output)
    _unused(output)
    result = evaluate_checkpoint(args.data, args.checkpoint, device=args.device,
                                 direct=args.direct, seed=args.seed, tolerance=args.tolerance,
                                 backend=getattr(args, "backend", "numpy"),
                                 initial_steps=getattr(args, "initial_steps", 128),
                                 max_steps=getattr(args, "max_steps", 8192))
    write_json(output, result)
    print(json.dumps({k: result[k] for k in ("mean_loss", "mean_bank_regret", "mean_linear_loss", "n_test_parents")}, indent=2))


def hardware_grow(args):
    from .generation import grow_hardware_partition, generate_problem, compile_embedding, validate_compilation
    hardware = json.loads(Path(args.hardware).read_text())
    rng = np.random.default_rng(args.seed)
    lengths = np.array([int(x) for x in args.lengths.split(",")])
    emb = grow_hardware_partition(int(hardware["n_qubits"]), hardware["edges"], lengths, rng)
    problem = generate_problem(len(lengths), args.family, rng, emb.quotient_edges)
    compiled = compile_embedding(problem, emb, args.chain_strength, rng)
    result = {"membership": emb.membership.tolist(), "active_hardware_edges": emb.hardware_edges.tolist(),
              "physical_h": compiled.physical.h.tolist(), "physical_J": compiled.physical.J.tolist(),
              "problem_J": compiled.problem_J.tolist(), "chain_J": compiled.chain_J.tolist(),
              "logical_h": problem.h.tolist(), "logical_edges": problem.edges.tolist(), "logical_J": problem.J.tolist(),
              "programmed_scale": compiled.programmed_scale, "metadata": emb.metadata,
              "note": "valid embedding construction, no quantum-hardness certificate"}
    if compiled.physical.n <= 16:
        result["validation"] = validate_compilation(compiled)
    else:
        result["validation"] = "exhaustive endpoint check not attempted above 16 active qubits"
    output = Path(args.output)
    _unused(output)
    write_json(output, result)
    print(f"saved hardware-growth instance: active={compiled.physical.n}, target_met={emb.metadata['target_met']}")


def benchmark_backend(args):
    from .physics import HamiltonianTerms, propagate
    from .schedules import Schedule
    n = args.qubits
    if not 2 <= n <= 20:
        raise ValueError("benchmark is capped at 2..20 qubits; certify memory before increasing")
    rng = np.random.default_rng(args.seed)
    terms = HamiltonianTerms(n, rng.normal(0, .2, n), np.array([(i, i+1) for i in range(n-1)]),
                             rng.normal(0, .5, n-1))
    result = {"qubits": n, "steps": args.steps, "backend": args.backend, "environment": environment(),
              "warning": "single propagation microbenchmark, not accepted spectral labels/sec"}
    # Missing CuPy/CUDA raises an actionable error, never silently falls back.
    propagate(terms, Schedule.linear(), 2., steps=4, backend=args.backend)
    started = perf_counter()
    measured = propagate(terms, Schedule.linear(), 2., steps=args.steps, backend=args.backend, step_doubling=True)
    result.update(elapsed_seconds=perf_counter()-started, norm_error=measured.norm_error,
                  step_doubling_error=measured.step_doubling_error)
    if args.backend == "cupy":
        import cupy as cp
        properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
        result["device_name"] = properties["name"].decode() if isinstance(properties["name"], bytes) else properties["name"]
        reference = propagate(terms, Schedule.linear(), 2., steps=2*args.steps)
        result["cpu_gpu_state_difference"] = float(np.linalg.norm(reference.state-measured.state))
    output = Path(args.output)
    _unused(output)
    write_json(output, result)
    print(json.dumps(result, indent=2))


def main(argv=None):
    import sys
    from .workflow_cli import COMMANDS, main as workflow_main
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in COMMANDS:
        return workflow_main(argv)
    parser = argparse.ArgumentParser(prog="annealctrl", epilog="Workflow commands: " + ", ".join(sorted(COMMANDS)) + ". Use annealctrl COMMAND --help.")
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate", help="build capped exact pilot labels")
    gen.add_argument("--config", required=True)
    gen.add_argument("--output", required=True)
    gen.add_argument("--resume", action="store_true")
    gen.add_argument("--backend", choices=["numpy", "cupy"], default="numpy")
    gen.add_argument("--workers", type=int, default=1)
    fit = sub.add_parser("train")
    fit.add_argument("--data", required=True)
    fit.add_argument("--output", required=True)
    fit.add_argument("--epochs", type=int, default=50)
    fit.add_argument("--patience", type=int, default=10)
    fit.add_argument("--width", type=int, default=64)
    fit.add_argument("--seed", type=int, default=0)
    fit.add_argument("--device", default="cpu")
    fit.add_argument("--threads", type=int, default=1)
    fit.add_argument("--response-weight", type=float, default=.05)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--data", required=True)
    ev.add_argument("--checkpoint", required=True)
    ev.add_argument("--output", required=True)
    ev.add_argument("--direct", action="store_true")
    ev.add_argument("--tolerance", type=float, default=5e-4)
    ev.add_argument("--seed", type=int, default=0)
    ev.add_argument("--device", default="cpu")
    ev.add_argument("--threads", type=int, default=1)
    hw = sub.add_parser("hardware-grow")
    hw.add_argument("--hardware", required=True, help="JSON n_qubits and edges, labels0..N-1")
    hw.add_argument("--lengths", required=True, help="comma-separated target sizes, achieved may be smaller")
    hw.add_argument("--family", choices=["spin_glass", "weighted_maxcut", "planted_loops", "weak_field"], default="spin_glass")
    hw.add_argument("--chain-strength", type=float, default=1.5)
    hw.add_argument("--seed", type=int, default=0)
    hw.add_argument("--output", required=True)
    bench = sub.add_parser("benchmark")
    bench.add_argument("--backend", choices=["numpy", "cupy"], default="numpy")
    bench.add_argument("--qubits", type=int, default=10)
    bench.add_argument("--steps", type=int, default=128)
    bench.add_argument("--seed", type=int, default=0)
    bench.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "generate":
        manifest = generate_dataset(json.loads(Path(args.config).read_text()), args.output,
                                    resume=args.resume, backend=args.backend, workers=args.workers)
        print(f"dataset complete: {manifest['record_count']} records in {manifest['elapsed_seconds']:.2f}s")
    else:
        {"train": train, "evaluate": evaluate, "hardware-grow": hardware_grow, "benchmark": benchmark_backend}[args.command](args)


if __name__ == "__main__":
    main()
