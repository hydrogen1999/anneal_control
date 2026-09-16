"""Fresh-dataset accepted-label throughput, with explicit CPU/GPU parity checks.

This measures complete generation, not only propagation kernels. It never
substitutes a backend or converts a failed numerical label into throughput.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .physics import backend_device_info, estimate_state_workspace
from .pipeline import _validate_config, environment, generate_dataset, load_records, source_fingerprint, write_json


def _scalar(record, key):
    return np.asarray(record[key]).item()


def _metadata(record):
    return json.loads(str(_scalar(record, "metadata_json")))


def _summarize(records: list[dict], config: dict, elapsed: float, workers: int) -> dict:
    if not records or not np.isfinite(elapsed) or elapsed <= 0:
        raise ValueError("profiling requires nonempty accepted records and positive elapsed time")
    label_count, max_state, max_norm = 0, 0., 0.
    max_steps, propagation_seconds, reference_seconds = 0, 0., 0.
    per_path, work_estimates = {}, []
    reference_errors = []
    for record in records:
        losses = np.asarray(record["candidate_losses"], dtype=float)
        errors = np.asarray(record["candidate_state_error"], dtype=float)
        norms = np.asarray(record["candidate_norm_error"], dtype=float)
        if (losses.ndim != 1 or not len(losses) or errors.shape != losses.shape
                or norms.shape != losses.shape or not np.isfinite(losses).all()
                or not np.isfinite(errors).all() or not np.isfinite(norms).all()
                or np.any(errors < 0) or np.any(norms < 0)
                or np.any(errors > config["label_state_tolerance"] * (1 + 1e-12))
                or np.any(norms > 1e-9)):
            raise ArithmeticError("profile dataset contains candidate labels that fail the numerical gate")
        label_count += len(losses)
        max_state = max(max_state, float(errors.max()))
        max_norm = max(max_norm, float(norms.max()))
        accepted_steps = int(np.max(record["candidate_steps"]))
        max_steps = max(max_steps, accepted_steps)
        propagation_seconds += float(np.sum(record["candidate_seconds"]))
        metadata = _metadata(record)
        reference_seconds += float(metadata.get("independent_reference_seconds", 0.))
        reference_errors.extend(metadata.get("independent_reference_errors", []))
        # Runtime variants share a cached teacher path. Never multiply the
        # recorded per-path teacher cost by the number of runtime records.
        path_id = str(_scalar(record, "record_id")).rsplit("_t", 1)[0]
        per_path.setdefault(path_id, metadata.get("teacher", {}))
        n_qubits = len(record["physical_h"])
        batch = min(config.get("candidate_batch_size", 1), len(losses))
        state = estimate_state_workspace(n_qubits, batch_size=batch, step_doubling=True)["estimated_peak_bytes"]
        state += 2 * accepted_steps * batch * 3 * 8
        saved_memory = metadata.get("memory", {})
        endpoint = float(saved_memory.get("endpoint_estimated_mib", 0.)) * 2**20
        state = max(state, float(saved_memory.get("state_estimated_mib", 0.)) * 2**20)
        dense_teacher = 0 if config.get("teacher", {}).get("mode", "exact") == "none" else 128 * (1 << n_qubits) ** 2
        work_estimates.append({"endpoint_bytes": int(endpoint), "state_bytes": int(state),
                               "dense_teacher_bytes": dense_teacher,
                               "numeric_work_bytes": int(endpoint + max(state, dense_teacher))})
    spectral_seconds = sum(float(teacher.get("spectral_seconds_per_path", 0.)) for teacher in per_path.values())
    active_workers = min(workers, len({str(_scalar(record, "parent_id")) for record in records}))
    maximum_work = max(item["numeric_work_bytes"] for item in work_estimates)
    return {
        "records": len(records), "physical_paths": len(per_path),
        "accepted_candidate_labels": label_count,
        "generation_wall_seconds": float(elapsed),
        "accepted_labels_per_second": label_count / elapsed,
        "timing_scope": "entire generate_dataset call: initialization, assembly, endpoint labels, spectral teacher/audit, convergence retries, independent dense dynamics audit and disk writes",
        "cost_components": {
            "spectral_including_random_audit_worker_seconds": spectral_seconds,
            "propagation_shared_cost_worker_seconds": propagation_seconds,
            "independent_dynamics_audit_worker_seconds": reference_seconds,
            "worker_seconds_not_additive_wall_time": workers > 1,
            "component_note": "Observed logged components; may exclude overhead. Runtime-shared teacher costs counted once per physical path.",
        },
        "max_state_error_diagnostic": max_state, "max_norm_error": max_norm,
        "max_independent_dense_state_error": max(reference_errors) if reference_errors else None,
        "max_accepted_integration_steps": max_steps,
        "all_candidate_labels_pass_numerical_gate": True,
        "state_error_is_certificate": False,
        "adaptive_paths_with_failed_sampled_checks": sum(
            teacher.get("mode") == "adaptive" and not teacher.get("sampled_checks_passed", False)
            for teacher in per_path.values()),
        "memory": {
            "estimated_numeric_work_peak_per_worker_bytes": maximum_work,
            "estimated_numeric_work_all_workers_bytes": maximum_work * active_workers,
            "max_endpoint_workspace_bytes": max(item["endpoint_bytes"] for item in work_estimates),
            "max_state_workspace_bytes": max(item["state_bytes"] for item in work_estimates),
            "max_dense_teacher_workspace_bytes": max(item["dense_teacher_bytes"] for item in work_estimates),
            "active_worker_upper_bound": active_workers, "measured_peak_memory": False,
            "scope": "Analytical numeric-array estimate; excludes Python/process overhead, coordinator dataset residency and allocator pools. Dense teacher estimate uses 128*(2**N)**2 bytes and remains CPU-only.",
        },
    }


def _compare(cpu: list[dict], gpu: list[dict]) -> dict:
    def index(records):
        indexed = {str(_scalar(record, "record_id")): record for record in records}
        if len(indexed) != len(records):
            raise ValueError("duplicate record IDs in profiling comparison")
        return indexed
    first, second = index(cpu), index(gpu)
    if set(first) != set(second):
        raise ValueError("CPU/GPU datasets have different record IDs")
    exact_fields = ("parent_id", "split", "family", "physical_h", "physical_edges", "physical_J",
                    "problem_J", "chain_J", "membership", "logical_h", "logical_edges", "logical_J",
                    "programmed_scale", "runtime", "chain_strength", "catalyst_strength",
                    "candidate_schedules", "candidate_tau", "candidate_ids")
    optional_physics = ("energy_scale", "xx_edges", "xx_weights")
    outcomes = ("candidate_losses", "candidate_success", "candidate_decoded_energy",
                "candidate_any_chain_break", "candidate_chain_break_fraction")
    differences = {field: [] for field in outcomes}
    error_radii = []
    for identifier in sorted(first):
        left, right = first[identifier], second[identifier]
        for field in exact_fields + optional_physics:
            if field not in left and field not in right and field in optional_physics:
                continue
            if field not in left or field not in right or not np.array_equal(left[field], right[field]):
                raise ValueError(f"CPU/GPU physics or waveform mismatch: {identifier}: {field}")
        for field in outcomes:
            if field not in left or field not in right:
                raise ValueError(f"missing CPU/GPU outcome {field}")
            a, b = np.asarray(left[field], dtype=float), np.asarray(right[field], dtype=float)
            if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError(f"invalid CPU/GPU outcome arrays: {field}")
            differences[field].extend(np.abs(a - b).ravel().tolist())
        a, b = np.asarray(left["candidate_state_error"]), np.asarray(right["candidate_state_error"])
        error_radii.extend((2 * a + a**2 + 2 * b + b**2).tolist())
    loss_differences = np.asarray(differences["candidate_losses"])
    return {
        "records": len(first), "same_record_ids": True, "same_physics_and_waveforms": True,
        "record_fingerprints_expected_to_differ": True,
        "outcome_differences": {field: {"max_absolute": float(np.max(values)),
                                          "mean_absolute": float(np.mean(values))}
                                for field, values in differences.items()},
        "loss_differences_within_combined_state_diagnostic_fraction": float(np.mean(
            loss_differences <= np.asarray(error_radii) + 1e-12)),
        "comparison_is_certificate": False,
        "note": "Combined state diagnostics are not rigorous bounds. Compare observed outcome differences and independent solver audits before claiming numerical parity.",
    }


def profile_generation(config: dict, output: str | Path, *, backends=("numpy",), workers: int = 1) -> dict:
    """Generate fresh identical configured workloads on explicitly requested backends.

    Every backend uses a separate new subdirectory. An error/interrupt writes
    partial status and retains completed datasets, then raises. This entry
    point deliberately refuses an existing output: a fresh measured workload
    must not inherit cached generation costs. Dataset recovery remains possible
    through the generation command, but is not comparable fresh-run timing.
    """
    _validate_config(config)
    requested = tuple(backends)
    if not requested or len(set(requested)) != len(requested) or any(item not in {"numpy", "cupy"} for item in requested):
        raise ValueError("backends must be unique explicitly requested numpy/cupy names")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if "cupy" in requested and workers != 1:
        raise ValueError("CuPy profiling requires workers=1; no implicit worker/backend changes")
    frozen = json.loads(json.dumps(deepcopy(config), allow_nan=False))
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    report_path = root / "profile.json"
    report = {
        "schema_version": 1, "status": "in_progress", "config": frozen,
        "config_hash": hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest(),
        "source_fingerprint": source_fingerprint(), "environment": environment(),
        "requested_backends": list(requested), "workers": workers,
        "backends": {backend: {"status": "planned", "dataset": str(root / backend)} for backend in requested},
        "cpu_gpu_comparison": None,
        "notes": [
            "No CPU fallback; unavailable requested backends fail explicitly.",
            "Complete spectral teacher and independent dense audit remain CPU-based even with CuPy propagation.",
            "Backend-dependent dataset fingerprints are expected. IDs, programmed physics and executed waveforms must agree.",
            "One fresh run per backend in declared order; device probes precede timing and can initialize the runtime. Generation-time allocation/JIT and thermal effects remain. Repeat profiles for speed claims.",
            "Adaptive audit seeds can differ with backend fingerprints; audits have identical configured budgets, not necessarily identical coordinates.",
        ],
        "recovery": "Artifacts remain intact after failure. Use a NEW output for a comparable complete profile; incomplete datasets can be resumed independently with matching generate configuration/source, but that recovery is not fresh-run timing.",
    }
    write_json(report_path, report)
    records_by_backend = {}
    active_backend = None
    try:
        for backend in requested:
            active_backend = backend
            entry = report["backends"][backend]
            entry["status"] = "running"
            write_json(report_path, report)
            entry["device_before"] = backend_device_info(backend)
            started = perf_counter()
            manifest = generate_dataset(frozen, root / backend, backend=backend, workers=workers)
            elapsed = perf_counter() - started
            records = load_records(root / backend)
            if source_fingerprint() != report["source_fingerprint"]:
                raise RuntimeError("source changed during profiling; measurements cannot use the frozen source identity")
            if manifest.get("source_fingerprint") != report["source_fingerprint"]:
                raise RuntimeError("generated dataset source identity differs from profiling source")
            if manifest.get("record_count") != len(records):
                raise ValueError("generated manifest count differs from accepted records")
            entry.update(_summarize(records, frozen, elapsed, workers))
            entry.update(status="complete", config_hash=manifest["config_hash"],
                         dataset_fingerprint=manifest.get("dataset_fingerprint"),
                         device_after=backend_device_info(backend))
            records_by_backend[backend] = records
            write_json(report_path, report)
        active_backend = None
        if {"numpy", "cupy"} <= records_by_backend.keys():
            comparison = _compare(records_by_backend["numpy"], records_by_backend["cupy"])
            cpu, gpu = report["backends"]["numpy"], report["backends"]["cupy"]
            comparison["observed_end_to_end_throughput_ratio_cupy_over_numpy"] = (
                gpu["accepted_labels_per_second"] / cpu["accepted_labels_per_second"])
            comparison["ratio_is_repeated_benchmark_or_speedup_claim"] = False
            report["cpu_gpu_comparison"] = comparison
        report["status"] = "complete"
        write_json(report_path, report)
        return report
    except BaseException as error:
        status = "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed"
        report.update(status=status, error={"type": type(error).__name__, "message": str(error),
                                           "backend": active_backend})
        if active_backend is not None:
            report["backends"][active_backend]["status"] = status
        write_json(report_path, report)
        raise
