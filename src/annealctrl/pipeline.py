"""Versioned, resumable pilot data generation. All candidate outcomes are retained.

This module intentionally uses capped exact spectral teachers and enumeration.
Larger-system approximations require new audited label contracts, not a flag
that silently changes the scientific meaning of success or ground state.
"""
from __future__ import annotations

import hashlib
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import platform
import socket
from time import perf_counter
from typing import Any

import numpy as np

from .generation import (compile_embedding, generate_problem, parent_splits,
                         synthetic_lift, validate_compilation, output_observables,
                         grow_hardware_partition, resolve_hardware, sample_chain_lengths,
                         sample_logical_support, logical_fingerprint)
from .physics import (AnnealPath, HamiltonianTerms, PropagationResult, propagate,
                      propagate_batch, dense_reference_propagate)
from .search import shared_candidate_bank
from .spectral import spectral_profile


SCHEMA_VERSION = 1


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot encode {type(value)}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, default=jsonable, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def _seed(master: int, *parts) -> int:
    digest = hashlib.sha256(json.dumps([master, *parts]).encode()).digest()
    return int.from_bytes(digest[:8], "little")


def environment() -> dict:
    versions = {}
    for package in ("numpy", "scipy", "networkx", "torch", "cupy", "cupy-cuda12x", "pytest"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": versions}


def _logical_sizes(cfg: dict) -> list[int]:
    values = cfg.get("logical_sizes", [cfg.get("logical_qubits")])
    if not isinstance(values, list) or not values or any(
            not isinstance(x, int) or isinstance(x, bool) or x < 2 for x in values):
        raise ValueError("logical_qubits or logical_sizes must contain integers >=2")
    if len(set(values)) != len(values):
        raise ValueError("logical_sizes contains duplicates")
    return values


def _validate_config(cfg: dict) -> None:
    required = {"seed", "parents", "families", "variants",
                "chain_strengths", "runtimes", "candidates", "spectral_points", "steps",
                "max_steps", "label_state_tolerance", "max_physical_qubits"}
    if required - set(cfg):
        raise ValueError(f"missing configuration keys {sorted(required - set(cfg))}")
    for key in ("seed", "parents", "candidates", "spectral_points", "steps", "max_steps"):
        if not isinstance(cfg[key], int) or isinstance(cfg[key], bool):
            raise ValueError(f"{key} must be an integer")
    sizes = _logical_sizes(cfg)
    if ("chain_lengths" in cfg) == ("chain_distribution" in cfg):
        raise ValueError("provide exactly one of chain_lengths or chain_distribution")
    if "chain_lengths" in cfg:
        for n in sizes:
            lengths = cfg["chain_lengths"]
            lengths = [lengths] * n if isinstance(lengths, int) else lengths
            if len(lengths) != n or any(not isinstance(x, int) or x < 1 for x in lengths):
                raise ValueError("one positive integer chain length required per logical variable")
            if sum(lengths) > cfg["max_physical_qubits"]:
                raise ValueError("chain lengths exceed max_physical_qubits")
    else:
        distribution = cfg["chain_distribution"]
        sample_chain_lengths(max(sizes), distribution["low"], distribution["high"],
                             np.random.default_rng(0), distribution.get("distribution", "uniform"),
                             distribution.get("exponent", 2.))
        if max(sizes) * distribution["low"] > cfg["max_physical_qubits"]:
            raise ValueError("minimum chain lengths exceed physical budget")
    if not cfg["families"] or len(set(cfg["families"])) != len(cfg["families"]):
        raise ValueError("families must be nonempty and unique")
    if cfg["parents"] < 3:
        raise ValueError("need >=3 parents")
    mode = cfg.get("teacher", {}).get("mode", "exact")
    if mode not in {"exact", "adaptive", "none"}:
        raise ValueError("teacher mode must be exact, adaptive, or none")
    cap = cfg["max_physical_qubits"]
    if not isinstance(cap, int) or not 2 <= cap <= 20:
        raise ValueError("max_physical_qubits must be an integer in 2..20")
    if mode != "none" and cap > 10:
        raise ValueError("full spectral teacher is capped at <=10 physical qubits; choose teacher none for outcome-only")
    if cap > 10 and not {"endpoint_max_qubits", "endpoint_memory_mb", "state_memory_mb"} <= cfg.keys():
        raise ValueError("outcome-only >10 requires explicit endpoint_max_qubits, endpoint_memory_mb, state_memory_mb")
    if not isinstance(cfg.get("endpoint_max_qubits", 20), int) or not cap <= cfg.get("endpoint_max_qubits", 20) <= 20:
        raise ValueError("endpoint_max_qubits must cover physical cap and be <=20")
    for name in ("endpoint_memory_mb", "state_memory_mb"):
        if not np.isfinite(cfg.get(name, 1024)) or cfg.get(name, 1024) <= 0:
            raise ValueError("memory budgets must be positive finite MiB")
    for name in ("h_limit", "j_limit"):
        if not np.isfinite(cfg.get(name, 1)) or cfg.get(name, 1) <= 0:
            raise ValueError("coefficient limits must be positive finite")
    if "endpoint_chunk_size" in cfg and (not isinstance(cfg["endpoint_chunk_size"], int) or cfg["endpoint_chunk_size"] < 1):
        raise ValueError("endpoint_chunk_size must be a positive integer")
    if not isinstance(cfg.get("candidate_batch_size", 1), int) or cfg.get("candidate_batch_size", 1) < 1:
        raise ValueError("candidate_batch_size must be a positive integer")
    route = cfg.get("generation_route", "synthetic_lift")
    if route not in {"synthetic_lift", "hardware_growth"}:
        raise ValueError("generation_route must be synthetic_lift or hardware_growth")
    if route == "hardware_growth":
        hw = cfg.get("hardware", {})
        if "topology" in hw:
            # Validate by building it once; the result is cached for generation.
            probe = resolve_hardware(hw, np.random.default_rng(0))
            if probe["n_qubits"] < max(sizes):
                raise ValueError("hardware patch is smaller than the largest logical size")
        elif not isinstance(hw.get("n_qubits"), int) or hw["n_qubits"] < max(sizes) or "edges" not in hw:
            raise ValueError("hardware requires n_qubits and edges, or topology and m, "
                             "with capacity for all logical seeds")
    if cfg["candidates"] < 2 or cfg["spectral_points"] < 3:
        raise ValueError("need >=2 candidates and >=3 spectral points")
    if not cfg["variants"] or not cfg["chain_strengths"] or not cfg["runtimes"]:
        raise ValueError("nonempty variants/strengths/runtimes required")
    for variant in cfg["variants"]:
        if not isinstance(variant, dict) or set(variant) - {"shape", "ports", "field_distribution", "coupling_distribution"}:
            raise ValueError("variant must contain only shape, ports, field_distribution, coupling_distribution")
        if variant.get("shape", "path") not in {"path", "star", "random_tree"}:
            raise ValueError("unknown variant shape")
        if not isinstance(variant.get("ports", 1), int) or variant.get("ports", 1) < 1:
            raise ValueError("variant ports must be a positive integer")
        if variant.get("field_distribution", "uniform") not in {"uniform", "concentrated"}:
            raise ValueError("field_distribution must be uniform or concentrated")
        if variant.get("coupling_distribution", "uniform") not in {"uniform", "random"}:
            raise ValueError("coupling_distribution must be uniform or random")
    if any(not np.isfinite(x) or x <= 0 for x in cfg["chain_strengths"] + cfg["runtimes"]):
        raise ValueError("positive finite strengths and dimensionless runtimes required")
    if cfg["steps"] < 1 or cfg["max_steps"] < 2 * cfg["steps"] or not np.isfinite(cfg["label_state_tolerance"]) or cfg["label_state_tolerance"] <= 0:
        raise ValueError("invalid convergence budget")
    if cfg["steps"] % 8:
        raise ValueError("steps must be a multiple of 8 to resolve fixed9 waveform knots")
    if cfg.get("catalyst_strength", 0) != 0:
        raise ValueError("pilot CLI currently uses X driver only; XX is tested through physics APIs")


def source_fingerprint() -> str:
    """Hash the local Python implementation, including module names and bytes."""
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


@contextmanager
def dataset_lock(root: Path):
    """One writer owns a dataset. A stale lock requires explicit operator review.

    Deliberately do not guess that a foreign PID/host lock is stale on NFS.
    """
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".generation.lock"
    try:
        fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise FileExistsError(f"dataset locked: {lock}; verify no writer is active before manually removing a stale lock") from error
    inode = os.fstat(fd).st_ino
    try:
        os.write(fd, json.dumps({"pid": os.getpid(), "host": socket.gethostname()}).encode())
        os.fsync(fd)
        yield
    finally:
        os.close(fd)
        if lock.exists() and lock.stat().st_ino == inode:
            lock.unlink()


def _atomic_npz(target: Path, payload: dict) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(f".npz.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def _payload_fingerprint(payload: dict) -> str:
    """Integrity digest of stored arrays (including labels), excluding itself."""
    digest = hashlib.sha256()
    for key in sorted(payload):
        if key == "payload_fingerprint":
            continue
        value = np.ascontiguousarray(payload[key])
        digest.update(json.dumps([key, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _lengths(cfg: dict, n: int, parent_index: int) -> tuple[np.ndarray, dict]:
    if "chain_lengths" in cfg:
        given = cfg["chain_lengths"]
        result = np.array([given] * n if isinstance(given, int) else given, dtype=int)
        return result, {"distribution": "fixed", "target_lengths": result.tolist()}
    settings = cfg["chain_distribution"]
    rng = np.random.default_rng(_seed(cfg["seed"], "chain_lengths", parent_index))
    hardware = cfg.get("hardware", {})
    capacity = hardware.get("n_qubits", hardware.get("patch_sites", cfg["max_physical_qubits"]))
    cap = min(cfg["max_physical_qubits"], capacity)
    for attempt in range(1024):
        result = sample_chain_lengths(n, settings["low"], settings["high"], rng,
                                      settings.get("distribution", "uniform"), settings.get("exponent", 2.))
        if result.sum() <= cap:
            return result, {**settings, "target_lengths": result.tolist(),
                            "sampling_condition": "sum_of_target_lengths_le_physical_budget",
                            "budget": cap, "rejected_draws": attempt}
    raise ValueError("chain distribution failed to fit physical budget after 1024 draws; change distribution/budget")


def _plan_parents(cfg: dict) -> list[dict]:
    """Generate and audit logical parents once, independent of worker count."""
    sizes = _logical_sizes(cfg)
    cells = [(family, n) for n in sizes for family in cfg["families"]]
    seen, parents = {}, []
    route = cfg.get("generation_route", "synthetic_lift")
    for index in range(cfg["parents"]):
        family, n = cells[index % len(cells)]
        lengths, length_metadata = _lengths(cfg, n, index)
        for attempt in range(128):
            fixed_embedding = None
            support_cfg = cfg.get("logical_support", {"kind": "complete"})
            if route == "hardware_growth":
                # A fresh patch per parent: different local regions of the chip
                # give the structural variety a single fixed region cannot.
                hw = resolve_hardware(cfg["hardware"],
                                      np.random.default_rng(_seed(cfg["seed"], "hardware_patch", index, attempt)))
                fixed_embedding = grow_hardware_partition(hw["n_qubits"], hw["edges"], lengths,
                    np.random.default_rng(_seed(cfg["seed"], "hardware_partition", index, attempt)))
                # Two mappings compose here and must not be confused: growth
                # returns active->patch indices, the patch carries patch->device
                # qubit ids. What provenance needs is active->device.
                patch_ids = hw.get("original_ids")
                if patch_ids is not None:
                    active = fixed_embedding.metadata["original_physical_ids"]
                    fixed_embedding.metadata["device_qubit_ids"] = [int(patch_ids[i]) for i in active]
                fixed_embedding.metadata.update(
                    {key: hw[key] for key in ("topology", "m", "patch_sites", "source_qubits",
                                              "is_commercial_topology", "is_full_device",
                                              "is_calibrated_device", "generator")
                     if key in hw})
                edges = fixed_embedding.quotient_edges
            else:
                edges = sample_logical_support(n, np.random.default_rng(_seed(cfg["seed"], "support", index, attempt)),
                    kind=support_cfg.get("kind", "complete"), edge_probability=support_cfg.get("edge_probability", .5))
            try:
                # Preserve legacy coefficient seeds for the first attempt.
                seed = _seed(cfg["seed"], "parent", index) if attempt == 0 else _seed(cfg["seed"], "parent_retry", index, attempt)
                problem = generate_problem(n, family, np.random.default_rng(seed), edges=edges)
            except ValueError as error:
                if family == "planted_loops" and "cyclic support" in str(error):
                    continue
                raise
            fingerprint = logical_fingerprint(problem)
            if fingerprint not in seen:
                break
        else:
            raise ValueError(f"cannot generate a unique valid parent {index}; support may lack cycles or distribution may collapse")
        parent_id = f"parent_{index:04d}"
        seen[fingerprint] = parent_id
        parents.append({"parent_id": parent_id, "index": index, "family": family, "size": n,
                        "problem": problem, "lengths": lengths, "length_metadata": length_metadata,
                        "embedding": fixed_embedding, "logical_fingerprint": fingerprint,
                        "generation_attempt": attempt,
                        "support_metadata": support_cfg if route == "synthetic_lift" else {"kind": "hardware_quotient"}})
    return parents


def _split_parents(parents: list[dict], cfg: dict) -> dict[str, str]:
    settings = cfg.get("split", {})
    held_families = set(settings.get("holdout_families", []))
    held_sizes = set(settings.get("holdout_sizes", []))
    if held_families - set(cfg["families"]) or held_sizes - set(_logical_sizes(cfg)):
        raise ValueError("holdout family/size absent from requested distribution")
    if not settings and len(_logical_sizes(cfg)) == 1:
        return parent_splits([p["family"] for p in parents], cfg["seed"])
    validation_fraction = settings.get("validation_fraction", .2)
    test_fraction = settings.get("test_fraction", .2)
    if not (0 < validation_fraction < 1 and 0 <= test_fraction < 1 and validation_fraction + test_fraction < 1):
        raise ValueError("invalid validation/test fractions")
    groups, mapping = {}, {}
    for parent in parents:
        if parent["family"] in held_families or parent["size"] in held_sizes:
            mapping[parent["parent_id"]] = "test"
        else:
            groups.setdefault((parent["family"], parent["size"]), []).append(parent["parent_id"])
    for group, identifiers in sorted(groups.items()):
        identifiers = list(identifiers)
        np.random.default_rng(_seed(cfg["seed"], "split", *group)).shuffle(identifiers)
        n_validation = max(1, int(len(identifiers) * validation_fraction))
        n_test = max(1, int(len(identifiers) * test_fraction)) if test_fraction > 0 else 0
        if len(identifiers) <= n_validation + n_test:
            raise ValueError(f"too few parents in family-size cell {group}; increase parents or lower holdout fractions")
        for i, identifier in enumerate(identifiers):
            mapping[identifier] = "validation" if i < n_validation else "test" if i < n_validation + n_test else "train"
    if set(mapping.values()) != {"train", "validation", "test"}:
        raise ValueError("split must leave nonempty train, validation, and test sets")
    return mapping


def _endpoint_budget(cfg: dict, terms: HamiltonianTerms) -> dict:
    """Conservative analytical estimates, not a peak-RSS measurement."""
    dimension = 1 << terms.n_qubits
    chunk = min(dimension, int(cfg.get("endpoint_chunk_size", 16384)))
    if chunk < 1:
        raise ValueError("endpoint_chunk_size must be positive")
    # Five outcome vectors plus bounded spin/energy temporaries and safety margin.
    endpoint_bytes = 48 * dimension + chunk * (16 * terms.n_qubits + 24 * len(terms.zz_edges) + 256)
    state_bytes = 320 * dimension * min(cfg.get("candidate_batch_size", 1), cfg["candidates"])
    endpoint_mb, state_mb = endpoint_bytes / 2**20, state_bytes / 2**20
    if endpoint_mb > cfg.get("endpoint_memory_mb", 1024):
        raise MemoryError(f"endpoint estimate {endpoint_mb:.1f} MiB exceeds per-worker endpoint_memory_mb")
    if state_mb > cfg.get("state_memory_mb", 1024):
        raise MemoryError(f"state estimate {state_mb:.1f} MiB exceeds per-worker state_memory_mb")
    return {"endpoint_estimated_mib": endpoint_mb, "state_estimated_mib": state_mb,
            "endpoint_chunk_size": chunk, "estimate_not_peak_rss": True}


def _spectral_arrays(points: list) -> dict:
    moments = np.array([[p.mu0, p.g_ss, p.d2] for p in points])
    mask = np.isfinite(moments) & np.array([p.moments_resolved for p in points])[:, None]
    values = {"response_s": np.array([p.s for p in points]),
              "response_moments": np.where(mask, moments, 0.), "response_mask": mask,
              "response_bins": np.stack([p.bin_masses for p in points]),
              "response_frequency_edges": points[0].frequency_edges}
    names = {"low_mass": "low_frequency_mass", "high_mass": "high_frequency_mass",
             "unresolved_mass": "unresolved_mass", "sum_rule_error": "sum_rule_error",
             "orthogonality_error": "orthogonality_error", "energy_resolution": "energy_resolution",
             "regularization_epsilon": "regularization_epsilon", "ground_band_spread": "ground_band_spread",
             "near_degenerate": "near_degenerate_band", "ground_rank": "ground_rank", "raw_gap": "raw_gap",
             "accessible_gap": "accessible_gap", "residual": "max_eigenpair_residual"}
    values.update({f"response_{name}": np.array([getattr(point, attribute) for point in points])
                   for name, attribute in names.items()})
    return values


def _spectral_labels(terms: HamiltonianTerms, cfg: dict, cache: Path, fingerprint: str) -> tuple[dict, dict]:
    if cache.exists():
        with np.load(cache, allow_pickle=False) as stored:
            if str(stored["fingerprint"].item()) != fingerprint:
                raise ValueError("spectral cache fingerprint mismatch")
            if "payload_fingerprint" not in stored or stored["payload_fingerprint"].item() != _payload_fingerprint(stored):
                raise ValueError("spectral cache payload fingerprint mismatch")
            return ({key: stored[key] for key in stored.files if key.startswith("response_")},
                    json.loads(str(stored["diagnostics_json"].item())))
    teacher = dict(cfg.get("teacher", {}))
    mode = teacher.pop("mode", "exact")
    grid = np.linspace(.02, .98, cfg["spectral_points"])
    start = perf_counter()
    if mode == "none":
        # Explicitly absent labels. False masks prevent invented zero targets.
        payload = {"response_s": grid, "response_moments": np.zeros((len(grid), 3)),
                   "response_mask": np.zeros((len(grid), 3), dtype=bool),
                   "response_bins": np.zeros((len(grid), 0)), "response_frequency_edges": np.empty(0),
                   "response_residual": np.zeros(len(grid))}
        diagnostics = {"mode": mode, "labels_available": False, "uniform_certificate": False}
    elif mode == "adaptive":
        from .spectral import adaptive_spectral_profile
        result = adaptive_spectral_profile(terms, initial_grid=np.linspace(0., 1., cfg["spectral_points"]), max_qubits=cfg["max_physical_qubits"],
            seed=_seed(cfg["seed"], "spectral", fingerprint), **teacher)
        payload = _spectral_arrays(result.points)
        audit_keys = {"s", "mu0", "g_ss", "d2", "ground_rank", "raw_gap", "moments_resolved",
                      "max_eigenpair_residual", "orthogonality_error", "unresolved_mass", "sum_rule_error"}
        diagnostics = {"mode": mode, **result.diagnostics,
                       "audit": [{key: value for key, value in point.items() if key in audit_keys}
                                 for point in result.to_dict()["audit_points"]]}
    else:
        points = spectral_profile(terms, grid=grid, max_qubits=cfg["max_physical_qubits"], **teacher)
        if max(p.max_eigenpair_residual for p in points) > 1e-8:
            raise ArithmeticError("spectral residual gate failed")
        payload = _spectral_arrays(points)
        diagnostics = {"mode": mode, "labels_available": True, "uniform_certificate": False,
                       "interpolation_audit": "not_implemented_fixed_grid_labels_only"}
    diagnostics["spectral_seconds_per_path"] = perf_counter() - start
    cache_payload = {**payload, "fingerprint": np.array(fingerprint),
                     "diagnostics_json": np.array(json.dumps(diagnostics, default=jsonable, allow_nan=False))}
    cache_payload["payload_fingerprint"] = np.array(_payload_fingerprint(cache_payload))
    _atomic_npz(cache, cache_payload)
    return payload, diagnostics


def generate_dataset(config: dict, destination: str | Path, *, resume: bool = False,
                     backend: str = "numpy", workers: int = 1) -> dict:
    """Generate auditable parent-disjoint data, with a single coordinating writer.

    CPU workers own independent parents. Worker count is an execution detail,
    excluded from scientific fingerprints; timing/environment metadata may differ.
    CuPy currently requires workers=1 to avoid implicit GPU oversubscription.
    """
    _validate_config(config)
    if backend not in {"numpy", "cupy"}:
        raise ValueError("backend must be numpy or cupy")
    if not isinstance(workers, int) or workers < 1 or (backend == "cupy" and workers != 1):
        raise ValueError("workers must be a positive integer; CuPy requires workers=1")
    cfg = json.loads(json.dumps(config, default=jsonable))
    cfg["backend"] = backend
    cfg_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    source_hash = source_fingerprint()
    root = Path(destination)
    manifest_path = root / "manifest.json"
    with dataset_lock(root):
        prior = None
        if any(p.name != ".generation.lock" for p in root.iterdir()):
            if not resume or not manifest_path.exists():
                raise FileExistsError("destination nonempty; choose a new directory or --resume matching manifest")
            prior = json.loads(manifest_path.read_text())
            if prior["config_hash"] != cfg_hash or prior["schema_version"] != SCHEMA_VERSION:
                raise ValueError("resume config/schema mismatch; refusing stale cache reuse")
            if prior.get("source_fingerprint") != source_hash:
                raise ValueError("resume source fingerprint mismatch; old/unversioned data remain readable but cannot be silently resumed")
            if prior["status"] == "complete":
                load_records(root)
                return prior
        parents = _plan_parents(cfg)
        splits = _split_parents(parents, cfg)
        for parent in parents:
            parent["split"] = splits[parent["parent_id"]]
        manifest = {"schema_version": SCHEMA_VERSION, "status": "in_progress", "config": cfg,
                    "config_hash": cfg_hash, "source_fingerprint": source_hash,
                    "environment": environment(), "execution": {"workers": workers, "backend": backend},
                    "splits": splits, "records": [], "scope": cfg.get("generation_route", "synthetic_lift") + "_closed_system",
                    "teacher_mode": cfg.get("teacher", {}).get("mode", "exact"),
                    "decoder": "majority_tie_plus_v1",
                    "parents": [{"parent_id": p["parent_id"], "family": p["family"], "logical_qubits": p["size"],
                                 "logical_fingerprint": p["logical_fingerprint"], "split": p["split"]} for p in parents],
                    "duplicate_audit": {"exact_labelled_coefficients": True, "duplicates": 0,
                                        "graph_or_gauge_isomorphism_checked": False},
                    "training_labels_not_deployment_inputs":
                    ["response_*", "candidate_losses", "ground_energy", "planted_spins"],
                    "budget_note": "spectral labels privileged; common bank independent of test spectrum"}
        write_json(manifest_path, manifest)
        started = perf_counter()
        try:
            arguments = [(cfg, root, cfg_hash, source_hash, p, resume, backend) for p in parents]
            if workers == 1:
                for args in arguments:
                    manifest["records"].extend(_generate_parent(*args))
                    write_json(manifest_path, manifest)
            else:
                with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
                    # Ordered map retains deterministic manifest order.
                    for rows in pool.map(_parent_job, arguments):
                        manifest["records"].extend(rows)
                        write_json(manifest_path, manifest)
        except Exception as error:
            manifest["error"] = {"type": type(error).__name__, "message": str(error)}
            manifest["elapsed_seconds"] = perf_counter() - started
            write_json(manifest_path, manifest)
            raise
        manifest.update(status="complete", elapsed_seconds=perf_counter() - started,
                        record_count=len(manifest["records"]))
        manifest["dataset_fingerprint"] = hashlib.sha256(json.dumps(
            [(r["record_id"], r["fingerprint"]) for r in manifest["records"]], sort_keys=True).encode()).hexdigest()
        write_json(manifest_path, manifest)
        return manifest


def _parent_job(arguments):
    return _generate_parent(*arguments)


def _candidate_results(terms: HamiltonianTerms, schedules: list, runtime: float,
                       cfg: dict, path: AnnealPath, backend: str):
    """Yield converged candidate states; batched costs are shared elapsed time.

    Retry only failed rows at doubled resolution. Never pick a different control
    because its simulation is difficult; exhausting the budget fails the task.
    """
    batch_size = cfg.get("candidate_batch_size", 1)
    if batch_size == 1:
        for index, schedule in enumerate(schedules):
            start = perf_counter()
            steps = cfg["steps"]
            while True:
                result = propagate(terms, schedule, runtime, steps=steps, path=path,
                                   backend=backend, step_doubling=True)
                if result.norm_error > 1e-9:
                    raise ArithmeticError("norm gate failed (state never renormalized)")
                if result.step_doubling_error <= cfg["label_state_tolerance"]:
                    break
                steps *= 2
                if 2 * steps > cfg["max_steps"]:
                    raise ArithmeticError(f"label convergence gate failed: candidate {index}")
            yield index, result, perf_counter() - start
        return
    for offset in range(0, len(schedules), batch_size):
        indices = list(range(offset, min(offset + batch_size, len(schedules))))
        pending, accepted, costs = indices.copy(), {}, {i: 0. for i in indices}
        steps = cfg["steps"]
        while pending:
            start = perf_counter()
            result = propagate_batch(terms, [schedules[i] for i in pending], runtime,
                steps=steps, path=path, backend=backend, step_doubling=True,
                state_tolerance=cfg["label_state_tolerance"], norm_tolerance=1e-9,
                max_state_bytes=int(cfg.get("state_memory_mb", 1024) * 2**20))
            elapsed_share = (perf_counter() - start) / len(pending)
            if np.max(result.norm_errors) > 1e-9:
                raise ArithmeticError("batched norm gate failed (state never renormalized)")
            retry = []
            for row, index in enumerate(pending):
                costs[index] += elapsed_share
                if result.accepted[row]:
                    accepted[index] = PropagationResult(result.states[row].copy(), float(result.norm_errors[row]),
                        result.steps, float(result.step_doubling_errors[row]), backend, result.method)
                else:
                    retry.append(index)
            pending = retry
            if pending:
                steps *= 2
                if 2 * steps > cfg["max_steps"]:
                    raise ArithmeticError(f"batched label convergence gate failed: candidates {pending}")
        for index in indices:
            yield index, accepted.pop(index), costs[index]


def _generate_parent(cfg: dict, root: Path, cfg_hash: str, source_hash: str,
                     parent: dict, resume: bool, backend: str) -> list[dict]:
    parent_index, parent_id, family = parent["index"], parent["parent_id"], parent["family"]
    problem = parent["problem"]
    splits = {parent_id: parent["split"]}
    records = []
    sampled_tau = np.linspace(0., 1., 9)
    path = AnnealPath()
    for variant_index, variant in enumerate(cfg["variants"]):
        emb = parent["embedding"] or synthetic_lift(problem, parent["lengths"],
                             np.random.default_rng(_seed(cfg["seed"], "embedding", parent_index, variant_index)),
                             shape=variant.get("shape", "path"), ports=variant.get("ports", 1))
        for strength_index, strength in enumerate(cfg["chain_strengths"]):
            compiled = compile_embedding(problem, emb, strength,
                np.random.default_rng(_seed(cfg["seed"], "distribution", parent_index, variant_index)),
                field_distribution=variant.get("field_distribution", "uniform"),
                coupling_distribution=variant.get("coupling_distribution", "uniform"),
                h_limit=cfg.get("h_limit", 2.), j_limit=cfg.get("j_limit", 1.))
            terms = HamiltonianTerms(compiled.physical.n, compiled.physical.h,
                                     compiled.physical.edges, compiled.physical.J)
            memory = _endpoint_budget(cfg, terms)
            validity = validate_compilation(compiled, max_qubits=cfg.get("endpoint_max_qubits", 20),
                                            chunk_size=memory["endpoint_chunk_size"])
            observables = output_observables(compiled, max_qubits=cfg.get("endpoint_max_qubits", 20),
                chunk_size=memory["endpoint_chunk_size"], ground_energy=validity["logical_ground_energy"])
            path_fingerprint = hashlib.sha256((compiled.fingerprint() + cfg_hash + source_hash).encode()).hexdigest()
            spectral_arrays, spectral_diagnostics = _spectral_labels(terms, cfg,
                root / "spectral_cache" / f"{parent_id}_e{variant_index}_k{strength_index}.npz", path_fingerprint)
            for runtime_index, runtime in enumerate(cfg["runtimes"]):
                record_id = f"{parent_id}_e{variant_index}_k{strength_index}_t{runtime_index}"
                target = root / "records" / f"{record_id}.npz"
                fingerprint = hashlib.sha256((compiled.fingerprint() + cfg_hash + source_hash + str(runtime)).encode()).hexdigest()
                record_meta = {"record_id": record_id, "parent_id": parent_id, "split": splits[parent_id],
                               "path": str(target.relative_to(root)), "fingerprint": fingerprint,
                               "logical_fingerprint": parent["logical_fingerprint"], "logical_qubits": problem.n}
                if resume and target.exists():
                    with np.load(target, allow_pickle=False) as old:
                        if str(old["fingerprint"].item()) != fingerprint:
                            raise ValueError("record cache fingerprint mismatch")
                        if "payload_fingerprint" not in old or old["payload_fingerprint"].item() != _payload_fingerprint(old):
                            raise ValueError("record cache payload fingerprint mismatch")
                    record_meta["file_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
                    records.append(record_meta)
                    continue
                bank = shared_candidate_bank(n=cfg["candidates"], n_segments=8, seed=cfg["seed"],
                                             runtime=runtime, max_slope=4. / runtime)
                # Sampled waveforms are the authoritative controls for this ML benchmark.
                # Reconstruct them before simulation, ensuring critic sees exactly the
                # same piecewise-linear waveform that generated the label (no aliasing).
                from .schedules import Schedule
                waves = np.stack([c.schedule(sampled_tau) for c in bank])
                executed = [Schedule(sampled_tau, wave) for wave in waves]
                labels, norms, errors, actual_steps, costs, refs = [], [], [], [], [], []
                reference_seconds = 0.0
                for candidate_index, result, elapsed in _candidate_results(terms, executed, runtime, cfg, path, backend):
                    schedule = executed[candidate_index]
                    probabilities = np.abs(result.state) ** 2
                    labels.append([float(probabilities @ observables[name]) for name in
                                   ("success", "decoded_energy", "any_chain_break", "chain_break_fraction")])
                    norms.append(result.norm_error)
                    errors.append(result.step_doubling_error)
                    actual_steps.append(result.steps)
                    costs.append(elapsed)
                    # Independent method on linear schedule of first runtime per path.
                    if candidate_index == 0 and runtime_index == 0 and compiled.physical.n <= 8:
                        reference_start = perf_counter()
                        ref = dense_reference_propagate(terms, schedule, runtime)
                        ref_error = float(np.linalg.norm(ref.state - result.state))
                        reference_seconds += perf_counter() - reference_start
                        refs.append(ref_error)
                        if ref_error > max(5 * cfg["label_state_tolerance"], 1e-7):
                            raise ArithmeticError("independent dense dynamics audit failed")
                labels = np.asarray(labels)
                metadata = {**validity, "variant": variant, "embedding": emb.metadata,
                            "problem_metadata": problem.metadata, "reference_status": "finite_shared_bank_best_not_global",
                            "time_units": "dimensionless_Eref_T_over_hbar", "path": "linear_X_plus_Z",
                            "candidate_rule": "common_bank_resampled9_then_simulated",
                            "candidate_batch_size": cfg.get("candidate_batch_size", 1),
                            "candidate_cost_rule": "shared_batch_elapsed_per_active_row" if cfg.get("candidate_batch_size", 1) > 1 else "per_candidate_elapsed",
                            "original_candidate_parameters": [c.parameters for c in bank],
                            "spectral_seconds_per_path": spectral_diagnostics["spectral_seconds_per_path"],
                            "teacher": spectral_diagnostics, "memory": memory,
                            "logical_support": parent["support_metadata"],
                            "variant_geometry_applicable": parent["embedding"] is None,
                            "chain_distribution": parent["length_metadata"],
                            "parent_generation_attempt": parent["generation_attempt"],
                            "source_fingerprint": source_hash,
                            "logical_fingerprint": parent["logical_fingerprint"],
                            "independent_reference_seconds": reference_seconds,
                            "independent_reference_errors": refs,
                            "slope_contract": "max_ds_dtau=4; not a fixed physical slew across runtimes",
                            "interpolation_audit": spectral_diagnostics.get("interpolation_audit", "see_teacher_diagnostics")}
                payload = dict(parent_id=np.array(parent_id), record_id=np.array(record_id),
                    split=np.array(splits[parent_id]), family=np.array(family), fingerprint=np.array(fingerprint),
                    physical_h=compiled.physical.h, physical_edges=compiled.physical.edges,
                    physical_J=compiled.physical.J, problem_J=compiled.problem_J, chain_J=compiled.chain_J,
                    membership=emb.membership, logical_h=problem.h, logical_edges=problem.edges, logical_J=problem.J,
                    programmed_scale=np.array(compiled.programmed_scale), runtime=np.array(runtime),
                    chain_strength=np.array(strength), catalyst_strength=np.array(0.),
                    **spectral_arrays,
                    source_fingerprint=np.array(source_hash), logical_fingerprint=np.array(parent["logical_fingerprint"]),
                    candidate_schedules=waves, candidate_tau=sampled_tau,
                    candidate_ids=np.array([c.candidate_id for c in bank]), candidate_losses=1. - labels[:, 0],
                    candidate_success=labels[:, 0], candidate_decoded_energy=labels[:, 1],
                    candidate_any_chain_break=labels[:, 2], candidate_chain_break_fraction=labels[:, 3],
                    candidate_norm_error=np.array(norms), candidate_state_error=np.array(errors),
                    candidate_steps=np.array(actual_steps), candidate_seconds=np.array(costs),
                    metadata_json=np.array(json.dumps(metadata, default=jsonable)))
                payload["payload_fingerprint"] = np.array(_payload_fingerprint(payload))
                _atomic_npz(target, payload)
                record_meta["file_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
                records.append(record_meta)
                print(f"accepted {record_id}: Nphys={compiled.physical.n}, candidates={len(bank)}", flush=True)
    return records


def load_records(root: str | Path, split: str | None = None) -> list[dict]:
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported dataset schema_version")
    if manifest["status"] != "complete":
        raise ValueError("incomplete dataset is not accepted for training")
    if manifest.get("record_count") != len(manifest["records"]):
        raise ValueError("manifest record_count mismatch")
    identifiers = [item["record_id"] for item in manifest["records"]]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("duplicate manifest record_id")
    from .evaluation import audit_parent_splits
    audit_parent_splits([item["parent_id"] for item in manifest["records"]],
                        [item["split"] for item in manifest["records"]])
    if split not in {None, "train", "validation", "test"}:
        raise ValueError("unknown split")
    result, seen_parents, seen_logical = [], {}, {}
    # Cross-split structural auditing uses manifest-only metadata; opening test
    # outcome records during training/tuning is deliberately forbidden.
    for item in manifest["records"]:
        parent_id, fingerprint = item["parent_id"], item.get("logical_fingerprint")
        if manifest.get("splits", {}).get(parent_id) != item["split"]:
            raise ValueError("manifest parent split mismatch")
        if "source_fingerprint" in manifest and fingerprint is None:
            raise ValueError("new manifest missing logical fingerprint")
        if fingerprint is not None:
            if parent_id in seen_parents and seen_parents[parent_id] != fingerprint:
                raise ValueError("manifest parent ID has inconsistent logical fingerprints")
            if fingerprint in seen_logical and seen_logical[fingerprint] != parent_id:
                raise ValueError("duplicate exact labelled logical parent under different IDs")
            seen_parents[parent_id], seen_logical[fingerprint] = fingerprint, parent_id
    for item in manifest["records"]:
        if split is not None and item["split"] != split:
            continue
        target = (root / item["path"]).resolve()
        if not target.is_relative_to(root.resolve()):
            raise ValueError("record path escapes dataset")
        if "file_sha256" in item and hashlib.sha256(target.read_bytes()).hexdigest() != item["file_sha256"]:
            raise ValueError("record file checksum mismatch")
        with np.load(target, allow_pickle=False) as arrays:
            record = {key: arrays[key] for key in arrays.files}
        if "payload_fingerprint" in record and record["payload_fingerprint"].item() != _payload_fingerprint(record):
            raise ValueError("record payload fingerprint mismatch")
        if str(record["parent_id"].item()) != item["parent_id"] or str(record["split"].item()) != item["split"]:
            raise ValueError("manifest/record split mismatch")
        if manifest.get("splits", {}).get(item["parent_id"]) != item["split"]:
            raise ValueError("manifest parent split mismatch")
        if str(record["fingerprint"].item()) != item["fingerprint"]:
            raise ValueError("manifest/record fingerprint mismatch")
        needed = {"physical_h", "physical_edges", "physical_J", "membership", "logical_h",
                  "logical_edges", "logical_J", "runtime", "candidate_losses", "candidate_schedules"}
        if needed - set(record):
            raise ValueError("missing required record arrays")
        from .generation import IsingProblem
        logical = IsingProblem(record["logical_h"], record["logical_edges"], record["logical_J"])
        actual_fingerprint = logical_fingerprint(logical)
        parent_id = item["parent_id"]
        if "logical_fingerprint" in record and record["logical_fingerprint"].item() != actual_fingerprint:
            raise ValueError("logical coefficient fingerprint mismatch")
        if item.get("logical_fingerprint", actual_fingerprint) != actual_fingerprint:
            raise ValueError("manifest/record logical fingerprint mismatch")
        if parent_id in seen_parents and seen_parents[parent_id] != actual_fingerprint:
            raise ValueError("one parent ID has different logical coefficients")
        if actual_fingerprint in seen_logical and seen_logical[actual_fingerprint] != parent_id:
            raise ValueError("duplicate exact labelled logical parent under different IDs")
        seen_parents[parent_id], seen_logical[actual_fingerprint] = actual_fingerprint, parent_id
        if "source_fingerprint" in manifest and str(record.get("source_fingerprint", "")) != manifest["source_fingerprint"]:
            raise ValueError("manifest/record source fingerprint mismatch")
        waves, losses = record["candidate_schedules"], record["candidate_losses"]
        if waves.ndim != 2 or waves.shape[1] != 9 or losses.shape != (len(waves),):
            raise ValueError("invalid candidate waveform/loss shapes")
        if not np.all(np.isfinite(waves)) or not np.all(np.isfinite(losses)):
            raise ValueError("nonfinite candidate waveform/loss")
        if np.any(losses < -1e-8) or np.any(losses > 1 + 1e-8):
            raise ValueError("success-loss values outside tolerance")
        if split is None or item["split"] == split:
            result.append(record)
    return result
