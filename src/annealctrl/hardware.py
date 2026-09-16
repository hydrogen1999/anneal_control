"""Offline hardware program validation and user-supplied sample ingestion.

Nothing in this module submits jobs, accesses credentials, or validates an
actual QPU execution. Device declarations must come from the intended device;
passing these checks is necessary, not sufficient, for vendor acceptance.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
from typing import Any

import numpy as np
from scipy.stats import beta

from .generation import Embedding, IsingProblem, all_spins
from .schedules import Schedule


def _integers(value: Any, name: str, *, ndim: int = 1) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != ndim or not np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.bool_):
        raise ValueError(f"{name} must be an integer array of dimension {ndim}")
    return array.astype(np.int64, copy=True)


def _hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _scalar(record: dict, key: str, default: Any = None) -> Any:
    return np.asarray(record[key]).item() if key in record else default


@dataclass(frozen=True)
class DeviceConstraints:
    device_id: str
    hardware_nodes: list[int]
    hardware_edges: list[list[int]]
    h_range: list[float]
    J_range: list[float]
    runtime_range: list[float]
    time_unit: str
    max_schedule_points: int
    max_slope: float
    coefficient_unit: str
    time_resolution: float | None = None
    s_resolution: float | None = None

    def __post_init__(self):
        nodes = _integers(self.hardware_nodes, "hardware_nodes")
        edges = np.asarray(self.hardware_edges)
        if edges.size == 0:
            edges = np.empty((0, 2), dtype=np.int64)
        edges = _integers(edges, "hardware_edges", ndim=2)
        if not self.device_id or not len(nodes) or len(set(nodes.tolist())) != len(nodes):
            raise ValueError("device_id and unique hardware_nodes are required")
        if np.any(nodes < 0) or edges.shape[1:] != (2,):
            raise ValueError("nonnegative hardware IDs and (E,2) edges are required")
        edge_pairs = [tuple(sorted(edge)) for edge in edges.tolist()]
        if any(u == v or u not in nodes or v not in nodes for u, v in edge_pairs) or len(set(edge_pairs)) != len(edge_pairs):
            raise ValueError("invalid or duplicate hardware edges")
        for name in ("h_range", "J_range", "runtime_range"):
            values = np.asarray(getattr(self, name), dtype=float)
            if values.shape != (2,) or not np.isfinite(values).all() or values[0] > values[1]:
                raise ValueError(f"{name} must be a finite ordered pair")
        if self.runtime_range[0] <= 0 or self.time_unit != "us":
            raise ValueError("hardware runtime_range must be positive; supported time_unit is us")
        if self.coefficient_unit != "dimensionless_ising":
            raise ValueError("hardware coefficients must be explicitly dimensionless_ising")
        if isinstance(self.max_schedule_points, bool) or not isinstance(self.max_schedule_points, (int, np.integer)) or self.max_schedule_points < 2:
            raise ValueError("max_schedule_points must be an integer >=2")
        if not np.isfinite(self.max_slope) or self.max_slope <= 0:
            raise ValueError("max_slope must be finite and positive in inverse us")
        for name in ("time_resolution", "s_resolution"):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")
        # Canonicalize inputs so hashing does not depend on tuple vs ndarray.
        object.__setattr__(self, "hardware_nodes", nodes.tolist())
        object.__setattr__(self, "hardware_edges", [list(edge) for edge in edge_pairs])
        object.__setattr__(self, "max_schedule_points", int(self.max_schedule_points))
        object.__setattr__(self, "max_slope", float(self.max_slope))
        for name in ("time_resolution", "s_resolution"):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, float(getattr(self, name)))
        for name in ("h_range", "J_range", "runtime_range"):
            object.__setattr__(self, name, np.asarray(getattr(self, name), dtype=float).tolist())

    @classmethod
    def from_mapping(cls, data: dict) -> "DeviceConstraints":
        try:
            return cls(**data)
        except TypeError as exc:
            raise ValueError("incomplete or unknown DeviceConstraints fields") from exc


def export_program(
    record: dict, schedule: Schedule, constraints: DeviceConstraints, *,
    physical_ids: Any, runtime: float, time_unit: str, tie_policy: str,
    gauge: Any = None, coefficient_scale: float = 1., logical_ground_energy: float | None = None,
) -> dict:
    """Build a hashed JSON-safe plan; no autoscaling or submission is performed.

`physical_ids[i]` names the device qubit corresponding to record qubit i.
`gauge='identity'` or explicit +/-1 values is mandatory. Gauge convention is
z_programmed[i]=gauge[i]*z_original[i]. The simulation runtime is NOT converted
to hardware us; the caller must provide a deliberate hardware runtime.
"""
    needed = {"physical_h", "physical_edges", "physical_J", "membership",
              "logical_h", "logical_edges", "logical_J", "record_id", "fingerprint"}
    if needed - record.keys():
        raise ValueError(f"missing record fields: {sorted(needed - record.keys())}")
    if float(_scalar(record, "catalyst_strength", 0.)) != 0:
        raise ValueError("hardware exporter does not support simulated XX catalysts")
    physical = IsingProblem(record["physical_h"], record["physical_edges"], record["physical_J"])
    logical = IsingProblem(record["logical_h"], record["logical_edges"], record["logical_J"])
    membership = _integers(record["membership"], "membership")
    embedding = Embedding(membership, physical.edges)
    if len(membership) != physical.n or embedding.n_logical != logical.n:
        raise ValueError("record embedding dimensions do not match coefficients")
    ids = _integers(physical_ids, "physical_ids")
    if ids.shape != (physical.n,) or len(set(ids.tolist())) != physical.n:
        raise ValueError("physical_ids must map each record qubit to a unique device qubit")
    if not set(ids.tolist()).issubset(constraints.hardware_nodes):
        raise ValueError("record maps to unavailable hardware qubits")
    allowed = {tuple(sorted(edge)) for edge in constraints.hardware_edges}
    mapped_edges = [[int(ids[i]), int(ids[j])] for i, j in physical.edges]
    if any(tuple(sorted(edge)) not in allowed for edge in mapped_edges):
        raise ValueError("programmed edge is unavailable on declared hardware topology")
    if time_unit != constraints.time_unit:
        raise ValueError("explicit hardware time_unit must match the device; simulation time is not us")
    if not np.isfinite(runtime) or not constraints.runtime_range[0] <= runtime <= constraints.runtime_range[1]:
        raise ValueError("runtime outside device range")
    if len(schedule.tau_knots) > constraints.max_schedule_points:
        raise ValueError("schedule exceeds device max_schedule_points")
    schedule.validate_slope(runtime=runtime, max_slope=constraints.max_slope)
    times = schedule.tau_knots * runtime
    for values, resolution, name in ((times, constraints.time_resolution, "time"),
                                     (schedule.s_knots, constraints.s_resolution, "s")):
        if resolution is not None and not np.allclose(values / resolution, np.rint(values / resolution), rtol=0., atol=1e-8):
            raise ValueError(f"schedule violates device {name} quantization; rounding is never automatic")
    if tie_policy not in {"plus", "minus", "reject"}:
        raise ValueError("tie_policy must explicitly be plus, minus, or reject")
    if isinstance(gauge, str) and gauge == "identity":
        gauges = np.ones(physical.n, dtype=np.int64)
    else:
        if gauge is None:
            raise ValueError("explicitly select gauge='identity' or one +/-1 value per physical qubit")
        gauges = _integers(gauge, "gauge")
        if gauges.shape != (physical.n,) or not np.isin(gauges, [-1, 1]).all():
            raise ValueError("gauge must contain one +/-1 per physical qubit")
    if not np.isfinite(coefficient_scale) or coefficient_scale <= 0:
        raise ValueError("coefficient_scale must be finite and positive")
    h = coefficient_scale * physical.h * gauges
    J = coefficient_scale * physical.J * gauges[physical.edges[:, 0]] * gauges[physical.edges[:, 1]]
    for values, limits, name in ((h, constraints.h_range, "h"), (J, constraints.J_range, "J")):
        if np.any(values < limits[0]) or np.any(values > limits[1]):
            raise ValueError(f"programmed {name} outside device range; no hidden autoscale")
    if logical_ground_energy is not None and not np.isfinite(logical_ground_energy):
        raise ValueError("logical_ground_energy must be finite")
    if logical.n <= 20:
        ground = float(logical.energy(all_spins(logical.n)).min())
        if logical_ground_energy is not None and not np.isclose(ground, logical_ground_energy, rtol=0., atol=1e-9):
            raise ValueError("provided logical_ground_energy disagrees with exact enumeration")
        ground_source = "exact_enumeration"
    else:
        ground = None if logical_ground_energy is None else float(logical_ground_energy)
        ground_source = "unavailable" if ground is None else "user_supplied_not_verified"
    source = {"record_id": str(_scalar(record, "record_id")),
              "fingerprint": str(_scalar(record, "fingerprint")), "physical_h": physical.h.tolist(),
              "physical_edges": physical.edges.tolist(), "physical_J": physical.J.tolist(),
              "logical_h": logical.h.tolist(), "logical_edges": logical.edges.tolist(),
              "logical_J": logical.J.tolist(), "membership": membership.tolist()}
    program = {
        "schema_version": 1, "status": "offline_plan_not_submitted_or_qpu_validated",
        "source_record": source, "source_parameter_hash": _hash(source),
        "device_constraints": asdict(constraints), "physical_ids": ids.tolist(),
        "h": h.tolist(), "edges": mapped_edges, "J": J.tolist(),
        "coefficient_unit": "dimensionless_ising", "coefficient_scale": float(coefficient_scale),
        "autoscale": False, "energy_convention": "sum_h_z_plus_sum_J_zz_no_offset",
        "schedule": [[float(t), float(s)] for t, s in zip(times, schedule.s_knots)],
        "runtime": float(runtime), "time_unit": time_unit,
        "source_simulation_runtime": _scalar(record, "runtime"),
        "time_mapping": "hardware_runtime_explicitly_selected_not_inferred_from_simulation",
        "gauge": gauges.tolist(), "gauge_convention": "z_programmed=gauge*z_original",
        "tie_policy": tie_policy, "logical_ground_energy": ground,
        "logical_ground_energy_source": ground_source,
    }
    program["program_hash"] = _hash(program)
    return program


def _binomial_ci(successes: int, shots: int, confidence: float) -> list[float]:
    alpha = 1 - confidence
    low = 0. if successes == 0 else float(beta.ppf(alpha / 2, successes, shots - successes + 1))
    high = 1. if successes == shots else float(beta.ppf(1 - alpha / 2, successes + 1, shots - successes))
    return [low, high]


def ingest_samples(program: dict, samples: dict, *, confidence: float = .95,
                   energy_tolerance: float = 1e-7) -> dict:
    """Check provenance/energies, undo gauge, decode chains, aggregate counts.

Rows are mappings {values: [...], count: positive integer, energy: number}.
`variable_order` explicitly lists the device IDs in values order. Binary data
must also declare binary_zero_spin (+1 or -1). Duplicate rows are aggregated,
not discarded; counts are never interpreted as independent jobs. Binomial
intervals assume IID shots, which may fail under device drift/correlation.
"""
    if not np.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0,1)")
    if not np.isfinite(energy_tolerance) or energy_tolerance <= 0:
        raise ValueError("energy_tolerance must be finite and positive")
    payload = {key: value for key, value in program.items() if key != "program_hash"}
    if program.get("schema_version") != 1 or program.get("program_hash") != _hash(payload):
        raise ValueError("program hash mismatch or unsupported schema")
    if program["source_parameter_hash"] != _hash(program["source_record"]):
        raise ValueError("source parameter hash mismatch")
    if samples.get("program_hash") != program["program_hash"]:
        raise ValueError("samples must identify the exact exported program_hash")
    if samples.get("energy_convention") != "programmed_ising":
        raise ValueError("energy_convention must be programmed_ising (no offset)")
    vartype = samples.get("vartype")
    if vartype not in {"SPIN", "BINARY"}:
        raise ValueError("vartype must explicitly be SPIN or BINARY")
    if vartype == "BINARY" and (isinstance(samples.get("binary_zero_spin"), bool) or samples.get("binary_zero_spin") not in {-1, 1}):
        raise ValueError("BINARY samples require explicit binary_zero_spin=+1 or -1")
    order = _integers(samples.get("variable_order"), "variable_order")
    ids = program["physical_ids"]
    if len(order) != len(ids) or len(set(order.tolist())) != len(ids) or set(order.tolist()) != set(ids):
        raise ValueError("variable_order must contain each programmed device qubit exactly once")
    positions = {int(node): index for index, node in enumerate(order)}
    reorder = [positions[node] for node in ids]
    source = program["source_record"]
    physical = IsingProblem(program["h"], source["physical_edges"], program["J"])
    logical = IsingProblem(source["logical_h"], source["logical_edges"], source["logical_J"])
    membership, gauge = np.asarray(source["membership"]), np.asarray(program["gauge"])
    if not isinstance(samples.get("rows"), list) or not samples["rows"]:
        raise ValueError("nonempty sample rows are required")
    aggregate: dict[tuple, dict] = {}
    for row in samples["rows"]:
        if not isinstance(row, dict) or not {"values", "count", "energy"}.issubset(row):
            raise ValueError("each sample row needs values, count, and energy")
        values = _integers(row["values"], "sample values")
        if values.shape != (len(ids),) or not np.isin(values, [-1, 1] if vartype == "SPIN" else [0, 1]).all():
            raise ValueError("sample values violate declared vartype or dimension")
        count = row["count"]
        if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count <= 0:
            raise ValueError("sample count must be a positive integer")
        try:
            reported_energy = float(row["energy"])
        except (TypeError, ValueError) as exc:
            raise ValueError("sample energy must be finite") from exc
        if not np.isfinite(reported_energy):
            raise ValueError("sample energy must be finite")
        spins = values[reorder] if vartype == "SPIN" else (1 - 2 * values[reorder]) * samples["binary_zero_spin"]
        calculated = float(physical.energy(spins))
        if not np.isclose(reported_energy, calculated, rtol=0., atol=energy_tolerance):
            raise ValueError("sample energy does not match programmed coefficients/order/gauge")
        key = tuple(int(v) for v in spins)
        if key not in aggregate:
            aggregate[key] = {"spins": spins, "count": 0, "energy": calculated}
        aggregate[key]["count"] += int(count)
    total = accepted = successes = broken_shots = 0
    chain_broken_count = energy_sum = programmed_energy_sum = 0.
    decoded_rows = []
    for row in aggregate.values():
        count, z = row["count"], row["spins"] * gauge
        sums = np.bincount(membership, weights=z, minlength=logical.n)
        lengths = np.bincount(membership, minlength=logical.n)
        broken = np.abs(sums) != lengths
        tied = sums == 0
        total += count
        broken_shots += count * int(broken.any())
        chain_broken_count += count * float(broken.mean())
        programmed_energy_sum += count * row["energy"]
        rejected = bool(tied.any() and program["tie_policy"] == "reject")
        decoded = np.where(sums > 0, 1, -1)
        decoded[tied] = 1 if program["tie_policy"] == "plus" else -1
        decoded_energy = None if rejected else float(logical.energy(decoded))
        if not rejected:
            accepted += count
            energy_sum += count * decoded_energy
            ground = program["logical_ground_energy"]
            if ground is not None:
                if decoded_energy < ground - energy_tolerance:
                    raise ValueError("decoded energy is below claimed logical ground; ground metadata is invalid")
                successes += count * int(abs(decoded_energy - ground) <= energy_tolerance)
        decoded_rows.append({"original_spins": z.tolist(), "count": count,
                             "decoded_spins": None if rejected else decoded.tolist(),
                             "decoded_energy": decoded_energy, "rejected_tie": rejected,
                             "broken_chain_fraction": float(broken.mean())})
    ground_known = program["logical_ground_energy"] is not None
    return {"schema_version": 1, "program_hash": program["program_hash"],
            "source_parameter_hash": program["source_parameter_hash"],
            "status": "user_supplied_samples_validated_offline_not_authenticated_qpu_results",
            "shots": total, "unique_physical_samples": len(aggregate), "input_rows": len(samples["rows"]),
            "accepted_shots": accepted, "rejected_tie_shots": total - accepted,
            "tie_policy": program["tie_policy"], "successes": successes if ground_known else None,
            "success_probability": successes / total if ground_known else None,
            "success_confidence_interval": _binomial_ci(successes, total, confidence) if ground_known else None,
            "confidence": confidence, "interval_method": "Clopper_Pearson_IID_binomial",
            "interval_scope": "counts_assumed_IID_no_device_drift_or_correlation_adjustment",
            "success_denominator": "all_shots_including_rejected_ties_as_failures",
            "logical_ground_energy_source": program["logical_ground_energy_source"],
            "mean_decoded_energy_accepted": energy_sum / accepted if accepted else None,
            "mean_programmed_energy": programmed_energy_sum / total,
            "any_chain_break_probability": broken_shots / total,
            "mean_chain_break_fraction": chain_broken_count / total, "decoded_samples": decoded_rows}
