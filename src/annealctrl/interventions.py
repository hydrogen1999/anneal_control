"""G3: paired embedding interventions and cross-control loss matrices.

The representation claim is causal, and the protocol (§7) states the standard it
has to meet: hold the logical objective, decoder, driver family, runtime,
candidate budget and numerical accuracy fixed, intervene on **one** declared
physical feature, and show that the *preferred control* changes — not merely that
a predicted parameter changes. Evidence that a schedule parameter moved is
compatible with both controls being equally good.

Three things this module refuses to do, each because the alternative produces a
number that looks like evidence and is not:

1. **Change two factors at once.** ``build_pairs`` rejects a change touching more
   than the one key its declared factor implies, and verifies afterwards what
   actually moved in the physical graph — a geometry change that relocated the
   boundary ports is not a geometry intervention.
2. **Confound an intervention with the coefficient scale.** Chain strength enters
   the same cap as the problem couplers, so raising it generally lowers the
   programmed scale α. Whenever α moves, a second ``scale_controlled`` pair is
   emitted with one common conservative α, and both are reported. Neither is
   called "the" effect.
3. **Clip a transfer penalty.** A control imported from the other arm can be
   better; that sign is information about the intervention's strength.

Every causal statement here is **inside the declared closed-system simulator**.
See ``docs/decisions/ADR-0003-single-factor-interventions-with-scale-control.md``.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np

from .generation import (
    CompiledInstance,
    IsingProblem,
    compile_embedding,
    conservative_common_scale,
    output_observables,
    synthetic_lift,
    validate_compilation,
)
from .headroom import _bootstrap, _describe, _parent_means
from .physics import AnnealPath, HamiltonianTerms
from .schedules import Schedule
from .search import CONTROL_FAMILIES, optimize_control_family
from .sweeps import SweepUnit, run_sweep
from .telemetry import _safe

# Declared factor -> the single specification key it is allowed to change.
FACTORS: dict[str, str] = {
    "geometry": "shape",
    "ports": "ports",
    "field_allocation": "field_distribution",
    "chain_strength": "chain_strength",
    "chain_length": "chain_lengths",
}
SIZE_CHANGING = frozenset({"chain_length"})

BASE_KEYS = frozenset({"shape", "ports", "field_distribution", "coupling_distribution",
                       "chain_strength"})


class VacuousIntervention(ValueError):
    """The declared change left the physical instance identical on both arms.

    Distinct from a confounded intervention: a ``random_tree`` shape can happen
    to reproduce the path on a short chain, which makes that one pair empty
    rather than wrong. A planner may skip and count these; it must never skip a
    confounded pair, which is why the two are different exceptions.
    """


HELD_FIXED = ("logical_coefficients", "logical_graph", "decoder", "driver_family",
              "runtime", "control_families", "objective_budget", "search_seed",
              "numerical_tolerance")


@dataclass(frozen=True)
class InterventionArm:
    label: str
    descriptor: dict
    compiled: CompiledInstance
    validation: dict

    @property
    def terms(self) -> HamiltonianTerms:
        physical = self.compiled.physical
        return HamiltonianTerms(physical.n, physical.h, physical.edges, physical.J)


@dataclass(frozen=True)
class InterventionPair:
    pair_id: str
    parent_id: str
    factor: str
    scale_arm: str
    runtime: float
    arms: tuple[InterventionArm, InterventionArm]
    held_fixed: tuple[str, ...]
    physical_size_matched: bool
    declared_size_change: bool
    scale_moved: bool
    metadata: dict = field(default_factory=dict)

    @property
    def fingerprint(self) -> str:
        payload = "␟".join([self.pair_id, self.factor, self.scale_arm, f"{self.runtime!r}",
                            *(arm.compiled.fingerprint() for arm in self.arms)])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# Pair construction
# --------------------------------------------------------------------------

def _slug(value: Any) -> str:
    """Filesystem-safe rendering of a changed value, for use in a pair id."""
    if isinstance(value, (list, tuple, np.ndarray)):
        return "-".join(_slug(item) for item in np.asarray(value).tolist())
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, float):
        return f"{value:g}".replace(".", "p").replace("-", "m")
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in str(value))


def _edge_sets(arm: InterventionArm) -> tuple[set, set]:
    owner = arm.compiled.embedding.membership
    edges = arm.compiled.embedding.hardware_edges.tolist()
    internal = {tuple(e) for e in edges if owner[e[0]] == owner[e[1]]}
    boundary = {tuple(e) for e in edges if owner[e[0]] != owner[e[1]]}
    return internal, boundary


def _raw_coefficients(arm: InterventionArm) -> dict[str, np.ndarray]:
    """Un-scale the compiled coefficients back to what was asked for."""
    scale = arm.compiled.programmed_scale
    return {"h": np.asarray(arm.compiled.physical.h) / scale,
            "problem_J": np.asarray(arm.compiled.problem_J) / scale,
            "chain_J": np.asarray(arm.compiled.chain_J) / scale}


def _audit_single_factor(factor: str, a: InterventionArm, b: InterventionArm) -> dict:
    """Verify the physical graph moved in exactly the way the factor claims."""
    internal_a, boundary_a = _edge_sets(a)
    internal_b, boundary_b = _edge_sets(b)
    same_membership = np.array_equal(a.compiled.embedding.membership, b.compiled.embedding.membership)
    report = {"membership_identical": bool(same_membership),
              "internal_edges_identical": internal_a == internal_b,
              "boundary_edges_identical": boundary_a == boundary_b}
    if factor == "geometry":
        if not same_membership or boundary_a != boundary_b:
            raise ValueError("geometry intervention moved the boundary ports or the chain membership; "
                             "that is more than one factor")
        if internal_a == internal_b:
            raise VacuousIntervention("geometry intervention produced identical intra-chain edges")
    elif factor == "ports":
        if not same_membership or internal_a != internal_b:
            raise ValueError("port intervention reshaped the chains; that is more than one factor")
        if boundary_a == boundary_b:
            raise VacuousIntervention("port intervention produced identical boundary edges")
    elif factor in {"field_allocation", "chain_strength"}:
        if not same_membership or internal_a != internal_b or boundary_a != boundary_b:
            raise ValueError(f"{factor} intervention changed the physical graph; "
                             "coefficient interventions must leave it identical")
        # Compare raw coefficients, not compiled ones: the programmed scale is a
        # legitimate consequence of a chain-strength change and must not be
        # mistaken for the coefficient drift this audit looks for.
        raw_a, raw_b = _raw_coefficients(a), _raw_coefficients(b)
        moved = {name for name in raw_a if not np.allclose(raw_a[name], raw_b[name])}
        expected = {"field_allocation": {"h"}, "chain_strength": {"chain_J"}}[factor]
        unexpected = moved - expected
        if unexpected:
            raise ValueError(
                f"{factor} intervention also moved {sorted(unexpected)} "
                f"(problem coupler / field / penalty drift). A coefficient intervention must change "
                f"only {sorted(expected)}; this usually means two allocations shared one random "
                "stream. Pass independent streams, as build_pairs does.")
        if not moved:
            raise VacuousIntervention(f"{factor} intervention left every raw coefficient identical")
        report["raw_coefficients_moved"] = sorted(moved)
    return report


def build_pairs(problem: IsingProblem, spec: Mapping[str, Any], *, lengths: Sequence[int],
                runtime: float, seed: int = 0, parent_id: str = "parent",
                h_limit: float = 2.0, j_limit: float = 1.0, allow_size_change: bool = False,
                max_qubits: int = 20) -> list[InterventionPair]:
    """Build the (A, B) arms of a single-factor embedding intervention.

    Returns one pair under each arm's own programmed scale, and — whenever that
    scale differs between the arms, always for ``chain_strength`` — a second pair
    compiled at one conservative common scale. Both are returned; the caller must
    report both.
    """
    if not isinstance(spec, Mapping):
        raise ValueError("spec must be a mapping with factor/base/change")
    factor = spec.get("factor")
    if factor not in FACTORS:
        raise ValueError(f"unknown intervention factor {factor!r}; allowed: {sorted(FACTORS)}")
    change = dict(spec.get("change") or {})
    if len(change) != 1:
        raise ValueError(f"an intervention must change exactly one key, got {sorted(change)}")
    expected = FACTORS[factor]
    if expected not in change:
        raise ValueError(f"factor {factor!r} intervenes on {expected!r}, not {sorted(change)[0]!r}")
    base = {"shape": "path", "ports": 1, "field_distribution": "uniform",
            "coupling_distribution": "uniform", "chain_strength": 1.5, **dict(spec.get("base") or {})}
    unknown = set(base) - BASE_KEYS
    if unknown:
        raise ValueError(f"unknown base keys: {sorted(unknown)}")
    if factor in SIZE_CHANGING and not allow_size_change:
        raise ValueError(f"factor {factor!r} changes the physical size; pass allow_size_change=True "
                         "and report it as a separate, declared experiment")
    if not np.isfinite(runtime) or runtime <= 0:
        raise ValueError("runtime must be finite and positive")

    lengths_a = np.asarray(lengths, dtype=int)
    lengths_b = np.asarray(change["chain_lengths"] if factor == "chain_length" else lengths, dtype=int)
    params_a = dict(base)
    params_b = {**base, **({} if factor == "chain_length" else change)}
    if factor != "chain_length" and params_a[expected] == params_b[expected]:
        raise ValueError(f"arm A and arm B must differ in {expected!r}")
    if factor == "chain_length" and np.array_equal(lengths_a, lengths_b):
        raise ValueError("arm A and arm B must differ in chain_lengths")

    def compile_arm(label, lengths_x, params, scale_override=None) -> InterventionArm:
        # Fresh, identically seeded streams per arm: the chain construction and
        # the port choice must not share entropy, or a shape change relocates
        # the ports (see synthetic_lift's port_rng).
        embedding = synthetic_lift(problem, lengths_x, np.random.default_rng(seed + 1),
                                   shape=params["shape"], ports=int(params["ports"]),
                                   port_rng=np.random.default_rng(seed + 2))
        compiled = compile_embedding(problem, embedding, float(params["chain_strength"]),
                                     np.random.default_rng(seed + 3),
                                     field_distribution=params["field_distribution"],
                                     coupling_distribution=params["coupling_distribution"],
                                     h_limit=h_limit, j_limit=j_limit, scale_override=scale_override,
                                     coupling_rng=np.random.default_rng(seed + 4))
        if compiled.physical.n > max_qubits:
            raise ValueError(f"arm {label} has {compiled.physical.n} physical qubits, above the "
                             f"exact-enumeration cap {max_qubits}")
        report = validate_compilation(compiled, max_qubits=max_qubits)
        descriptor = {**params, "chain_lengths": lengths_x.tolist(),
                      "physical_n": int(compiled.physical.n),
                      "programmed_scale": float(compiled.programmed_scale),
                      "aligned_offset": float(compiled.aligned_offset),
                      "scale_override": None if scale_override is None else float(scale_override)}
        return InterventionArm(label, descriptor, compiled, report)

    arm_a = compile_arm("A", lengths_a, params_a)
    arm_b = compile_arm("B", lengths_b, params_b)
    graph_audit = _audit_single_factor(factor, arm_a, arm_b)

    size_matched = arm_a.compiled.physical.n == arm_b.compiled.physical.n
    if factor not in SIZE_CHANGING and not size_matched:
        raise ValueError(f"factor {factor!r} changed the physical size without declaring it")
    scale_moved = not np.isclose(arm_a.compiled.programmed_scale, arm_b.compiled.programmed_scale,
                                 rtol=0, atol=1e-12)

    base_meta = {"spec": {"factor": factor, "base": base, "change": change},
                 "graph_audit": graph_audit, "seed": seed,
                 "h_limit": h_limit, "j_limit": j_limit,
                 "scale_applies_to": "H_Z_only",
                 "causal_scope": "inside the declared closed-system simulator only",
                 "is_commercial_hardware": False}

    # The factor alone is not a unique key: a config may declare two changes to
    # the same factor (star and random_tree geometry, say), and identically named
    # pairs would collide in the sweep plan and on disk.
    pair_id = f"{parent_id}__{factor}_{_slug(change[expected])}__t{runtime:g}"
    built = [InterventionPair(pair_id=pair_id, parent_id=parent_id, factor=factor,
                              scale_arm="total_compiled_effect", runtime=float(runtime),
                              arms=(arm_a, arm_b), held_fixed=HELD_FIXED,
                              physical_size_matched=size_matched,
                              declared_size_change=factor in SIZE_CHANGING,
                              scale_moved=scale_moved, metadata=dict(base_meta))]

    if scale_moved or factor == "chain_strength":
        common = conservative_common_scale(arm_a.compiled, arm_b.compiled)
        controlled = (compile_arm("A", lengths_a, params_a, scale_override=common),
                      compile_arm("B", lengths_b, params_b, scale_override=common))
        _audit_single_factor(factor, *controlled)
        built.append(InterventionPair(
            pair_id=f"{pair_id}__scale_controlled", parent_id=parent_id, factor=factor,
            scale_arm="scale_controlled", runtime=float(runtime), arms=controlled,
            held_fixed=(*HELD_FIXED, "programmed_scale"),
            physical_size_matched=size_matched, declared_size_change=factor in SIZE_CHANGING,
            scale_moved=False,
            metadata={**base_meta, "common_programmed_scale": common,
                      "why": ("the intervened factor also moved the coefficient scale; this arm holds "
                              "the global H_Z scale fixed so the two effects are not conflated")}))
    return built


PLAN_KEYS = frozenset({"seed", "parents", "families", "logical_qubits", "logical_support",
                       "chain_lengths", "runtimes", "splits", "base", "interventions",
                       "search", "max_physical_qubits", "allow_size_change", "h_limit", "j_limit"})


def plan_intervention_pairs(config: Mapping[str, Any] | str | Path, *,
                            allow_test_parents: bool = False) -> tuple[list[InterventionPair], dict]:
    """Generate logical parents and expand them into declared single-factor pairs.

    Parents are split with the same ``parent_splits`` rule the training pipeline
    uses, and only train/validation parents are used by default. A mechanism
    experiment built on test parents is not automatically wrong, but it has to be
    a declared choice, because its pairs share logical objectives with the
    held-out evaluation.
    """
    from .generation import generate_problem, parent_splits, sample_logical_support

    if isinstance(config, (str, Path)):
        config = json.loads(Path(config).read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise ValueError("intervention plan configuration must be a mapping or a path to one")
    unknown = set(config) - PLAN_KEYS
    if unknown:
        raise ValueError(f"unknown intervention plan keys: {sorted(unknown)}")

    seed = int(config.get("seed", 0))
    n_parents = int(config.get("parents", 12))
    families = list(config.get("families") or ["spin_glass"])
    logical_n = int(config.get("logical_qubits", 3))
    lengths = np.asarray(config.get("chain_lengths") or [1] * logical_n, dtype=int)
    runtimes = [float(value) for value in (config.get("runtimes") or [2.0])]
    splits = list(config.get("splits") or ["train", "validation"])
    specs = list(config.get("interventions") or [])
    base = dict(config.get("base") or {})
    cap = int(config.get("max_physical_qubits", 10))
    h_limit, j_limit = float(config.get("h_limit", 2.0)), float(config.get("j_limit", 1.0))
    allow_size_change = bool(config.get("allow_size_change", False))

    if not specs:
        raise ValueError("at least one intervention specification is required")
    if n_parents < 1 or logical_n < 2 or lengths.shape != (logical_n,):
        raise ValueError("parents>=1, logical_qubits>=2 and one chain length per logical qubit required")
    if "test" in splits and not allow_test_parents:
        raise ValueError("intervention pairs on test parents share logical objectives with the held-out "
                         "evaluation; pass allow_test_parents=True to declare that deliberately")
    invalid = set(splits) - {"train", "validation", "test"}
    if invalid:
        raise ValueError(f"unknown splits: {sorted(invalid)}")
    largest = int(max(lengths.sum(), *(np.asarray(spec.get("change", {}).get("chain_lengths", lengths),
                                                  dtype=int).sum() for spec in specs)))
    if largest > cap:
        raise ValueError(f"planned embeddings reach {largest} physical qubits, above the declared "
                         f"cap {cap}; exact enumeration and full-state propagation are exponential")

    assignment = [families[index % len(families)] for index in range(n_parents)]
    split_of = parent_splits(assignment, seed=seed)
    support = sample_logical_support(logical_n, np.random.default_rng(seed),
                                     **(config.get("logical_support") or {"kind": "complete"}))

    pairs: list[InterventionPair] = []
    used: list[dict] = []
    vacuous: list[dict] = []
    for index, family in enumerate(assignment):
        parent_id = f"parent_{index:04d}"
        split = split_of[parent_id]
        if split not in splits:
            continue
        problem = generate_problem(logical_n, family,
                                   np.random.default_rng(seed + 1_000_003 * index), support)
        used.append({"parent_id": parent_id, "family": family, "split": split})
        for runtime in runtimes:
            for spec in specs:
                try:
                    built = build_pairs(problem, {**spec, "base": {**base, **dict(spec.get("base") or {})}},
                                        lengths=lengths, runtime=runtime,
                                        seed=seed + 7 * index, parent_id=parent_id,
                                        h_limit=h_limit, j_limit=j_limit,
                                        allow_size_change=allow_size_change, max_qubits=cap)
                except VacuousIntervention as error:
                    # e.g. a random_tree that happened to reproduce the path on a
                    # short chain. Counted, never silently dropped; a confounded
                    # pair raises a plain ValueError and still aborts the plan.
                    vacuous.append({"parent_id": parent_id, "runtime": runtime,
                                    "factor": spec.get("factor"), "change": spec.get("change"),
                                    "reason": str(error)})
                    continue
                pairs.extend(built)
    if not pairs:
        raise ValueError(f"no parents fell in splits {splits}; nothing to intervene on")

    plan = {"schema_version": 1, "seed": seed, "n_parents": len(used), "parents": used,
            "splits": sorted({entry["split"] for entry in used}),
            "includes_test_parents": any(entry["split"] == "test" for entry in used),
            "runtimes": runtimes, "factors": sorted({pair.factor for pair in pairs}),
            "n_pairs": len(pairs),
            "scale_arm_counts": dict(sorted(Counter(pair.scale_arm for pair in pairs).items())),
            "chain_lengths": lengths.tolist(), "logical_qubits": logical_n,
            "max_physical_qubits": cap, "search": dict(config.get("search") or {}),
            "n_vacuous_skipped": len(vacuous), "vacuous_skipped": vacuous,
            "is_commercial_hardware": False}
    return pairs, plan


# --------------------------------------------------------------------------
# Cross-control matrix
# --------------------------------------------------------------------------

def transfer_penalties(loss_matrix: Mapping[str, float]) -> tuple[float, float]:
    """Cost of importing the other arm's selected control, signed and never clipped.

    A negative value means the imported control beat this arm's own best found,
    which says the equal-budget search on this arm was the weaker of the two. That
    is information about the search, and clipping it to zero would hide a case
    where the intervention story is weaker than it looks.
    """
    return (float(loss_matrix["B_on_A"]) - float(loss_matrix["A_on_A"]),
            float(loss_matrix["A_on_B"]) - float(loss_matrix["B_on_B"]))


def _arm_scorer(arm: InterventionArm, runtime: float, settings: Mapping[str, Any]):
    """A loss function for one arm. Observables are built once and reused."""
    from .benchmarking import score_schedule

    context = (arm.terms, AnnealPath(),
               output_observables(arm.compiled, max_qubits=settings["max_qubits"],
                                  ground_energy=arm.validation["logical_ground_energy"]))
    stub = {"runtime": np.array(float(runtime))}
    calls: list[dict] = []

    def score(schedule: Schedule) -> dict:
        result = score_schedule(stub, schedule, physics_context=context,
                                backend=settings["backend"], tolerance=settings["tolerance"],
                                initial_steps=settings["initial_steps"], max_steps=settings["max_steps"],
                                max_ds_dtau=settings["max_ds_dtau"])
        calls.append(result)
        return result

    return score, calls


def cross_control_matrix(pair: InterventionPair, *, families: Sequence[str] = ("linear", "one_window"),
                         budget: int = 32, seed: int = 0, ambiguity_margin: float = 1.0,
                         backend: str = "numpy", tolerance: float = 5e-4,
                         initial_steps: int = 128, max_steps: int = 8192,
                         max_ds_dtau: float = 4.0, max_qubits: int = 20) -> dict:
    """Search each arm at equal budget, then execute each arm's control on the other.

    A **swap** requires both directions to be decisive: importing B's control into
    A must cost more than A's own numerical ambiguity, and vice versa. If only one
    direction is decisive the pair is ``one_sided`` — informative, but not a clean
    preference reversal. If neither is, it is censored. Ties are therefore handled
    by the same resolution rule that governs headroom, not by an unstable argmin.
    """
    families = tuple(families)
    if not families or len(set(families)) != len(families) or any(f not in CONTROL_FAMILIES for f in families):
        raise ValueError("families must be a nonempty unique subset of CONTROL_FAMILIES")
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        raise ValueError("budget must be a positive integer")
    if not np.isfinite(ambiguity_margin) or ambiguity_margin < 0:
        raise ValueError("ambiguity_margin must be finite and nonnegative")

    settings = {"backend": backend, "tolerance": tolerance, "initial_steps": initial_steps,
                "max_steps": max_steps, "max_ds_dtau": max_ds_dtau, "max_qubits": max_qubits}
    runtime = pair.runtime
    began = perf_counter()
    arm_a, arm_b = pair.arms
    score_a, calls_a = _arm_scorer(arm_a, runtime, settings)
    score_b, calls_b = _arm_scorer(arm_b, runtime, settings)

    def search(score, label):
        best_loss, best = float("inf"), None
        per_family = {}
        for index, name in enumerate(families):
            result = optimize_control_family(lambda s: score(s)["loss"], name, budget=budget,
                                             runtime=runtime, max_slope=max_ds_dtau / runtime,
                                             seed=seed + CONTROL_FAMILIES.index(name) * 1009,
                                             split="train")
            per_family[name] = {"best_loss": result.best.loss,
                                "n_evaluations": result.n_evaluations,
                                "search_seconds": result.elapsed_seconds}
            if result.best.loss < best_loss:
                best_loss, best = result.best.loss, (name, result.best.candidate)
        return {"label": label, "best_found_loss": best_loss, "best_family": best[0],
                "best_candidate_id": best[1].candidate_id,
                "best_parameters": best[1].parameters,
                "best_waveform": best[1].schedule.to_dict(),
                "family_best_loss": {k: v["best_loss"] for k, v in per_family.items()},
                "objective_calls": sum(v["n_evaluations"] for v in per_family.values()),
                "search_seconds": sum(v["search_seconds"] for v in per_family.values())}, best[1].schedule

    found_a, wave_a = search(score_a, "A")
    found_b, wave_b = search(score_b, "B")

    own_a = calls_a[int(np.argmin([c["loss"] for c in calls_a]))]
    own_b = calls_b[int(np.argmin([c["loss"] for c in calls_b]))]
    cross_a = score_a(wave_b)     # B's selected control executed on arm A
    cross_b = score_b(wave_a)     # A's selected control executed on arm B

    matrix = {"A_on_A": float(found_a["best_found_loss"]), "B_on_A": float(cross_a["loss"]),
              "A_on_B": float(cross_b["loss"]), "B_on_B": float(found_b["best_found_loss"])}
    penalty_a, penalty_b = transfer_penalties(matrix)
    ambiguity_a = float(own_a["loss_ambiguity_indicator"]) + float(cross_a["loss_ambiguity_indicator"])
    ambiguity_b = float(own_b["loss_ambiguity_indicator"]) + float(cross_b["loss_ambiguity_indicator"])
    decisive_a = bool(penalty_a > ambiguity_margin * ambiguity_a)
    decisive_b = bool(penalty_b > ambiguity_margin * ambiguity_b)
    identical = bool(np.array_equal(wave_a.tau_knots, wave_b.tau_knots)
                     and np.array_equal(wave_a.s_knots, wave_b.s_knots))
    resolution = ("resolved" if decisive_a and decisive_b else
                  "one_sided" if decisive_a or decisive_b else "censored_numerical")

    return _safe({
        "schema_version": 1, "pair_id": pair.pair_id, "parent_id": pair.parent_id,
        "factor": pair.factor, "scale_arm": pair.scale_arm, "runtime": runtime,
        "families": list(families), "budget_per_tunable_family": budget, "seed": seed,
        "arm_A": {**found_a, "descriptor": arm_a.descriptor},
        "arm_B": {**found_b, "descriptor": arm_b.descriptor},
        "loss_matrix": matrix,
        "transfer_penalty_on_A": float(penalty_a), "transfer_penalty_on_B": float(penalty_b),
        "mean_transfer_penalty": float((penalty_a + penalty_b) / 2),
        "combined_ambiguity_on_A": ambiguity_a, "combined_ambiguity_on_B": ambiguity_b,
        "decisive_on_A": decisive_a, "decisive_on_B": decisive_b,
        "ambiguity_margin": float(ambiguity_margin),
        "resolution_status": resolution,
        "selected_waveform_identical": identical,
        "preferred_control_swapped": bool(resolution == "resolved" and not identical),
        "physical_size_matched": pair.physical_size_matched,
        "declared_size_change": pair.declared_size_change,
        "scale_moved": pair.scale_moved,
        "held_fixed": list(pair.held_fixed),
        "graph_audit": pair.metadata.get("graph_audit"),
        "objective_calls": int(found_a["objective_calls"] + found_b["objective_calls"] + 2),
        "cross_executions": 2,
        "wall_seconds": perf_counter() - began,
        "reference_status": "best_found_within_evaluated_candidates",
        "causal_scope": "inside the declared closed-system simulator only",
        "penalties_are_signed_and_unclipped": True,
    })


# --------------------------------------------------------------------------
# Sweep and aggregation
# --------------------------------------------------------------------------

def sweep_interventions(pairs: Sequence[InterventionPair], *, output: str | Path,
                        families: Sequence[str] = ("linear", "one_window", "two_window", "eight_bin"),
                        budget: int = 32, seed: int = 0, ambiguity_margin: float = 1.0,
                        backend: str = "numpy", tolerance: float = 5e-4,
                        initial_steps: int = 128, max_steps: int = 8192, max_ds_dtau: float = 4.0,
                        max_qubits: int = 20, resume: bool = False, dry_run: bool = False,
                        on_error: str = "raise", shard: int = 0, shard_count: int = 1) -> dict:
    """Run ``cross_control_matrix`` over constructed pairs through the sweep driver."""
    from .experiments import source_hash

    pairs = list(pairs)
    if not pairs:
        raise ValueError("sweep_interventions requires a nonempty pair list")
    settings = {"families": list(families), "budget": budget, "seed": seed,
                "ambiguity_margin": ambiguity_margin, "backend": backend, "tolerance": tolerance,
                "initial_steps": initial_steps, "max_steps": max_steps,
                "max_ds_dtau": max_ds_dtau, "max_qubits": max_qubits}
    by_id = {pair.pair_id: pair for pair in pairs}
    if len(by_id) != len(pairs):
        raise ValueError("pair ids must be unique within a sweep")
    units = [SweepUnit(unit_id=pair.pair_id, fingerprint=pair.fingerprint, payload=pair.pair_id)
             for pair in pairs]
    if shard_count > 1:
        from .sweeps import select_shard
        units = select_shard(units, shard, shard_count)

    if dry_run:
        per_unit = 2 * sum(1 if name == "linear" else budget for name in families) + 2
        return {"dry_run": True, "planned": len(pairs),
                "requested_objective_calls_per_unit": per_unit,
                "requested_objective_calls_total": per_unit * len(pairs),
                "settings": settings,
                "note": "requested workload only; no control was evaluated and no output written"}

    def worker(unit: SweepUnit) -> dict:
        return cross_control_matrix(by_id[unit.payload], **settings)

    return run_sweep(units, worker, output=output, settings=settings, command="intervention-sweep",
                     source_hash=source_hash(), resume=resume, on_error=on_error,
                     extra_manifest={"n_pairs": len(pairs),
                                     "factors": sorted({pair.factor for pair in pairs}),
                                     "scale_arms": sorted({pair.scale_arm for pair in pairs}),
                                     "scope": "paired interventions inside the declared closed-system simulator"})


def _penalty_block(rows: Sequence[Mapping[str, Any]], *, bootstrap_resamples: int, seed: int) -> dict:
    """Penalty statistics over pairs decisive in at least one direction.

    ``n_included_pairs`` is deliberately a different population from the swap
    denominator: a swap needs BOTH directions to be decisive, while a penalty is
    a measured quantity as soon as one direction is. Naming them the same would
    invite a reader to divide one by the other.
    """
    included = [row for row in rows if row.get("resolution_status") != "censored_numerical"]
    values, _ = _parent_means(included, lambda row: row.get("mean_transfer_penalty"))
    block = _describe(values)
    block["parent_bootstrap_ci"] = _bootstrap(values, n_resamples=bootstrap_resamples, seed=seed)
    block["n_pairs"] = len(rows)
    block["n_included_pairs"] = len(included)
    block["n_resolved_pairs"] = sum(1 for row in rows if row.get("resolution_status") == "resolved")
    block["n_one_sided_pairs"] = sum(1 for row in rows if row.get("resolution_status") == "one_sided")
    block["inclusion_rule"] = "decisive in at least one direction; censored pairs excluded"
    return block


def _swap_stats(rows: Sequence[Mapping[str, Any]]) -> tuple[float | None, int]:
    eligible = [row for row in rows if row.get("resolution_status") == "resolved"]
    if not eligible:
        return None, 0
    swaps = sum(1 for row in eligible if row.get("preferred_control_swapped"))
    return swaps / len(eligible), len(eligible)


def aggregate_interventions(rows: Sequence[Mapping[str, Any]], *, bootstrap_resamples: int = 2000,
                            seed: int = 0) -> dict:
    """Parent-level intervention summary, split by scale arm and by factor.

    Scale arms are never pooled: ``total_compiled_effect`` and ``scale_controlled``
    answer different questions, and a single averaged penalty across them would
    describe neither.
    """
    rows = list(rows)
    if not rows:
        raise ValueError("aggregate_interventions requires a nonempty row set")
    sizes = {bool(row.get("physical_size_matched")) for row in rows}
    if len(sizes) != 1:
        raise ValueError("refusing to pool size-matched and size-changed pairs; a declared physical "
                         "size change is a separate experiment")
    statuses = {str(row.get("resolution_status")) for row in rows}
    allowed = {"resolved", "one_sided", "censored_numerical"}
    if not statuses <= allowed:
        raise ValueError(f"unknown resolution_status values: {sorted(statuses - allowed)}")

    censored = [row for row in rows if row["resolution_status"] == "censored_numerical"]
    swap_rate, denominator = _swap_stats(rows)
    arms = sorted({str(row.get("scale_arm")) for row in rows})
    factors = sorted({str(row.get("factor")) for row in rows})

    summary = {
        "schema_version": 1,
        "n_pairs": len(rows),
        "n_parents": len({str(row["parent_id"]) for row in rows}),
        "n_censored_pairs": len(censored),
        "censored_fraction": len(censored) / len(rows),
        "n_one_sided_pairs": sum(1 for row in rows if row["resolution_status"] == "one_sided"),
        "swap_rate": swap_rate, "swap_rate_denominator": denominator,
        "identical_waveform_pairs": sum(1 for row in rows if row.get("selected_waveform_identical")),
        "physical_size_matched": sizes.pop(),
        "by_scale_arm": {arm: {"transfer_penalty": _penalty_block(
            [row for row in rows if row.get("scale_arm") == arm],
            bootstrap_resamples=bootstrap_resamples, seed=seed),
            "swap_rate": _swap_stats([row for row in rows if row.get("scale_arm") == arm])[0]}
            for arm in arms},
        "by_factor": {factor: {"transfer_penalty": _penalty_block(
            [row for row in rows if row.get("factor") == factor],
            bootstrap_resamples=bootstrap_resamples, seed=seed),
            "swap_rate": _swap_stats([row for row in rows if row.get("factor") == factor])[0]}
            for factor in factors},
        "factor_counts": dict(sorted(Counter(str(row.get("factor")) for row in rows).items())),
        "total_objective_calls": int(sum(int(row.get("objective_calls", 0)) for row in rows)),
        "verdict": "no_resolved_preference_change" if swap_rate is None else
                   ("preference_changes_measured" if swap_rate > 0 else "no_preference_change_at_resolution"),
        "scope": ("paired interventions inside the declared closed-system simulator; transfer penalties "
                  "are signed and unclipped, and scale arms are reported separately"),
    }
    return _safe(summary)


def intervention_report(sweep_dir: str | Path, *, output: str | Path | None = None,
                        bootstrap_resamples: int = 2000, seed: int = 0) -> dict:
    """Aggregate a completed ``intervention-sweep`` and write JSON plus Markdown."""
    from .pipeline import write_json
    from .sweeps import load_rows

    roots = [Path(part) for part in ([sweep_dir] if isinstance(sweep_dir, (str, Path)) else sweep_dir)]
    root = roots[0]
    rows = [row for part in roots for row in load_rows(part)]
    hashes = {row.get("settings_hash") for row in rows}
    if len(hashes) > 1:
        raise ValueError(f"refusing to aggregate shards computed under different settings: {sorted(hashes)}")
    successful = [row["result"] for row in rows if row.get("status") == "ok"]
    recovered = {row["unit_id"] for row in rows if row.get("status") == "ok"}
    outstanding = [row for row in rows if row.get("status") == "failed" and row["unit_id"] not in recovered]
    if not successful:
        raise ValueError(f"{root} contains no successful rows to aggregate")

    summary = aggregate_interventions(successful, bootstrap_resamples=bootstrap_resamples, seed=seed)
    summary["failed_units"] = len(outstanding)
    summary["failed_unit_ids"] = sorted({row["unit_id"] for row in outstanding})
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        summary["sweep"] = {key: manifest.get(key) for key in
                            ("command", "settings", "settings_hash", "source_hash", "status")}
    destination = Path(output) if output is not None else root / "report"
    write_json(destination / "summary.json", summary)
    (destination / "INTERVENTIONS.md").write_text(_intervention_markdown(summary), encoding="utf-8")
    return summary


def _cell(value, digits=6):
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def _rate(value) -> str:
    return "n/a" if value is None else f"{float(value):.1%}"


def _intervention_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Paired embedding interventions (G3)",
        "",
        "Software output of `annealctrl intervention-sweep`. Every effect below is measured "
        "**inside the declared closed-system simulator**; none of it is evidence about "
        "unmeasured hardware. Transfer penalties are signed and unclipped.",
        "",
        f"- Pairs: {summary['n_pairs']} over {summary['n_parents']} independent logical parents",
        f"- Physical size matched: {summary['physical_size_matched']}",
        f"- Censored by numerical resolution: {summary['n_censored_pairs']} "
        f"({summary['censored_fraction']:.1%})",
        f"- One-sided (decisive in a single direction): {summary['n_one_sided_pairs']}",
        f"- Identical selected waveform on both arms: {summary['identical_waveform_pairs']}",
        f"- Preferred-control swap rate: {_rate(summary['swap_rate'])} "
        f"over {summary['swap_rate_denominator']} resolved pairs",
        f"- Failed units excluded: {summary.get('failed_units', 0)}",
        f"- Verdict: **{summary['verdict']}**",
        "",
        "## Transfer penalty by scale arm",
        "",
        "| scale arm | pairs | incl. | swaps / resolved | mean | p50 | p90 | bootstrap CI |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for arm, block in summary["by_scale_arm"].items():
        penalty = block["transfer_penalty"]
        ci = penalty["parent_bootstrap_ci"]
        rate = block["swap_rate"]
        swaps = "n/a" if rate is None else f"{round(rate * penalty['n_resolved_pairs'])}/{penalty['n_resolved_pairs']}"
        lines.append(
            f"| `{arm}` | {penalty['n_pairs']} | {penalty['n_included_pairs']} | {swaps} | "
            f"{_cell(penalty['mean'])} | {_cell(penalty['quantiles']['p50'])} | "
            f"{_cell(penalty['quantiles']['p90'])} | "
            f"[{_cell(ci['low'], 4)}, {_cell(ci['high'], 4)}] |")
    lines += ["", "## Transfer penalty by factor", "",
              "| factor | pairs | incl. | swaps / resolved | mean | p90 |",
              "|---|---:|---:|---:|---:|---:|"]
    for factor, block in summary["by_factor"].items():
        penalty = block["transfer_penalty"]
        rate = block["swap_rate"]
        swaps = "n/a" if rate is None else f"{round(rate * penalty['n_resolved_pairs'])}/{penalty['n_resolved_pairs']}"
        lines.append(f"| `{factor}` | {penalty['n_pairs']} | {penalty['n_included_pairs']} | {swaps} | "
                     f"{_cell(penalty['mean'])} | {_cell(penalty['quantiles']['p90'])} |")
    lines += [
        "",
        "## Reading this table",
        "",
        "- A **swap** requires both directions to be decisive: importing B's control into A "
        "must cost more than A's own numerical ambiguity, and vice versa. A pair decisive in "
        "only one direction is reported as `one_sided`, not as a preference reversal.",
        "- `total_compiled_effect` uses each arm's own programmed scale, as a device would. "
        "`scale_controlled` compiles both arms at one conservative common scale so the "
        "intervened factor moves and the global H_Z scale does not. They answer different "
        "questions and are never pooled.",
        "- **pairs** is every pair; **incl.** is those decisive in at least one direction, which is the population behind the penalty statistics; **swaps / resolved** counts pairs decisive in BOTH directions. The three columns are different populations and the swap fraction must be read against its own denominator, not against `incl.`.",
        "- A zero swap rate with a well-resolved population is a result: it says the "
        "intervention does not change which control is preferred on this distribution.",
        "",
        f"Total objective calls charged: {summary['total_objective_calls']}.",
        "",
    ]
    return "\n".join(lines)
