"""Does the control ranking survive decoherence, or is it a closed-system artefact?

Every loss elsewhere in this project comes from unitary evolution of a pure
state. The first question a reviewer asks about that is whether the ordering of
controls is an artefact of simulating no environment at all: a schedule that
lingers near a small gap buys adiabaticity in a closed system and buys the bath
more time in an open one, so the two effects pull in opposite directions and the
closed-system answer is not obviously the robust one.

This module asks the question with the independent Lindblad solver already in
``adapters``, on the same records, with the same waveforms and the same decoder,
and reports the answer whichever way it comes out.

Two limits are structural and neither is worked around. A density matrix costs
O(4**N), so this runs on the smallest records and not on the campaign; and the
noise model is local dephasing at a declared rate, which is **not** a calibrated
device model -- no thermal bath, no per-qubit calibration, no working-graph
exclusions, no measured T1 or T2. A result here says whether the ordering is
fragile under a simple, explicitly chosen perturbation. It says nothing about
what a real annealer would do.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .telemetry import _safe


def open_loss(record: Mapping[str, Any], schedule, *, dephasing_rate: float = 0.0,
              relaxation_rate: float = 0.0, max_qubits: int = 6, runtime: float | None = None,
              **solver) -> float:
    """1 − logical success under local dephasing, decoded exactly as elsewhere.

    The decoder is ``output_observables``' ``success`` indicator over computational
    basis states, so this is the same quantity the closed-system scorer reports
    and the two are directly comparable at rate zero.
    """
    from .adapters import simulate_lindblad
    from .benchmarking import record_physics

    if not np.isfinite(dephasing_rate) or dephasing_rate < 0:
        raise ValueError("dephasing_rate must be finite and nonnegative")
    if not np.isfinite(relaxation_rate) or relaxation_rate < 0:
        raise ValueError("relaxation_rate must be finite and nonnegative")
    n = len(np.asarray(record["physical_h"]))
    if n > max_qubits:
        raise ValueError(f"density matrix for {n} qubits needs {4 ** n} entries, above the "
                         f"{max_qubits}-qubit cap; raise max_qubits deliberately or use a "
                         "smaller record")

    terms, path, observables = record_physics(record)
    duration = float(record["runtime"]) if runtime is None else float(runtime)
    rates = {}
    if dephasing_rate:
        rates["dephasing_rates"] = float(dephasing_rate)
    if relaxation_rate:
        rates["relaxation_rates"] = float(relaxation_rate)
    result = simulate_lindblad(terms, schedule, duration, path=path, max_qubits=max_qubits, **rates, **solver)
    probabilities = np.asarray(result.probabilities, dtype=float)
    return float(1.0 - probabilities @ np.asarray(observables["success"], dtype=float))


def robustness_sweep(records: Sequence[Mapping[str, Any]], waveforms: Mapping[str, Any], *,
                     rates: Sequence[float], max_qubits: int = 6,
                     relaxation_rate: float = 0.0, **solver) -> dict:
    """Every waveform on every record at every rate, with the ordering checked.

    ``rates`` must include zero. Without the noiseless point there is no
    reference to measure degradation against, and a table of losses at three
    nonzero rates cannot say whether anything degraded at all.
    """
    records, waveforms = list(records), dict(waveforms)
    if len(waveforms) < 2:
        raise ValueError("robustness_sweep needs at least two waveforms to order")
    rates = [float(rate) for rate in rates]
    if not any(rate == 0.0 for rate in rates):
        raise ValueError("rates must include zero: degradation is measured against the "
                         "noiseless point")

    by_rate: dict[str, dict] = {}
    for rate in rates:
        losses = {name: [] for name in waveforms}
        for record in records:
            for name, schedule in waveforms.items():
                losses[name].append(open_loss(record, schedule, dephasing_rate=rate,
                                              relaxation_rate=relaxation_rate,
                                              max_qubits=max_qubits, **solver))
        by_rate[str(rate)] = {
            "mean_loss": {name: float(np.mean(values)) for name, values in losses.items()},
            "per_record": {name: [float(v) for v in values] for name, values in losses.items()},
            "n_records": len(records)}

    best = {rate: min(block["mean_loss"], key=block["mean_loss"].get)
            for rate, block in by_rate.items()}
    reference = best[str(0.0)]
    order_at_zero = sorted(by_rate[str(0.0)]["mean_loss"],
                           key=by_rate[str(0.0)]["mean_loss"].get)
    preserved = all(sorted(block["mean_loss"], key=block["mean_loss"].get) == order_at_zero
                    for block in by_rate.values())

    return _safe({
        "schema_version": 1,
        "n_records": len(records),
        "rates": rates,
        "waveforms": sorted(waveforms),
        "by_rate": by_rate,
        "best_at_each_rate": best,
        "best_is_unchanged": all(name == reference for name in best.values()),
        "ordering_preserved": bool(preserved),
        "ordering_at_zero": order_at_zero,
        "relaxation_rate": relaxation_rate,
        "scope": ("local dephasing at a declared rate, not a calibrated device model: no thermal "
                  "bath, no measured T1 or T2, no per-qubit calibration, no working-graph "
                  "exclusions. Says whether the ordering is fragile under an explicitly chosen "
                  "perturbation, not what a real annealer would do."),
    })


def _linear_index(bank: np.ndarray) -> int:
    """The bank entry closest to the identity ramp, by sup norm."""
    ramp = np.linspace(0.0, 1.0, bank.shape[1])
    return int(np.argmin(np.abs(bank - ramp).max(axis=1)))


def bank_loss_table(records: Sequence[Mapping[str, Any]], *, rates: Sequence[float],
                    max_qubits: int = 6, relaxation_rate: float = 0.0, **solver) -> dict:
    """Every bank candidate of every record at every rate, solved once.

    The density-matrix solves are the entire cost of an open-system study and
    they do not depend on who is selecting: a second training seed picks a
    different index out of the same table. Computing the table once and scoring
    selectors against it keeps a multi-seed result the same price as one seed,
    and guarantees every seed is compared on identical numbers rather than on
    two runs of a stochastic solver.
    """
    from .schedules import Schedule

    records = list(records)
    if not records:
        raise ValueError("bank_loss_table requires a nonempty record set")
    rates = [float(rate) for rate in rates]
    if not any(rate == 0.0 for rate in rates):
        raise ValueError("rates must include zero: degradation is measured against the "
                         "noiseless point")

    table: dict[str, dict[str, list[float]]] = {}
    for record in records:
        if "candidate_schedules" not in record:
            raise ValueError(f"record {record.get('record_id')!r} carries no "
                             "candidate_schedules to select from")
        bank = np.asarray(record["candidate_schedules"], dtype=float)
        if bank.ndim != 2 or not bank.size:
            raise ValueError("candidate_schedules must be a nonempty 2-D bank")
        grid = np.linspace(0.0, 1.0, bank.shape[1])
        key = str(record["record_id"])
        table[key] = {str(rate): [
            open_loss(record, Schedule(grid, wave), dephasing_rate=rate,
                      relaxation_rate=relaxation_rate, max_qubits=max_qubits, **solver)
            for wave in bank] for rate in rates}
    return {"schema_version": 1, "rates": rates, "relaxation_rate": relaxation_rate,
            "max_qubits": max_qubits, "n_records": len(records), "losses": table}


def selection_robustness(records: Sequence[Mapping[str, Any]], *, select,
                         rates: Sequence[float], oracle_selects: bool = False,
                         table: Mapping[str, Any] | None = None,
                         max_qubits: int = 6, relaxation_rate: float = 0.0,
                         bootstrap_resamples: int = 2000, seed: int = 0, **solver) -> dict:
    """Does a rule trained without an environment still choose well once there is one?

    ``robustness_sweep`` asks whether a *fixed* waveform's ranking survives
    dephasing. That is not the question a learned method has to answer. A
    learned critic was fit to closed-system labels, so the environment can hurt
    it twice: the control it likes may degrade, and the control it *should* have
    liked may change. Only the second is a failure of the learning.

    ``select(record)`` returns the bank index the rule picks. It is called once,
    before any noise is simulated, and never sees a rate -- that asymmetry is
    deliberate, because it is exactly the deployment condition. Set
    ``oracle_selects`` to let the choice see the noise instead; that upper bound
    is what selection regret is measured against.

    Two quantities are reported per rate, both at parent level:

        advantage_vs_linear = selected loss − linear-candidate loss
        selection_regret    = selected loss − best bank loss *at that rate*

    The first says whether the method still pays off. The second isolates how
    much of any shortfall is the closed-system critic choosing wrongly, as
    opposed to every control simply degrading together.
    """
    from .headroom import _bootstrap, _describe, _parent_means
    from .schedules import Schedule

    records = list(records)
    if not records:
        raise ValueError("selection_robustness requires a nonempty record set")
    rates = [float(rate) for rate in rates]
    if not any(rate == 0.0 for rate in rates):
        raise ValueError("rates must include zero: degradation is measured against the "
                         "noiseless point")

    banks, chosen, linear_at = [], [], []
    for record in records:
        if "candidate_schedules" not in record:
            raise ValueError(f"record {record.get('record_id')!r} carries no "
                             "candidate_schedules to select from")
        bank = np.asarray(record["candidate_schedules"], dtype=float)
        if bank.ndim != 2 or not bank.size:
            raise ValueError("candidate_schedules must be a nonempty 2-D bank")
        pick = select(record)
        if not oracle_selects:
            if isinstance(pick, bool) or not isinstance(pick, (int, np.integer)) \
                    or not 0 <= int(pick) < bank.shape[0]:
                raise ValueError(f"select returned {pick!r}, not an index into the "
                                 f"{bank.shape[0]}-entry bank")
        banks.append(bank)
        chosen.append(None if oracle_selects else int(pick))
        linear_at.append(_linear_index(bank))

    by_rate, rows = {}, []
    for rate in rates:
        per_record = []
        for record, bank, pick, linear in zip(records, banks, chosen, linear_at):
            if table is None:
                grid = np.linspace(0.0, 1.0, bank.shape[1])
                losses = np.array([
                    open_loss(record, Schedule(grid, wave), dephasing_rate=rate,
                              relaxation_rate=relaxation_rate, max_qubits=max_qubits, **solver)
                    for wave in bank])
            else:
                losses = np.asarray(
                    table["losses"][str(record["record_id"])][str(rate)], dtype=float)
                if losses.shape[0] != bank.shape[0]:
                    raise ValueError(
                        f"table holds {losses.shape[0]} losses for record "
                        f"{record.get('record_id')!r} but its bank has {bank.shape[0]}")
            index = int(losses.argmin()) if pick is None else pick
            per_record.append({
                "record_id": str(record.get("record_id")),
                "parent_id": str(record["parent_id"]),
                "dephasing_rate": rate,
                "selected_index": index, "linear_index": linear,
                "best_index_at_rate": int(losses.argmin()),
                "selected_loss": float(losses[index]),
                "linear_loss": float(losses[linear]),
                "best_loss_at_rate": float(losses.min()),
                "advantage_vs_linear": float(losses[index] - losses[linear]),
                "selection_regret": float(losses[index] - losses.min()),
                "selected_is_best_at_rate": bool(index == int(losses.argmin()))})
        rows.extend(per_record)

        block: dict[str, Any] = {"n_records": len(per_record)}
        for name in ("advantage_vs_linear", "selection_regret", "selected_loss", "linear_loss"):
            values, parents = _parent_means(per_record, lambda row, name=name: row[name])
            described = _describe(values)
            described["parent_bootstrap_ci"] = _bootstrap(
                values, n_resamples=bootstrap_resamples, seed=seed)
            block[name] = described
            block["n_parents"] = len(parents)
        block["selected_is_best_fraction"] = float(
            np.mean([row["selected_is_best_at_rate"] for row in per_record]))
        by_rate[str(rate)] = block

    zero = by_rate["0.0"]
    return _safe({
        "schema_version": 1,
        "n_records": len(records),
        "n_parents": len({str(r["parent_id"]) for r in records}),
        "rates": rates,
        "selection_saw_the_noise": bool(oracle_selects),
        "by_rate": by_rate,
        "rows": rows,
        # A single number for the abstract: how much of the noiseless advantage
        # is left at the largest rate tested.
        "advantage_retained_at_max_rate": (
            None if not zero["advantage_vs_linear"]["mean"]
            else float(by_rate[str(max(rates))]["advantage_vs_linear"]["mean"]
                       / zero["advantage_vs_linear"]["mean"])),
        "relaxation_rate": relaxation_rate,
        "reused_precomputed_table": table is not None,
        "scope": ("the selection is made once from closed-system features and never sees a "
                  "rate, which is the deployment condition. Local dephasing at a declared "
                  "rate is not a calibrated device model: no thermal bath, no measured T1 or "
                  "T2, no per-qubit calibration, no working-graph exclusions."),
    })


def checkpoint_selector(checkpoint, *, device: str = "cpu"):
    """The bank index a trained critic picks, for use as ``selection_robustness``' rule.

    Loaded once and reused across records, so the cost of this sweep stays in the
    density-matrix solves where it belongs.
    """
    import torch

    from .learning import load_checkpoint
    from .models import graph_from_record

    model, normalizer = load_checkpoint(checkpoint, device=device)
    model.eval()

    def select(record: Mapping[str, Any]) -> int:
        graph = normalizer.transform(graph_from_record(record, device=device))
        bank = torch.as_tensor(np.asarray(record["candidate_schedules"], dtype=float),
                               dtype=torch.float32, device=device)
        with torch.no_grad():
            return int(model.predict_losses(graph, bank).argmin())

    return select
