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
              relaxation_rate: float = 0.0, max_qubits: int = 8, runtime: float | None = None,
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
    result = simulate_lindblad(terms, schedule, duration, path=path, **rates, **solver)
    probabilities = np.asarray(result.probabilities, dtype=float)
    return float(1.0 - probabilities @ np.asarray(observables["success"], dtype=float))


def robustness_sweep(records: Sequence[Mapping[str, Any]], waveforms: Mapping[str, Any], *,
                     rates: Sequence[float], max_qubits: int = 8,
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
