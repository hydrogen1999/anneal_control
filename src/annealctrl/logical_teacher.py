"""Schedule construction from the LOGICAL spectrum, executed on the embedded system.

The design document names Tx-NQDT (Lu et al. 2026) as "the strongest overlap in
spectrum-guided hardware scheduling" and records the one structural difference
that matters here: *"The reported spectrum is logical, rather than the complete
physical embedded system."*

Its construction -- transition matrix elements, a normalised time density, and
inverse cumulative reconstruction -- is already what ``physics_baselines``
builds. So the faithful comparison is not a second implementation of that
machinery; it is the same machinery fed the **logical** Hamiltonian and then
asked to control the **physical embedded** one, which is the task.

That also makes this the sharpest available test of the project's central
thesis. If a schedule derived from the logical spectrum controls the embedded
system as well as one derived from its true physical spectrum, the embedding
carries no control-relevant information and the thesis fails. If it does not,
the gap is measured in the units the rest of the evidence uses.

This is an information-path comparison, deliberately labelled *inspired by*
rather than presented as the published method's result: no neural quantum
state, no variational reconstruction, and the logical spectrum here is exact
rather than approximated.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def logical_terms(record: Mapping[str, Any], *, scale: str = "programmed"):
    """The logical Ising problem as a Hamiltonian, with no chains and no embedding.

    ``scale`` decides which of two different questions this asks, and the
    default is not the obvious one. The raw logical coefficients differ from
    the physical ones by **both** the embedding and a common rescaling by
    ``programmed_scale``, and the document lists re-embedding and common
    rescaling as *separate* falsification axes. Feeding raw coefficients
    therefore confounds them: on this dataset every one-to-one record with
    ``programmed_scale == 1`` reproduces the physical schedule exactly, and
    every one with a smaller scale does not, by up to 0.034 in loss.

    ``"programmed"`` applies ``programmed_scale`` so that a one-to-one
    embedding yields the identical Hamiltonian and isolates the embedding.
    ``"raw"`` keeps the unscaled problem, which is closer to what a solver of
    the stated logical problem would see, and is reported separately.
    """
    from .physics import HamiltonianTerms

    if scale not in {"programmed", "raw"}:
        raise ValueError("scale must be 'programmed' or 'raw'")
    factor = 1.0
    if scale == "programmed":
        factor = float(np.asarray(record["programmed_scale"]).item())
        if not np.isfinite(factor) or factor <= 0:
            raise ValueError("programmed_scale must be finite and positive")
    h = np.asarray(record["logical_h"], dtype=float) * factor
    edges = np.asarray(record["logical_edges"], dtype=int).reshape(-1, 2)
    weights = np.asarray(record["logical_J"], dtype=float) * factor
    if edges.shape[0] != weights.shape[0]:
        raise ValueError("logical_edges and logical_J must agree in length")
    return HamiltonianTerms(int(h.size), h, edges, weights)


def logical_spectrum_schedule(record: Mapping[str, Any], *, method: str = "d2",
                              runtime: float | None = None, max_slope: float = 4.0,
                              max_qubits: int = 10, scale: str = "programmed", **kwargs):
    """Build the teacher schedule from the logical spectrum alone.

    The anneal path keeps the record's own catalyst and energy scale: changing
    them as well would confound the spectrum's provenance with a different
    physical path.
    """
    from .benchmarking import _scalar
    from .physics import AnnealPath
    from .physics_baselines import exact_teacher_baseline

    duration = float(_scalar(record, "runtime")) if runtime is None else float(runtime)
    path = AnnealPath(catalyst_strength=float(_scalar(record, "catalyst_strength", 0.0)),
                      energy_scale=float(_scalar(record, "energy_scale", 1.0)))
    return exact_teacher_baseline(logical_terms(record, scale=scale), method, runtime=duration,
                                  max_slope=max_slope, path=path, max_qubits=max_qubits,
                                  **kwargs)


def compare_spectra(record: Mapping[str, Any], *, method: str = "d2", max_slope: float = 4.0,
                    max_qubits: int = 10, tolerance: float = 5e-4, max_steps: int = 8192,
                    scale: str = "programmed", **kwargs) -> dict:
    """Score logical- and physical-spectrum teachers on the same embedded task.

    Both schedules are evaluated by the same simulator on the same physical
    record, so the only thing that differs is which spectrum built them. A
    schedule that the teacher could not resolve is reported as ``None`` rather
    than replaced by a fallback.
    """
    from .benchmarking import record_physics, score_schedule
    from .physics_baselines import exact_teacher_baseline
    from .telemetry import _safe
    from .schedules import Schedule

    terms, path, _ = record_physics(record)
    duration = float(np.asarray(record["runtime"]).item())
    physical = exact_teacher_baseline(terms, method, runtime=duration, max_slope=max_slope,
                                      path=path, max_qubits=max_qubits, **kwargs)
    logical = logical_spectrum_schedule(record, method=method, runtime=duration,
                                        max_slope=max_slope, max_qubits=max_qubits,
                                        scale=scale, **kwargs)

    def score(result):
        if result.schedule is None:
            return None
        return float(score_schedule(record, result.schedule, tolerance=tolerance,
                                    max_steps=max_steps, max_ds_dtau=max_slope * (1 + 1e-6)
                                    )["loss"])

    linear = float(score_schedule(record, Schedule.linear(), tolerance=tolerance,
                                  max_steps=max_steps, max_ds_dtau=max_slope * (1 + 1e-6))["loss"])
    physical_loss, logical_loss = score(physical), score(logical)
    return _safe({
        "record_id": str(np.asarray(record["record_id"]).item()),
        "parent_id": str(np.asarray(record["parent_id"]).item()),
        "n_logical": int(np.asarray(record["logical_h"]).size),
        "n_physical": int(np.asarray(record["physical_h"]).size),
        "method": method, "logical_scale": scale,
        "one_to_one_embedding": int(np.asarray(record["logical_h"]).size)
                                == int(np.asarray(record["physical_h"]).size),
        "physical_spectrum_loss": physical_loss,
        "logical_spectrum_loss": logical_loss,
        "linear_loss": linear,
        "logical_minus_physical": (None if physical_loss is None or logical_loss is None
                                   else logical_loss - physical_loss),
        "physical_status": physical.status, "logical_status": logical.status,
        "scope": ("both schedules control the same embedded physical record; only the "
                  "spectrum that built them differs. Inspired by the logical-spectrum "
                  "information path of Tx-NQDT, not a reimplementation of it."),
    })
