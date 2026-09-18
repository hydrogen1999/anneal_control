"""Frozen learned decisions under small, explicitly phenomenological noise.

This is a perturbation experiment, not thermal annealing or device validation.
The policy is never adapted to noisy target labels. One checkpoint is assessed
per study; repeat identical target records for each predeclared training seed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .experiments import content_hash, output_lock, source_hash
from .pipeline import load_records, write_json


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _summarize(rows, bootstrap_resamples):
    from .headroom import _bootstrap
    values = {(r["record_id"], r["method"], tuple(r["rates"])): r["loss"] for r in rows}
    result = []
    for method, rates in sorted({(r["method"], tuple(r["rates"])) for r in rows}):
        parents = {}
        for row in rows:
            if row["method"] != method or tuple(row["rates"]) != rates:
                continue
            rid = row["record_id"]
            parents.setdefault(row["logical_fingerprint"], []).append([
                row["loss"], row["loss"] - values[rid, "linear", rates],
                row["loss"] - values[rid, method, (0., 0.)]])
        means = np.asarray([np.mean(v, axis=0) for _, v in sorted(parents.items())])
        result.append({"method": method, "rates": list(rates), "parents": len(parents),
                       "mean_loss": float(means[:, 0].mean()),
                       "vs_linear": {"mean": float(means[:, 1].mean()),
                                     "ci": _bootstrap(means[:, 1], n_resamples=bootstrap_resamples, seed=0)},
                       "vs_noiseless": {"mean": float(means[:, 2].mean()),
                                        "ci": _bootstrap(means[:, 2], n_resamples=bootstrap_resamples, seed=0)}})
    return result


def run_noise_study(config, output, *, resume=False):
    import torch
    from .adapters import simulate_lindblad
    from .benchmarking import record_physics, score_schedule
    from .learning import records_content_digest
    from .models import graph_from_record
    from .schedules import Schedule
    from .transfer import (_load_transfer_model, source_global_schedule,
                           validate_transfer_provenance, validate_source_reference,
                           record_logical_fingerprint)

    if isinstance(config, (str, Path)):
        file = Path(config).resolve()
        cfg = json.loads(file.read_text())
        for key in ("data", "source_data", "checkpoint"):
            if key in cfg:
                cfg[key] = str((file.parent / cfg[key]).resolve())
    else:
        cfg = dict(config)
    allowed = {"schema_version", "data", "source_data", "checkpoint", "split", "rates", "max_qubits",
               "overflow", "device", "rtol", "atol", "tolerance", "bootstrap_resamples"}
    if {k for k in cfg if not k.startswith("_")} - allowed or cfg.get("schema_version", 1) != 1:
        raise ValueError("invalid noise study keys/schema")
    rates = np.asarray(cfg.get("rates", [[0., 0.], [.02, 0.], [0., .02]]), dtype=float)
    if (rates.ndim != 2 or rates.shape[1] != 2 or not np.isfinite(rates).all() or (rates < 0).any()
            or not any(np.array_equal(r, [0., 0.]) for r in rates)
            or len({tuple(r) for r in rates}) != len(rates)):
        raise ValueError("rates must be unique nonnegative [dephasing, relaxation] pairs including [0,0]")
    cap, overflow = cfg.get("max_qubits", 6), cfg.get("overflow", "error")
    if type(cap) is not int or not 1 <= cap <= 8 or overflow not in {"error", "exclude"}:
        raise ValueError("max_qubits must be 1..8 and overflow error/exclude")
    if cfg.get("split", "test") not in {"test", "validation"}:
        raise ValueError("noise study requires heldout test/validation records")
    rtol, atol, tolerance = (float(cfg.get(k, default)) for k, default in
                            (("rtol", 1e-8), ("atol", 1e-10), ("tolerance", 1e-5)))
    if any(not np.isfinite(x) or x <= 0 for x in (rtol, atol, tolerance)):
        raise ValueError("solver tolerances must be finite and positive")
    resamples = cfg.get("bootstrap_resamples", 2000)
    if type(resamples) is not int or resamples < 1:
        raise ValueError("bootstrap_resamples must be positive")
    records = load_records(cfg["data"], cfg.get("split", "test"))
    sources = load_records(cfg["source_data"], "train") + load_records(cfg["source_data"], "validation")
    payload = torch.load(cfg["checkpoint"], map_location="cpu", weights_only=True)
    provenance = validate_transfer_provenance(payload, records, source_records=sources)
    source_reference = validate_source_reference(payload, sources)
    requested_digest = records_content_digest(records)
    excluded = [{"record_id": str(r["record_id"]), "parent_id": str(r["parent_id"]),
                 "physical_n": len(r["physical_h"]), "reason": "declared_density_matrix_cap"}
                for r in records if len(r["physical_h"]) > cap]
    if excluded and overflow == "error":
        raise ValueError(f"{len(excluded)} records exceed the declared density matrix cap")
    records = [r for r in records if len(r["physical_h"]) <= cap]
    if not records:
        raise ValueError("no records remain after the declared cap")
    wave_global, global_info = source_global_schedule(sources)
    device = cfg.get("device", "cpu")
    model, normalizer = _load_transfer_model(payload, device)
    root = Path(output)
    identity = {"config": cfg, "source_hash": source_hash(), "checkpoint_sha256": _sha(cfg["checkpoint"]),
                "target_content": records_content_digest(records), "requested_target_content": requested_digest,
                "excluded_records": excluded, "source_content": records_content_digest(sources)}
    with output_lock(root):
        manifest_path = root / "study.json"
        if manifest_path.exists():
            if not resume:
                raise FileExistsError("noise study exists; use --resume")
            manifest = json.loads(manifest_path.read_text())
            if manifest["identity"] != identity:
                raise ValueError("noise config/source/checkpoint/data changed")
        else:
            manifest = {"schema_version": 1, "identity": identity, "status": "running", "units": {},
                        "excluded_records": excluded, "provenance": provenance, "source_reference": source_reference,
                        "global_baseline": global_info}
            write_json(manifest_path, manifest)
        # Freeze every decision before running any noisy target outcome.
        decisions = {}
        for record in records:
            graph = normalizer.transform(graph_from_record(record, device=device))
            with torch.no_grad():
                bank = torch.as_tensor(record["candidate_schedules"], dtype=torch.float32, device=device)
                proposals = model(graph)["proposal_schedules"]
                scores = [model.predict_losses(graph, waves) for waves in (bank, proposals)]
                if not all(torch.isfinite(s).all() for s in scores) or not torch.isfinite(proposals).all():
                    raise ArithmeticError("nonfinite learned noise-study decisions")
                waves = [bank[scores[0].argmin()].cpu().double().numpy(),
                         proposals[scores[1].argmin()].cpu().double().numpy()]
            choices = {"linear": Schedule.linear(),
                       "source_global": Schedule(np.linspace(0., 1., len(wave_global)), wave_global)}
            for method, wave in zip(("bank", "direct"), waves):
                wave = wave.copy()
                wave[0], wave[-1] = 0., 1.
                schedule = Schedule(np.linspace(0., 1., len(wave)), wave)
                schedule.validate_slope(float(record["runtime"]), model.max_ds_dtau * (1 + 1e-6) / float(record["runtime"]))
                choices[method] = schedule
            decisions[str(record["record_id"])] = choices
        decision_json = {rid: {m: {"tau_knots": s.tau_knots.tolist(), "s_knots": s.s_knots.tolist()}
                               for m, s in choices.items()} for rid, choices in decisions.items()}
        if "decisions_sha256" in manifest and manifest["decisions_sha256"] != content_hash(decision_json):
            raise ValueError("frozen learned decisions changed")
        manifest["decisions_sha256"] = content_hash(decision_json)
        write_json(root / "decisions.json", decision_json)
        write_json(manifest_path, manifest)
        rows = []
        for record in records:
            rid, parent = str(record["record_id"]), str(record["parent_id"])
            terms, path, observables = record_physics(record)
            success = np.asarray(observables["success"])
            for method, schedule in decisions[rid].items():
                for phi, relax in rates:
                    key = content_hash([rid, method, float(phi), float(relax)])
                    state = manifest["units"].setdefault(key, {"attempts": []})
                    target = root / "records" / f"{key}.json"
                    if state.get("status") == "complete":
                        if not target.exists() or _sha(target) != state["sha256"]:
                            raise ValueError("noise result artifact changed")
                        rows.append(json.loads(target.read_text()))
                        continue
                    for attempt in state["attempts"]:
                        if attempt["status"] == "running":
                            attempt.update(status="interrupted", cost_complete=False)
                    attempt = {"status": "running", "solver_calls_started": 0, "rhs_evaluations": 0,
                               "cost_complete": True}
                    state["attempts"].append(attempt)
                    write_json(manifest_path, manifest)
                    started = perf_counter()
                    try:
                        results = []
                        for refinement in (1., .1):
                            attempt["solver_calls_started"] += 1
                            write_json(manifest_path, manifest)
                            result = simulate_lindblad(terms, schedule, float(record["runtime"]), path=path,
                                dephasing_rates=float(phi), relaxation_rates=float(relax), max_qubits=cap,
                                rtol=rtol * refinement, atol=atol * refinement)
                            attempt["rhs_evaluations"] += result.diagnostics["rhs_evaluations"]
                            results.append(result)
                        losses = [float(1 - result.probabilities @ success) for result in results]
                        difference = abs(losses[0] - losses[1])
                        if difference > tolerance:
                            raise ArithmeticError("Lindblad tolerance refinement failed")
                        closed = None
                        if phi == relax == 0:
                            attempt["closed_objective_calls"] = 1
                            write_json(manifest_path, manifest)
                            closed = score_schedule(record, schedule, tolerance=tolerance / 4,
                                                    max_steps=65536, max_ds_dtau=model.max_ds_dtau * (1 + 1e-6))
                            if abs(closed["loss"] - losses[1]) > tolerance:
                                raise ArithmeticError("independent zero-noise parity gate failed")
                        row = {"record_id": rid, "parent_id": parent,
                               "logical_fingerprint": record_logical_fingerprint(record), "physical_n": len(record["physical_h"]),
                               "method": method, "rates": [float(phi), float(relax)], "loss": losses[1],
                               "refinement_difference": difference, "closed_parity": closed,
                               "solver_diagnostics": [r.diagnostics for r in results]}
                        if source_hash() != identity["source_hash"] or _sha(cfg["checkpoint"]) != identity["checkpoint_sha256"]:
                            raise RuntimeError("source/checkpoint drift during noise study")
                        write_json(target, row)
                        state.update(status="complete", sha256=_sha(target))
                        attempt["status"] = "complete"
                        rows.append(row)
                    except BaseException as error:
                        attempt.update(status="failed", cost_complete=False,
                                       error={"type": type(error).__name__, "message": str(error)})
                        manifest["status"] = "failed"
                        raise
                    finally:
                        attempt["observed_seconds"] = perf_counter() - started
                        write_json(manifest_path, manifest)
        attempts = [a for state in manifest["units"].values() for a in state["attempts"]]
        report = {"schema_version": 1, "status": "complete", "rows": rows,
                  "summary": _summarize(rows, resamples), "excluded_records": excluded,
                  "checkpoint_sha256": identity["checkpoint_sha256"],
                  "costs": {"solver_calls_started": sum(a["solver_calls_started"] for a in attempts),
                            "known_rhs_evaluations": sum(a["rhs_evaluations"] for a in attempts),
                            "closed_objective_calls": sum(a.get("closed_objective_calls", 0) for a in attempts),
                            "complete": all(a["cost_complete"] for a in attempts)},
                  "scope": "phenomenological local dephasing/computational-basis relaxation; no thermal or device claim",
                  "inference": "descriptive parent bootstrap conditional on one checkpoint; no multiplicity correction",
                  "numerics": "tolerance refinement and zero-noise solver agreement are diagnostics, not certificates"}
        write_json(root / "summary.json", report)
        manifest["status"] = "complete"
        write_json(manifest_path, manifest)
        return report
