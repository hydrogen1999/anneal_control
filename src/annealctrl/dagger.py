"""One round of dataset aggregation, to put the critic back on its own distribution.

The measured problem. The critic is trained on each record's candidate bank and
then deployed to rank waveforms its own policy emits. Those are different
manifolds: the bank uses residual-softmax durations plus window and pause
closures, the policy uses capped-simplex water filling, and 92% of policy
waveforms sit further from the bank than a typical bank waveform sits from its
own nearest neighbour (0.133 against 0.079, max-norm over nine knots). The
consequence is a Spearman correlation of only 0.50-0.62 between predicted and
true losses *on the policy's own three proposals*, and a ranking regret of
0.015-0.021.

The fix that does not work, measured before this one: adding random candidates
from the policy's decoder to the shared bank moved the distance from 0.133 to
0.127. An eight-dimensional waveform space is not coverable by a bank of sixty
four, so scattering more points in it is the wrong move.

The fix that does. Take the trained model's **actual** proposals on the training
and validation records, score them with the simulator, and append them to those
records' banks. This is standard dataset aggregation, one round, and it is honest
about its cost: every appended candidate is a real propagation, charged and
reported. Measured over three seeds against a matched ``collect_bank_extension``
control, it is worth -0.0226 in selected loss on held-out parents, consistently
(-0.0195, -0.0238, -0.0243), with 78% of the total gain attributable to
aggregation and 22% to the bank merely growing from 64 candidates to 67.

**The mechanism is not the one this module was written to address.** The
hypothesis above was critic distribution shift, so the expected signature was a
rise in rank correlation. It is not what happens. Against the control, rho moves
by +0.003 on average and is negative on one seed of three; almost all of the
apparent rho gain (0.575 to 0.630) comes from the larger bank, which the control
arm reproduces. What aggregation actually moves is **generation**: the policy's
own best proposal improves by -0.0194 and the generation gap closes by -0.0183,
while ranking regret improves by only -0.0031.

The plausible reading is that the policy and critic share an encoder and are
trained jointly, so appending truthfully-labelled proposals changes the policy
head's targets more than it recalibrates the critic. That is a hypothesis about
a measurement, not a second measurement, and it is not claimed as established.

Boundaries. Proposals are collected on train and validation records only; a test
record's proposals are never labelled or trained on. The augmented bank is a
different bank from the one the frontier instrument uses, so a headroom number
computed against it is not comparable to one computed against the original.
"""
from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np

from .telemetry import _safe


def collect_labelled_proposals(data_dir, checkpoint, *, split: str = "train", device: str = "cpu",
                               backend: str = "numpy", tolerance: float = 5e-4,
                               initial_steps: int = 128, max_steps: int = 8192,
                               max_ds_dtau: float = 4.0) -> dict:
    """Score a trained policy's own proposals on a split, keeping every waveform.

    Refuses the test split outright: labelling a held-out record's proposals and
    training on them is leakage, whatever it is called afterwards.
    """
    import torch

    from .benchmarking import score_schedule
    from .learning import load_checkpoint
    from .models import graph_from_record
    from .pipeline import load_records
    from .schedules import Schedule

    if split == "test":
        raise ValueError("refusing to label test-split proposals: training on them is leakage")

    records = load_records(data_dir, split)
    if not records:
        raise ValueError(f"split {split!r} of {data_dir} contains no records")
    model, normalizer = load_checkpoint(checkpoint, device=device)
    model.eval()

    began = perf_counter()
    collected: dict[str, dict[str, np.ndarray]] = {}
    propagations, infeasible, per_record = 0, 0, None
    for record in records:
        graph = normalizer.transform(graph_from_record(record, device=device))
        with torch.no_grad():
            proposals = model(graph)["proposal_schedules"].detach().cpu().double().numpy()
        waveforms, losses = [], []
        for row in proposals:
            wave = row.copy()
            wave[0], wave[-1] = 0.0, 1.0
            schedule = Schedule(np.linspace(0.0, 1.0, len(wave)), wave)
            try:
                outcome = score_schedule(record, schedule, backend=backend, tolerance=tolerance,
                                         initial_steps=initial_steps, max_steps=max_steps,
                                         max_ds_dtau=max_ds_dtau * (1 + 1e-6))
            except ValueError:
                infeasible += 1
                continue
            waveforms.append(wave)
            losses.append(float(outcome["loss"]))
            propagations += 1
        if not waveforms:
            continue
        per_record = len(waveforms) if per_record is None else min(per_record, len(waveforms))
        collected[str(np.asarray(record["record_id"]).item())] = {
            "waveforms": np.asarray(waveforms, dtype=float),
            "losses": np.asarray(losses, dtype=float)}

    if not collected:
        raise ValueError("no feasible proposals were collected; nothing to aggregate")
    return {"records": collected, "n_records": len(collected), "split": split,
            "checkpoint": str(checkpoint), "proposals_per_record": int(per_record),
            "propagations": propagations, "infeasible_proposals": infeasible,
            "wall_seconds": perf_counter() - began,
            "scope": "true simulator outcomes of the policy's own proposals, for training only"}


def augment_records(records: Sequence[Mapping[str, Any]], collected: Mapping[str, Any]) -> list[dict]:
    """Append the collected proposals to each record's candidate bank.

    Every original candidate is preserved at its original index, so anything that
    refers to a bank position still means what it meant. Each record receives the
    same number of new candidates, because training batches over a fixed bank
    size and a ragged bank cannot be stacked.
    """
    per_record = int(collected["proposals_per_record"])
    if per_record < 1:
        raise ValueError("collected proposals must supply at least one per record")
    augmented = []
    for record in records:
        record_id = str(np.asarray(record["record_id"]).item())
        entry = collected["records"].get(record_id)
        if entry is None:
            raise ValueError(f"record {record_id!r} has no collected proposals; "
                             "aggregate over the same split the records come from")
        waveforms = np.asarray(entry["waveforms"], dtype=float)[:per_record]
        losses = np.asarray(entry["losses"], dtype=float)[:per_record]
        stored = np.asarray(record["candidate_schedules"], dtype=float)
        if waveforms.shape[1] != stored.shape[1]:
            raise ValueError("proposal knot count differs from the stored bank's")

        new = dict(record)
        new["candidate_schedules"] = np.vstack((stored, waveforms))
        new["candidate_losses"] = np.concatenate(
            (np.asarray(record["candidate_losses"], dtype=float), losses))
        new["candidate_success"] = np.concatenate(
            (np.asarray(record["candidate_success"], dtype=float), 1.0 - losses))
        new["candidate_ids"] = np.asarray(
            [str(x) for x in np.asarray(record["candidate_ids"])]
            + [f"dagger_{i:04d}" for i in range(per_record)])
        # Fields the learner reads per candidate must stay the same length. The
        # appended rows carry no chain-break or timing labels, so they are filled
        # with the record's own means rather than invented zeros.
        for key in ("candidate_decoded_energy", "candidate_any_chain_break",
                    "candidate_chain_break_fraction", "candidate_norm_error",
                    "candidate_state_error", "candidate_steps", "candidate_seconds"):
            if key not in record:
                continue
            column = np.asarray(record[key], dtype=float)
            new[key] = np.concatenate((column, np.full(per_record, float(column.mean()))))
        augmented.append(new)
    return augmented


def collect_bank_extension(data_dir, *, split: str = "train", n_extra: int = 3,
                           bank_seed: int, bank_size: int, n_segments: int = 8,
                           backend: str = "numpy", tolerance: float = 5e-4,
                           initial_steps: int = 128, max_steps: int = 8192,
                           max_ds_dtau: float = 4.0) -> dict:
    """The control arm: extend each bank with more of *the same* distribution.

    Aggregation changes two things at once -- the bank grows, and the new
    entries come from the policy. Without this arm a reader cannot tell which
    one did the work, and "our method helps" would be indistinguishable from
    "sixty seven candidates beat sixty four".

    The bank is a deterministic Sobol sequence, so asking for ``bank_size +
    n_extra`` reproduces the original entries and appends the next points of the
    same sequence. That identity is checked rather than assumed: if the
    regenerated prefix does not match what the record stores, the bank was built
    with different parameters and every downstream comparison would be void.
    """
    from .benchmarking import score_schedule
    from .pipeline import load_records
    from .schedules import Schedule
    from .search import shared_candidate_bank

    if split == "test":
        raise ValueError("refusing to label test-split candidates: training on them is leakage")
    if n_extra < 1:
        raise ValueError("n_extra must be at least one")

    records = load_records(data_dir, split)
    if not records:
        raise ValueError(f"split {split!r} of {data_dir} contains no records")

    began = perf_counter()
    collected: dict[str, dict[str, np.ndarray]] = {}
    propagations, infeasible = 0, 0
    for record in records:
        runtime = float(np.asarray(record["runtime"]).item())
        stored = np.asarray(record["candidate_schedules"], dtype=float)
        tau = np.linspace(0.0, 1.0, stored.shape[1])
        bank = shared_candidate_bank(n=bank_size + n_extra, n_segments=n_segments, seed=bank_seed,
                                     runtime=runtime, max_slope=max_ds_dtau / runtime)
        waves = np.stack([candidate.schedule(tau) for candidate in bank])
        if waves.shape[0] < bank_size + n_extra:
            raise ValueError(f"bank generator returned {waves.shape[0]} candidates, "
                             f"fewer than the {bank_size + n_extra} requested")
        deviation = float(np.abs(waves[:bank_size] - stored[:bank_size]).max())
        if deviation > 1e-9:
            raise ValueError(
                f"regenerated bank prefix differs from the stored bank by {deviation:.3e} on "
                f"record {record['record_id']}; the stored bank was not built with seed="
                f"{bank_seed}, n={bank_size}, n_segments={n_segments}, max_slope="
                f"{max_ds_dtau}/runtime, so this control arm would not be matched")

        waveforms, losses = [], []
        for wave in waves[bank_size:bank_size + n_extra]:
            try:
                outcome = score_schedule(record, Schedule(tau, wave), backend=backend,
                                         tolerance=tolerance, initial_steps=initial_steps,
                                         max_steps=max_steps, max_ds_dtau=max_ds_dtau * (1 + 1e-6))
            except ValueError:
                infeasible += 1
                continue
            waveforms.append(np.asarray(wave, dtype=float))
            losses.append(float(outcome["loss"]))
            propagations += 1
        if len(waveforms) != n_extra:
            raise ValueError(f"record {record['record_id']} yielded {len(waveforms)} of {n_extra} "
                             "extension candidates; a ragged control arm is not matched")
        collected[str(np.asarray(record["record_id"]).item())] = {
            "waveforms": np.asarray(waveforms, dtype=float),
            "losses": np.asarray(losses, dtype=float)}

    return {"records": collected, "n_records": len(collected), "split": split,
            "proposals_per_record": int(n_extra), "propagations": propagations,
            "infeasible_proposals": infeasible, "wall_seconds": perf_counter() - began,
            "bank_seed": bank_seed, "bank_size": bank_size,
            "scope": ("control arm: the next points of the same Sobol bank, scored the same way, "
                      "so only the source of the extra candidates differs from aggregation")}
