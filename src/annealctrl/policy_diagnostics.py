"""Why a direct policy underperforms: generation, or ranking?

The campaign found every method's direct proposal barely beating the matched
linear schedule (≈0.590 against 0.601) while the same model selecting from a
fixed bank reached ≈0.545. Reported that way the result is uninformative,
because it conflates two failures with different fixes:

**Generation.** The policy cannot produce a control as good as the bank's best.
No ranker can rescue that; the policy head or its loss needs work.

**Ranking.** The policy *does* produce a good control, and the critic picks the
wrong one of its own proposals. That is a distribution-shift failure — the critic
is trained on bank waveforms and must extrapolate to waveforms it has never
scored — and it is fixable without touching the policy.

Separating them costs one extra true simulator call per proposal, which is small
against what the comparison buys. Two quantities do the work:

    ranking_regret        = selected proposal loss − best proposal loss
    generation_gap_vs_bank = best proposal loss    − bank-selected loss

The first is what the critic threw away. The second is what the policy could
never have reached. Their sum is the whole shortfall against bank selection.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .headroom import _bootstrap, _describe, _parent_means
from .telemetry import _safe


def _rank_correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    """Spearman rho without scipy.stats overhead; None when undefined."""
    if a.size < 2 or np.all(a == a[0]) or np.all(b == b[0]):
        return None
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    denominator = float(np.sqrt((ra**2).sum() * (rb**2).sum()))
    return None if denominator == 0 else float((ra @ rb) / denominator)


def decompose_proposals(*, true_losses: Sequence[float], predicted_losses: Sequence[float],
                        selected_index: int, bank_loss: float, linear_loss: float) -> dict:
    """Split one record's direct-policy shortfall into ranking and generation.

    ``true_losses`` must be real simulator outcomes of every proposal, obtained
    after the critic made its choice. Scoring them does not change the reported
    selection; it only reveals what the selection cost.
    """
    true = np.asarray(true_losses, dtype=float)
    predicted = np.asarray(predicted_losses, dtype=float)
    if true.ndim != 1 or not true.size or true.shape != predicted.shape:
        raise ValueError("true and predicted losses must be nonempty and the same length")
    if not np.isfinite(true).all() or not np.isfinite(predicted).all():
        raise ValueError("losses must be finite")
    if isinstance(selected_index, bool) or not isinstance(selected_index, int) \
            or not 0 <= selected_index < true.size:
        raise ValueError(f"selected_index must index the {true.size} proposals")

    best = float(true.min())
    selected = float(true[selected_index])
    return _safe({
        "n_proposals": int(true.size),
        "selected_proposal_loss": selected,
        "best_proposal_loss": best,
        "worst_proposal_loss": float(true.max()),
        "proposal_spread": float(true.max() - true.min()),
        "bank_loss": float(bank_loss),
        "linear_loss": float(linear_loss),
        # What the critic threw away among its own proposals.
        "ranking_regret": selected - best,
        # What no ranker could have recovered.
        "generation_gap_vs_bank": best - float(bank_loss),
        "oracle_beats_bank": bool(best < float(bank_loss)),
        "selected_beats_linear": bool(selected < float(linear_loss)),
        "oracle_beats_linear": bool(best < float(linear_loss)),
        "critic_rank_correlation": _rank_correlation(predicted, true),
        "scope": ("proposal losses are true simulator outcomes scored after selection; "
                  "the selection itself saw no outcome"),
    })


def aggregate_proposal_decomposition(rows: Sequence[Mapping[str, Any]], *,
                                     bootstrap_resamples: int = 2000, seed: int = 0) -> dict:
    """Parent-level attribution of the direct-policy shortfall.

    ``dominant_cause`` names whichever term is larger on average. It is a
    description of this population, not a claim that the other term is zero.
    """
    rows = list(rows)
    if not rows:
        raise ValueError("aggregate_proposal_decomposition requires a nonempty row set")

    blocks = {}
    for key in ("ranking_regret", "generation_gap_vs_bank", "selected_proposal_loss",
                "best_proposal_loss", "bank_loss", "proposal_spread"):
        values, _ = _parent_means(rows, lambda row, key=key: row.get(key))
        block = _describe(values)
        block["parent_bootstrap_ci"] = _bootstrap(values, n_resamples=bootstrap_resamples, seed=seed)
        blocks[key] = block

    correlations = [float(row["critic_rank_correlation"]) for row in rows
                    if row.get("critic_rank_correlation") is not None]
    ranking = blocks["ranking_regret"]["mean"] or 0.0
    generation = blocks["generation_gap_vs_bank"]["mean"] or 0.0

    return _safe({
        "schema_version": 1,
        "n_records": len(rows),
        "n_parents": len({str(row["parent_id"]) for row in rows}),
        **blocks,
        "mean_critic_rank_correlation": float(np.mean(correlations)) if correlations else None,
        "n_records_with_rank_correlation": len(correlations),
        "oracle_beats_bank_fraction": sum(1 for row in rows if row.get("oracle_beats_bank")) / len(rows),
        "oracle_beats_linear_fraction": sum(1 for row in rows if row.get("oracle_beats_linear")) / len(rows),
        "selected_beats_linear_fraction": sum(1 for row in rows if row.get("selected_beats_linear")) / len(rows),
        "dominant_cause": "critic_ranking" if ranking > generation else "policy_generation",
        "attribution": {"ranking_regret": ranking, "generation_gap_vs_bank": generation,
                        "total_shortfall_vs_bank": ranking + generation},
        "scope": ("a decomposition of this population's shortfall, not a claim that the "
                  "smaller term is zero"),
    })


def diagnose_checkpoint(data_dir, checkpoint, *, split: str = "test", device: str = "cpu",
                        backend: str = "numpy", tolerance: float = 5e-4,
                        initial_steps: int = 128, max_steps: int = 8192,
                        max_ds_dtau: float = 4.0, record_ids: Sequence[str] | None = None) -> dict:
    """Score every proposal of a trained policy, not only the one its critic picked.

    The selection is unchanged and still sees no outcome; the extra propagations
    happen afterwards and only reveal what the selection cost. The 1e-6 relative
    slope slack is the same construction tolerance ``evaluate_checkpoint`` uses
    for direct proposals.
    """
    import torch

    from .benchmarking import score_schedule
    from .learning import load_checkpoint
    from .models import graph_from_record
    from .pipeline import load_records
    from .schedules import Schedule

    records = load_records(data_dir, split)
    if record_ids is not None:
        wanted = set(record_ids)
        records = [r for r in records if str(np.asarray(r["record_id"]).item()) in wanted]
    if not records:
        raise ValueError(f"split {split!r} of {data_dir} contains no records")

    model, normalizer = load_checkpoint(checkpoint, device=device)
    model.eval()
    rows, infeasible = [], 0
    for record in records:
        graph = normalizer.transform(graph_from_record(record, device=device))
        with torch.no_grad():
            proposals = model(graph)["proposal_schedules"]
            predicted = model.predict_losses(graph, proposals)
            bank = torch.as_tensor(record["candidate_schedules"], dtype=torch.float32, device=device)
            bank_predicted = model.predict_losses(graph, bank)
            bank_index = int(bank_predicted.argmin())
            selected = int(predicted.argmin())

        stored = np.asarray(record["candidate_losses"], dtype=float)
        true_losses, skipped = [], False
        for row in proposals.detach().cpu().double().numpy():
            wave = row.copy()
            wave[0], wave[-1] = 0.0, 1.0
            schedule = Schedule(np.linspace(0.0, 1.0, len(wave)), wave)
            try:
                outcome = score_schedule(record, schedule, backend=backend, tolerance=tolerance,
                                         initial_steps=initial_steps, max_steps=max_steps,
                                         max_ds_dtau=max_ds_dtau * (1 + 1e-6))
            except ValueError:
                skipped = True
                break
            true_losses.append(float(outcome["loss"]))
        if skipped:
            infeasible += 1
            continue

        linear_index = int(np.argmin([
            float(np.abs(np.asarray(w) - np.linspace(0.0, 1.0, len(w))).max())
            for w in np.asarray(record["candidate_schedules"])]))
        decomposition = decompose_proposals(
            true_losses=true_losses,
            predicted_losses=[float(x) for x in predicted.detach().cpu().tolist()],
            selected_index=selected, bank_loss=float(stored[bank_index]),
            linear_loss=float(stored[linear_index]))
        rows.append({**decomposition,
                     "record_id": str(np.asarray(record["record_id"]).item()),
                     "parent_id": str(np.asarray(record["parent_id"]).item()),
                     "split": split})

    return _safe({"schema_version": 1, "split": split, "checkpoint": str(checkpoint),
                  "n_records": len(rows), "n_infeasible_records": infeasible,
                  "rows": rows,
                  "note": ("infeasible proposals are counted and excluded; every reported "
                           "loss is a true simulator outcome scored after selection")})
