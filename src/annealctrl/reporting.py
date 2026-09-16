"""Paper-facing tables/figures from measured records, never fabricated results."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .evaluation import audit_parent_splits, paired_parent_bootstrap
from .pipeline import load_records, write_json


def _scalar(record, key):
    return np.asarray(record[key]).item()


def _stats(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0, "min": None, "median": None, "p90": None, "max": None, "mean": None}
    return {"count": len(values), "min": float(values.min()), "median": float(np.median(values)),
            "p90": float(np.quantile(values, .9)), "max": float(values.max()), "mean": float(values.mean())}


def audit_dataset(data_dir, *, split=None) -> dict:
    records = load_records(data_dir, split)
    parents = [str(_scalar(r, "parent_id")) for r in records]
    splits = [str(_scalar(r, "split")) for r in records]
    split_counts = audit_parent_splits(parents, splits)
    rows, masks, step_errors, headroom = [], [], [], []
    for record in records:
        losses = np.asarray(record["candidate_losses"], dtype=float)
        masks.extend(np.asarray(record["response_mask"], dtype=bool).ravel().tolist())
        step_errors.extend(np.asarray(record.get("candidate_state_error", []), dtype=float).ravel().tolist())
        headroom.append(float(losses[0] - losses.min()))
        rows.append({"record_id": str(_scalar(record, "record_id")), "parent_id": str(_scalar(record, "parent_id")),
            "split": str(_scalar(record, "split")), "family": str(_scalar(record, "family")),
            "logical_n": len(record["logical_h"]), "physical_n": len(record["physical_h"]),
            "runtime": float(record["runtime"]), "scale": float(record["programmed_scale"]),
            "chain_lengths": np.bincount(record["membership"]).tolist(),
            "linear_loss": float(losses[0]), "best_bank_loss": float(losses.min()),
            "best_candidate": int(losses.argmin()), "headroom_vs_linear": headroom[-1]})
    manifest = json.loads((Path(data_dir) / "manifest.json").read_text())
    return {"records": rows, "split_scope": split or "all", "n_records": len(rows), "n_parents": len(set(parents)),
        "parents_per_split": split_counts, "family_record_counts": dict(Counter(r["family"] for r in rows)),
        "logical_size_record_counts": dict(Counter(str(r["logical_n"]) for r in rows)),
        "physical_size_record_counts": dict(Counter(str(r["physical_n"]) for r in rows)),
        "resolved_response_fraction": float(np.mean(masks)) if masks else None,
        "state_error_diagnostic": _stats(step_errors), "finite_bank_headroom": _stats(headroom),
        "headroom_fraction_above_0p01": float(np.mean(np.asarray(headroom) > .01)),
        "best_candidate_counts": dict(Counter(str(r["best_candidate"]) for r in rows)),
        "generation_seconds": manifest.get("elapsed_seconds"),
        "warning": "Descriptive audit within split_scope. Do not choose hyperparameters from test headroom. Finite bank != control optimum."}


def _parent_mean(rows, key):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["parent_id"]].append(float(row[key]))
    return float(np.mean([np.mean(values) for values in grouped.values()]))


def aggregate_evaluations(evaluations: list[dict], *, bootstrap_resamples: int = 2000) -> dict:
    """Average seeds per record, then variants per parent. Never treat seeds as parents."""
    if not evaluations:
        raise ValueError("No completed evaluations to report")
    groups = defaultdict(list)
    seen = set()
    for result in evaluations:
        identity = (result["method"], int(result["training_seed"]))
        if identity in seen:
            raise ValueError("duplicate method/seed evaluation")
        seen.add(identity)
        groups[result["method"]].append(result)
    output, plot_rows = [], []
    reference_ids = None
    reference_truth = {}
    for method, runs in sorted(groups.items()):
        mode_rows = {"bank": []}
        have_direct = ["direct_policy" in run for run in runs]
        if any(have_direct) and not all(have_direct):
            raise ValueError("Direct evaluation missing for some seeds")
        if all(have_direct):
            mode_rows["direct"] = []
        for run in sorted(runs, key=lambda x: x["training_seed"]):
            rows = run["records"]
            ids = [r["record_id"] for r in rows]
            if len(ids) != len(set(ids)):
                raise ValueError("Duplicate test record in evaluation")
            if reference_ids is None:
                reference_ids = set(ids)
            if set(ids) != reference_ids:
                raise ValueError("Methods/seeds must be evaluated on exactly the same records")
            for row in rows:
                truth = (row["parent_id"], float(row["linear_loss"]), float(row["global_loss"]), float(row["bank_best_loss"]))
                if row["record_id"] in reference_truth and reference_truth[row["record_id"]] != truth:
                    raise ValueError("Reference outcomes/parents changed across methods or seeds")
                reference_truth[row["record_id"]] = truth
            mode_rows["bank"].append([{**r, "loss": r["selected_loss"]} for r in rows])
            if "direct" in mode_rows:
                direct = {r["record_id"]: r for r in run["direct_policy"]["records"]}
                if set(direct) != set(ids):
                    raise ValueError("Direct rows must exactly match bank rows")
                mode_rows["direct"].append([{**r, "loss": direct[r["record_id"]]["loss"]} for r in rows])
        for mode, per_seed in mode_rows.items():
            by_record = defaultdict(list)
            for rows in per_seed:
                for row in rows:
                    by_record[row["record_id"]].append(row)
            averaged = [{**items[0], "loss": float(np.mean([i["loss"] for i in items]))}
                        for _, items in sorted(by_record.items())]
            parents = [r["parent_id"] for r in averaged]
            differences = {}
            for baseline in ("linear_loss", "global_loss"):
                if len(set(parents)) >= 2:
                    differences[baseline] = asdict(paired_parent_bootstrap(
                        [r["loss"] for r in averaged], [r[baseline] for r in averaged], parents,
                        seed=0, n_resamples=bootstrap_resamples))
                else:
                    differences[baseline] = {"mean_difference": _parent_mean(averaged, "loss") - _parent_mean(averaged, baseline),
                        "ci_low": None, "ci_high": None, "note": "fewer than two independent parents"}
            seed_means = [_parent_mean(rows, "loss") for rows in per_seed]
            row = {"method": method, "mode": mode, "seeds": len(runs), "parents": len(set(parents)),
                "records": len(averaged), "mean_loss": _parent_mean(averaged, "loss"),
                "seed_std_loss": float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else None,
                "mean_success": 1 - _parent_mean(averaged, "loss"),
                "mean_linear_loss": _parent_mean(averaged, "linear_loss"),
                "mean_global_loss": _parent_mean(averaged, "global_loss"),
                "mean_bank_best_loss": _parent_mean(averaged, "bank_best_loss"),
                "paired_differences": differences,
                "mean_training_seconds": float(np.mean([r.get("training_seconds", 0) for r in runs])),
                "training_cost_complete": all(r.get("training_cost_complete", False) for r in runs)}
            output.append(row)
            plot_rows.extend({**r, "method": method, "mode": mode} for r in averaged)
    return {"summary": output, "record_means": plot_rows,
        "estimand": "equal parent after averaging training seeds and within-parent variants",
        "uncertainty": "Parent bootstrap conditional on observed training seeds; seed SD is separate, not a joint CI.",
        "claim": "Measured run results only. No automatic significance, quantum advantage or conference-readiness claim."}


def _tex_escape(text):
    mapping = {"_": r"\_", "&": r"\&", "%": r"\%", "#": r"\#", "$": r"\$", "{": r"\{", "}": r"\}"}
    return "".join(mapping.get(c, c) for c in str(text))


def build_report(experiment_dir, *, plots: bool = True, bootstrap_resamples: int = 2000) -> dict:
    root = Path(experiment_dir)
    manifest = json.loads((root / "experiment.json").read_text())
    required = {(m["name"], s) for m in manifest["config"]["methods"] for s in manifest["config"].get("seeds", [0])}
    completed = {(v["method"], v["seed"]) for v in manifest["runs"].values() if v.get("evaluated")}
    if completed != required:
        raise ValueError("Paper report requires exactly all predeclared method/seed evaluations")
    paths = [root / state["evaluation"] for state in manifest["runs"].values() if state.get("evaluated")]
    results = [json.loads(p.read_text()) for p in sorted(paths)]
    if {(r["method"], r["training_seed"]) for r in results} != required or any(
            r.get("experiment_config_hash") != manifest["config_hash"] for r in results):
        raise ValueError("Evaluation method/seed/config provenance does not match experiment")
    aggregate = aggregate_evaluations(results, bootstrap_resamples=bootstrap_resamples)
    report = root / "paper"
    report.mkdir(parents=True, exist_ok=True)
    write_json(report / "results.json", aggregate)
    audit = audit_dataset(root / "data")
    write_json(report / "data_audit.json", audit)
    lines = ["# Measured experiment results", "", aggregate["claim"], "", aggregate["estimand"] + ".",
             aggregate["uncertainty"], "", "Lower loss is better; loss = 1 - decoded logical success.", "",
             "| Method | Mode | Seeds | Parents | Loss | Seed SD | Δ vs global [parent 95% CI] |",
             "|---|---|---:|---:|---:|---:|---| "]
    tex = [r"\begin{tabular}{llrrr}", r"\hline", r"Method & Mode & Seeds & Loss & $\Delta$ vs global \\", r"\hline"]
    for row in aggregate["summary"]:
        paired = row["paired_differences"]["global_loss"]
        sd = "—" if row["seed_std_loss"] is None else f"{row['seed_std_loss']:.5f}"
        ci = "unavailable" if paired["ci_low"] is None else f"[{paired['ci_low']:.5f}, {paired['ci_high']:.5f}]"
        lines.append(f"| {row['method']} | {row['mode']} | {row['seeds']} | {row['parents']} | {row['mean_loss']:.5f} | {sd} | {paired['mean_difference']:+.5f} {ci} |")
        tex.append(f"{_tex_escape(row['method'])} & {row['mode']} & {row['seeds']} & {row['mean_loss']:.5f} & {paired['mean_difference']:+.5f} " + r"\\")
    tex.extend([r"\hline", r"\end{tabular}"])
    lines.extend(["", "## Reproducibility", "", f"Config hash: `{manifest['config_hash']}`.",
        f"Source hash: `{manifest['source_hash']}`.",
        "All predeclared methods/seeds are included; negative results are retained.",
        "Best-bank outcomes are privileged references, not deployable methods or continuous optima.",
        "Resumed training costs are flagged incomplete: observed attempts may omit work before a hard process kill.",
        "Plots show observed values only, with no smoothing or manufactured runs.", ""])
    (report / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    (report / "table_results.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    if plots:
        _plots(report, aggregate, audit)
    return aggregate


def _plots(folder, results, audit):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Plot export requires pip install -e '.[plots]', or report.plots=false") from error
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 200, "svg.fonttype": "none"})
    rows = results["summary"]
    fig, ax = plt.subplots(figsize=(6.4, max(2.8, len(rows) * .35)))
    for i, row in enumerate(rows):
        diff = row["paired_differences"]["global_loss"]
        value = diff["mean_difference"]
        errors = None if diff["ci_low"] is None else [[max(0, value - diff["ci_low"])], [max(0, diff["ci_high"] - value)]]
        ax.errorbar(value, i, xerr=errors, fmt="o", color="#285b8a", capsize=3)
    ax.axvline(0, color="#777777", linewidth=.8, linestyle="--")
    ax.set_yticks(range(len(rows)), [f"{r['method']} / {r['mode']}" for r in rows])
    ax.set_xlabel("Loss difference versus validation-tuned global control\nNegative favors model; 95% parent bootstrap")
    ax.invert_yaxis()
    fig.tight_layout()
    for suffix in ("svg", "png"):
        fig.savefig(folder / f"paired_global.{suffix}")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    grouped = defaultdict(list)
    for row in results["record_means"]:
        grouped[(row["method"], row["mode"], row["parent_id"])].append(row["loss"] - row["bank_best_loss"])
    curves = defaultdict(list)
    for (method, mode, _), values in grouped.items():
        curves[(method, mode)].append(float(np.mean(values)))
    for (method, mode), values in sorted(curves.items()):
        x = np.sort(values)
        ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", label=f"{method}/{mode}")
    ax.set(xlabel="Parent-mean loss minus finite-bank best", ylabel="Empirical cumulative fraction", ylim=(0, 1.03))
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    for suffix in ("svg", "png"):
        fig.savefig(folder / f"regret_cdf.{suffix}")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    sizes = audit["physical_size_record_counts"]
    axes[0].bar(list(sizes), list(sizes.values()), color="#285b8a")
    axes[0].set(xlabel="Physical qubits", ylabel="Task count")
    axes[1].hist([r["headroom_vs_linear"] for r in audit["records"]], bins=12, color="#587a48", edgecolor="white")
    axes[1].set(xlabel="Linear loss − best-bank loss", ylabel="Task count")
    fig.suptitle("Dataset diagnostics — descriptive, not independent task samples", fontsize=10)
    fig.tight_layout()
    for suffix in ("svg", "png"):
        fig.savefig(folder / f"data_diagnostics.{suffix}")
    plt.close(fig)
