"""Rebuild audited paper evidence from committed record-level outcomes.

Needs NumPy/SciPy and package source, not PyTorch or original server paths.
Run from any directory: python /path/to/repo/scripts/rebuild_evidence.py --check
No training/simulation is launched. Historical inputs are never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from annealctrl.contrasts import contrast_matrix, encoder_information_contrast, paired_method_contrast
from annealctrl.paper_table import assemble_comparison

INPUTS = {
    "synthetic": "reports/heldout_2026-09-17/heldout_records.json",
    "pegasus": "reports/pegasus_learned_2026-09-18/heldout_records.json",
    "sobol": "reports/comparison_2026-09-17/testref_rows.json",
    "bayesian": "reports/comparison_2026-09-17/bayes_rows.json",
    "policy_gradient": "reports/comparison_2026-09-17/policy_gradient_rows.json",
    "transfer": "reports/pegasus_learned_2026-09-18/transfer.json",
}
DEPENDENCIES = ["scripts/rebuild_evidence.py", "src/annealctrl/paper_table.py",
                "src/annealctrl/contrasts.py", "src/annealctrl/headroom.py"]


def _json(value):
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _balanced(rows):
    """Refuse unequal record populations even when their parent sets match."""
    panels = {}
    for row in rows:
        if row.get("split") != "test":
            raise ValueError("reanalysis requires held-out test rows")
        panel = panels.setdefault((row["method"], row["mode"]), {})
        record = str(row["record_id"])
        if record in panel:
            raise ValueError("duplicate learned record within method/mode")
        panel[record] = str(row["parent_id"])
    expected = next(iter(panels.values()), None)
    if not expected or any(panel != expected for panel in panels.values()):
        raise ValueError("all learned methods/modes must cover the same record-parent population")



UPSTREAM_TRANSFER_PATH = "reports/pegasus_learned_2026-09-18/transfer.json"
UPSTREAM_TRANSFER_REVISION = "974207aa24cb8ea02cf800649e0fa354a43f7587"


def audit_transfer_aggregates(raw: bytes) -> dict[str, str]:
    """Describe the archived aggregates without inventing paired observations."""
    data = json.loads(raw)
    summaries, entries, verdict_conflicts = [], [], []
    for method, rows in data.items():
        if not isinstance(rows, list) or not rows:
            raise ValueError("transfer archive must contain nonempty aggregate lists")
        losses, deltas = [], []
        for index, row in enumerate(rows, start=1):
            contrast = row["vs_linear"]
            interval = contrast["parent_bootstrap_ci"]
            numeric = [row["mean_selected_loss"], row["mean_linear_loss"],
                       contrast["mean_difference"], interval["low"], interval["high"]]
            if not all(math.isfinite(float(value)) for value in numeric) or interval["low"] > interval["high"]:
                raise ValueError("transfer aggregate values and ordered intervals must be finite")
            if not math.isclose(row["mean_selected_loss"] - row["mean_linear_loss"],
                                contrast["mean_difference"], abs_tol=1e-12, rel_tol=0):
                raise ValueError("transfer loss and difference disagree")
            favourable = interval["high"] < 0
            unfavorable = interval["low"] > 0
            interpretation = "CI_favours_selector" if favourable else "CI_favours_linear" if unfavorable else "not_separated"
            entry = {"method": method, "array_position_one_based": index,
                     "training_seed_id": None, "checkpoint_sha256": None,
                     "n_records_reported": row["n_records"], "n_parents_reported": row["n_parents"],
                     "selected_loss": row["mean_selected_loss"], "linear_loss": row["mean_linear_loss"],
                     "difference": contrast["mean_difference"], "archived_ci_low": interval["low"],
                     "archived_ci_high": interval["high"], "archived_verdict": row["verdict"],
                     "archived_separated": contrast["separated"],
                     "interval_interpretation": interpretation,
                     "interval_recomputed": False}
            entries.append(entry)
            losses.append(row["mean_selected_loss"])
            deltas.append(contrast["mean_difference"])
            if row["verdict"] == "beats_linear" and not favourable:
                verdict_conflicts.append({"method": method, "array_position_one_based": index,
                                          "reason": "beats_linear verdict despite interval including zero"})
        summaries.append({"method": method, "n_entries": len(rows),
                          "mean_of_reported_selected_losses": sum(losses) / len(losses),
                          "mean_of_reported_differences": sum(deltas) / len(deltas),
                          "entries_with_ci_favouring_selector": sum(row["vs_linear"]["parent_bootstrap_ci"]["high"] < 0 for row in rows),
                          "pooled_confidence_interval": None,
                          "scope": "descriptive mean across unlabeled entries, not a pooled parent/seed analysis"})
    output = {"schema_version": 1, "archive_revision": UPSTREAM_TRANSFER_REVISION,
              "archive_path": UPSTREAM_TRANSFER_PATH, "archive_sha256": hashlib.sha256(raw).hexdigest(),
              "reported_design": "commit message describes synthetic 3–10-qubit training to Pegasus 10–14-qubit evaluation with five encoders and three seeds; JSON has no seed or checkpoint identifiers",
              "scope": "aggregate-only audit; no new simulation and no recomputed confidence intervals",
              "summaries": summaries, "entries": entries, "verdict_conflicts": verdict_conflicts,
              "missing_evidence": ["paired record/parent/seed outcomes", "training-seed IDs", "checkpoint hashes",
                                   "logical-fingerprint disjointness receipts", "source/target dataset manifests",
                                   "global fixed-schedule target baseline", "between-method multiplicity-corrected contrasts"]}
    lines = ["# Upstream transfer aggregate audit — 2026-09-18", "",
             f"Upstream commit `{UPSTREAM_TRANSFER_REVISION}` adds `{UPSTREAM_TRANSFER_PATH}`. This is new measured aggregate evidence, not a completed independently auditable transfer campaign. Its commit message describes synthetic 3–10-qubit training and Pegasus 10–14-qubit evaluation, five encoders and three seeds. The file contains five method lists of three aggregate entries, without seed identities or checkpoint hashes.", "",
             f"Input SHA-256: `{output['archive_sha256']}`. The original JSON is preserved unchanged. This audit checks its arithmetic and interval/verdict consistency; it does not reconstruct raw outcomes or bootstrap new intervals.", "",
             "## Descriptive method averages", "",
             "Every entry reports 144 target records and 12 logical parents, linear loss 0.7070519901, and best-in-bank loss 0.6115838701. Matching counts and means do not independently prove identical record identities. Best-in-bank is an outcome-informed finite-bank diagnostic, not a global control optimum.", "",
             "| Method | Mean selected loss across three entries | Mean selected − linear | Entries whose archived CI excludes zero in favour of selection |", "|---|---:|---:|---:|"]
    for row in sorted(summaries, key=lambda row: row["mean_of_reported_selected_losses"]):
        lines.append(f"| {row['method']} | {row['mean_of_reported_selected_losses']:.6f} | {row['mean_of_reported_differences']:+.6f} | {row['entries_with_ci_favouring_selector']}/{row['n_entries']} |")
    lines += ["", "The hierarchical methods have favourable point estimates, and each of their three archived entry-level parent intervals favours selection over linear. Summary and physical each have one interval crossing zero. Logical has mixed entry-level effects. These are descriptive observations; the arithmetic mean does not have a recoverable pooled confidence interval from this file. Entries are not presumed to be independent test populations, and all 15 intervals are pointwise rather than corrected as one family.", "",
              "## All archived entry-level intervals", "",
              "Entry numbers are one-based array positions, **not seed IDs**. Differences are selected loss minus linear loss; negative is favourable. Each archived interval reports 8,000 parent resamples and explicitly excludes training-seed uncertainty.", "",
              "| Method | Array entry | Difference | Archived 95% parent CI | Interpretation |", "|---|---:|---:|---|---|"]
    for row in entries:
        lines.append(f"| {row['method']} | {row['array_position_one_based']} | {row['difference']:+.6f} | [{row['archived_ci_low']:+.6f}, {row['archived_ci_high']:+.6f}] | {row['interval_interpretation']} |")
    lines += ["", "## Two legacy verdict errors", "",
              "The summary entry 1 interval is [−0.024154, +0.015091]; physical entry 2 is [−0.047005, +0.001171]. Both store `separated: false` but `verdict: beats_linear`. Their intervals do not establish improvement. This is the legacy verdict bug, not evidence of a successful corrected-code run. A non-separated result also does not establish equivalence to linear.", "",
              "## What is still required", "",
              "The aggregate JSON contains no raw paired record/parent outcomes, training-seed IDs, checkpoint hashes, source/target dataset manifests, logical fingerprints, or guard receipts. The commit message reports zero overlap under a content fingerprint, but the historical guard used compiled-record identity rather than logical-problem identity. This artifact cannot independently establish logical-parent disjointness. No leakage is demonstrated by the absence of provenance; the required validation remains missing.", "",
              "There is no target evaluation of the fixed global baseline learned on the source, and no archived paired hierarchy-versus-summary contrast. The apparent ranking reversal is a hypothesis worth testing; it does not establish hierarchy superiority or an architecture contribution. Separate per-method intervals against linear cannot answer the between-method question. Do not assign seed IDs from array positions, bootstrap the three means as if they were independent parents, average CI endpoints into a pooled CI, or infer Holm decisions from these summaries.", "",
              "Rerun the corrected transfer protocol using explicit source checkpoints, frozen source normalizers, disjoint logical fingerprints, shared target rows, the source-selected global baseline, and retained per-record/per-parent/per-seed outcomes. Then compute parent and crossed parent-by-seed contrasts with a declared multiplicity family. Until those artifacts are available, the new result is **promising aggregate transfer evidence with incomplete provenance**.", "",
              "After this upstream input is present in the working tree, reproduce this audit with `python scripts/rebuild_evidence.py`; `--check` verifies it alongside the other evidence outputs. `UPSTREAM_TRANSFER.json` contains the machine-readable arithmetic audit and all 15 archived interval interpretations.", ""]
    return {"UPSTREAM_TRANSFER.md": "\n".join(lines), "UPSTREAM_TRANSFER.json": _json(output)}


def build(root: Path, *, resamples: int = 20000) -> dict[str, str]:
    data = {name: json.loads((root / path).read_text()) for name, path in INPUTS.items()}
    learned = {name: data[name]["record_means"] for name in ("synthetic", "pegasus")}
    for rows in learned.values():
        _balanced(rows)
    source_hashes = {path: hashlib.sha256((root / path).read_bytes()).hexdigest()
                     for path in DEPENDENCIES}
    provenance = {"input_sha256": {path: hashlib.sha256((root / path).read_bytes()).hexdigest()
                                   for path in INPUTS.values()},
                  "analysis_sha256": source_hashes, "bootstrap_resamples": resamples,
                  "seed": 0, "scope": "reanalysis of archived outcomes only; no new training or simulator queries",
                  "seed_uncertainty": "raw per-training-seed test panels are absent from these archives; intervals condition on archived seed averages"}
    table = assemble_comparison(learned["synthetic"], data["sobol"]["rows"],
                                searches={name: data[name]["rows"] for name in ("bayesian", "policy_gradient")},
                                bootstrap_resamples=resamples, seed=0)
    table["provenance"] = provenance
    contrasts = {"provenance": provenance, "datasets": {}}
    for name, rows in learned.items():
        result = {"bank_pairs": contrast_matrix(rows, mode="bank", bootstrap_resamples=resamples, seed=0),
                  "direct_pairs": contrast_matrix(rows, mode="direct", bootstrap_resamples=resamples, seed=0),
                  "exploratory_embedding_information": encoder_information_contrast(rows, bootstrap_resamples=resamples, seed=0),
                  "summary_vs_global": {}}
        for mode in ("bank", "direct"):
            subset = [row for row in rows if row["method"] == "summary" and row["mode"] == mode]
            augmented = subset + [{**row, "method": "global", "loss": row["global_loss"]} for row in subset]
            result["summary_vs_global"][mode] = paired_method_contrast(augmented, "global", "summary", mode=mode,
                                                                       bootstrap_resamples=resamples, seed=0)
        contrasts["datasets"][name] = result
    search_rows = [{**row, "method": name, "mode": "search", "loss": row["best_found_loss"]}
                   for name in ("sobol", "bayesian", "policy_gradient") for row in data[name]["rows"]]
    contrasts["search_pairs"] = contrast_matrix(search_rows, mode="search", bootstrap_resamples=resamples, seed=0)
    ci = lambda x: f"[{x['low']:+.6f}, {x['high']:+.6f}]"
    lines = ["# Audited evidence reanalysis — 2026-09-18", "",
             "Authoritative replacement for the historical comparison table and historical contrast p-values. All values below are regenerated from committed record-level outcomes. Lower loss is better; loss is one minus decoded logical success. No training or new simulation is claimed.", "",
             "## Common synthetic test population", "",
             f"{table['n_records']} records, {table['n_parents']} logical parents. Means give parents equal weight. Teacher rows are conditional and appear separately below. Search arms have matched record identities and per-record objective budgets; the archive does not retain search seeds, so seed matching is not asserted.", "",
             "| Cost class | Method | Mean loss | 95% parent CI | Records |", "|---|---|---:|---|---:|"]
    for row in table["rows"]:
        if row["cost_class"] == "privileged_spectrum":
            continue
        lines.append(f"| {row['cost_class']} | {row['method']} | {row['mean_loss']:.6f} | {ci(row['parent_bootstrap_ci'])} | {row['n_records']} |")
    lines += ["", "No global ranking across cost classes is defined. Offline training and label costs are not included in the zero deployment-query count. The 257-call references are budget-limited methods, not certified control optima.", "",
              "## Audited teacher populations", "",
              "Only finite teacher losses with `sampled_point_audit_passed` are eligible. Each contrast uses identical records for teacher and learned method. A sampled-point audit is a numerical diagnostic, not a continuous-path certificate.", "",
              "| Teacher | Eligible records | Excluded records |", "|---|---:|---:|"]
    for name, population in table["teacher_populations"].items():
        lines.append(f"| {name} | {population['n_eligible_records']} | {population['n_excluded_records']} |")
    lines += ["", "| Learned method | Teacher | Parents | Learned − teacher | 95% parent CI |", "|---|---|---:|---:|---|"]
    for row in table["matched_teacher_contrasts"]:
        if row["method"].startswith("summary/"):
            lines.append(f"| {row['method']} | {row['teacher']} | {row['n_parents']} | {row['mean_difference']:+.6f} | {ci(row['parent_bootstrap_ci'])} |")
    lines += ["", "These intervals are descriptive and unadjusted, conditional on teacher eligibility and archived learned seed averages. Different teacher rows concern different populations. They do not establish equal-cost superiority or hardware performance.", "",
              "## Regenerated encoder contrasts", "",
              "The ten encoder pairs form a separate Holm family for each dataset and mode. Approximate centered-null bootstrap p-values are corrected; intervals remain pointwise. Non-rejection does not establish equivalence. The pooled information effect is exploratory and outside those Holm families.", "",
              "| Dataset | Contrast | Difference | 95% parent CI | Parents |", "|---|---|---:|---|---:|"]
    for name, result in contrasts["datasets"].items():
        for label, row in [("aware − blind (exploratory)", result["exploratory_embedding_information"]),
                           ("summary bank − global", result["summary_vs_global"]["bank"]),
                           ("summary direct − global", result["summary_vs_global"]["direct"])]:
            lines.append(f"| {name} | {label} | {row['mean_difference']:+.6f} | [{row['ci_low']:+.6f}, {row['ci_high']:+.6f}] | {row['n_parents']} |")
    lines += ["", "The two Pegasus hierarchy arms have no spectral-response labels in that dataset and are degenerate as an auxiliary-loss ablation. No architecture superiority, cross-topology transfer, equivalence, or minimum required sample size follows from these tables.", "",
              "## Reproduction and remaining artifact limits", "", "```bash", "python scripts/rebuild_evidence.py", "python scripts/rebuild_evidence.py --check", "```", "",
              "The command uses NumPy/SciPy and source code without importing PyTorch. JSON outputs include input/analysis hashes, all contrasts, exact teacher eligibility IDs, and search population audits. `--check` fails on any changed/missing output; it never silently refreshes artifacts.", "",
              "Historical aggregation and proposal-count archives retain aggregate statistics but not the raw paired parent-by-seed panels needed to reconstruct new confidence intervals. The aggregation campaign also augmented validation; its gain is not isolated to train-only acquisition. The independent frozen-validation pilot remains a negative result. The regenerated comparisons do not repair missing raw training traces or prove data-regeneration identity. GPU crossover raw profiles and full manifest comparison inputs are not archived. Upstream commit 974207a now adds aggregate transfer measurements; see UPSTREAM_TRANSFER.md for their provisional interpretation and missing provenance. Budget-curve and new multi-seed acquisition campaigns must supply their own completed raw artifacts before being claimed.", ""]
    outputs = {"comparison_table.json": _json(table), "contrasts.json": _json(contrasts), "RESULTS.md": "\n".join(lines)}
    transfer_path = root / UPSTREAM_TRANSFER_PATH
    outputs.update(audit_transfer_aggregates(transfer_path.read_bytes()))
    return outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/evidence_audit_2026-09-18")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--bootstrap-resamples", type=int, default=20000)
    args = parser.parse_args(argv)
    generated = build(ROOT, resamples=args.bootstrap_resamples)
    if args.check:
        mismatch = [name for name, content in generated.items()
                    if not (args.output / name).exists() or (args.output / name).read_text() != content]
        if mismatch:
            raise SystemExit("Evidence differs or is missing: " + ", ".join(mismatch))
        print("Verified audited evidence against committed inputs and current analysis code.")
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        for name, content in generated.items():
            (args.output / name).write_text(content)
        print(f"Wrote {len(generated)} audited outputs to {args.output}")


if __name__ == "__main__":
    main()
