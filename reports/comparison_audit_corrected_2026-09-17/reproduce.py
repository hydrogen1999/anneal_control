"""Regenerate the corrected table from immutable archived record-level inputs."""
import hashlib
import json
from pathlib import Path

from annealctrl.paper_table import assemble_comparison


def main():
    destination = Path(__file__).resolve().parent
    root = destination.parents[1]
    learned_path = root / "reports/heldout_2026-09-17/heldout_records.json"
    reference_path = root / "reports/comparison_2026-09-17/testref_rows.json"
    learned = json.loads(learned_path.read_text())["record_means"]
    reference = json.loads(reference_path.read_text())["rows"]
    table = assemble_comparison(learned, reference, bootstrap_resamples=20000, seed=0)
    table["provenance"] = {
        "archived_input_revision": "2d48229bf3a7d2e869b3bdbf0290e06f6c9d14f3",
        "input_sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in (learned_path, reference_path)},
        "paper_table_sha256": hashlib.sha256((root / "src/annealctrl/paper_table.py").read_bytes()).hexdigest(),
        "bootstrap_resamples": 20000, "seed": 0,
        "scope": "reanalysis of existing outcomes; no new training or simulator queries"}
    (destination / "comparison_table.json").write_text(json.dumps(table, indent=2, allow_nan=False) + "\n")
    lines = ["# Audit-corrected teacher comparison", "",
             "This reanalysis excludes spectral controls without a passed sampled-point audit, including finite losses whose interpolation audit failed. Historical artifacts are preserved.", "",
             "The learned and teacher means below use exactly the same teacher-specific records and equal logical-parent weight. Negative differences favour the learned method. Teachers require privileged spectral computation; learned policies require offline training. These are different cost classes, not equal-cost competitors.", "",
             "| Teacher | Eligible records | Excluded records |", "|---|---:|---:|"]
    for teacher, population in table["teacher_populations"].items():
        lines.append(f"| {teacher} | {population['n_eligible_records']} | {population['n_excluded_records']} |")
    lines += ["", "| Learned mode | Teacher | Parents | Learned loss | Teacher loss | Difference | 95% parent CI |",
              "|---|---|---:|---:|---:|---:|---|"]
    for row in table["matched_teacher_contrasts"]:
        if not row["method"].startswith("summary/"):
            continue
        ci = row["parent_bootstrap_ci"]
        lines.append(f"| {row['method']} | {row['teacher']} | {row['n_parents']} | {row['learned_mean_loss']:.6f} | {row['teacher_mean_loss']:.6f} | {row['mean_difference']:+.6f} | [{ci['low']:+.6f}, {ci['high']:+.6f}] |")
    lines += ["", "All encoder/mode contrasts, eligible record IDs, status counts and input hashes are in `comparison_table.json`.", "",
              "Intervals are descriptive, unadjusted parent bootstraps conditional on the archived learned seed averages; they do not include training-seed uncertainty. Passed sampled-point audits are numerical diagnostics, not continuous-path certificates. Conditional teacher rows are excluded from full-population rankings. This correction does not establish hardware performance or a general advantage over spectral schedules.", "",
              "Reproduce from repository root:", "", "```bash", "PYTHONPATH=src python reports/comparison_audit_corrected_2026-09-17/reproduce.py", "```", ""]
    (destination / "RESULTS.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
