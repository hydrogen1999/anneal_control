"""Assemble the experiments section's numbers into one JSON file per table.

Writing a paper from twenty report directories invites transcription error, and
the transcription error this project already paid for was exactly that: a
number carried into prose from the wrong artifact. So the tables are exported
mechanically, each file naming its sources and their hashes, and each carrying
the scope note that must travel with the numbers.

These files are inputs to the draft and to figure rendering. They are derived:
delete and regenerate freely, never hand-edit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
EFFECT = "reports/effect_size_2026-09-19/artifacts"
PEG = "reports/pegasus240_2026-09-20/artifacts"
POOL = "reports/pool_selection_2026-09-20/artifacts"
AUDIT = "reports/evidence_audit_2026-09-18"
RUNG3 = "reports/spectral_rung3_2026-09-20"


def load(rel: str):
    return json.loads((ROOT / rel).read_text())


def digest(rel: str) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()[:16]


def block(table: str, caption: str, sources: list[str], payload: dict, scope: str) -> dict:
    return {"schema_version": 1, "table": table, "caption_hint": caption,
            "sources": sources, "source_sha256": {s: digest(s) for s in sources},
            **payload, "scope": scope}


# --- populations -------------------------------------------------------------

def populations() -> dict:
    return block(
        "populations", "Two instance populations and one replication target.", [],
        {"rows": [
            {"name": "synthetic", "topology": "generated hardware-like", "parents": 240,
             "records": 4320, "split_parents": [144, 48, 48], "logical_sizes": [3, 4, 5],
             "physical_qubits": [3, 10], "runtimes": [1.0, 4.0, 12.0], "candidates": 64,
             "training_seeds": 5, "resolved_response_fraction": 0.7992608836907082},
            {"name": "pegasus240", "topology": "Pegasus", "parents": 240,
             "records": 2880, "split_parents": [144, 48, 48], "logical_sizes": [5, 6, 7],
             "physical_qubits": [10, 14], "runtimes": [4.0, 12.0], "candidates": 64,
             "training_seeds": 3, "resolved_response_fraction": 0.0},
            {"name": "pegasus12", "topology": "Pegasus", "parents": 96,
             "records": 1152, "split_parents": [None, None, 12], "logical_sizes": [5, 6, 7],
             "physical_qubits": [10, 14], "runtimes": [4.0, 12.0], "candidates": 64,
             "training_seeds": 3, "resolved_response_fraction": 0.0,
             "role": "replication target for pegasus240; independent draw, no shared parent"}],
         "instance_families": ["spin_glass", "weighted_maxcut", "planted_loops", "weak_field"]},
        "resolved_response_fraction 0.0 is why the auxiliary-physics arm is vacuous on "
        "both Pegasus sets; the two hierarchy variants are the same model there.")


# --- effect size -------------------------------------------------------------

def effect_size() -> dict:
    srcs = [f"{EFFECT}/synth_both.json", f"{PEG}/effect_size_both.json"]
    rows = []
    for name, rel in (("synthetic", srcs[0]), ("pegasus240", srcs[1])):
        bank = load(rel)["by_reference"]["bank_oracle"]
        pooled = bank["pooled_across_seeds"]
        rows.append({"population": name, "n_parents": bank["n_parents"],
                     "n_training_seeds": bank["n_training_seeds"],
                     "gain_vs_linear": bank["mean_gain_vs_linear"],
                     "cohens_d_pooled": pooled["cohens_d"],
                     "cohens_d_mean_per_seed": bank["mean_cohens_d"],
                     "parents_won_pooled": pooled["parents_won"],
                     "parents_won_worst_seed": pooled["worst_seed_parents_won"]})
    return block("effect_size",
                 "Bank selection against a matched linear ramp, held-out parents.",
                 srcs, {"rows": rows},
                 "Mean of parent means; d is pooled over training seeds before the "
                 "statistic. Worst-seed win rate is reported because pooling can exceed it.")


# --- encoder contrast --------------------------------------------------------

def encoder_contrast() -> dict:
    src = f"{AUDIT}/contrasts.json"
    data = load(src)["datasets"]
    out = {}
    for name in ("synthetic", "pegasus", "pegasus240"):
        if name not in data:
            continue
        cm = data[name]["bank_pairs"]
        info = data[name]["exploratory_embedding_information"]
        out[name] = {
            "n_separated_after_correction": cm["n_separated_after_correction"],
            "n_comparisons": cm["n_comparisons"],
            "nonseparated_maximal_sets": cm["nonseparated_maximal_sets"],
            "pairs": [{"a": p["method_a"], "b": p["method_b"],
                       "mean_difference_b_minus_a": p["mean_difference"],
                       "ci_low": p["ci_low"], "ci_high": p["ci_high"],
                       "p_value_holm": p["p_value_holm"],
                       "separated_after_correction": p["separated_after_correction"],
                       "n_parents": p["n_parents"]} for p in cm["pairs"]],
            "pooled_aware_minus_blind": {
                "mean": info["mean_difference"], "ci_low": info["ci_low"],
                "ci_high": info["ci_high"], "separated": info["separated"],
                "parents_favouring_aware": info.get("parents_favouring_aware"),
                "aware_methods": info["aware_methods"], "blind_methods": info["blind_methods"]}}
    return block("encoder_contrast",
                 "All ten encoder pairs, Holm-corrected as one family, bank selection.",
                 [src], {"populations": out},
                 "Difference is method_b minus method_a; negative favours method_b for loss. "
                 "On Pegasus the hierarchy_physics/hierarchy_outcome pair is DEGENERATE "
                 "(resolved response 0.0, same model), so the family has ten nominal and "
                 "nine informative members and the correction is conservative. "
                 "Non-separation is not equivalence.")


# --- reads -------------------------------------------------------------------

def reads() -> dict:
    srcs = [f"{EFFECT}/tts_synth.json", f"{PEG}/time_to_solution.json"]
    rows = []
    for name, rel in (("synthetic", srcs[0]), ("pegasus240", srcs[1])):
        payload = load(rel)
        ratio = payload["read_ratios"]["linear_reads_over_learned_reads"]
        # summary is keyed by method name, not a list of rows.
        summary = payload.get("summary", {})
        rows.append({"population": name,
                     "reads_by_method": {k: {m: v.get(m) for m in
                                             ("parent_mean_reads", "mean_loss",
                                              "record_quantiles", "n_censored",
                                              "n_parents")}
                                         for k, v in summary.items()},
                     "linear_over_learned_parent_mean": ratio["parent_mean"],
                     "ci_low": ratio["parent_bootstrap_ci"]["low"],
                     "ci_high": ratio["parent_bootstrap_ci"]["high"],
                     "learned_needs_fewer_in_parents": ratio["learned_needs_fewer_in_parents"],
                     "n_parents": ratio["n_parents"],
                     "record_median_ratio": ratio["record_median"]})
    return block("reads", "Reads to 99 % confidence, n_q = ceil(log(1-q)/log(1-p)).",
                 srcs, {"rows": rows},
                 "p is a SIMULATED success probability from exact propagation, so the "
                 "binomial interval for hardware counts does not apply and is not used; "
                 "uncertainty is across parents. On Pegasus the one non-win is a TIE "
                 "(parent_0186, 3.00 reads either way): n_q is a ceiling.")


# --- call equivalent ---------------------------------------------------------

def call_equivalent() -> dict:
    srcs = [f"{EFFECT}/synth_equiv_bank.json", f"{EFFECT}/synth_equiv_bank_ceiling.json",
            f"{EFFECT}/synth_equiv_direct.json", f"{EFFECT}/equiv_g2_frontier_test_bayes.json",
            f"{EFFECT}/equiv_g2_frontier_test_pg.json",
            f"{PEG}/equiv_bank_learned_gain.json", f"{PEG}/equiv_bank_bank_ceiling.json",
            f"{PEG}/equiv_direct_learned_gain.json"]
    labels = [("synthetic", "bank", "learned_gain", srcs[0]),
              ("synthetic", "bank", "bank_ceiling", srcs[1]),
              ("synthetic", "direct", "learned_gain", srcs[2]),
              ("synthetic_bayes", "bank", "learned_gain", srcs[3]),
              ("synthetic_policy_gradient", "bank", "learned_gain", srcs[4]),
              ("pegasus240", "bank", "learned_gain", srcs[5]),
              ("pegasus240", "bank", "bank_ceiling", srcs[6]),
              ("pegasus240", "direct", "learned_gain", srcs[7])]
    rows = []
    for population, mode, target, rel in labels:
        d = load(rel)
        rows.append({"population": population, "mode": mode, "target": target,
                     "n_parents": d["n_parents"], "n_resolved": d["n_resolved"],
                     "n_censored": d["n_censored"], "n_no_gain": d.get("n_no_gain"),
                     "median_calls": d["median_calls"], "mean_calls": d["mean_calls"],
                     "ci_low": d["parent_bootstrap_ci"]["low"],
                     "ci_high": d["parent_bootstrap_ci"]["high"],
                     "censor_at": d["censor_at"]})
    return block("call_equivalent",
                 "One forward pass priced in instance-specific simulator calls.",
                 srcs, {"rows": rows},
                 "The equivalent is the smallest grid budget whose headroom reaches the "
                 "gain, an upper bound between grid points. n_no_gain parents have NO "
                 "equivalent and are excluded from the mean, never averaged in at the "
                 "grid's first budget. The unit moves with search strategy: report the "
                 "range, and the conservative claim is the minimum.")


# --- pool axis ---------------------------------------------------------------

def pool_axis() -> dict:
    srcs = []
    rows = []
    for population, prefix, seeds, pool in (
            ("synthetic", "synth_seed", range(5), "proposals+linear"),
            ("synthetic", "synth_designed", range(5), "proposals+designed_library"),
            ("pegasus240", "peg240_seed", range(3), "proposals+linear"),
            ("pegasus240", "peg240_designed", range(3), "proposals+designed_library")):
        for s in seeds:
            rel = f"{POOL}/{prefix}{s}.json"
            srcs.append(rel)
            d = load(rel)
            rows.append({"population": population, "pool": pool, "seed": s,
                         "n_parents": d["n_parents"], "cost_class": d["cost_class"],
                         "pool_prefixes": d.get("pool_prefixes", ["linear"]),
                         "proposal_only_vs_linear": d["proposal_only_vs_linear"],
                         "pool_vs_linear": d["pool_vs_linear"],
                         "pool_vs_proposal_only": d["pool_vs_proposal_only"],
                         "fallback_rate_records": d["fallback_rate_records"],
                         "parents_rescued": d["parents_rescued"],
                         "parents_harmed": d["parents_harmed"]})
    return block("pool_axis",
                 "What the critic may choose between, and what it is worth.",
                 srcs, {"rows": rows},
                 "Negative favours the first named. amortised throughout: proposals plus "
                 "fixed waveforms, one forward pass, no simulation. The no-harm property "
                 "belongs to the two-element pool and is STRUCTURAL there -- the only "
                 "alternative is the reference itself -- and does not transfer to the "
                 "designed library. Physics-derived members are excluded because they "
                 "would be privileged_spectrum.")


# --- menu size and factorisation --------------------------------------------

def menu_size() -> dict:
    srcs = [f"{EFFECT}/synth_menu.json", f"{PEG}/menu_size.json"]
    rows = []
    for name, rel in (("synthetic", srcs[0]), ("pegasus240", srcs[1])):
        d = load(rel)
        rows.append({"population": name, "bank_size": d["bank_size"],
                     "n_parents": d["n_parents"], "sizes": d["sizes"],
                     "curve": d["curve"], "across_draw_std": d.get("across_draw_std"),
                     "last_doubling_gain": d["last_doubling_gain"],
                     "search_headroom": d.get("search_headroom"),
                     "draws": d["draws"]})
    return block("menu_size", "Headroom against the size of a FIXED menu of controls.",
                 srcs, {"rows": rows},
                 "One fixed menu shared by every record, subsets drawn at random. Below "
                 "the full bank this UNDERSTATES a purpose-designed menu of the same "
                 "size, so the saturation is a conservative reading. Each curve's "
                 "endpoint reproduces that population's bank-oracle headroom.")


def factorisation() -> dict:
    srcs = [f"{EFFECT}/synth_both.json", f"{PEG}/effect_size_both.json"]
    rows = []
    for name, rel in (("synthetic", srcs[0]), ("pegasus240", srcs[1])):
        d = load(rel)
        rows.append({"population": name, **{k: v for k, v in d["decomposition"].items()
                                            if k not in ("schema_version", "scope")}})
    return block("factorisation",
                 "share of findable = selector efficiency x bank coverage.",
                 srcs, {"rows": rows},
                 "A share is comparable only to a share against the SAME reference. The "
                 "previously published 60 % vs 83 % compared a 257-call frontier headroom "
                 "against a 64-candidate bank-oracle headroom and is withdrawn.")


def seed_stability() -> dict:
    srcs = [f"{EFFECT}/peg240_policy_seed_stability.json", f"{PEG}/policy_seed_stability.json"]
    rel = srcs[1]
    d = load(rel)
    return block("seed_stability",
                 "Per-training-seed stability of bank selection and direct generation.",
                 [rel], {"population": "pegasus240", "modes": d["modes"],
                         "difference_definition": d["difference_definition"]},
                 "A sign change between seeds is training instability, not parent "
                 "disagreement; a pooled interval cannot distinguish the two.")


def negatives() -> dict:
    src = f"{RUNG3}/rung3_contrasts.json"
    data = load(src)
    rung = {}
    for mode in ("bank", "direct"):
        cm = data[mode]
        rung[mode] = {"n_separated_after_correction": cm["n_separated_after_correction"],
                      "n_comparisons": cm["n_comparisons"],
                      "nonseparated_maximal_sets": cm["nonseparated_maximal_sets"],
                      "pairs": [{"a": p["method_a"], "b": p["method_b"],
                                 "mean_difference_b_minus_a": p["mean_difference"],
                                 "ci_low": p["ci_low"], "ci_high": p["ci_high"],
                                 "p_value_holm": p["p_value_holm"],
                                 "separated_after_correction": p["separated_after_correction"]}
                                for p in cm["pairs"]]}
    return block("negatives_spectral_ladder",
                 "Spectral resolution as a supervision target (rung 3).",
                 [src], {"rung3": rung},
                 "Arms differ in one training key. Zero of three separate in either mode. "
                 "Bounds any bins-over-moments effect under 2 % of the method's own gain; "
                 "a non-separation is not equivalence. The teacher-side rungs 1-2 live in "
                 "reports/spectral_ladder_2026-09-19/ and are not machine-exported.")


TABLES = {"populations": populations, "effect_size": effect_size,
          "encoder_contrast": encoder_contrast, "reads": reads,
          "call_equivalent": call_equivalent, "pool_axis": pool_axis,
          "menu_size": menu_size, "factorisation": factorisation,
          "seed_stability": seed_stability, "negatives_spectral_ladder": negatives}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="docs/paper/results")
    args = parser.parse_args(argv)
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)

    index = {"schema_version": 1, "note": __doc__.strip().splitlines()[0], "tables": {}}
    for name, builder in TABLES.items():
        payload = builder()
        path = out / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        index["tables"][name] = {"file": f"{name}.json",
                                 "caption_hint": payload["caption_hint"],
                                 "sources": payload["sources"]}
        print(f"  wrote {path.relative_to(ROOT)}")
    (out / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(f"  wrote {(out / 'index.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
