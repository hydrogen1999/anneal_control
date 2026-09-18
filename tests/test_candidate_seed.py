"""Fresh problems can share a frozen action bank for honest transfer."""
import numpy as np

from annealctrl.pipeline import generate_dataset, load_records


def test_action_seed_is_independent_of_logical_problem_seed(tmp_path):
    cfg = {"seed": 701, "candidate_seed": 891, "parents": 6, "families": ["spin_glass"],
           "logical_qubits": 2, "chain_lengths": [1, 1], "variants": [{"shape": "path", "ports": 1}],
           "chain_strengths": [1.5], "runtimes": [.3], "candidates": 16, "spectral_points": 3,
           "teacher": {"mode": "none"}, "steps": 16, "max_steps": 1024,
           "label_state_tolerance": .001, "max_physical_qubits": 2}
    generate_dataset(cfg, tmp_path / "a")
    generate_dataset({**cfg, "seed": 702}, tmp_path / "b")
    a, b = load_records(tmp_path / "a"), load_records(tmp_path / "b")
    np.testing.assert_array_equal(a[0]["candidate_schedules"], b[0]["candidate_schedules"])
    assert {str(r["logical_fingerprint"]) for r in a}.isdisjoint(
        {str(r["logical_fingerprint"]) for r in b})
