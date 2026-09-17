"""Real Pegasus/Zephyr connectivity, and the honest labelling of a local patch."""
import numpy as np
import pytest

dnx = pytest.importorskip("dwave_networkx")

from annealctrl.generation import (  # noqa: E402
    Embedding,
    commercial_topology,
    grow_hardware_partition,
    resolve_hardware,
    topology_patch,
)

# Documented sizes of the shipped Advantage / Advantage2 graphs.
FULL = {("pegasus", 16): (5640, 40484), ("zephyr", 15): (7440, 71736)}


def test_full_topologies_match_the_vendor_generator():
    for (name, m), (nodes, edges) in FULL.items():
        topology = commercial_topology(name, m)
        assert topology["n_qubits"] == nodes
        assert len(topology["edges"]) == edges
        assert topology["is_commercial_topology"] is True


def test_labels_are_contiguous_and_original_ids_are_kept():
    topology = commercial_topology("pegasus", 3)
    assert sorted({int(x) for edge in topology["edges"] for x in edge}) == list(
        range(topology["n_qubits"]))
    # Pegasus labels start at 2 and are not contiguous; the mapping must survive.
    assert topology["original_ids"][0] != 0
    assert len(topology["original_ids"]) == topology["n_qubits"]


def test_an_unknown_topology_is_refused():
    with pytest.raises(ValueError, match="topology"):
        commercial_topology("chimera_but_misspelled", 4)


def test_a_nonpositive_size_parameter_is_refused():
    for bad in (0, -1):
        with pytest.raises(ValueError, match="m must"):
            commercial_topology("pegasus", bad)


# --- patches -----------------------------------------------------------------

def test_a_patch_is_connected_and_the_requested_size():
    patch = topology_patch("pegasus", 16, 60, np.random.default_rng(0))
    assert patch["n_qubits"] == 60
    import networkx as nx
    graph = nx.Graph()
    graph.add_nodes_from(range(patch["n_qubits"]))
    graph.add_edges_from([tuple(e) for e in patch["edges"]])
    assert nx.is_connected(graph)


def test_a_patch_states_that_it_is_not_a_full_device():
    patch = topology_patch("zephyr", 15, 40, np.random.default_rng(1))
    assert patch["is_commercial_topology"] is True
    assert patch["is_full_device"] is False
    assert patch["source_qubits"] == 7440
    assert patch["patch_sites"] == 40
    assert "local patch" in patch["note"]


def test_different_seeds_select_different_regions():
    a = topology_patch("pegasus", 16, 50, np.random.default_rng(0))
    b = topology_patch("pegasus", 16, 50, np.random.default_rng(1))
    assert a["original_ids"] != b["original_ids"]


def test_a_patch_larger_than_the_device_is_refused():
    with pytest.raises(ValueError, match="patch"):
        topology_patch("pegasus", 2, 10**6, np.random.default_rng(0))


def test_patch_edges_are_a_subgraph_of_the_real_topology():
    full = commercial_topology("pegasus", 3)
    real = {tuple(sorted((full["original_ids"][i], full["original_ids"][j])))
            for i, j in full["edges"]}
    patch = topology_patch("pegasus", 3, 30, np.random.default_rng(2))
    for i, j in patch["edges"]:
        pair = tuple(sorted((patch["original_ids"][i], patch["original_ids"][j])))
        assert pair in real, "a patch edge must exist in the device graph"


# --- growth inside a real topology ------------------------------------------

def test_chains_grown_in_a_real_patch_form_a_valid_embedding():
    patch = topology_patch("pegasus", 16, 80, np.random.default_rng(3))
    embedding = grow_hardware_partition(patch["n_qubits"], patch["edges"],
                                        np.array([2, 2, 2, 2, 2, 2]), np.random.default_rng(4))
    assert isinstance(embedding, Embedding)
    assert embedding.metadata["route"] == "fixed_hardware_growth"
    # The whole point: chains grown in a real patch actually touch each other.
    assert len(embedding.quotient_edges) > 0


def test_grown_degrees_never_exceed_the_topology_degree_bound():
    patch = topology_patch("pegasus", 16, 80, np.random.default_rng(5))
    embedding = grow_hardware_partition(patch["n_qubits"], patch["edges"],
                                        np.array([2, 2, 2, 2]), np.random.default_rng(6))
    degrees = np.bincount(embedding.hardware_edges.reshape(-1),
                          minlength=len(embedding.membership))
    assert degrees.max() <= 15, "Pegasus degree is at most 15"


# --- resolve_hardware --------------------------------------------------------

def test_resolve_accepts_an_explicit_edge_list_unchanged():
    spec = {"n_qubits": 4, "edges": [[0, 1], [1, 2], [2, 3]]}
    resolved = resolve_hardware(spec, np.random.default_rng(0))
    assert resolved["n_qubits"] == 4
    assert resolved.get("is_commercial_topology", False) is False


def test_resolve_builds_a_patch_from_a_topology_spec():
    resolved = resolve_hardware({"topology": "pegasus", "m": 16, "patch_sites": 48},
                                np.random.default_rng(7))
    assert resolved["n_qubits"] == 48 and resolved["is_commercial_topology"] is True


def test_resolve_refuses_a_spec_that_is_neither():
    with pytest.raises(ValueError, match="hardware"):
        resolve_hardware({"m": 16}, np.random.default_rng(0))


def test_resolve_refuses_a_spec_that_is_both():
    with pytest.raises(ValueError, match="exactly one"):
        resolve_hardware({"topology": "pegasus", "m": 16, "n_qubits": 4,
                          "edges": [[0, 1]]}, np.random.default_rng(0))


# --- end to end through the pipeline -----------------------------------------

def test_a_dataset_can_be_generated_on_a_real_pegasus_patch(tmp_path):
    import json
    from annealctrl.pipeline import generate_dataset, load_records

    config = {"seed": 4242, "parents": 6, "families": ["spin_glass", "weighted_maxcut"],
              "logical_qubits": 4, "chain_lengths": [2, 2, 2, 2],
              "generation_route": "hardware_growth",
              "hardware": {"topology": "pegasus", "m": 16, "patch_sites": 64},
              "variants": [{"field_distribution": "uniform", "coupling_distribution": "uniform"}],
              "chain_strengths": [1.5], "runtimes": [2.0], "candidates": 4,
              "candidate_batch_size": 4, "spectral_points": 3, "teacher": {"mode": "none"},
              "steps": 16, "max_steps": 1024, "label_state_tolerance": 0.005,
              "max_physical_qubits": 8}
    generate_dataset(config, tmp_path / "data")
    records = load_records(tmp_path / "data", "train")
    assert records
    meta = json.loads(str(records[0]["metadata_json"]))
    embedding = meta["embedding"]
    assert embedding["topology"] == "pegasus"
    assert embedding["is_commercial_topology"] is True
    assert embedding["is_full_device"] is False
    assert embedding["source_qubits"] == 5640
    # active -> device, composed from active -> patch and patch -> device.
    assert len(embedding["device_qubit_ids"]) == len(records[0]["membership"])
    # Pegasus labels are sparse: P16 has 5640 nodes but ids run past 5700, so the
    # meaningful check is membership of the real node set, not a numeric bound.
    real_nodes = set(dnx.pegasus_graph(16).nodes())
    assert set(embedding["device_qubit_ids"]) <= real_nodes


def test_each_parent_lands_on_a_different_patch(tmp_path):
    import json
    from annealctrl.pipeline import generate_dataset, load_records

    config = {"seed": 77, "parents": 6, "families": ["spin_glass"],
              "logical_qubits": 3, "chain_lengths": [2, 2, 2],
              "generation_route": "hardware_growth",
              "hardware": {"topology": "zephyr", "m": 15, "patch_sites": 48},
              "variants": [{"field_distribution": "uniform"}],
              "chain_strengths": [1.5], "runtimes": [2.0], "candidates": 4,
              "spectral_points": 3, "teacher": {"mode": "none"}, "steps": 16,
              "max_steps": 1024, "label_state_tolerance": 0.005, "max_physical_qubits": 6}
    generate_dataset(config, tmp_path / "data")
    records = load_records(tmp_path / "data")
    patches = {tuple(json.loads(str(r["metadata_json"]))["embedding"]["device_qubit_ids"])
               for r in records}
    assert len(patches) > 1, "parents must sample different regions of the chip"
