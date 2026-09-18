# A second real topology: headroom holds on Zephyr too

`configs/data_zephyr.json` builds instances on genuine D-Wave **Zephyr Z15**
connectivity from `dwave-networkx` (7440 qubits, 71736 couplers) — a denser
graph than Pegasus P16, and a second independent device topology.

## Result: both splits resolved, nothing censored

| split | records | parents | censored | headroom | 95% CI |
|---|---:|---:|---:|---:|---|
| validation | 24 | 12 | **0.0%** | 0.1479 | [0.1150, 0.1867] |
| train | 144 | 72 | **0.0%** | 0.1339 | [0.1183, 0.1495] |

Verdict `resolved_headroom_present` on both.

### Stratified by physical size (train, the larger split)

| physical qubits | records | parents | mean headroom |
|---:|---:|---:|---:|
| 9 | 2 | 1 | 0.0932 |
| 10 | 46 | 23 | 0.1559 |
| 11 | 2 | 1 | 0.0817 |
| 12 | 46 | 23 | 0.1426 |
| 13 | 4 | 2 | 0.1063 |
| 14 | 44 | 22 | 0.1084 |

Validation, independently: 0.1178 / 0.1830 / 0.1430 at 10 / 12 / 14.

## Read against Pegasus

| topology | validation headroom | 10q | 12q | 14q |
|---|---:|---:|---:|---:|
| Pegasus P16 | 0.1441 | 0.1362 | 0.1649 | 0.1312 |
| **Zephyr Z15** | **0.1479** | 0.1178 | 0.1830 | 0.1430 |

Two independently generated device topologies, built from different vendor
graphs, agree on the headline number to within 0.004 and on the size ladder in
shape. That the 3–10 qubit synthetic result was not an artefact of the synthetic
graph is now supported by two real ones rather than one.

## What this does not cover

**Connectivity is not a device.** Vendor coupling graphs with no working-graph
exclusions, no per-qubit calibration, no noise model, no readout error. No QPU
job was submitted. The open-system check in `open_system_2026-09-17/` is a
perturbation study at hand-chosen rates, not a substitute.

**Budget 32, not 64.** `frontier_scaleup.json`, so roughly 97 objective calls per
record. A smaller budget gives a weaker best-found reference and therefore
*understates* headroom; these numbers are not directly comparable to the main
campaign's 257-call figures.

**The size ladder is uneven.** 9, 11 and 13 qubits carry 1–2 parents each and
should not be read as points on a curve; only 10, 12 and 14 have enough parents
to compare.

**Mild decline with size on train** (0.1559, 0.1426, 0.1084 at 10, 12, 14) that
validation does not reproduce (0.1178, 0.1830, 0.1430). With 22–23 parents per
cell the two splits disagree on the trend, so "headroom persists" is supported
and "headroom is flat in size" is not.

**No spectral oracle here.** A full teacher caps at 10 physical qubits, so the
privileged-baseline comparison does not extend to this dataset.
