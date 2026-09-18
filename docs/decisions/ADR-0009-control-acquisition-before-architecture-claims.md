# ADR-0009: Separate information, architecture, acquisition and deployment claims

Status: accepted implementation decision, 2026-09-17.

## Evidence

Archived parent-paired results support physical embedding information for bank
selection. They do not support a bank-selection advantage for a particular
embedding-aware architecture. A single-seed proposal-labelling improvement does
not isolate policy-dependent acquisition from additional labels, decoder-family
coverage or optimization randomness. Direct policy deployment also competes
with amortized bank selection, which needs no online simulator queries.

## Decision

Use summary moments as the primary efficient encoder and retain hierarchy as an
ablation. Implement a frozen one-round study with CONTROL, BANKEXT,
DECODER-RANDOM and POLICY arms; equal additional label counts for the three
acquisition arms, identical retraining recipes/seeds, original validation/test
banks, and full acquisition provenance. Add fixed-proposal critic comparisons
and synchronized standalone deployment timing.

Retain negative outcomes. Do not call scalar simulator relabelling canonical
DAgger or inherit its imitation-learning guarantees. Do not claim equivalence
from nonsignificance. Do not compare bank/direct online latency while omitting
offline labels and training.

Scale evidence must come from archived repeated runs with load/precision checks.
Physical state dimension, logical optimization size and hardware execution are
different quantities. A documented literature-method adaptation is not a
reproduction of the original system or hardware experiment.

## Consequences

The code can reject the proposed mechanism. POLICY may fail to beat matched
random controls, and direct may remain less useful than bank selection. Such a
result changes the paper contribution rather than justifying an unplanned test
subset or a revised claim after seeing test outcomes. Larger/OOD experiments
must use fresh, frozen evaluation populations after pilot design decisions.
