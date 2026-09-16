# Neural input contract: implementation and limits

For v0.2 encoder variants, train-only normalization and exact checkpoint resume,
see `training.md`. The detailed tensor schema below still applies to the default
hierarchical encoder. Adaptive response-query coordinates affect the auxiliary
branch only, not policy or critic selection.

`models.py` implements a small signed PyTorch model without PyG. Default width
64, three physical message layers, sum/mean/count chain pooling, and two logical
message layers retain a bank of physical, chain, and logical tokens. Typed
embeddings encode roles, never qubit identifiers. Queries attend to this bank;
there is no compulsory scalar-difficulty bottleneck.

## Input contract

One `GraphInput` describes one instance. Edges in the dataclass are bidirected.
The NPZ helper expects undirected `[E,2]` arrays, each edge once.

| Field | Shape | Meaning |
|---|---|---|
| `node_features` | N × 7 | h, abs(h), degree, signed J load, absolute J load, internal degree, boundary absolute load |
| `edge_index`, `edge_features` | 2 × E, E × 3 | Signed J, abs(J), same-chain relation |
| `membership` | N | Physical-to-logical relation; contiguous labels, every chain nonempty |
| `logical_node_features` | L × 2 | Logical h, abs(h) |
| `logical_edge_index`, `logical_edge_features` | 2 × E_L, E_L × 2 | Logical signed J, abs(J) |
| `context` | 2 | Physical runtime, programmed coefficient scale |
| `path_queries` | Q × 7 | s, a, b, c, da/ds, db/ds, dc/ds |

`graph_from_record` currently supports a=1-s, b=s, c=0. It explicitly rejects
nonzero catalysts because an XX driver graph has not been added to this feature
contract. It does not silently model a different Hamiltonian. Custom drivers
and objective/device conditioning require extending the input contract before
combining those experiments in one model. Pilot training assumes one declared
objective and one ideal driver family. `programmed_scale` defaults to one only
for legacy dimensionless records; generated records should always provide it.

No ground states, eigenvalues, moments, outcomes, record IDs, or parent/split IDs
enter the encoder. The response sampling positions are queries; policy and
critic inference do not use response queries or labels. Signed channels remain
present at zero field. This is a signed baseline, not an exactly gauge-invariant
network and not a guaranteed classifier of arbitrary long-loop frustration.

## Three heads

1. The auxiliary head predicts three nonnegative response moments using
   softplus. A log1p-Huber loss ignores nonfinite, negative, or masked labels.
2. The direct head predicts three waveform proposals and mixture logits. A
   capped-simplex water-filling decoder guarantees monotonicity, endpoints 0/1,
   and a declared maximum normalized slope. Default: nine equally spaced time
   knots and max ds/dtau=4. This is not a D-Wave schedule validator: actual time,
   hardware slope limits, minimum segment spacing, and knot limits must be
   checked again by the deployment control layer. Strict pauses and ordered
   bang controls are not separate learned action families in this pilot.
3. The critic conditions on the actual nine waveform values and their eight
   segment slopes, attends to graph tokens, and predicts candidate loss. It
   does not condition solely on nominal slow-window parameters.

The direct policy is distilled against a soft distribution over all stored
candidate losses: lower loss has higher mass, with ties retaining multiple
modes. A mixture of waveform-distance kernels evaluated on the candidate bank
is the implemented surrogate. It is not mean-parameter regression; it is also
not differentiable Schrödinger propagation or proof that off-bank proposals
have good outcomes. Its bandwidth must be tuned on validation. Small datasets
can still exhibit mode collapse. The critic includes Huber outcomes and pairwise
ranking, omitting differences below a declared ranking tolerance.
When records contain explicit loss uncertainties, pairwise differences below
the sum of the two uncertainties are omitted and label temperature is not set
below the largest uncertainty. For pilot success-loss records with only a
step-doubling state-distance diagnostic delta, training uses 2 delta + delta² as
an ambiguity indicator. This is a numerical heuristic, not a certified error
bound against the exact state, and is not valid unmodified for unbounded losses.

## Training and reporting boundary

`fit_records(train_records, validation_records, ...)` requires explicit split
metadata and disjoint logical-parent IDs. It fits affine normalization only on
training graphs, preserves runtime/scale as recoverable channels, uses AdamW
and gradient clipping, and selects checkpoints by validation finite-bank regret.
`FeatureNormalizer.fit` is a lower-level utility; callers bypassing `fit_records`
are responsible for the same train-only restriction.

Checkpoints contain model configuration, weights, train-only statistics, best
epoch, training configuration, optimizer state, and split provenance. The loader
uses PyTorch `weights_only=True`. A checkpoint is sufficient for inference; exact
mid-epoch continuation and distributed sampler/RNG state are not implemented.

`evaluate_records` ranks a **fixed pre-evaluated bank**. Its regret is relative to
that bank only. It never assigns a direct proposal the label of its nearest
candidate, never interpolates unknown physical outcomes, and never calls its
finite-bank reference a continuous-control optimum. Direct proposals must be
decoded, validated, and propagated by the true simulator for separate evaluation.
Validation-bank early stopping evaluates critic selection, not direct proposals.

To run outcome-only ablations, set `policy_weight=0` and `response_weight=0`;
keep graph width, layer count, tuning budget, training data, and evaluation bank
matched. The auxiliary and policy heads remain present but untrained. Logical-only,
flat physical, summary-statistics, invariant-gauge, device-adapter, uncertainty,
adjoint-intervention, distributed training, and hardware likelihood comparisons
remain separate milestones. No such result is claimed by this pilot.

Tests cover physical/logical permutation invariance, zero-field sign sensitivity,
bounded monotone decoding, finite gradients, soft-target sign, two-mode kernel
distillation, unresolved response masking, train-only normalization, parent-split
leakage checks, oracle-free inputs, and a tiny train/checkpoint/evaluation cycle.
If Torch is unavailable, these tests are visibly skipped, not counted as passes.
