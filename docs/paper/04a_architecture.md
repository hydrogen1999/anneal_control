# 4a. Architecture as implemented

Source of truth: `src/annealctrl/models.py`. Every dimension below is the
default used in every experiment reported in the paper.

## 4a.1 Inputs

The model never sees an outcome, a spectrum, or a solution. It sees the
compiled problem and the task conditions.

**Physical node features** (7 per qubit) — all derived from the *embedded*
Hamiltonian:

| # | feature | why it is there |
|---|---|---|
| 1 | $h_i$ | signed local field |
| 2 | $\lvert h_i\rvert$ | magnitude, separately, so sign is not the only route |
| 3 | degree | |
| 4 | signed load $\sum_j J_{ij}$ | frustration is a signed quantity |
| 5 | absolute load $\sum_j \lvert J_{ij}\rvert$ | |
| 6 | **internal degree** | edges to the same chain |
| 7 | **boundary load** $\sum_j \lvert J_{ij}\rvert\,[\,\text{different chain}\,]$ | the chain/problem split |

Features 6–7 are the embedding made explicit: they separate intra-chain
penalty structure from the problem couplings that cross chains.

**Physical edge features** (3): $J_{ij}$, $\lvert J_{ij}\rvert$, and a binary
**same-chain** indicator. Edges are stored once and symmetrised into both
directions.

**Logical features**: node $(h_v, \lvert h_v\rvert)$, edge
$(J_{uv}, \lvert J_{uv}\rvert)$.

**Membership** maps each physical qubit to its chain.

**Context** (2): runtime $T$ and programmed scale $\alpha$.

**Path queries** (7 per sampled $s$): $(s,\,a(s),\,b(s),\,c(s),\,a',\,b',\,c')$
— the path coefficients and their derivatives at that point, *not* any spectral
label of it.

## 4a.2 Blocks

`MLP(d_in → w → d_out)` is `Linear, SiLU, Linear` with one hidden layer of
width $w$.

**Signed message layer.** For width $w$ and edge dimension $e$:

```
m_ij = MLP(2w + e → w → w)  applied to  [x_i, x_j, edge_ij]
x_i' = LayerNorm( x_i + MLP(2w → w → w)([x_i, Σ_j m_ji]) )
```

Sum aggregation with a residual and LayerNorm. Raw **signed** edge channels
enter every message, so the layer can distinguish a frustrated loop from an
unfrustrated one even when every $h_i=0$.

**Attention.** `MultiheadAttention(width, heads = 4)`, queries attending over
the token bank.

## 4a.3 Encoder variants — the ablation

All four share the heads; they differ only in how the token bank and summary
are produced.

### `hierarchical` (full token bank)

```
x        = MLP(7 → w → w)(node_features)
x        = SignedMessageLayer × 3        on the physical graph
chain    = MLP(2w+1 → w → w)( [Σ_{i∈C} x_i , mean_{i∈C} x_i , |C|] )
logical  = MLP(w+2 → w → w)( [chain , logical_node_features] )
logical  = SignedMessageLayer × 2        on the logical graph
tokens   = [ x + τ_phys ; chain + τ_chain ; logical + τ_log ]
summary  = MLP(3w+3+2 → w → w)( [Σx , Σchain , Σlogical , sizes(3) , context(2)] )
```

$\tau$ are three learned **type** embeddings — physical, chain, logical roles,
never node identities.

### `physical`

Physical message passing only; no chain pooling, no logical graph.
`summary = MLP(w+1+2 → w → w)([Σx, |V|, context])`.

### `logical`  (the embedding-blind control)

Logical message passing only, on logical node/edge features.
`summary = MLP(w+2 → w → w)([Σ logical, |V_L|, context[:1]])`.

Note **`context[:1]`**: this branch receives the runtime and **not the
programmed scale**. Chain cardinalities, the physical graph, compiled
coefficients and $\alpha$ all stop here. That is what makes it a control on
*embedding information* rather than a smaller network.

### `summary`  (statistics, no message passing)

Mean and standard deviation of each of the four feature blocks, plus node and
edge counts, chain-length mean and max, and the context:

```
dim = 2(7 + 3 + 2 + 2) + 6 + 2 = 36  →  MLP(36 → w → w)
tokens = summary.unsqueeze(0)     # a single token
```

This variant is embedding-aware — chain lengths and the same-chain channel are
in its statistics — while doing no graph computation at all. §5.1 finds it is
not separated from `hierarchical`.

## 4a.4 Heads

**Spectral auxiliary.** `query = MLP(7 → w → w)(path_queries) + summary`;
`response = softplus( MLP(2w → w → r)( [attend(query, tokens), query] ) )`,
with $r = 3$ moments (or 8 bins, §7.1).

**Policy.** A learned query vector $q_\pi\in\mathbb R^{w}$:

```
features  = [ attend(q_π + summary, tokens) , summary ]
raw       = MLP(2w → w → 3 × 9)(features) → reshape (3 proposals, 9 points)
proposals = monotone_samples(raw[:, :-1])
```

**`monotone_samples` is where feasibility is enforced, by construction rather
than by penalty.** Capped-simplex water filling turns logits into increments
that are non-negative, sum to one, and are each at most
$\max(\mathrm ds/\mathrm d\tau)/n$. Every emitted waveform therefore satisfies
the slope bound exactly; no proposal has to be rejected for infeasibility.

**Critic.** For a batch of candidate waveforms $\theta$ on the common uniform
$\tau$ grid:

```
features  = [ θ , diff(θ)·(n−1) ]          # values AND slopes, 2·9−1 = 17
action    = MLP(17 → w → w)(features)
attended  = attend(action + summary, tokens)
Q̂(θ)      = MLP(3w → w → 1)( [action, attended, summary] )
```

The critic is instance-conditioned twice: through attention over the token
bank and through the summary concatenated directly. It consumes a waveform as
**actual values and slopes**, not as fitted family parameters, so it can score
a candidate from any family — including ones no policy generated.

## 4a.5 The optional bottleneck (§7.2)

When `bottleneck_dim = k`, every route from graph to control is forced through
$k$ non-negative numbers:

```
profile = softplus(Linear(w → k)(summary))
squeezed = MLP(k → w → w)(profile)
tokens, summary = squeezed.reshape(1, −1), squeezed
```

The token bank is collapsed to a **single** token so attention cannot carry
per-node information around the constriction. These two layers **add**
parameters, so a loss measured here cannot be attributed to a smaller model —
which is the point of the ablation.

## 4a.6 Parameter count and cost

Width 64, 3 physical and 2 logical message layers, 4 attention heads, 3
proposals, 9 schedule points. Deployment is one encoder pass plus one critic
pass per candidate; candidates are scored in a batch. **No simulation and no
eigendecomposition occur at deployment**, which is what places the method in
the `amortised` cost class (§2.4).
