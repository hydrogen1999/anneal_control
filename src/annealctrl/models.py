"""Small signed, hierarchical controller; PyTorch is an optional dependency.

No eigenvalues, ground states, measured outcomes, parent IDs, or split IDs enter
the model. GraphInput describes one programmed system, not a padded minibatch.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import math
from typing import Mapping, Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class GraphInput:
    node_features: Tensor
    edge_index: Tensor
    edge_features: Tensor
    membership: Tensor
    logical_node_features: Tensor
    logical_edge_index: Tensor
    logical_edge_features: Tensor
    context: Tensor
    path_queries: Tensor

    def to(self, device: str | torch.device) -> "GraphInput":
        return GraphInput(**{f.name: getattr(self, f.name).to(device) for f in fields(self)})

    def validate(self) -> None:
        n, l = len(self.node_features), len(self.logical_node_features)
        if n == 0 or l == 0 or self.membership.shape != (n,):
            raise ValueError("Nonempty physical/logical nodes and membership[N] required")
        if self.membership.dtype != torch.long:
            raise ValueError("membership must use torch.long")
        if not torch.equal(torch.unique(self.membership), torch.arange(l, device=self.membership.device)):
            raise ValueError("Every logical node must have a nonempty, contiguously indexed chain")
        for idx, feat, count in ((self.edge_index, self.edge_features, n),
                                 (self.logical_edge_index, self.logical_edge_features, l)):
            if idx.ndim != 2 or idx.shape[0] != 2 or idx.shape[1] != len(feat):
                raise ValueError("Expected edge_index[2,E], edge_features[E,F]")
            if idx.dtype != torch.long or (idx.numel() and (idx.min() < 0 or idx.max() >= count)):
                raise ValueError("Invalid edge indices")
        for f in fields(self):
            value = getattr(self, f.name)
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError(f"Nonfinite model input: {f.name}")


def graph_from_record(record: Mapping[str, Any], *, device: str | torch.device = "cpu") -> GraphInput:
    """Convert the pilot NPZ schema without consulting outcome/response labels.

    Undirected edge arrays are [E,2], each pair stored once. The initial helper
    supports a(s)=1-s,b(s)=s, no catalyst; other drivers must use GraphInput
    explicitly with a corresponding extension of the edge feature contract.
    """
    def tensor(key: str, dtype: torch.dtype = torch.float32) -> Tensor:
        return torch.as_tensor(record[key], dtype=dtype, device=device)

    if float(record.get("catalyst_strength", 0.0)) != 0.0:
        raise ValueError("Nonzero catalyst requires an explicit driver-graph feature extension")
    if float(record.get("energy_scale", 1.0)) != 1.0:
        raise ValueError("Nonunit energy_scale requires an explicit path-context feature extension")
    h, j = tensor("physical_h"), tensor("physical_J")
    edges = tensor("physical_edges", torch.long).reshape(-1, 2)
    membership = tensor("membership", torch.long)
    if len(edges) != len(j):
        raise ValueError("physical_edges and physical_J lengths differ")
    idx = torch.cat((edges.T, edges.flip(1).T), dim=1)
    jj = torch.cat((j, j))
    same_chain = (membership[idx[0]] == membership[idx[1]]).float()
    degree, signed_load, absolute_load, internal_degree, boundary_load = [torch.zeros_like(h) for _ in range(5)]
    for target, values in ((degree, torch.ones_like(jj)), (signed_load, jj),
                           (absolute_load, jj.abs()), (internal_degree, same_chain),
                           (boundary_load, jj.abs() * (1 - same_chain))):
        target.index_add_(0, idx[1], values)
    nodes = torch.stack((h, h.abs(), degree, signed_load, absolute_load,
                         internal_degree, boundary_load), dim=-1)
    edge_features = torch.stack((jj, jj.abs(), same_chain), dim=-1)
    logical_h, logical_j = tensor("logical_h"), tensor("logical_J")
    logical_edges = tensor("logical_edges", torch.long).reshape(-1, 2)
    if len(logical_edges) != len(logical_j):
        raise ValueError("logical_edges and logical_J lengths differ")
    logical_idx = torch.cat((logical_edges.T, logical_edges.flip(1).T), dim=1)
    logical_jj = torch.cat((logical_j, logical_j))
    # s positions are queries, not their expensive spectral labels.
    s = torch.as_tensor(record.get("response_s", [0.0, 0.25, 0.5, 0.75, 1.0]),
                        dtype=torch.float32, device=device)
    zeros, ones = torch.zeros_like(s), torch.ones_like(s)
    queries = torch.stack((s, 1 - s, s, zeros, -ones, ones, zeros), dim=-1)
    scale = float(record.get("programmed_scale", record.get("physical_scale", 1.0)))
    runtime = float(record["runtime"])
    if runtime <= 0 or scale <= 0:
        raise ValueError("runtime and programmed_scale must be positive")
    graph = GraphInput(nodes, idx, edge_features, membership,
                       torch.stack((logical_h, logical_h.abs()), dim=-1), logical_idx,
                       torch.stack((logical_jj, logical_jj.abs()), dim=-1),
                       torch.tensor([runtime, scale], dtype=torch.float32, device=device), queries)
    graph.validate()
    return graph


def _mlp(input_dim: int, width: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, width), nn.SiLU(), nn.Linear(width, output_dim))


class SignedMessageLayer(nn.Module):
    """Messages retain raw signed edge channels even when every h_i is zero."""
    def __init__(self, width: int, edge_dim: int):
        super().__init__()
        self.message = _mlp(2 * width + edge_dim, width, width)
        self.update = _mlp(2 * width, width, width)
        self.norm = nn.LayerNorm(width)

    def forward(self, x: Tensor, edge_index: Tensor, edge_features: Tensor) -> Tensor:
        src, dst = edge_index
        aggregate = torch.zeros_like(x)
        if src.numel():
            messages = self.message(torch.cat((x[src], x[dst], edge_features), dim=-1))
            aggregate.index_add_(0, dst, messages)
        return self.norm(x + self.update(torch.cat((x, aggregate), dim=-1)))


def monotone_samples(logits: Tensor, *, max_ds_dtau: float = 4.0) -> Tensor:
    """Decode to equal-time piecewise-linear samples with bounded segment slope.

    Capped-simplex water filling gives increments >=0, sum=1, and increments
    <=max_ds_dtau/n. Differentiable almost everywhere; knot count is fixed.
    This is a normalized-time envelope, not a device-specific approval check.
    """
    bins = logits.shape[-1]
    if bins < 1 or not math.isfinite(max_ds_dtau) or max_ds_dtau < 1 or not torch.isfinite(logits).all():
        raise ValueError("Finite logits and max_ds_dtau >= 1 required")
    if max_ds_dtau == 1:
        increments = torch.ones_like(logits) / bins + logits * 0
    else:
        cap = min(float(max_ds_dtau) / bins, 1.0)
        # Clamping relative logits avoids underflow after high-weight bins saturate.
        weights = torch.exp((logits - logits.max(dim=-1, keepdim=True).values).clamp(min=-30))
        saturated = torch.zeros_like(logits, dtype=torch.bool)
        increments = weights / weights.sum(dim=-1, keepdim=True)
        for _ in range(bins):
            free_weights = weights * (~saturated)
            remaining = (1 - cap * saturated.sum(dim=-1, keepdim=True)).clamp(min=0)
            free = remaining * free_weights / free_weights.sum(dim=-1, keepdim=True).clamp_min(1e-30)
            increments = torch.where(saturated, torch.full_like(free, cap), free)
            new_saturated = saturated | (free > cap)
            if torch.equal(new_saturated, saturated):
                break
            saturated = new_saturated
    cumulative = increments.cumsum(dim=-1)
    # Set the endpoint exactly; tiny floating error in the last slope is tolerated.
    return torch.cat((torch.zeros_like(cumulative[..., :1]), cumulative[..., :-1],
                      torch.ones_like(cumulative[..., :1])), dim=-1)


class AnnealController(nn.Module):
    """Physical -> chain -> logical token bank with three cooperating heads."""
    def __init__(self, *, node_dim: int = 7, edge_dim: int = 3,
                 logical_node_dim: int = 2, logical_edge_dim: int = 2,
                 context_dim: int = 2, query_dim: int = 7, width: int = 64,
                 physical_layers: int = 3, logical_layers: int = 2,
                 proposals: int = 3, schedule_points: int = 9,
                 response_dim: int = 3, max_ds_dtau: float = 4.0,
                 encoder_variant: str = "hierarchical"):
        super().__init__()
        dimensions = (node_dim, edge_dim, logical_node_dim, logical_edge_dim, context_dim,
                      query_dim, width, proposals, schedule_points, response_dim)
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in dimensions):
            raise ValueError("Feature dimensions, width, proposals, and schedule_points must be positive integers")
        if width % 4 or width < 4 or proposals < 1 or schedule_points < 2:
            raise ValueError("width must be divisible by four; positive proposals and >=2 points")
        if encoder_variant not in {"hierarchical", "physical", "logical", "summary"}:
            raise ValueError("encoder_variant must be hierarchical, physical, logical, or summary")
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in (physical_layers, logical_layers)):
            raise ValueError("Layer counts must be nonnegative and response_dim positive")
        if not math.isfinite(max_ds_dtau) or max_ds_dtau < 1:
            raise ValueError("max_ds_dtau must be finite and at least one")
        if encoder_variant == "logical" and context_dim != 2:
            raise ValueError("Logical baseline expects context=[runtime, programmed_scale]")
        self.config = dict(node_dim=node_dim, edge_dim=edge_dim, logical_node_dim=logical_node_dim,
                           logical_edge_dim=logical_edge_dim, context_dim=context_dim,
                           query_dim=query_dim, width=width, physical_layers=physical_layers,
                           logical_layers=logical_layers, proposals=proposals,
                           schedule_points=schedule_points, response_dim=response_dim,
                           max_ds_dtau=max_ds_dtau, encoder_variant=encoder_variant)
        self.width, self.proposals, self.schedule_points = width, proposals, schedule_points
        self.max_ds_dtau, self.encoder_variant = max_ds_dtau, encoder_variant
        if encoder_variant in {"hierarchical", "physical"}:
            self.node_encoder = _mlp(node_dim, width, width)
            self.physical_layers = nn.ModuleList([SignedMessageLayer(width, edge_dim) for _ in range(physical_layers)])
        if encoder_variant in {"hierarchical", "logical"}:
            self.logical_layers = nn.ModuleList([SignedMessageLayer(width, logical_edge_dim) for _ in range(logical_layers)])
        if encoder_variant == "hierarchical":
            self.chain_encoder = _mlp(2 * width + 1, width, width)
            self.logical_encoder = _mlp(width + logical_node_dim, width, width)
            # Type embeddings are physical/chain/logical roles, never arbitrary node IDs.
            self.type_embedding = nn.Parameter(torch.randn(3, width) * 0.02)
            self.summary_encoder = _mlp(3 * width + 3 + context_dim, width, width)
        # Keep hierarchical parameter names unchanged so v0.1 checkpoints load.
        # Alternative encoders share the control heads, not hidden physical tokens.
        if encoder_variant == "logical":
            self.baseline_node_encoder = _mlp(logical_node_dim, width, width)
            self.baseline_summary_encoder = _mlp(width + 2, width, width)
        elif encoder_variant == "physical":
            self.baseline_summary_encoder = _mlp(width + 1 + context_dim, width, width)
        elif encoder_variant == "summary":
            statistics_dim = 2 * (node_dim + edge_dim + logical_node_dim + logical_edge_dim) + 6 + context_dim
            self.baseline_summary_encoder = _mlp(statistics_dim, width, width)
        self.query_encoder = _mlp(query_dim, width, width)
        self.attention = nn.MultiheadAttention(width, 4, batch_first=True, dropout=0.0)
        self.response_head = _mlp(2 * width, width, response_dim)
        self.policy_query = nn.Parameter(torch.randn(1, width) * 0.02)
        self.policy_head = _mlp(2 * width, width, proposals * schedule_points)
        # Actual waveform values and slopes, not fitted window parameters or oracles.
        self.schedule_encoder = _mlp(2 * schedule_points - 1, width, width)
        self.critic_head = _mlp(3 * width, width, 1)

    def encode(self, graph: GraphInput) -> tuple[Tensor, Tensor]:
        graph.validate()
        if self.encoder_variant == "logical":
            # No chain cardinalities, physical graph, compiled coefficients, or
            # programmed scale enter this branch. Runtime is a task condition.
            logical = self.baseline_node_encoder(graph.logical_node_features)
            for layer in self.logical_layers:
                logical = layer(logical, graph.logical_edge_index, graph.logical_edge_features)
            summary = self.baseline_summary_encoder(torch.cat((logical.sum(0),
                       logical.new_tensor([len(logical)]), graph.context[:1])))
            return logical, summary
        if self.encoder_variant == "summary":
            def moments(values: Tensor) -> Tensor:
                if len(values) == 0:
                    return values.new_zeros(2 * values.shape[-1])
                return torch.cat((values.mean(0), values.std(0, unbiased=False)))
            lengths = torch.bincount(graph.membership).to(graph.node_features.dtype)
            counts = graph.node_features.new_tensor([len(graph.node_features), graph.edge_index.shape[1] / 2,
                      len(graph.logical_node_features), graph.logical_edge_index.shape[1] / 2])
            features = torch.cat((moments(graph.node_features), moments(graph.edge_features),
                       moments(graph.logical_node_features), moments(graph.logical_edge_features),
                       counts, torch.stack((lengths.mean(), lengths.max())), graph.context))
            summary = self.baseline_summary_encoder(features)
            return summary.unsqueeze(0), summary
        x = self.node_encoder(graph.node_features)
        for layer in self.physical_layers:
            x = layer(x, graph.edge_index, graph.edge_features)
        if self.encoder_variant == "physical":
            summary = self.baseline_summary_encoder(torch.cat((x.sum(0), x.new_tensor([len(x)]), graph.context)))
            return x, summary
        l = len(graph.logical_node_features)
        chain_sum = x.new_zeros(l, self.width).index_add_(0, graph.membership, x)
        count = torch.bincount(graph.membership, minlength=l).to(x.dtype).unsqueeze(-1)
        chain = self.chain_encoder(torch.cat((chain_sum, chain_sum / count, count), dim=-1))
        logical = self.logical_encoder(torch.cat((chain, graph.logical_node_features), dim=-1))
        for layer in self.logical_layers:
            logical = layer(logical, graph.logical_edge_index, graph.logical_edge_features)
        tokens = torch.cat((x + self.type_embedding[0], chain + self.type_embedding[1],
                            logical + self.type_embedding[2]), dim=0)
        sizes = x.new_tensor([len(x), len(chain), len(logical)])
        summary = self.summary_encoder(torch.cat((x.sum(0), chain.sum(0), logical.sum(0),
                                                  sizes, graph.context), dim=-1))
        return tokens, summary

    def _attend(self, queries: Tensor, tokens: Tensor) -> Tensor:
        return self.attention(queries.unsqueeze(0), tokens.unsqueeze(0), tokens.unsqueeze(0),
                              need_weights=False)[0].squeeze(0)

    def predict_losses(self, graph: GraphInput, schedules: Tensor,
                       encoded: tuple[Tensor, Tensor] | None = None) -> Tensor:
        if schedules.ndim != 2 or schedules.shape[1] != self.schedule_points:
            raise ValueError(f"Expected schedules[B,{self.schedule_points}] on a common uniform tau grid")
        if not torch.isfinite(schedules).all():
            raise ValueError("Candidate waveform contains nonfinite values")
        tokens, summary = self.encode(graph) if encoded is None else encoded
        features = torch.cat((schedules, torch.diff(schedules, dim=-1) * (self.schedule_points - 1)), dim=-1)
        action = self.schedule_encoder(features)
        attended = self._attend(action + summary, tokens)
        return self.critic_head(torch.cat((action, attended, summary.expand(len(action), -1)), dim=-1)).squeeze(-1)

    def forward(self, graph: GraphInput, schedules: Tensor | None = None) -> dict[str, Tensor]:
        tokens, summary = self.encode(graph)
        query = self.query_encoder(graph.path_queries) + summary
        response_features = torch.cat((self._attend(query, tokens), query), dim=-1)
        response = F.softplus(self.response_head(response_features))
        policy_features = torch.cat((self._attend(self.policy_query + summary, tokens).squeeze(0), summary), dim=-1)
        raw = self.policy_head(policy_features).reshape(self.proposals, self.schedule_points)
        proposals = monotone_samples(raw[:, :-1], max_ds_dtau=self.max_ds_dtau)
        result = {"proposal_schedules": proposals, "proposal_logits": raw[:, -1], "response": response}
        if schedules is not None:
            result["predicted_losses"] = self.predict_losses(graph, schedules, (tokens, summary))
        return result
