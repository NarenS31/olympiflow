"""XTraffic ST-GNN — Graph WaveNet base + our three paper modifications.

WHY Graph WaveNet (Wu et al., IJCAI 2019) as the base:
Traffic prediction needs to model (a) how congestion spreads *across* the road
network (spatial) and (b) how it evolves *over time* (temporal). Graph WaveNet is
the cleanest architecture that does both: dilated causal 1-D convolutions along
time, graph convolutions across space, stacked in residual blocks. We build it
from the paper description (not by copying a repo) so we understand every line,
then bolt on three contributions:

  MOD 1 — Semantic co-movement edges: a learned graph on top of the physical one,
          blended by a single learnable knob alpha. Lets the model discover pairs
          of roads that move together even without a physical connection.
  MOD 2 — Multi-scale temporal convolution: parallel dilated branches (1,2,4,8)
          inside each block instead of one dilation, so a single block sees several
          time horizons at once.
  MOD 3 — Heterogeneous feature fusion: a per-modality encoder (traffic / weather /
          events / transit) with learnable gates, so missing modalities zero out
          instead of crashing — essential for cross-city transfer (Chicago lacks
          some feeds).

Layout convention inside the network: tensors are CHANNELS-FIRST as
    x : [B, C, N, T]
    B = batch, C = feature channels, N = nodes, T = time steps.
This matches nn.Conv2d, which convolves over the last two dims (N, T); we only ever
convolve along T (kernel width 1 on N), so a conv is a per-node temporal filter.

Python 3.9 compatible (typing.Optional / Dict, no `X | Y`).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# MOD 3 — Heterogeneous feature fusion
# ---------------------------------------------------------------------------
class HeteroFusion(nn.Module):
    """Encode each data modality separately, then sum with learnable gates.

    Early fusion: every modality is projected to the same `out_channels` width by
    its own small linear encoder, multiplied by a per-modality sigmoid gate, and
    summed. A modality that is absent for a city is passed as None -> it simply
    contributes nothing (its gate term is skipped). This is what lets a model
    trained on LA (all modalities) run on Chicago (fewer) with zero code changes.

    modality_dims maps a modality name -> number of input feature channels it has.
    For Phase 2 only "traffic" (speed + time-of-day = 2 channels) carries real data;
    weather/events/transit are declared here but fed None until Phase 6.
    """

    def __init__(self, modality_dims: Dict[str, int], out_channels: int):
        super().__init__()
        # One 1x1 conv (== per-node, per-timestep linear layer) per modality.
        # Conv2d with kernel (1,1) over [B, C_in, N, T] -> [B, out_channels, N, T].
        self.encoders = nn.ModuleDict(
            {name: nn.Conv2d(dim, out_channels, kernel_size=(1, 1))
             for name, dim in modality_dims.items()}
        )
        # One scalar gate parameter per modality, squashed by sigmoid at use time.
        # Init at 0.0 -> sigmoid(0)=0.5, a neutral half-open gate the model can
        # push toward 0 (ignore) or 1 (rely on) during training.
        self.gate_logits = nn.ParameterDict(
            {name: nn.Parameter(torch.zeros(1)) for name in modality_dims}
        )

    def forward(self, modalities: Dict[str, Optional[torch.Tensor]]) -> torch.Tensor:
        # modalities[name] : [B, C_name, N, T]  or None if that feed is absent.
        fused = None  # will become [B, out_channels, N, T]
        for name, encoder in self.encoders.items():
            x = modalities.get(name, None)
            if x is None:
                continue  # missing modality -> contributes nothing (no crash)
            gate = torch.sigmoid(self.gate_logits[name])   # scalar in (0,1)
            enc = encoder(x) * gate                         # [B, out_channels, N, T]
            fused = enc if fused is None else fused + enc
        if fused is None:
            raise ValueError("HeteroFusion received no present modalities.")
        return fused  # [B, out_channels, N, T]

    def gate_values(self) -> Dict[str, float]:
        """Current sigmoid(gate) per modality — logged over training (paper figure)."""
        return {name: float(torch.sigmoid(p).item())
                for name, p in self.gate_logits.items()}


# ---------------------------------------------------------------------------
# Graph convolution (diffusion) over a set of support matrices
# ---------------------------------------------------------------------------
class GraphConv(nn.Module):
    """Order-K diffusion graph convolution over one or more [N, N] supports.

    For each support A we form A^1 x, A^2 x, ... A^order x (information diffusing
    1,2,...,order hops out) plus the input itself, concatenate all these along the
    channel dim, and mix them with a 1x1 conv. This is the Graph WaveNet GCN.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 n_supports: int, order: int = 2, dropout: float = 0.3):
        super().__init__()
        self.order = order
        self.dropout = dropout
        # Input to the final mixer = original x + (order hops) * (n_supports) copies.
        mix_in = in_channels * (1 + order * n_supports)
        self.mix = nn.Conv2d(mix_in, out_channels, kernel_size=(1, 1))

    @staticmethod
    def _nconv(x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # Diffuse features one hop: for each node m, sum over neighbours n of A[n,m].
        # x:[B,C,N,T], A:[N,N] -> [B,C,N,T]. einsum over the node axis.
        return torch.einsum("bcnt,nm->bcmt", (x, A)).contiguous()

    def forward(self, x: torch.Tensor, supports: List[torch.Tensor]) -> torch.Tensor:
        out = [x]                       # keep the 0-hop (self) term
        for A in supports:              # each A : [N, N]
            xk = x
            for _ in range(self.order):
                xk = self._nconv(xk, A)  # [B,C,N,T] diffused one more hop
                out.append(xk)
        h = torch.cat(out, dim=1)        # [B, C*(1+order*n_supports), N, T]
        h = self.mix(h)                  # [B, out_channels, N, T]
        h = F.dropout(h, self.dropout, training=self.training)
        return h


# ---------------------------------------------------------------------------
# MOD 2 — Multi-scale temporal convolution (one spatial-temporal block)
# ---------------------------------------------------------------------------
class STBlock(nn.Module):
    """One spatial-temporal block: multi-scale gated TCN, then graph conv.

    Base Graph WaveNet uses ONE dilated causal conv per block. MOD 2 runs FOUR in
    parallel at dilations [1,2,4,8] and concatenates them, so a single block sees
    multiple temporal ranges simultaneously. At 5-min resolution with kernel width
    2, the receptive field of a dilation-d branch is 2*d steps back:
        dilation 1 -> ~10 min,  2 -> ~20 min,  4 -> ~40 min,  8 -> ~80 min.
    We LEFT-pad each branch so the output time length is preserved (causal: no peek
    at the future), which keeps every block's T fixed at 12 and avoids the
    receptive-field length bookkeeping of vanilla Graph WaveNet — easier to read
    and shape-safe.
    """

    KERNEL = 2  # temporal kernel width

    def __init__(self, residual_channels: int, dilation_channels: int,
                 skip_channels: int, n_supports: int, gcn_order: int,
                 dropout: float, dilations=(1, 2, 4, 8)):
        super().__init__()
        # MOD 2 ablation knob: `dilations` is the set of parallel temporal scales.
        # Default (1,2,4,8) = full multi-scale; the ablation passes (1,) = a single
        # dilation (vanilla single-scale temporal conv) to measure MOD 2's value.
        self.dilations = tuple(dilations)
        # Per-branch gated conv: a "filter" conv (tanh) and a "gate" conv (sigmoid).
        # Each branch outputs dilation_channels; k branches -> concat k*dilation.
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        for d in self.dilations:
            self.filter_convs.append(
                nn.Conv2d(residual_channels, dilation_channels,
                          kernel_size=(1, self.KERNEL), dilation=(1, d)))
            self.gate_convs.append(
                nn.Conv2d(residual_channels, dilation_channels,
                          kernel_size=(1, self.KERNEL), dilation=(1, d)))
        multi = dilation_channels * len(self.dilations)  # concatenated width

        # Project the concatenated multi-scale features back to residual width,
        # then hand to the graph conv (spatial mixing).
        self.gconv = GraphConv(multi, residual_channels, n_supports,
                               order=gcn_order, dropout=dropout)
        # Skip path: a 1x1 conv exporting this block's contribution to the readout.
        self.skip_conv = nn.Conv2d(multi, skip_channels, kernel_size=(1, 1))
        self.bn = nn.BatchNorm2d(residual_channels)

    def forward(self, x: torch.Tensor, supports: List[torch.Tensor]):
        # x : [B, residual_channels, N, T]
        residual = x
        branches = []
        for d, fconv, gconv in zip(self.dilations, self.filter_convs, self.gate_convs):
            pad = (self.KERNEL - 1) * d          # left pad => causal, T preserved
            xp = F.pad(x, (pad, 0))              # pad only the time (last) dim on left
            filt = torch.tanh(fconv(xp))         # [B, dilation_channels, N, T]
            gate = torch.sigmoid(gconv(xp))      # [B, dilation_channels, N, T]
            branches.append(filt * gate)         # gated activation
        h = torch.cat(branches, dim=1)           # [B, 4*dilation_channels, N, T]

        skip = self.skip_conv(h)                 # [B, skip_channels, N, T]
        h = self.gconv(h, supports)              # [B, residual_channels, N, T]
        h = h + residual                         # residual connection
        h = self.bn(h)                           # stabilise training
        return h, skip


# ---------------------------------------------------------------------------
# The full model
# ---------------------------------------------------------------------------
class XTrafficSTGNN(nn.Module):
    """Graph WaveNet + semantic edges (MOD 1) + multi-scale (MOD 2) + fusion (MOD 3).

    forward() accepts a modality dict (MOD 3) and returns z-scored speed forecasts
    of shape [B, T_out, N]. Loss/metrics inverse-transform to mph outside the model.
    """

    def __init__(self, num_nodes: int, physical_adj: torch.Tensor,
                 modality_dims: Dict[str, int],
                 residual_channels: int = 32, dilation_channels: int = 32,
                 skip_channels: int = 64, end_channels: int = 128,
                 n_blocks: int = 4, embed_dim: int = 10,
                 gcn_order: int = 2, dropout: float = 0.3,
                 out_len: int = 12,
                 use_semantic: bool = True, use_multiscale: bool = True):
        super().__init__()
        self.num_nodes = num_nodes
        self.out_len = out_len
        # --- Phase 7 ablation switches (default True = full model, unchanged). ---
        # use_semantic  : MOD 1. False -> the first support is the plain physical
        #                 adjacency (no learned semantic co-movement blend).
        # use_multiscale: MOD 2. False -> each block uses a single temporal dilation
        #                 (1,) instead of the parallel (1,2,4,8) multi-scale set.
        self.use_semantic = use_semantic
        dilations = (1, 2, 4, 8) if use_multiscale else (1,)

        # --- physical adjacency: a fixed [N,N] buffer, row-normalised so a GCN hop
        #     is a weighted average over neighbours (moves with the model to GPU). ---
        A = physical_adj.clone().float()
        A = A + torch.eye(num_nodes)             # add self-loops
        A = A / A.sum(dim=1, keepdim=True).clamp(min=1e-6)  # row-normalise
        self.register_buffer("physical_adj", A)  # [N, N]

        # --- MOD 3: fusion front-end. Output width = residual_channels. ---
        self.fusion = HeteroFusion(modality_dims, residual_channels)

        # --- MOD 1: semantic co-movement embeddings E, and the blend knob alpha. ---
        # A_sem = softmax(relu(E @ E^T)); blended A_final = a*A_phys + (1-a)*A_sem,
        # a = sigmoid(alpha_logit) so it is a single scalar always in (0,1).
        self.sem_embed = nn.Parameter(torch.randn(num_nodes, embed_dim) * 0.01)
        self.alpha_logit = nn.Parameter(torch.zeros(1))  # sigmoid(0)=0.5 to start

        # --- base Graph WaveNet adaptive adjacency (a second, purely learned graph) ---
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, embed_dim) * 0.01)
        self.nodevec2 = nn.Parameter(torch.randn(num_nodes, embed_dim) * 0.01)

        # supports fed to every block: [A_final (phys+semantic), A_adaptive] -> 2
        n_supports = 2
        self.blocks = nn.ModuleList([
            STBlock(residual_channels, dilation_channels, skip_channels,
                    n_supports, gcn_order, dropout, dilations=dilations)
            for _ in range(n_blocks)
        ])

        # --- readout head: collapse time to 1 with a (1, T) conv, then map to T_out. ---
        self.end_conv_1 = nn.Conv2d(skip_channels, end_channels, kernel_size=(1, 1))
        self.time_collapse = nn.Conv2d(end_channels, end_channels,
                                       kernel_size=(1, out_len))  # T -> 1
        self.end_conv_2 = nn.Conv2d(end_channels, out_len, kernel_size=(1, 1))

    # -- MOD 1 helper: build the blended physical+semantic support --
    def _semantic_support(self) -> torch.Tensor:
        # MOD 1 ablation: with semantic edges OFF, the support is just the physical
        # adjacency (no learned co-movement graph blended in).
        if not self.use_semantic:
            return self.physical_adj                                           # [N,N]
        A_sem = F.softmax(F.relu(self.sem_embed @ self.sem_embed.t()), dim=1)  # [N,N]
        a = torch.sigmoid(self.alpha_logit)                                    # scalar
        return a * self.physical_adj + (1.0 - a) * A_sem                       # [N,N]

    def _adaptive_support(self) -> torch.Tensor:
        return F.softmax(F.relu(self.nodevec1 @ self.nodevec2.t()), dim=1)     # [N,N]

    def current_alpha(self) -> float:
        """sigmoid(alpha) — logged each epoch (physical-vs-semantic reliance figure)."""
        return float(torch.sigmoid(self.alpha_logit).item())

    def forward(self, modalities: Dict[str, Optional[torch.Tensor]]) -> torch.Tensor:
        # Each present modality arrives as [B, T, N, C]; convert to channels-first
        # [B, C, N, T] that the conv stack expects.
        mods_cf = {}
        for name, x in modalities.items():
            if x is None:
                mods_cf[name] = None
            else:
                mods_cf[name] = x.permute(0, 3, 2, 1).contiguous()  # [B,C,N,T]

        x = self.fusion(mods_cf)                     # [B, residual_channels, N, T]

        supports = [self._semantic_support(), self._adaptive_support()]  # 2 x [N,N]

        skip_total = 0
        for block in self.blocks:
            x, skip = block(x, supports)             # x:[B,res,N,T] skip:[B,skip,N,T]
            skip_total = skip + skip_total           # accumulate skip contributions

        h = F.relu(skip_total)                       # [B, skip_channels, N, T]
        h = F.relu(self.end_conv_1(h))               # [B, end_channels, N, T]
        h = self.time_collapse(h)                    # [B, end_channels, N, 1]
        h = self.end_conv_2(h)                       # [B, out_len,   N, 1]
        out = h.squeeze(-1)                          # [B, out_len, N]
        return out                                   # z-scored speed forecasts
