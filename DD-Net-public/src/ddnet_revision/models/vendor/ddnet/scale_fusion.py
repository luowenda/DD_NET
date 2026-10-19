"""Sample-independent attention over neighborhood scales at each local node."""

import torch
from torch import nn


class NodeScaleGate(nn.Module):
    def __init__(self, channels, scale_gates, hidden_channels=16, adaptive=True):
        super().__init__()
        if channels <= 0 or hidden_channels <= 0:
            raise ValueError("Scale gate channel counts must be positive")
        active = torch.tensor(scale_gates, dtype=torch.bool)
        if active.ndim != 1 or not active.any():
            raise ValueError("At least one neighborhood scale must be enabled")
        self.register_buffer("active", active, persistent=False)
        self.adaptive = bool(adaptive)
        self.scorer = nn.Sequential(
            nn.Linear(channels * len(scale_gates), hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, len(scale_gates)),
        )
        # Equal initial weights recover the existing fusion without rescaling it.
        nn.init.zeros_(self.scorer[-1].weight)
        nn.init.zeros_(self.scorer[-1].bias)

    def weights(self, branches):
        if len(branches) != self.active.numel():
            raise ValueError("Scale gate requires one feature tensor per scale")
        logits = self.scorer(torch.cat(branches, dim=-1))
        if not self.adaptive:
            logits = logits * 0.0
        logits = logits.float().masked_fill(~self.active, -torch.inf)
        return logits.softmax(dim=-1).to(branches[0].dtype)

    def forward(self, branches):
        weights = self.weights(branches)
        # K * (1/K) = 1, including single-scale controls with a disabled branch.
        multipliers = weights * self.active.sum().to(weights.dtype)
        return torch.cat(
            [branch * multipliers[:, index:index + 1]
             for index, branch in enumerate(branches)],
            dim=-1,
        )
