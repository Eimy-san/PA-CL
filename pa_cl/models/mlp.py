"""2-layer / 3-layer MLP used for Continual-Permuted-MNIST in the
loss-of-plasticity protocol of Dohare et al. (2024, Nature).

The hidden layers are named 'h1', 'h2', ... so PA-CL hooks can find
them by name from the YAML config.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int = 784,
        hidden_dims: tuple[int, ...] = (2000, 2000),
        out_dim: int = 10,
        bias: bool = True,
        act: str = "relu",
    ):
        super().__init__()
        self.act = nn.ReLU() if act == "relu" else nn.GELU()
        prev = in_dim
        self.hidden = nn.ModuleDict()
        for i, h in enumerate(hidden_dims, start=1):
            self.hidden[f"h{i}"] = nn.Linear(prev, h, bias=bias)
            prev = h
        self.head = nn.Linear(prev, out_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.flatten(start_dim=1)
        h = x
        for name, layer in self.hidden.items():
            h = self.act(layer(h))
        return self.head(h)