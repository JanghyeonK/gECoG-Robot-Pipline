# models/shared_mlp.py
import torch
import torch.nn as nn
from typing import Tuple

class SharedMLP(nn.Module):
    """
    4-layer MLP: in_dim -> h1 -> h2 -> h3 -> n_shared
    """
    def __init__(
        self,
        in_dim: int = 160,
        out_dim: int = 128,
        hidden_dims: Tuple[int, int, int] = (256, 256, 256),
    ):
        super().__init__()
        assert len(hidden_dims) == 3, "hidden_dims must be 3 to build a 4-layer MLP."
        h1, h2, h3 = hidden_dims
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1), nn.GELU(),
            nn.Linear(h1, h2),     nn.GELU(),
            nn.Linear(h2, h3),     nn.GELU(),
            nn.Linear(h3, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B, out_dim)
