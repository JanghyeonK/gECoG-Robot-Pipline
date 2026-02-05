# models/multitask_binary.py
import torch
import torch.nn as nn
from typing import Tuple

from .mappers import build_mapper
from .shared_mlp import SharedMLP

class MultiTaskBinaryModel(nn.Module):
    """
    Pipeline:
      - mapper:  (B,5,32,H,W) -> (B,160)
      - gating:  z_scalar * info_vectors[i]  # (7,160) non-trainable buffer, differs per head
      - shared:  (B,160) -> (B,N_shared)
      - concat:  (B,160+N_shared)
      - heads:   7 binary logits -> (B,7)
    """
    def __init__(
        self,
        n_bands: int = 5,
        n_channels: int = 32,
        shared_out_dims: int = 128,
        hidden_dims: Tuple[int, int, int] = (256, 256, 256),
        num_heads: int = 7,
        mapper_name="rescnn",   # <- select by string
    ):
        super().__init__()
        self.mapper = build_mapper(mapper_name, n_bands=n_bands, n_channels=n_channels)
        self.shared = SharedMLP(in_dim=n_bands * n_channels,
                                out_dim=shared_out_dims,
                                hidden_dims=hidden_dims)

        self.c160 = n_bands * n_channels
        self.concat_dim = self.c160 + shared_out_dims
        self.num_heads = num_heads

        # === Per-classifier information vectors (7,160), non-trainable ===
        self.register_buffer("info_vectors", torch.ones(num_heads, self.c160))

        # === Classification heads ===
        self.heads = nn.ModuleList([
            nn.Linear(self.concat_dim, 1) for _ in range(num_heads)
        ])

    def forward(self, x: torch.Tensor):
        """
        x: (B, 5, 32, H, W)
        return:
          logits: (B, num_heads)
          aux: dict
        """
        z_scalar = self.mapper(x)  # (B,160)

        logits_list = []
        for i, head in enumerate(self.heads):
            z_scalar_g = z_scalar * self.info_vectors[i].unsqueeze(0)  # (B,160)
            z_shared_i = self.shared(z_scalar_g)                       # (B,N_shared)
            z_all_i = torch.cat([z_scalar_g, z_shared_i], dim=1)       # (B,160+N_shared)
            logits_list.append(head(z_all_i))                           # (B,1)

        logits = torch.cat(logits_list, dim=1)  # (B,num_heads)
        return logits, {"z_scalar": z_scalar}

    # ===== Set information vectors (7,160) from external source =====
    @torch.no_grad()
    def set_info_vectors(self, vecs: torch.Tensor):
        """ vecs: (num_heads, 160) """
        vecs = vecs.reshape(self.num_heads, self.c160).to(
            self.info_vectors.device, self.info_vectors.dtype
        )
        self.info_vectors.copy_(vecs)
