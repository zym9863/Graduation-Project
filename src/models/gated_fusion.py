from __future__ import annotations

import torch
from torch import nn


class GatedFusionLayer(nn.Module):
    def __init__(self, siglip_dim: int, news_value_dim: int, output_dim: int) -> None:
        super().__init__()
        self.siglip_projection = nn.Linear(siglip_dim * 2, output_dim)
        self.value_projection = nn.Linear(news_value_dim, output_dim)
        self.siglip_gate = nn.Linear(siglip_dim * 2, output_dim)
        self.value_gate = nn.Linear(news_value_dim, output_dim)

    def forward(self, text_emb: torch.Tensor, image_emb: torch.Tensor, news_value: torch.Tensor) -> torch.Tensor:
        siglip_feature = torch.cat([text_emb, image_emb], dim=-1)
        siglip_repr = self.siglip_projection(siglip_feature)
        value_repr = self.value_projection(news_value)
        gate = torch.sigmoid(self.siglip_gate(siglip_feature) + self.value_gate(news_value))
        return gate * siglip_repr + (1.0 - gate) * value_repr