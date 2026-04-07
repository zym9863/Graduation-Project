from __future__ import annotations

import torch
from torch import nn


class CrossModalCrossAttentionLayer(nn.Module):
    def __init__(
        self,
        siglip_dim: int,
        news_value_dim: int,
        output_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=siglip_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_projection = nn.Linear(siglip_dim, output_dim)
        self.value_projection = nn.Linear(news_value_dim, output_dim)
        self.attn_gate = nn.Linear(siglip_dim, output_dim)
        self.value_gate = nn.Linear(news_value_dim, output_dim)

    def forward(self, text_emb: torch.Tensor, image_emb: torch.Tensor, news_value: torch.Tensor) -> torch.Tensor:
        # Single-step cross-attention: text is query, image is key/value.
        query = text_emb.unsqueeze(1)
        key_value = image_emb.unsqueeze(1)
        attended, _ = self.cross_attention(query, key_value, key_value)
        attended = attended.squeeze(1)

        attn_repr = self.attn_projection(attended)
        value_repr = self.value_projection(news_value)
        gate = torch.sigmoid(self.attn_gate(attended) + self.value_gate(news_value))
        return gate * attn_repr + (1.0 - gate) * value_repr
