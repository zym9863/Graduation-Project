from __future__ import annotations

import torch
from torch import nn


class AdditiveAttention(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        self.query = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        scores = self.query(torch.tanh(self.projection(inputs))).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=-1)
        if mask is not None:
            weights = weights * mask.float()
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return torch.sum(weights.unsqueeze(-1) * inputs, dim=1)


class NRMSUserEncoder(nn.Module):
    def __init__(self, news_repr_dim: int, num_attention_heads: int, dropout: float) -> None:
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            embed_dim=news_repr_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.additive_attention = AdditiveAttention(news_repr_dim)

    def forward(self, history_repr: torch.Tensor, history_mask: torch.Tensor | None = None) -> torch.Tensor:
        safe_mask = history_mask
        empty_rows = None

        if safe_mask is not None:
            safe_mask = safe_mask.clone()
            empty_rows = ~safe_mask.any(dim=1)
            if empty_rows.any():
                safe_mask[empty_rows, 0] = True

        attn_output, _ = self.self_attention(
            history_repr,
            history_repr,
            history_repr,
            key_padding_mask=None if safe_mask is None else ~safe_mask,
            need_weights=False,
        )
        attn_output = self.dropout(attn_output)
        user_repr = self.additive_attention(attn_output, safe_mask)

        if empty_rows is not None and empty_rows.any():
            user_repr[empty_rows] = 0.0

        return user_repr