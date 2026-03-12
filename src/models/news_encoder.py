from __future__ import annotations

import torch
from torch import nn

from src.models.gated_fusion import GatedFusionLayer


class ConcatNewsEncoder(nn.Module):
    def __init__(
        self,
        siglip_dim: int,
        news_value_dim: int,
        category_emb_dim: int,
        news_repr_dim: int,
        num_categories: int,
        num_subcategories: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.category_embedding = nn.Embedding(num_categories, category_emb_dim, padding_idx=0)
        self.subcategory_embedding = nn.Embedding(num_subcategories, category_emb_dim, padding_idx=0)
        input_dim = siglip_dim * 2 + news_value_dim + category_emb_dim * 2
        self.projection = nn.Sequential(
            nn.Linear(input_dim, news_repr_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        text_emb: torch.Tensor,
        image_emb: torch.Tensor,
        news_value: torch.Tensor,
        category_ids: torch.Tensor,
        subcategory_ids: torch.Tensor,
    ) -> torch.Tensor:
        category_emb = self.category_embedding(category_ids)
        subcategory_emb = self.subcategory_embedding(subcategory_ids)
        combined = torch.cat([text_emb, image_emb, news_value, category_emb, subcategory_emb], dim=-1)
        return self.projection(combined)


class GatedNewsEncoder(nn.Module):
    def __init__(
        self,
        siglip_dim: int,
        news_value_dim: int,
        category_emb_dim: int,
        news_repr_dim: int,
        num_categories: int,
        num_subcategories: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.category_embedding = nn.Embedding(num_categories, category_emb_dim, padding_idx=0)
        self.subcategory_embedding = nn.Embedding(num_subcategories, category_emb_dim, padding_idx=0)
        self.gated_fusion = GatedFusionLayer(siglip_dim=siglip_dim, news_value_dim=news_value_dim, output_dim=news_repr_dim)
        self.output_projection = nn.Sequential(
            nn.Linear(news_repr_dim + category_emb_dim * 2, news_repr_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        text_emb: torch.Tensor,
        image_emb: torch.Tensor,
        news_value: torch.Tensor,
        category_ids: torch.Tensor,
        subcategory_ids: torch.Tensor,
    ) -> torch.Tensor:
        fused = self.gated_fusion(text_emb, image_emb, news_value)
        category_emb = self.category_embedding(category_ids)
        subcategory_emb = self.subcategory_embedding(subcategory_ids)
        return self.output_projection(torch.cat([fused, category_emb, subcategory_emb], dim=-1))