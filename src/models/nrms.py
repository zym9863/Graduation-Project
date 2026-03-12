from __future__ import annotations

import torch
from torch import nn

from src.models.news_encoder import ConcatNewsEncoder, GatedNewsEncoder
from src.models.user_encoder import NRMSUserEncoder


class NRMSModel(nn.Module):
    def __init__(
        self,
        num_categories: int,
        num_subcategories: int,
        siglip_dim: int = 768,
        news_value_dim: int = 5,
        category_emb_dim: int = 64,
        news_repr_dim: int = 256,
        num_attention_heads: int = 16,
        dropout: float = 0.2,
        fusion: str = "concat",
    ) -> None:
        super().__init__()

        if fusion == "concat":
            self.news_encoder = ConcatNewsEncoder(
                siglip_dim=siglip_dim,
                news_value_dim=news_value_dim,
                category_emb_dim=category_emb_dim,
                news_repr_dim=news_repr_dim,
                num_categories=num_categories,
                num_subcategories=num_subcategories,
                dropout=dropout,
            )
        elif fusion == "gate":
            self.news_encoder = GatedNewsEncoder(
                siglip_dim=siglip_dim,
                news_value_dim=news_value_dim,
                category_emb_dim=category_emb_dim,
                news_repr_dim=news_repr_dim,
                num_categories=num_categories,
                num_subcategories=num_subcategories,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unsupported fusion mode: {fusion}")

        self.user_encoder = NRMSUserEncoder(
            news_repr_dim=news_repr_dim,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
        )

    def _encode_news_batch(
        self,
        text_emb: torch.Tensor,
        image_emb: torch.Tensor,
        news_value: torch.Tensor,
        category_ids: torch.Tensor,
        subcategory_ids: torch.Tensor,
    ) -> torch.Tensor:
        original_shape = text_emb.shape[:-1]
        flat_text = text_emb.reshape(-1, text_emb.size(-1))
        flat_image = image_emb.reshape(-1, image_emb.size(-1))
        flat_value = news_value.reshape(-1, news_value.size(-1))
        flat_category = category_ids.reshape(-1)
        flat_subcategory = subcategory_ids.reshape(-1)

        encoded = self.news_encoder(
            text_emb=flat_text,
            image_emb=flat_image,
            news_value=flat_value,
            category_ids=flat_category,
            subcategory_ids=flat_subcategory,
        )
        return encoded.reshape(*original_shape, encoded.size(-1))

    def forward(
        self,
        history_text: torch.Tensor,
        history_image: torch.Tensor,
        history_value: torch.Tensor,
        history_category: torch.Tensor,
        history_subcategory: torch.Tensor,
        history_mask: torch.Tensor,
        candidate_text: torch.Tensor,
        candidate_image: torch.Tensor,
        candidate_value: torch.Tensor,
        candidate_category: torch.Tensor,
        candidate_subcategory: torch.Tensor,
    ) -> torch.Tensor:
        history_repr = self._encode_news_batch(
            text_emb=history_text,
            image_emb=history_image,
            news_value=history_value,
            category_ids=history_category,
            subcategory_ids=history_subcategory,
        )
        candidate_repr = self._encode_news_batch(
            text_emb=candidate_text,
            image_emb=candidate_image,
            news_value=candidate_value,
            category_ids=candidate_category,
            subcategory_ids=candidate_subcategory,
        )
        user_repr = self.user_encoder(history_repr, history_mask)
        return torch.einsum("bd,bkd->bk", user_repr, candidate_repr)