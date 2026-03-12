from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from src.data.preprocess import parse_behaviors_file


def _coerce_vector(value: Any, dim: int) -> torch.Tensor:
    if value is None:
        return torch.zeros(dim, dtype=torch.float32)
    tensor = torch.as_tensor(value, dtype=torch.float32).flatten()
    if tensor.numel() == dim:
        return tensor
    if tensor.numel() > dim:
        return tensor[:dim]
    padded = torch.zeros(dim, dtype=torch.float32)
    padded[: tensor.numel()] = tensor
    return padded


def _pick_feature(entry: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in entry:
            return entry[key]
    return None


@dataclass(slots=True)
class NewsFeatureStore:
    news: dict[str, dict[str, Any]]
    cat2id: dict[str, int]
    subcat2id: dict[str, int]
    siglip_features: dict[str, Any]
    news_value_scores: dict[str, Any]
    siglip_dim: int = 768
    news_value_dim: int = 5

    @classmethod
    def from_files(
        cls,
        news: dict[str, dict[str, Any]],
        cat2id: dict[str, int],
        subcat2id: dict[str, int],
        feature_file: Path,
        news_value_file: Path,
        siglip_dim: int = 768,
        news_value_dim: int = 5,
    ) -> "NewsFeatureStore":
        siglip_features: dict[str, Any] = {}
        if feature_file.exists():
            siglip_features = torch.load(feature_file, map_location="cpu", weights_only=False)

        news_value_scores: dict[str, Any] = {}
        if news_value_file.exists():
            news_value_scores = json.loads(news_value_file.read_text(encoding="utf-8"))

        return cls(
            news=news,
            cat2id=cat2id,
            subcat2id=subcat2id,
            siglip_features=siglip_features,
            news_value_scores=news_value_scores,
            siglip_dim=siglip_dim,
            news_value_dim=news_value_dim,
        )

    def encode_news(self, news_id: str | None) -> dict[str, torch.Tensor]:
        article = self.news.get(news_id or "", {})
        siglip_entry = self.siglip_features.get(news_id or "", {})
        if not isinstance(siglip_entry, dict):
            siglip_entry = {}

        value_entry = self.news_value_scores.get(news_id or "", [0.0] * self.news_value_dim)

        return {
            "text": _coerce_vector(
                _pick_feature(siglip_entry, ("text_emb", "text", "title_emb")),
                self.siglip_dim,
            ),
            "image": _coerce_vector(
                _pick_feature(siglip_entry, ("image_emb", "image", "img_emb")),
                self.siglip_dim,
            ),
            "value": _coerce_vector(value_entry, self.news_value_dim),
            "category": torch.tensor(self.cat2id.get(article.get("category", ""), 0), dtype=torch.long),
            "subcategory": torch.tensor(self.subcat2id.get(article.get("subcategory", ""), 0), dtype=torch.long),
        }

    def stack_news(self, news_ids: list[str], pad_to: int) -> dict[str, torch.Tensor]:
        trimmed = news_ids[:pad_to]
        encoded = [self.encode_news(news_id) for news_id in trimmed]
        while len(encoded) < pad_to:
            encoded.append(self.encode_news(None))

        return {
            "text": torch.stack([item["text"] for item in encoded], dim=0),
            "image": torch.stack([item["image"] for item in encoded], dim=0),
            "value": torch.stack([item["value"] for item in encoded], dim=0),
            "category": torch.stack([item["category"] for item in encoded], dim=0),
            "subcategory": torch.stack([item["subcategory"] for item in encoded], dim=0),
            "mask": torch.tensor([True] * len(trimmed) + [False] * (pad_to - len(trimmed)), dtype=torch.bool),
        }


class NRMSTrainDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        behaviors_file: Path,
        feature_store: NewsFeatureStore,
        max_history_len: int,
        npratio: int,
        seed: int = 42,
        behavior_limit: int | None = None,
    ) -> None:
        self.feature_store = feature_store
        self.max_history_len = max_history_len
        self.npratio = npratio
        self.rng = random.Random(seed)
        behaviors = parse_behaviors_file(behaviors_file)
        if behavior_limit is not None:
            behaviors = behaviors[:behavior_limit]
        self.samples = self._build_samples(behaviors)

    def _build_samples(self, behaviors: list[dict[str, Any]]) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        for behavior in behaviors:
            history = [news_id for news_id in behavior["history"] if news_id in self.feature_store.news]
            positives = [news_id for news_id, label in behavior["impressions"] if label == 1 and news_id in self.feature_store.news]
            negatives = [news_id for news_id, label in behavior["impressions"] if label == 0 and news_id in self.feature_store.news]
            if not positives or not negatives:
                continue

            for positive in positives:
                if len(negatives) >= self.npratio:
                    sampled_negatives = self.rng.sample(negatives, self.npratio)
                else:
                    sampled_negatives = [self.rng.choice(negatives) for _ in range(self.npratio)]
                samples.append(
                    {
                        "history": history,
                        "positive": positive,
                        "negatives": sampled_negatives,
                    }
                )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        history = self.feature_store.stack_news(sample["history"][-self.max_history_len :], self.max_history_len)
        candidates = self.feature_store.stack_news([sample["positive"], *sample["negatives"]], self.npratio + 1)
        return {
            "history_text": history["text"],
            "history_image": history["image"],
            "history_value": history["value"],
            "history_category": history["category"],
            "history_subcategory": history["subcategory"],
            "history_mask": history["mask"],
            "candidate_text": candidates["text"],
            "candidate_image": candidates["image"],
            "candidate_value": candidates["value"],
            "candidate_category": candidates["category"],
            "candidate_subcategory": candidates["subcategory"],
            "label": torch.tensor(0, dtype=torch.long),
        }


class NRMSImpressionDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        behaviors_file: Path,
        feature_store: NewsFeatureStore,
        max_history_len: int,
        behavior_limit: int | None = None,
    ) -> None:
        self.feature_store = feature_store
        self.max_history_len = max_history_len
        behaviors = parse_behaviors_file(behaviors_file)
        if behavior_limit is not None:
            behaviors = behaviors[:behavior_limit]
        self.impressions = self._build_impressions(behaviors)

    def _build_impressions(self, behaviors: list[dict[str, Any]]) -> list[dict[str, Any]]:
        impressions: list[dict[str, Any]] = []
        for behavior in behaviors:
            history = [news_id for news_id in behavior["history"] if news_id in self.feature_store.news]
            candidate_ids = [news_id for news_id, _ in behavior["impressions"] if news_id in self.feature_store.news]
            labels = [label for news_id, label in behavior["impressions"] if news_id in self.feature_store.news]
            if not candidate_ids:
                continue
            impressions.append(
                {
                    "impression_id": behavior["impression_id"],
                    "history": history,
                    "candidate_ids": candidate_ids,
                    "labels": labels,
                }
            )
        return impressions

    def __len__(self) -> int:
        return len(self.impressions)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.impressions[index]
        history = self.feature_store.stack_news(sample["history"][-self.max_history_len :], self.max_history_len)
        candidates = self.feature_store.stack_news(sample["candidate_ids"], len(sample["candidate_ids"]))
        return {
            "impression_id": sample["impression_id"],
            "history_text": history["text"],
            "history_image": history["image"],
            "history_value": history["value"],
            "history_category": history["category"],
            "history_subcategory": history["subcategory"],
            "history_mask": history["mask"],
            "candidate_text": candidates["text"],
            "candidate_image": candidates["image"],
            "candidate_value": candidates["value"],
            "candidate_category": candidates["category"],
            "candidate_subcategory": candidates["subcategory"],
            "labels": torch.tensor(sample["labels"], dtype=torch.long),
        }