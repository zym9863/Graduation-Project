from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch


@dataclass(slots=True)
class ExperimentConfig:
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])

    train_dir: Path = field(init=False)
    dev_dir: Path = field(init=False)
    image_dir: Path = field(init=False)
    data_dir: Path = field(init=False)
    processed_dir: Path = field(init=False)
    analytics_dir: Path = field(init=False)
    analytics_figures_dir: Path = field(init=False)
    analytics_tables_dir: Path = field(init=False)
    feature_file: Path = field(init=False)
    news_value_file: Path = field(init=False)
    metadata_file: Path = field(init=False)

    siglip_model_name: str = "google/siglip-base-patch16-224"
    siglip_max_length: int = 64
    siglip_dim: int = 768

    news_value_dim: int = 5
    category_emb_dim: int = 64
    news_repr_dim: int = 256
    max_history_len: int = 50
    num_attention_heads: int = 16
    dropout: float = 0.2

    batch_size: int = 64
    eval_batch_size: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    epochs: int = 5
    npratio: int = 4
    seed: int = 42
    num_workers: int = 0

    def __post_init__(self) -> None:
        self.train_dir = self.project_root / "MINDsmall_train"
        self.dev_dir = self.project_root / "MINDsmall_dev"
        self.image_dir = self.project_root / "newData"
        self.data_dir = self.project_root / "data"
        self.processed_dir = self.data_dir / "processed"
        self.analytics_dir = self.processed_dir / "analytics"
        self.analytics_figures_dir = self.analytics_dir / "figures"
        self.analytics_tables_dir = self.analytics_dir / "tables"
        self.feature_file = self.data_dir / "news_siglip_features.pt"
        self.news_value_file = self.data_dir / "news_value_scores.json"
        self.metadata_file = self.processed_dir / "metadata.json"

    @property
    def device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.analytics_dir.mkdir(parents=True, exist_ok=True)
        self.analytics_figures_dir.mkdir(parents=True, exist_ok=True)
        self.analytics_tables_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, object]:
        serialized: dict[str, object] = {}
        for key, value in asdict(self).items():
            serialized[key] = str(value) if isinstance(value, Path) else value
        return serialized