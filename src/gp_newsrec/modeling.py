from __future__ import annotations

import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .constants import DEFAULT_SEED, PROMPT_VERSION, VALUE_DIMENSIONS
from .io import ensure_dir, read_jsonl
from .metrics import aggregate_ranking_metrics


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    mode: str
    seed: int
    data_dir: Path
    feature_dir: Path
    value_labels_path: Path
    output_dir: Path
    epochs: int
    batch_size: int
    learning_rate: float
    hidden_dim: int
    dropout: float
    max_history: int
    eval_batch_size: int
    device: str


def load_config(path: str | Path) -> ExperimentConfig:
    import yaml

    with Path(path).open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    experiment = raw["experiment"]
    paths = raw["paths"]
    training = raw["training"]
    batch_size = int(training.get("batch_size", 256))
    return ExperimentConfig(
        name=str(experiment["name"]),
        mode=str(experiment["mode"]),
        seed=int(experiment.get("seed", DEFAULT_SEED)),
        data_dir=Path(paths.get("data_dir", "artifacts/data")),
        feature_dir=Path(paths.get("feature_dir", "artifacts/features/siglip")),
        value_labels_path=Path(paths.get("value_labels", "artifacts/labels/news_value_labels.jsonl")),
        output_dir=Path(paths.get("output_dir", f"artifacts/models/{experiment['name']}")),
        epochs=int(training.get("epochs", 3)),
        batch_size=batch_size,
        learning_rate=float(training.get("learning_rate", 1e-3)),
        hidden_dim=int(training.get("hidden_dim", 256)),
        dropout=float(training.get("dropout", 0.1)),
        max_history=int(training.get("max_history", 50)),
        eval_batch_size=int(training.get("eval_batch_size", max(batch_size, 1024))),
        device=str(training.get("device", "auto")),
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _load_feature_bundle(feature_dir: Path, mode: str, value_labels_path: Path) -> tuple[np.ndarray, dict[str, int]]:
    with (feature_dir / "news_ids.json").open("r", encoding="utf-8") as file:
        news_ids = json.load(file)
    text = np.load(feature_dir / "text.npy").astype("float32")
    if mode == "text":
        features = text
    else:
        image = np.load(feature_dir / "image.npy").astype("float32")
        features = np.concatenate([text, image], axis=1)
        if mode == "value":
            value_map = _load_value_vectors(value_labels_path)
            values = np.stack([value_map.get(news_id, _missing_value_vector()) for news_id in news_ids]).astype(
                "float32"
            )
            features = np.concatenate([features, values], axis=1)
    return features, {news_id: idx for idx, news_id in enumerate(news_ids)}


def _missing_value_vector() -> np.ndarray:
    return np.array([0.0] * len(VALUE_DIMENSIONS) + [1.0], dtype="float32")


def _load_value_vectors(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    vectors: dict[str, np.ndarray] = {}
    for record in read_jsonl(path):
        if record.get("prompt_version") != PROMPT_VERSION:
            continue
        scores = record.get("scores", {})
        values = [float(scores.get(dim, 0)) / 3.0 for dim in VALUE_DIMENSIONS]
        values.append(0.0)
        vectors[str(record["news_id"])] = np.array(values, dtype="float32")
    return vectors


class TrainDataset:
    def __init__(
        self,
        path: Path,
        features: np.ndarray,
        news_index: dict[str, int],
        max_history: int,
    ) -> None:
        self.features = features
        self.news_index = news_index
        self.max_history = max_history
        self.records = [
            record for record in read_jsonl(path) if str(record["candidate"]) in self.news_index
        ]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        record = self.records[idx]
        history_ids = [news_id for news_id in record["history"][-self.max_history :] if news_id in self.news_index]
        dim = self.features.shape[1]
        history = np.zeros((self.max_history, dim), dtype="float32")
        mask = np.zeros((self.max_history,), dtype="float32")
        for pos, news_id in enumerate(history_ids):
            history[pos] = self.features[self.news_index[news_id]]
            mask[pos] = 1.0
        candidate = self.features[self.news_index[record["candidate"]]]
        return history, mask, candidate, float(record["label"])


def collate_train(batch: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]]) -> tuple[Any, Any, Any, Any]:
    import torch

    histories, masks, candidates, labels = zip(*batch, strict=True)
    return (
        torch.from_numpy(np.stack(histories)),
        torch.from_numpy(np.stack(masks)),
        torch.from_numpy(np.stack(candidates)),
        torch.tensor(labels, dtype=torch.float32),
    )


def build_model(input_dim: int, hidden_dim: int, dropout: float) -> Any:
    import torch

    class RecModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_dim, hidden_dim),
            )

        def encode_user(self, history: Any, mask: Any) -> Any:
            hist = self.encoder(history)
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            return (hist * mask.unsqueeze(-1)).sum(dim=1) / denom

        def score_user_candidate(self, user: Any, candidate: Any) -> Any:
            cand = self.encoder(candidate)
            return (user * cand).sum(dim=1) / math.sqrt(cand.shape[-1])

        def forward(self, history: Any, mask: Any, candidate: Any) -> Any:
            user = self.encode_user(history, mask)
            return self.score_user_candidate(user, candidate)

    return RecModel()


def _resolve_device(config: ExperimentConfig, torch: Any) -> str:
    device = "cuda" if config.device == "auto" and torch.cuda.is_available() else config.device
    return "cpu" if device == "auto" else device


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def train(config_path: str | Path) -> dict[str, float]:
    import torch
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm

    config = load_config(config_path)
    set_seed(config.seed)
    device = _resolve_device(config, torch)

    _log(f"[train:{config.name}] loading features and training samples...")
    features, news_index = _load_feature_bundle(config.feature_dir, config.mode, config.value_labels_path)
    dataset = TrainDataset(config.data_dir / "train_samples.jsonl", features, news_index, config.max_history)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device == "cuda",
        collate_fn=collate_train,
    )
    model = build_model(features.shape[1], config.hidden_dim, config.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss()
    _log(
        f"[train:{config.name}] device={device} samples={len(dataset)} "
        f"batches/epoch={len(loader)} input_dim={features.shape[1]}"
    )

    model.train()
    last_loss = 0.0
    for epoch in range(config.epochs):
        total_loss = 0.0
        total = 0
        progress = tqdm(
            loader,
            total=len(loader),
            desc=f"{config.name} epoch {epoch + 1}/{config.epochs}",
            unit="batch",
            dynamic_ncols=True,
            file=sys.stderr,
        )
        for history, mask, candidate, label in progress:
            history = history.to(device)
            mask = mask.to(device)
            candidate = candidate.to(device)
            label = label.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(history, mask, candidate)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(label)
            total += len(label)
            progress.set_postfix(loss=f"{total_loss / max(total, 1):.4f}")
        last_loss = total_loss / max(total, 1)
        _log(f"[train:{config.name}] epoch {epoch + 1}/{config.epochs} loss={last_loss:.6f}")

    ensure_dir(config.output_dir)
    checkpoint = {
        "model_state": model.state_dict(),
        "input_dim": features.shape[1],
        "hidden_dim": config.hidden_dim,
        "dropout": config.dropout,
        "mode": config.mode,
        "name": config.name,
    }
    torch.save(checkpoint, config.output_dir / "model.pt")
    _log(f"[train:{config.name}] evaluating checkpoint...")
    metrics = _evaluate_model(config, model, features, news_index, device)
    metrics["train_loss"] = last_loss
    with (config.output_dir / "metrics.json").open("w", encoding="utf-8", newline="\n") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
    return metrics


def _predict_group(
    model: Any,
    features: np.ndarray,
    news_index: dict[str, int],
    history_ids: list[str],
    candidate_ids: list[str],
    max_history: int,
    device: str,
) -> list[float]:
    import torch

    dim = features.shape[1]
    history = np.zeros((1, max_history, dim), dtype="float32")
    mask = np.zeros((1, max_history), dtype="float32")
    filtered_history = [news_id for news_id in history_ids[-max_history:] if news_id in news_index]
    for idx, news_id in enumerate(filtered_history):
        history[0, idx] = features[news_index[news_id]]
        mask[0, idx] = 1.0
    valid_candidates = [news_id for news_id in candidate_ids if news_id in news_index]
    if not valid_candidates:
        return [0.0] * len(candidate_ids)
    candidate = np.stack([features[news_index[news_id]] for news_id in valid_candidates]).astype("float32")
    history_tensor = torch.from_numpy(np.repeat(history, len(valid_candidates), axis=0)).to(device)
    mask_tensor = torch.from_numpy(np.repeat(mask, len(valid_candidates), axis=0)).to(device)
    candidate_tensor = torch.from_numpy(candidate).to(device)
    with torch.inference_mode():
        raw_scores = model(history_tensor, mask_tensor, candidate_tensor).detach().cpu().numpy().tolist()
    by_id = dict(zip(valid_candidates, raw_scores, strict=True))
    return [float(by_id.get(news_id, 0.0)) for news_id in candidate_ids]


def evaluate_config(config_path: str | Path) -> dict[str, float]:
    import torch

    config = load_config(config_path)
    device = _resolve_device(config, torch)
    features, news_index = _load_feature_bundle(config.feature_dir, config.mode, config.value_labels_path)
    checkpoint_path = config.output_dir / "model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(int(checkpoint["input_dim"]), int(checkpoint["hidden_dim"]), float(checkpoint["dropout"]))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    return _evaluate_model(config, model, features, news_index, device)


def _evaluate_model(
    config: ExperimentConfig,
    model: Any,
    features: np.ndarray,
    news_index: dict[str, int],
    device: str,
) -> dict[str, float]:
    from tqdm.auto import tqdm

    model.eval()
    records = list(read_jsonl(config.data_dir / "dev_impressions.jsonl"))
    _log(
        f"[eval:{config.name}] impressions={len(records)} "
        f"candidate_batch_size={config.eval_batch_size}"
    )

    def iter_groups() -> Iterable[tuple[list[int], list[float]]]:
        chunk: list[dict[str, Any]] = []
        candidate_count = 0
        progress = tqdm(
            total=len(records),
            desc=f"{config.name} eval",
            unit="impression",
            dynamic_ncols=True,
            file=sys.stderr,
        )
        try:
            for record in records:
                chunk.append(record)
                candidate_count += sum(1 for news_id in record["candidates"] if news_id in news_index)
                if candidate_count >= config.eval_batch_size or len(chunk) >= config.eval_batch_size:
                    yield from _score_eval_chunk(
                        model,
                        features,
                        news_index,
                        chunk,
                        config.max_history,
                        device,
                    )
                    progress.update(len(chunk))
                    chunk = []
                    candidate_count = 0
            if chunk:
                yield from _score_eval_chunk(
                    model,
                    features,
                    news_index,
                    chunk,
                    config.max_history,
                    device,
                )
                progress.update(len(chunk))
        finally:
            progress.close()

    metrics = aggregate_ranking_metrics(iter_groups())
    model.train()
    return metrics


def _score_eval_chunk(
    model: Any,
    features: np.ndarray,
    news_index: dict[str, int],
    records: list[dict[str, Any]],
    max_history: int,
    device: str,
) -> list[tuple[list[int], list[float]]]:
    import torch

    dim = features.shape[1]
    histories = np.zeros((len(records), max_history, dim), dtype="float32")
    masks = np.zeros((len(records), max_history), dtype="float32")
    candidate_rows: list[np.ndarray] = []
    candidate_groups: list[int] = []
    candidate_positions: list[int] = []
    groups: list[tuple[list[int], list[float]]] = []

    for group_idx, record in enumerate(records):
        history_ids = [news_id for news_id in record["history"][-max_history:] if news_id in news_index]
        for pos, news_id in enumerate(history_ids):
            histories[group_idx, pos] = features[news_index[news_id]]
            masks[group_idx, pos] = 1.0

        candidate_ids = list(record["candidates"])
        labels = [int(label) for label in record["labels"]]
        scores = [0.0] * len(candidate_ids)
        groups.append((labels, scores))
        for pos, news_id in enumerate(candidate_ids):
            candidate_idx = news_index.get(news_id)
            if candidate_idx is None:
                continue
            candidate_rows.append(features[candidate_idx])
            candidate_groups.append(group_idx)
            candidate_positions.append(pos)

    if not candidate_rows:
        return groups

    with torch.inference_mode():
        history_tensor = torch.from_numpy(histories).to(device)
        mask_tensor = torch.from_numpy(masks).to(device)
        candidate_tensor = torch.from_numpy(np.stack(candidate_rows).astype("float32")).to(device)
        group_tensor = torch.tensor(candidate_groups, dtype=torch.long, device=device)
        users = model.encode_user(history_tensor, mask_tensor)
        candidates = model.encoder(candidate_tensor)
        raw_scores = (users.index_select(0, group_tensor) * candidates).sum(dim=1) / math.sqrt(candidates.shape[-1])
        scores = raw_scores.detach().cpu().numpy().tolist()

    for score, group_idx, pos in zip(scores, candidate_groups, candidate_positions, strict=True):
        groups[group_idx][1][pos] = float(score)
    return groups


def evaluate_many(
    config_paths: Iterable[str | Path],
    output_path: str | Path = "artifacts/reports/ablation.csv",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for config_path in config_paths:
        try:
            config = load_config(config_path)
            metrics = evaluate_config(config_path)
        except FileNotFoundError:
            continue
        row = {"experiment": config.name, "mode": config.mode}
        row.update(metrics)
        rows.append(row)
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    fieldnames = ["experiment", "mode", "auc", "mrr", "ndcg5", "ndcg10"]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    return rows
