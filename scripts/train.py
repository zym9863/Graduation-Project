from __future__ import annotations

import argparse
import json
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import NRMSImpressionDataset, NRMSTrainDataset, NewsFeatureStore
from src.data.preprocess import build_category_maps, load_news_corpus
from src.models.nrms import NRMSModel
from src.utils.config import ExperimentConfig
from src.utils.metrics import compute_ranking_metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 NRMS 多模态推荐模型。")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--scheduler", choices=["none", "plateau", "cosine"], default="none")
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--scheduler-patience", type=int, default=1)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--fusion", choices=["concat", "gate", "text_only"], default="concat")
    parser.add_argument("--checkpoint", type=str, default="data/processed/nrms_latest.pt")
    parser.add_argument("--eval-dev", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--behavior-limit", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    return parser.parse_args(argv)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {name: tensor.to(device) for name, tensor in batch.items()}


def evaluate_model(model: NRMSModel, dataset: NRMSImpressionDataset, device: str) -> dict[str, float]:
    model.eval()
    label_groups: list[list[int]] = []
    score_groups: list[list[float]] = []

    with torch.inference_mode():
        for sample in tqdm(dataset, desc="Evaluating", leave=False):
            scores = model(
                history_text=sample["history_text"].unsqueeze(0).to(device),
                history_image=sample["history_image"].unsqueeze(0).to(device),
                history_value=sample["history_value"].unsqueeze(0).to(device),
                history_category=sample["history_category"].unsqueeze(0).to(device),
                history_subcategory=sample["history_subcategory"].unsqueeze(0).to(device),
                history_mask=sample["history_mask"].unsqueeze(0).to(device),
                candidate_text=sample["candidate_text"].unsqueeze(0).to(device),
                candidate_image=sample["candidate_image"].unsqueeze(0).to(device),
                candidate_value=sample["candidate_value"].unsqueeze(0).to(device),
                candidate_category=sample["candidate_category"].unsqueeze(0).to(device),
                candidate_subcategory=sample["candidate_subcategory"].unsqueeze(0).to(device),
            ).squeeze(0)
            label_groups.append(sample["labels"].tolist())
            score_groups.append(scores.cpu().tolist())

    return compute_ranking_metrics(label_groups, score_groups)


def build_scheduler(
    scheduler_name: str,
    optimizer: torch.optim.Optimizer,
    *,
    use_dev_metric: bool,
    total_epochs: int,
    factor: float,
    patience: int,
) -> torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    if scheduler_name == "none":
        return None
    if scheduler_name == "plateau":
        mode = "max" if use_dev_metric else "min"
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
        )
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_epochs, 1))


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = ExperimentConfig()
    config.ensure_directories()
    set_seed(config.seed)

    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate

    news = load_news_corpus(config.train_dir / "news.tsv", config.dev_dir / "news.tsv")
    if args.limit is not None:
        news = dict(list(news.items())[: args.limit])

    cat2id, subcat2id = build_category_maps(news)
    feature_store = NewsFeatureStore.from_files(
        news=news,
        cat2id=cat2id,
        subcat2id=subcat2id,
        feature_file=config.feature_file,
        news_value_file=config.news_value_file,
        siglip_dim=config.siglip_dim,
        news_value_dim=config.news_value_dim,
    )

    train_dataset = NRMSTrainDataset(
        behaviors_file=config.train_dir / "behaviors.tsv",
        feature_store=feature_store,
        max_history_len=config.max_history_len,
        npratio=config.npratio,
        seed=config.seed,
        behavior_limit=args.behavior_limit,
    )
    if len(train_dataset) == 0:
        raise RuntimeError("Training dataset is empty. Check feature coverage or behavior limits.")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )

    dev_dataset = None
    if args.eval_dev:
        dev_dataset = NRMSImpressionDataset(
            behaviors_file=config.dev_dir / "behaviors.tsv",
            feature_store=feature_store,
            max_history_len=config.max_history_len,
            behavior_limit=args.behavior_limit,
        )

    model = NRMSModel(
        num_categories=len(cat2id) + 1,
        num_subcategories=len(subcat2id) + 1,
        siglip_dim=config.siglip_dim,
        news_value_dim=config.news_value_dim,
        category_emb_dim=config.category_emb_dim,
        news_repr_dim=config.news_repr_dim,
        num_attention_heads=config.num_attention_heads,
        dropout=config.dropout,
        fusion=args.fusion,
    ).to(config.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = build_scheduler(
        args.scheduler,
        optimizer,
        use_dev_metric=dev_dataset is not None,
        total_epochs=config.epochs,
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
    )
    checkpoint_path = config.project_root / args.checkpoint
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_score = float("-inf")
    max_patience = args.patience if args.patience is not None and args.patience > 0 else None
    epochs_without_improvement = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        seen_samples = 0

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs}"), start=1):
            batch = move_batch_to_device(batch, config.device)
            scores = model(
                history_text=batch["history_text"],
                history_image=batch["history_image"],
                history_value=batch["history_value"],
                history_category=batch["history_category"],
                history_subcategory=batch["history_subcategory"],
                history_mask=batch["history_mask"],
                candidate_text=batch["candidate_text"],
                candidate_image=batch["candidate_image"],
                candidate_value=batch["candidate_value"],
                candidate_category=batch["candidate_category"],
                candidate_subcategory=batch["candidate_subcategory"],
            )
            loss = F.cross_entropy(scores, batch["label"])
            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            running_loss += loss.item() * scores.size(0)
            seen_samples += scores.size(0)
            if args.max_steps is not None and step >= args.max_steps:
                break

        epoch_loss = running_loss / max(seen_samples, 1)
        print(f"epoch={epoch} lr={optimizer.param_groups[0]['lr']:.6g} loss={epoch_loss:.4f}")

        metrics = None
        if dev_dataset is not None:
            metrics = evaluate_model(model, dev_dataset, config.device)
            print(json.dumps(metrics, ensure_ascii=False, indent=2))

        monitor_score = metrics["auc"] if metrics is not None else -epoch_loss
        scheduler_metric = metrics["auc"] if metrics is not None else epoch_loss
        improved = monitor_score > best_score + args.min_delta
        if improved:
            best_score = monitor_score
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "fusion": args.fusion,
                    "epoch": epoch,
                    "cat2id": cat2id,
                    "subcat2id": subcat2id,
                    "config": config.to_dict(),
                    "train_loss": epoch_loss,
                    "metrics": metrics,
                },
                checkpoint_path,
            )
            print(f"Saved checkpoint to {checkpoint_path}")
        else:
            epochs_without_improvement += 1

        if scheduler is not None:
            if args.scheduler == "plateau":
                scheduler.step(scheduler_metric)
            else:
                scheduler.step()

        if max_patience is not None and epochs_without_improvement >= max_patience:
            print(
                f"Early stopping triggered at epoch {epoch} after {epochs_without_improvement} non-improving epoch(s)."
            )
            print(f"Best monitored score={best_score:.6f}")
            break


if __name__ == "__main__":
    main()