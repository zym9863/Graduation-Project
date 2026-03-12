from __future__ import annotations

import argparse
import json

import torch
from tqdm import tqdm

from src.data.dataset import NRMSImpressionDataset, NewsFeatureStore
from src.data.preprocess import build_category_maps, load_news_corpus
from src.models.nrms import NRMSModel
from src.utils.config import ExperimentConfig
from src.utils.metrics import compute_ranking_metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在开发集上评估模型。")
    parser.add_argument("--checkpoint", type=str, default="data/processed/nrms_latest.pt")
    parser.add_argument("--fusion", choices=["concat", "gate"], default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--behavior-limit", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = ExperimentConfig()

    checkpoint_path = config.project_root / args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

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
    dataset = NRMSImpressionDataset(
        behaviors_file=config.dev_dir / "behaviors.tsv",
        feature_store=feature_store,
        max_history_len=config.max_history_len,
        behavior_limit=args.behavior_limit,
    )

    fusion = args.fusion or checkpoint.get("fusion", "concat")
    model = NRMSModel(
        num_categories=len(cat2id) + 1,
        num_subcategories=len(subcat2id) + 1,
        siglip_dim=config.siglip_dim,
        news_value_dim=config.news_value_dim,
        category_emb_dim=config.category_emb_dim,
        news_repr_dim=config.news_repr_dim,
        num_attention_heads=config.num_attention_heads,
        dropout=config.dropout,
        fusion=fusion,
    ).to(config.device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    label_groups: list[list[int]] = []
    score_groups: list[list[float]] = []

    with torch.inference_mode():
        for sample in tqdm(dataset, desc="Evaluating"):
            scores = model(
                history_text=sample["history_text"].unsqueeze(0).to(config.device),
                history_image=sample["history_image"].unsqueeze(0).to(config.device),
                history_value=sample["history_value"].unsqueeze(0).to(config.device),
                history_category=sample["history_category"].unsqueeze(0).to(config.device),
                history_subcategory=sample["history_subcategory"].unsqueeze(0).to(config.device),
                history_mask=sample["history_mask"].unsqueeze(0).to(config.device),
                candidate_text=sample["candidate_text"].unsqueeze(0).to(config.device),
                candidate_image=sample["candidate_image"].unsqueeze(0).to(config.device),
                candidate_value=sample["candidate_value"].unsqueeze(0).to(config.device),
                candidate_category=sample["candidate_category"].unsqueeze(0).to(config.device),
                candidate_subcategory=sample["candidate_subcategory"].unsqueeze(0).to(config.device),
            ).squeeze(0)
            label_groups.append(sample["labels"].tolist())
            score_groups.append(scores.cpu().tolist())

    metrics = compute_ranking_metrics(label_groups, score_groups)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()