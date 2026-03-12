from __future__ import annotations

import argparse

import torch
from tqdm import tqdm

from src.data.preprocess import load_news_corpus
from src.features.siglip_extractor import SigLIPFeatureExtractor, build_siglip_text, load_news_image
from src.utils.config import ExperimentConfig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 SigLIP 离线提取图文特征。")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = ExperimentConfig()
    config.ensure_directories()
    output_path = config.feature_file if args.output is None else config.project_root / args.output

    news = load_news_corpus(config.train_dir / "news.tsv", config.dev_dir / "news.tsv")
    items = list(news.items())
    if args.limit is not None:
        items = items[: args.limit]

    extractor = SigLIPFeatureExtractor(
        model_name=config.siglip_model_name,
        device=config.device,
        max_length=config.siglip_max_length,
    )

    features: dict[str, dict[str, torch.Tensor]] = {}
    for start in tqdm(range(0, len(items), args.batch_size), desc="Extracting SigLIP features"):
        batch = items[start : start + args.batch_size]
        news_ids = [news_id for news_id, _ in batch]
        texts = [build_siglip_text(article) for _, article in batch]
        images = [load_news_image(config.image_dir, news_id) for news_id in news_ids]
        text_embeddings, image_embeddings = extractor.encode_batch(texts=texts, images=images)

        for index, news_id in enumerate(news_ids):
            features[news_id] = {
                "text_emb": text_embeddings[index],
                "image_emb": image_embeddings[index],
            }

    torch.save(features, output_path)
    print(f"Saved {len(features)} feature entries to {output_path}")


if __name__ == "__main__":
    main()