from __future__ import annotations

import argparse
import json
import os
import time

from tqdm import tqdm

from src.data.preprocess import load_news_corpus
from src.features.news_value_annotator import NewsValueAnnotator
from src.utils.config import ExperimentConfig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="离线标注新闻价值五要素。")
    parser.add_argument("--provider", choices=["heuristic", "openai-compatible"], default="heuristic")
    parser.add_argument("--model", type=str, default=os.getenv("NEWS_VALUE_MODEL", "deepseek-chat"))
    parser.add_argument("--base-url", type=str, default=os.getenv("NEWS_VALUE_API_BASE"))
    parser.add_argument("--api-key", type=str, default=os.getenv("NEWS_VALUE_API_KEY"))
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = ExperimentConfig()
    config.ensure_directories()

    news = load_news_corpus(config.train_dir / "news.tsv", config.dev_dir / "news.tsv")
    items = list(news.items())
    if args.limit is not None:
        items = items[: args.limit]

    existing_scores: dict[str, list[int]] = {}
    if config.news_value_file.exists() and not args.overwrite:
        existing_scores = json.loads(config.news_value_file.read_text(encoding="utf-8"))

    annotator = NewsValueAnnotator(
        model=args.model,
        provider=args.provider,
        base_url=args.base_url,
        api_key=args.api_key,
    )

    for news_id, article in tqdm(items, desc="Annotating news values"):
        if news_id in existing_scores and not args.overwrite:
            continue
        existing_scores[news_id] = annotator.annotate(article)
        if args.sleep > 0:
            time.sleep(args.sleep)

    config.news_value_file.write_text(json.dumps(existing_scores, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(existing_scores)} news value entries to {config.news_value_file}")


if __name__ == "__main__":
    main()