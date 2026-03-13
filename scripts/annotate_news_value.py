from __future__ import annotations

import argparse
import json
import os
import time

from tqdm import tqdm

from src.data.preprocess import load_news_corpus
from src.features.news_value_annotator import NewsValueAnnotator
from src.utils.config import ExperimentConfig
from src.utils.env import load_project_env


VALUE_DIMENSIONS = ("timeliness", "importance", "prominence", "proximity", "interest")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    load_project_env()

    parser = argparse.ArgumentParser(description="离线标注新闻价值五要素。")
    parser.add_argument("--provider", choices=["heuristic", "openai-compatible"], default="heuristic")
    parser.add_argument("--model", type=str, default=os.getenv("NEWS_VALUE_MODEL", "ZhipuAI/GLM-5"))
    parser.add_argument("--base-url", type=str, default=os.getenv("NEWS_VALUE_API_BASE", "https://api-inference.modelscope.cn/v1"))
    parser.add_argument("--api-key", type=str, default=os.getenv("NEWS_VALUE_API_KEY") or os.getenv("MODELSCOPE_TOKEN"))
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--single-json", type=str, default=None, help="单条新闻JSON字符串。")
    parser.add_argument("--single-title", type=str, default=None, help="单条新闻标题。")
    parser.add_argument("--single-abstract", type=str, default=None, help="单条新闻摘要。")
    parser.add_argument("--single-category", type=str, default=None, help="单条新闻类别。")
    parser.add_argument("--single-subcategory", type=str, default=None, help="单条新闻子类别。")

    args = parser.parse_args(argv)
    args.single_mode = any(
        value is not None
        for value in (
            args.single_json,
            args.single_title,
            args.single_abstract,
            args.single_category,
            args.single_subcategory,
        )
    )
    if args.single_mode and (args.limit is not None or args.overwrite):
        parser.error("--single-* 参数不能与 --limit/--overwrite 同时使用。")
    if args.single_mode and args.single_json is None and not (args.single_title or args.single_abstract):
        parser.error("单条模式至少需要 --single-json，或提供 --single-title/--single-abstract。")

    return args


def build_single_article(args: argparse.Namespace) -> dict[str, str]:
    article: dict[str, str] = {
        "title": "",
        "abstract": "",
        "category": "",
        "subcategory": "",
    }

    if args.single_json:
        loaded = json.loads(args.single_json)
        if not isinstance(loaded, dict):
            raise ValueError("--single-json 必须是对象，例如 {'title': '...'}。")
        for key in article:
            value = loaded.get(key)
            if value is not None:
                article[key] = str(value)

    if args.single_title is not None:
        article["title"] = args.single_title
    if args.single_abstract is not None:
        article["abstract"] = args.single_abstract
    if args.single_category is not None:
        article["category"] = args.single_category
    if args.single_subcategory is not None:
        article["subcategory"] = args.single_subcategory

    return article


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    annotator = NewsValueAnnotator(
        model=args.model,
        provider=args.provider,
        base_url=args.base_url,
        api_key=args.api_key,
    )

    if args.single_mode:
        article = build_single_article(args)
        scores = annotator.annotate(article)
        mapped_scores = {name: score for name, score in zip(VALUE_DIMENSIONS, scores)}
        print("=== Single News Value Case ===")
        print("Input article:")
        print(json.dumps(article, ensure_ascii=False, indent=2))
        print("\nValue scores:")
        print(json.dumps(mapped_scores, ensure_ascii=False, indent=2))
        print(f"vector: {scores}")
        return

    config = ExperimentConfig()
    config.ensure_directories()

    news = load_news_corpus(config.train_dir / "news.tsv", config.dev_dir / "news.tsv")
    items = list(news.items())
    if args.limit is not None:
        items = items[: args.limit]

    existing_scores: dict[str, list[int]] = {}
    if config.news_value_file.exists() and not args.overwrite:
        existing_scores = json.loads(config.news_value_file.read_text(encoding="utf-8"))

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