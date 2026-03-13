from __future__ import annotations

import argparse
import json

from src.data.analytics import build_dataset_statistics
from src.data.preprocess import build_category_maps, load_news_corpus, parse_behaviors_file, summarize_corpus
from src.utils.config import ExperimentConfig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建类别映射和语料统计。")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    parse_args(argv)
    config = ExperimentConfig()
    config.ensure_directories()

    news = load_news_corpus(config.train_dir / "news.tsv", config.dev_dir / "news.tsv")
    train_behaviors = parse_behaviors_file(config.train_dir / "behaviors.tsv")
    dev_behaviors = parse_behaviors_file(config.dev_dir / "behaviors.tsv")
    cat2id, subcat2id = build_category_maps(news)
    statistics = build_dataset_statistics(news, train_behaviors, dev_behaviors)

    payload = {
        "summary": summarize_corpus(news, train_behaviors, dev_behaviors),
        "statistics": statistics,
        "cat2id": cat2id,
        "subcat2id": subcat2id,
    }

    config.metadata_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved metadata to {config.metadata_file}")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()