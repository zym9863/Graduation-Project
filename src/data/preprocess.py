from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def parse_news_file(filepath: Path) -> dict[str, dict[str, Any]]:
    news: dict[str, dict[str, Any]] = {}
    with filepath.open("r", encoding="utf-8") as file:
        for raw_line in file:
            parts = raw_line.rstrip("\n").split("\t")
            if len(parts) < 8:
                continue
            news_id, category, subcategory, title, abstract, url, title_entities, abstract_entities = parts[:8]
            news[news_id] = {
                "news_id": news_id,
                "category": category,
                "subcategory": subcategory,
                "title": title,
                "abstract": abstract,
                "url": url,
                "title_entities": _load_json_field(title_entities),
                "abstract_entities": _load_json_field(abstract_entities),
            }
    return news


def parse_behaviors_file(filepath: Path) -> list[dict[str, Any]]:
    behaviors: list[dict[str, Any]] = []
    with filepath.open("r", encoding="utf-8") as file:
        for raw_line in file:
            parts = raw_line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            history = parts[3].split() if parts[3] else []
            impressions = []
            for item in parts[4].split():
                news_id, label = item.rsplit("-", 1)
                impressions.append((news_id, int(label)))
            behaviors.append(
                {
                    "impression_id": parts[0],
                    "user_id": parts[1],
                    "time": parts[2],
                    "history": history,
                    "impressions": impressions,
                }
            )
    return behaviors


def build_category_maps(news_dict: dict[str, dict[str, Any]]) -> tuple[dict[str, int], dict[str, int]]:
    categories = sorted({item["category"] for item in news_dict.values()})
    subcategories = sorted({item["subcategory"] for item in news_dict.values()})
    cat2id = {name: index + 1 for index, name in enumerate(categories)}
    subcat2id = {name: index + 1 for index, name in enumerate(subcategories)}
    return cat2id, subcat2id


def load_news_corpus(*news_files: Path) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for news_file in news_files:
        merged.update(parse_news_file(news_file))
    return merged


def summarize_corpus(
    news_dict: dict[str, dict[str, Any]],
    train_behaviors: list[dict[str, Any]],
    dev_behaviors: list[dict[str, Any]],
) -> dict[str, int]:
    return {
        "unique_news": len(news_dict),
        "categories": len({item["category"] for item in news_dict.values()}),
        "subcategories": len({item["subcategory"] for item in news_dict.values()}),
        "train_behaviors": len(train_behaviors),
        "dev_behaviors": len(dev_behaviors),
    }


def save_json(filepath: Path, payload: dict[str, Any]) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_json_field(raw_value: str) -> Any:
    if not raw_value:
        return []
    try:
        return json.loads(raw_value)
    except json.JSONDecodeError:
        return []