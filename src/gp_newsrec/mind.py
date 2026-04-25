from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

from .constants import DEFAULT_SEED
from .io import ensure_dir, write_jsonl


@dataclass(frozen=True)
class NewsRecord:
    news_id: str
    category: str
    subcategory: str
    title: str
    abstract: str
    url: str
    title_entities: str
    abstract_entities: str
    image_path: str | None = None

    @property
    def text_for_encoder(self) -> str:
        parts = [self.category, self.subcategory, self.title, self.abstract]
        return " [SEP] ".join(part for part in parts if part)


@dataclass(frozen=True)
class BehaviorRecord:
    impression_id: str
    user_id: str
    time: str
    history: list[str]
    impressions: list[tuple[str, int]]


def parse_news_line(line: str, image_dir: str | Path | None = None) -> NewsRecord:
    fields = line.rstrip("\n").split("\t")
    if len(fields) != 8:
        raise ValueError(f"Expected 8 news fields, got {len(fields)}: {line[:100]!r}")
    news_id, category, subcategory, title, abstract, url, title_entities, abstract_entities = fields
    image_path = None
    if image_dir is not None:
        candidate = Path(image_dir) / f"{news_id}.jpg"
        if candidate.exists():
            image_path = str(candidate)
    return NewsRecord(
        news_id=news_id,
        category=category,
        subcategory=subcategory,
        title=title,
        abstract=abstract,
        url=url,
        title_entities=title_entities,
        abstract_entities=abstract_entities,
        image_path=image_path,
    )


def parse_behavior_line(line: str) -> BehaviorRecord:
    fields = line.rstrip("\n").split("\t")
    if len(fields) != 5:
        raise ValueError(f"Expected 5 behavior fields, got {len(fields)}: {line[:100]!r}")
    impression_id, user_id, time, history_raw, impressions_raw = fields
    history = [item for item in history_raw.split() if item]
    impressions: list[tuple[str, int]] = []
    for item in impressions_raw.split():
        news_id, label = item.rsplit("-", 1)
        impressions.append((news_id, int(label)))
    return BehaviorRecord(
        impression_id=impression_id,
        user_id=user_id,
        time=time,
        history=history,
        impressions=impressions,
    )


def iter_news(path: str | Path, image_dir: str | Path | None = None) -> Iterator[NewsRecord]:
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield parse_news_line(line, image_dir=image_dir)


def iter_behaviors(path: str | Path) -> Iterator[BehaviorRecord]:
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield parse_behavior_line(line)


def load_news_map(
    train_news_path: str | Path,
    dev_news_path: str | Path,
    image_dir: str | Path,
) -> dict[str, NewsRecord]:
    news: dict[str, NewsRecord] = {}
    for path in [train_news_path, dev_news_path]:
        for record in iter_news(path, image_dir=image_dir):
            news.setdefault(record.news_id, record)
    return news


def _sample_training_records(
    behaviors: Iterable[BehaviorRecord],
    negative_ratio: int,
    max_impressions: int | None,
    seed: int,
) -> Iterator[dict]:
    rng = random.Random(seed)
    seen = 0
    for behavior in behaviors:
        if max_impressions is not None and seen >= max_impressions:
            break
        seen += 1
        positives = [news_id for news_id, label in behavior.impressions if label == 1]
        negatives = [news_id for news_id, label in behavior.impressions if label == 0]
        if not positives or not negatives:
            continue
        rng.shuffle(negatives)
        for positive in positives:
            yield {
                "impression_id": behavior.impression_id,
                "user_id": behavior.user_id,
                "history": behavior.history,
                "candidate": positive,
                "label": 1,
            }
            for negative in negatives[:negative_ratio]:
                yield {
                    "impression_id": behavior.impression_id,
                    "user_id": behavior.user_id,
                    "history": behavior.history,
                    "candidate": negative,
                    "label": 0,
                }


def _grouped_eval_records(
    behaviors: Iterable[BehaviorRecord],
    max_impressions: int | None,
) -> Iterator[dict]:
    seen = 0
    for behavior in behaviors:
        if max_impressions is not None and seen >= max_impressions:
            break
        seen += 1
        candidates = [news_id for news_id, _ in behavior.impressions]
        labels = [label for _, label in behavior.impressions]
        if any(labels) and len(candidates) >= 2:
            yield {
                "impression_id": behavior.impression_id,
                "user_id": behavior.user_id,
                "history": behavior.history,
                "candidates": candidates,
                "labels": labels,
            }


def compute_news_frequency(
    train_behaviors_path: str | Path,
    dev_behaviors_path: str | Path,
) -> Counter[str]:
    counter: Counter[str] = Counter()
    for path in [train_behaviors_path, dev_behaviors_path]:
        for behavior in iter_behaviors(path):
            counter.update(behavior.history)
            counter.update(news_id for news_id, _ in behavior.impressions)
    return counter


def prepare_data(
    train_dir: str | Path = "MINDsmall_train",
    dev_dir: str | Path = "MINDsmall_dev",
    image_dir: str | Path = "newData",
    output_dir: str | Path = "artifacts/data",
    negative_ratio: int = 4,
    max_train_impressions: int | None = None,
    max_dev_impressions: int | None = None,
    seed: int = DEFAULT_SEED,
) -> dict[str, int]:
    train_dir = Path(train_dir)
    dev_dir = Path(dev_dir)
    image_dir = Path(image_dir)
    output_dir = ensure_dir(output_dir)

    news = load_news_map(train_dir / "news.tsv", dev_dir / "news.tsv", image_dir=image_dir)
    missing_images = [news_id for news_id, record in news.items() if record.image_path is None]
    if missing_images:
        raise FileNotFoundError(
            f"{len(missing_images)} MIND news items do not have matching images; "
            f"first missing id: {missing_images[0]}"
        )

    news_path = output_dir / "news.jsonl"
    train_path = output_dir / "train_samples.jsonl"
    dev_path = output_dir / "dev_impressions.jsonl"
    frequency_path = output_dir / "news_frequency.json"

    news_count = write_jsonl(news_path, (asdict(record) for record in news.values()))
    train_count = write_jsonl(
        train_path,
        _sample_training_records(
            iter_behaviors(train_dir / "behaviors.tsv"),
            negative_ratio=negative_ratio,
            max_impressions=max_train_impressions,
            seed=seed,
        ),
    )
    dev_count = write_jsonl(
        dev_path,
        _grouped_eval_records(
            iter_behaviors(dev_dir / "behaviors.tsv"),
            max_impressions=max_dev_impressions,
        ),
    )

    frequency = compute_news_frequency(train_dir / "behaviors.tsv", dev_dir / "behaviors.tsv")
    with frequency_path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(dict(frequency.most_common()), file, ensure_ascii=False)

    return {
        "news": news_count,
        "train_samples": train_count,
        "dev_impressions": dev_count,
        "missing_images": len(missing_images),
    }
