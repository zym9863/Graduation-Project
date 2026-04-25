from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

from .constants import PROMPT_VERSION, VALUE_DIMENSIONS, VALUE_DIMENSION_CN
from .io import append_jsonl, ensure_dir, read_jsonl


@dataclass(frozen=True)
class ValueLabel:
    news_id: str
    prompt_version: str
    scores: dict[str, int]
    total: int
    reason: str


def build_value_prompt(news: dict[str, Any]) -> str:
    dimensions = "\n".join(
        f"- {name}: {VALUE_DIMENSION_CN[name]}，0-3 分" for name in VALUE_DIMENSIONS
    )
    return f"""你是新闻传播学研究助理。请基于新闻价值理论为一条英文新闻打分。

只使用新闻的类别、标题和摘要，不要评估时效性和接近性，因为数据集中缺少可靠发布时间、地理位置和用户位置。

维度:
{dimensions}

评分规则:
0=几乎没有该价值，1=较弱，2=明显，3=强。

请只输出 JSON，不要输出 Markdown。格式:
{{
  "scores": {{
    "importance": 0,
    "prominence": 0,
    "conflict": 0,
    "novelty": 0,
    "human_interest": 0
  }},
  "reason": "不超过40个中文字的理由"
}}

新闻:
category: {news.get("category", "")}
subcategory: {news.get("subcategory", "")}
title: {news.get("title", "")}
abstract: {news.get("abstract", "")}
"""


def validate_value_payload(news_id: str, payload: dict[str, Any]) -> ValueLabel:
    raw_scores = payload.get("scores")
    if not isinstance(raw_scores, dict):
        raise ValueError("LLM value label response must contain object field 'scores'.")
    scores: dict[str, int] = {}
    for dim in VALUE_DIMENSIONS:
        value = raw_scores.get(dim)
        if not isinstance(value, int) or value < 0 or value > 3:
            raise ValueError(f"Invalid score for {dim}: {value!r}")
        scores[dim] = value
    reason = str(payload.get("reason", "")).strip()
    return ValueLabel(
        news_id=news_id,
        prompt_version=PROMPT_VERSION,
        scores=scores,
        total=sum(scores.values()),
        reason=reason,
    )


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text


def _chat_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def request_value_label(news: dict[str, Any], timeout: float = 60.0) -> ValueLabel:
    api_key = os.environ.get("LLM_API_KEY")
    base_url = os.environ.get("LLM_BASE_URL")
    model = os.environ.get("LLM_MODEL")
    if not api_key or not base_url or not model:
        raise RuntimeError("LLM_API_KEY, LLM_BASE_URL, and LLM_MODEL must be set.")

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You return strict JSON only."},
            {"role": "user", "content": build_value_prompt(news)},
        ],
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    with httpx.Client(timeout=timeout) as client:
        response = client.post(_chat_url(base_url), headers=headers, json=payload)
        response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    parsed = json.loads(_strip_json_fences(content))
    return validate_value_payload(str(news["news_id"]), parsed)


def load_value_cache(path: str | Path) -> dict[str, ValueLabel]:
    cache_path = Path(path)
    if not cache_path.exists():
        return {}
    labels: dict[str, ValueLabel] = {}
    for record in read_jsonl(cache_path):
        if record.get("prompt_version") == PROMPT_VERSION:
            labels[record["news_id"]] = ValueLabel(
                news_id=record["news_id"],
                prompt_version=record["prompt_version"],
                scores={dim: int(record["scores"][dim]) for dim in VALUE_DIMENSIONS},
                total=int(record["total"]),
                reason=str(record.get("reason", "")),
            )
    return labels


def load_news_records(path: str | Path) -> dict[str, dict[str, Any]]:
    return {record["news_id"]: record for record in read_jsonl(path)}


def load_frequency(path: str | Path) -> dict[str, int]:
    with Path(path).open("r", encoding="utf-8") as file:
        return {str(key): int(value) for key, value in json.load(file).items()}


def label_values(
    data_dir: str | Path = "artifacts/data",
    output_path: str | Path = "artifacts/labels/news_value_labels.jsonl",
    max_news: int = 3000,
    sleep_seconds: float = 0.0,
) -> dict[str, int]:
    data_dir = Path(data_dir)
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    news = load_news_records(data_dir / "news.jsonl")
    frequency = load_frequency(data_dir / "news_frequency.json")
    cache = load_value_cache(output_path)

    ordered_ids = [
        news_id
        for news_id, _ in sorted(frequency.items(), key=lambda item: item[1], reverse=True)
        if news_id in news
    ][:max_news]
    created = 0
    skipped = 0
    for news_id in ordered_ids:
        if news_id in cache:
            skipped += 1
            continue
        label = request_value_label(news[news_id])
        append_jsonl(output_path, asdict(label))
        cache[news_id] = label
        created += 1
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    return {"created": created, "cached_in_target": skipped, "target": len(ordered_ids), "available": len(cache)}
