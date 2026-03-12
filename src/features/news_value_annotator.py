from __future__ import annotations

import json
import re
from dataclasses import dataclass

import httpx


SYSTEM_PROMPT = """你是新闻价值标注器。请基于给定新闻内容，对以下五个要素分别打 1 到 5 分整数：timeliness、importance、prominence、proximity、interest。只返回 JSON。"""


def heuristic_news_value_scores(article: dict[str, str]) -> list[int]:
    text = f"{article.get('title', '')} {article.get('abstract', '')}".lower()
    timeliness = 4 if any(word in text for word in ("breaking", "today", "latest", "new", "just")) else 3
    importance = 4 if article.get("category") in {"news", "finance", "health"} else 3
    prominence = 4 if re.search(r"[A-Z][a-z]+ [A-Z][a-z]+", article.get("title", "")) else 3
    proximity = 3
    interest = 4 if article.get("category") in {"sports", "entertainment", "lifestyle"} else 3
    return [timeliness, importance, prominence, proximity, interest]


def parse_news_value_response(raw_content: str) -> list[int]:
    parsed = json.loads(raw_content)

    if isinstance(parsed, list):
        values = parsed
    else:
        values = [
            parsed.get("timeliness") or parsed.get("时新性"),
            parsed.get("importance") or parsed.get("重要性"),
            parsed.get("prominence") or parsed.get("显著性"),
            parsed.get("proximity") or parsed.get("接近性"),
            parsed.get("interest") or parsed.get("趣味性"),
        ]

    normalized: list[int] = []
    for value in values:
        score = int(value)
        normalized.append(max(1, min(score, 5)))
    if len(normalized) != 5:
        raise ValueError("News value response must contain exactly five scores.")
    return normalized


@dataclass(slots=True)
class NewsValueAnnotator:
    model: str
    provider: str = "heuristic"
    base_url: str | None = None
    api_key: str | None = None
    timeout: float = 60.0

    def annotate(self, article: dict[str, str]) -> list[int]:
        if self.provider == "heuristic" or not self.api_key:
            return heuristic_news_value_scores(article)

        if self.provider != "openai-compatible":
            raise ValueError(f"Unsupported provider: {self.provider}")

        if not self.base_url:
            raise ValueError("base_url is required for openai-compatible provider.")

        payload = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "title": article.get("title", ""),
                            "abstract": article.get("abstract", ""),
                            "category": article.get("category", ""),
                            "subcategory": article.get("subcategory", ""),
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
        }

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                self.base_url.rstrip("/") + "/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
        content = data["choices"][0]["message"]["content"]
        return parse_news_value_response(content)