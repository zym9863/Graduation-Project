from __future__ import annotations

import json
from dataclasses import dataclass

from openai import OpenAI


SYSTEM_PROMPT = """Score the given news with 1-5 integers using title, abstract, category, subcategory.
1=low, 3=medium, 5=high.
conflict: disagreement/confrontation severity.
importance: impact scope on society/economy/public life.
prominence: fame of involved people/orgs/places.
proximity: relevance to ordinary people's daily life.
interest: attractiveness/readability.
Return JSON only with keys: conflict, importance, prominence, proximity, interest."""


def parse_news_value_response(raw_content: str) -> list[int]:
    parsed = json.loads(raw_content)

    if isinstance(parsed, list):
        values = parsed
    else:
        values = [
            parsed.get("conflict") or parsed.get("冲突性"),
            parsed.get("importance") or parsed.get("重要性"),
            parsed.get("prominence") or parsed.get("显著性"),
            parsed.get("proximity") or parsed.get("接近性"),
            parsed.get("interest") or parsed.get("趣味性"),
        ]

    if len(values) != 5:
        raise ValueError("News value response must contain exactly five scores.")

    normalized: list[int] = []
    for value in values:
        if value is None:
            raise ValueError("News value response contains empty score.")
        score = int(value)
        normalized.append(max(1, min(score, 5)))
    return normalized


@dataclass(slots=True)
class NewsValueAnnotator:
    model: str
    provider: str = "openai-compatible"
    base_url: str | None = None
    api_key: str | None = None
    timeout: float = 60.0

    def annotate(self, article: dict[str, str]) -> list[int]:
        if self.provider != "openai-compatible":
            raise ValueError(f"Unsupported provider: {self.provider}")

        if not self.api_key:
            raise ValueError("api_key is required for openai-compatible provider.")

        if not self.base_url:
            raise ValueError("base_url is required for openai-compatible provider.")

        client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout)
        response = client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
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
        )

        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty completion content from news value provider.")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        return parse_news_value_response(content)