from __future__ import annotations

import json
import re
from dataclasses import dataclass

from openai import OpenAI


SYSTEM_PROMPT = """\
你是新闻价值标注器。请基于给定新闻的标题、摘要和类别信息，对以下五个维度分别打 1-5 分整数。

## 评分维度

1. conflict（冲突性）：新闻是否涉及对抗、争端、分歧或紧张关系。
   - 1 分：无任何冲突元素
   - 3 分：存在温和分歧或潜在矛盾
   - 5 分：涉及激烈对抗、战争、诉讼等严重冲突

2. importance（重要性）：事件对社会、经济或公众生活的影响程度。
   - 1 分：影响极小，仅涉及个别人
   - 3 分：影响一定范围的群体或行业
   - 5 分：影响国家或全球范围的重大事件

3. prominence（显著性）：涉及人物、机构或地点的知名度。
   - 1 分：涉及普通个人或无名机构
   - 3 分：涉及地区知名人物或机构
   - 5 分：涉及国际级知名人物、领导人或机构

4. proximity（接近性）：事件与普通大众日常生活的心理关联程度。
   - 1 分：与大多数人生活无关
   - 3 分：与部分人群的日常生活相关
   - 5 分：直接关系到每个人的切身利益

5. interest（趣味性）：内容的吸引力和可读性。
   - 1 分：枯燥乏味，无吸引力
   - 3 分：有一定趣味或话题性
   - 5 分：极具吸引力，引人入胜

## 输出格式

只返回 JSON 对象，格式如下：
{"conflict": <int>, "importance": <int>, "prominence": <int>, "proximity": <int>, "interest": <int>}"""


_CONFLICT_KEYWORDS = frozenset((
    "war", "conflict", "dispute", "protest", "lawsuit", "attack", "oppose",
    "fight", "debate", "crisis", "strike", "clash", "battle", "sue",
    "accuse", "condemn", "threat", "sanction", "ban", "arrest",
))

_PROXIMITY_KEYWORDS = frozenset((
    "price", "tax", "job", "employment", "wage", "salary", "health",
    "hospital", "school", "education", "housing", "rent", "food",
    "inflation", "insurance", "retirement", "pension", "traffic", "commute",
))


def heuristic_news_value_scores(article: dict[str, str]) -> list[int]:
    text = f"{article.get('title', '')} {article.get('abstract', '')}".lower()
    conflict = 4 if any(word in text for word in _CONFLICT_KEYWORDS) else 2
    importance = 4 if article.get("category") in {"news", "finance", "health"} else 3
    prominence = 4 if re.search(r"[A-Z][a-z]+ [A-Z][a-z]+", article.get("title", "")) else 3
    proximity = 4 if any(word in text for word in _PROXIMITY_KEYWORDS) else 3
    interest = 4 if article.get("category") in {"sports", "entertainment", "lifestyle"} else 3
    return [conflict, importance, prominence, proximity, interest]


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
    provider: str = "heuristic"
    base_url: str | None = None
    api_key: str | None = None
    timeout: float = 60.0

    def annotate(self, article: dict[str, str]) -> list[int]:
        if self.provider == "heuristic":
            return heuristic_news_value_scores(article)

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