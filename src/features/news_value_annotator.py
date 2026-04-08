from __future__ import annotations

import io
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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

    def _pick_value(payload: dict[str, Any], *keys: str) -> Any:
        for key in keys:
            if key in payload:
                return payload[key]
        return None

    if isinstance(parsed, list):
        values = parsed
    else:
        values = [
            _pick_value(parsed, "conflict", "冲突性"),
            _pick_value(parsed, "importance", "重要性"),
            _pick_value(parsed, "prominence", "显著性"),
            _pick_value(parsed, "proximity", "接近性"),
            _pick_value(parsed, "interest", "趣味性"),
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
class BatchAnnotationResult:
    scores: dict[str, list[int]]
    failures: dict[str, str]
    batch_job_id: str
    status: str
    input_file_id: str
    output_file_id: str | None
    error_file_id: str | None


@dataclass(slots=True)
class NewsValueAnnotator:
    model: str
    provider: str = "openai-compatible"
    base_url: str | None = None
    api_key: str | None = None
    timeout: float = 60.0
    batch_endpoint: str = "/v1/chat/completions"
    completion_window: str = "24h"
    poll_interval: float = 60.0

    def _create_client(self) -> OpenAI:
        if not self.api_key:
            raise ValueError("api_key is required for news value provider.")

        if not self.base_url:
            raise ValueError("base_url is required for news value provider.")

        return OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout)

    def _build_messages(self, article: dict[str, str]) -> list[dict[str, str]]:
        return [
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
        ]

    @staticmethod
    def _to_text(file_content: Any) -> str:
        text_attr = getattr(file_content, "text", None)
        if callable(text_attr):
            return str(text_attr())
        if isinstance(text_attr, str):
            return text_attr

        content_attr = getattr(file_content, "content", None)
        if isinstance(content_attr, bytes):
            return content_attr.decode("utf-8")
        if isinstance(content_attr, str):
            return content_attr

        read = getattr(file_content, "read", None)
        if callable(read):
            value = read()
            if isinstance(value, bytes):
                return value.decode("utf-8")
            return str(value)

        return str(file_content)

    @staticmethod
    def _extract_message_content(line_obj: dict[str, Any]) -> str:
        response = line_obj.get("response")
        if not isinstance(response, dict):
            raise ValueError("Missing response object in batch output line.")

        status_code = response.get("status_code")
        if status_code != 200:
            raise ValueError(f"Unexpected status code: {status_code}")

        body = response.get("body")
        if not isinstance(body, dict):
            raise ValueError("Missing response body in batch output line.")

        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("Missing response choices in batch output line.")

        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise ValueError("Invalid first choice in batch output line.")

        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise ValueError("Missing assistant message in batch output line.")

        content = message.get("content")
        if content is None:
            raise ValueError("Empty assistant message content in batch output line.")
        if isinstance(content, str):
            return content
        return json.dumps(content, ensure_ascii=False)

    @staticmethod
    def _parse_error_message(line_obj: dict[str, Any]) -> str:
        error = line_obj.get("error")
        if isinstance(error, str) and error:
            return error
        if isinstance(error, dict):
            message = error.get("message")
            code = error.get("code")
            if message and code:
                return f"{code}: {message}"
            if message:
                return str(message)
            if code:
                return str(code)
            return json.dumps(error, ensure_ascii=False)
        return "Unknown batch error"

    def annotate(self, article: dict[str, str]) -> list[int]:
        if self.provider != "openai-compatible":
            raise ValueError(f"Unsupported provider: {self.provider}")

        client = self._create_client()
        response = client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=self._build_messages(article),
        )

        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty completion content from news value provider.")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        return parse_news_value_response(content)

    def annotate_batch(
        self,
        articles_by_id: list[tuple[str, dict[str, str]]],
        *,
        input_file_path: Path | None = None,
        output_file_path: Path | None = None,
        error_file_path: Path | None = None,
    ) -> BatchAnnotationResult:
        if self.provider != "aliyun-batch":
            raise ValueError("annotate_batch requires provider='aliyun-batch'.")
        if not articles_by_id:
            raise ValueError("annotate_batch requires at least one article.")

        client = self._create_client()

        request_lines: list[str] = []
        for news_id, article in articles_by_id:
            request_obj = {
                "custom_id": news_id,
                "method": "POST",
                "url": self.batch_endpoint,
                "body": {
                    "model": self.model,
                    "temperature": 0,
                    "response_format": {"type": "json_object"},
                    "messages": self._build_messages(article),
                },
            }
            request_lines.append(json.dumps(request_obj, ensure_ascii=False))

        input_jsonl = "\n".join(request_lines)
        if input_file_path is not None:
            input_file_path.parent.mkdir(parents=True, exist_ok=True)
            input_file_path.write_text(input_jsonl + "\n", encoding="utf-8")

        buffer = io.BytesIO((input_jsonl + "\n").encode("utf-8"))
        buffer.name = "news_value_batch_input.jsonl"
        input_file = client.files.create(file=buffer, purpose="batch")

        batch = client.batches.create(
            input_file_id=input_file.id,
            endpoint=self.batch_endpoint,
            completion_window=self.completion_window,
        )

        terminal_statuses = {"completed", "failed", "cancelled", "canceled", "expired"}
        while getattr(batch, "status", None) not in terminal_statuses:
            time.sleep(max(1.0, self.poll_interval))
            batch = client.batches.retrieve(batch.id)

        if batch.status != "completed":
            raise RuntimeError(f"Batch job ended with status '{batch.status}'.")

        output_file_id = getattr(batch, "output_file_id", None)
        if not output_file_id:
            raise RuntimeError("Completed batch job missing output_file_id.")

        output_text = self._to_text(client.files.content(output_file_id))
        if output_file_path is not None:
            output_file_path.parent.mkdir(parents=True, exist_ok=True)
            output_file_path.write_text(output_text, encoding="utf-8")

        scores: dict[str, list[int]] = {}
        failures: dict[str, str] = {}

        for line in output_text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            line_obj = json.loads(stripped)
            custom_id = str(line_obj.get("custom_id") or "")
            if not custom_id:
                continue
            try:
                message_content = self._extract_message_content(line_obj)
                scores[custom_id] = parse_news_value_response(message_content)
            except Exception as exc:  # noqa: BLE001 - need robust per-line failure collection
                failures[custom_id] = str(exc)

        error_file_id = getattr(batch, "error_file_id", None)
        if error_file_id:
            error_text = self._to_text(client.files.content(error_file_id))
            if error_file_path is not None:
                error_file_path.parent.mkdir(parents=True, exist_ok=True)
                error_file_path.write_text(error_text, encoding="utf-8")

            for line in error_text.splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                line_obj = json.loads(stripped)
                custom_id = str(line_obj.get("custom_id") or "")
                if not custom_id:
                    continue
                failures[custom_id] = self._parse_error_message(line_obj)

        return BatchAnnotationResult(
            scores=scores,
            failures=failures,
            batch_job_id=batch.id,
            status=batch.status,
            input_file_id=input_file.id,
            output_file_id=output_file_id,
            error_file_id=error_file_id,
        )