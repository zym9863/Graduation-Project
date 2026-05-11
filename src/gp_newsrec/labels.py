from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from .constants import PROMPT_VERSION, VALUE_DIMENSIONS, VALUE_DIMENSION_CN
from .io import append_jsonl, ensure_dir, read_jsonl, write_jsonl


DEFAULT_ALIYUN_BATCH_MODEL = "qwen3.5-flash"
DEFAULT_ALIYUN_BATCH_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
ALIYUN_BATCH_CHAT_ENDPOINT = "/v1/chat/completions"
TERMINAL_BATCH_STATUSES = {"completed", "failed", "expired", "cancelled"}


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


def _api_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _value_chat_body(news: dict[str, Any], model: str, enable_thinking: bool | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You return strict JSON only."},
            {"role": "user", "content": build_value_prompt(news)},
        ],
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
    }
    if enable_thinking is not None:
        body["enable_thinking"] = enable_thinking
    return body


def request_value_label(news: dict[str, Any], timeout: float = 60.0) -> ValueLabel:
    api_key = os.environ.get("LLM_API_KEY")
    base_url = os.environ.get("LLM_BASE_URL")
    model = os.environ.get("LLM_MODEL")
    if not api_key or not base_url or not model:
        raise RuntimeError("LLM_API_KEY, LLM_BASE_URL, and LLM_MODEL must be set.")

    payload = _value_chat_body(news, model)
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


def _ordered_news_ids(news: dict[str, dict[str, Any]], frequency: dict[str, int], max_news: int) -> list[str]:
    return [
        news_id
        for news_id, _ in sorted(frequency.items(), key=lambda item: item[1], reverse=True)
        if news_id in news
    ][:max_news]


def _sample_news_ids(sample_path: str | Path, news: dict[str, dict[str, Any]], max_news: int) -> list[str]:
    ids: list[str] = []
    missing: list[str] = []
    seen: set[str] = set()
    for record in read_jsonl(sample_path):
        news_id = str(record.get("news_id", "")).strip()
        if not news_id or news_id in seen:
            continue
        seen.add(news_id)
        if news_id not in news:
            missing.append(news_id)
            continue
        ids.append(news_id)
        if len(ids) >= max_news:
            break
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"Sample file contains news IDs not found in data: {preview}")
    return ids


def build_aliyun_batch_request(
    news: dict[str, Any],
    model: str = DEFAULT_ALIYUN_BATCH_MODEL,
) -> dict[str, Any]:
    return {
        "custom_id": str(news["news_id"]),
        "method": "POST",
        "url": ALIYUN_BATCH_CHAT_ENDPOINT,
        "body": _value_chat_body(news, model, enable_thinking=False),
    }


def prepare_aliyun_batch_input(
    data_dir: str | Path = "artifacts/data",
    output_path: str | Path = "artifacts/labels/news_value_labels.jsonl",
    input_path: str | Path = "artifacts/labels/batches/input.jsonl",
    max_news: int = 3000,
    model: str = DEFAULT_ALIYUN_BATCH_MODEL,
    sample_path: str | Path | None = None,
) -> dict[str, Any]:
    data_dir = Path(data_dir)
    output_path = Path(output_path)
    input_path = Path(input_path)
    news = load_news_records(data_dir / "news.jsonl")
    cache = load_value_cache(output_path)
    if sample_path is None:
        frequency = load_frequency(data_dir / "news_frequency.json")
        ordered_ids = _ordered_news_ids(news, frequency, max_news)
    else:
        ordered_ids = _sample_news_ids(sample_path, news, max_news)
    pending_ids = [news_id for news_id in ordered_ids if news_id not in cache]
    request_count = write_jsonl(
        input_path,
        (build_aliyun_batch_request(news[news_id], model=model) for news_id in pending_ids),
    )
    return {
        "request_count": request_count,
        "cached_in_target": len(ordered_ids) - len(pending_ids),
        "target": len(ordered_ids),
        "available": len(cache),
        "sample_path": str(sample_path) if sample_path is not None else "",
    }


def _default_batch_run_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("artifacts/labels/batches") / f"news-value-{timestamp}"


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")


def _get_aliyun_api_key() -> str:
    api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("LLM_API_KEY")
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY or LLM_API_KEY must be set for Aliyun Batch.")
    return api_key


def _upload_batch_file(client: httpx.Client, base_url: str, api_key: str, input_path: Path) -> str:
    headers = {"Authorization": f"Bearer {api_key}"}
    with input_path.open("rb") as file:
        response = client.post(
            _api_url(base_url, "/files"),
            headers=headers,
            data={"purpose": "batch"},
            files={"file": (input_path.name, file, "application/jsonl")},
        )
    response.raise_for_status()
    return str(response.json()["id"])


def _create_batch_job(
    client: httpx.Client,
    base_url: str,
    api_key: str,
    input_file_id: str,
    completion_window: str,
    request_count: int,
) -> dict[str, Any]:
    payload = {
        "input_file_id": input_file_id,
        "endpoint": ALIYUN_BATCH_CHAT_ENDPOINT,
        "completion_window": completion_window,
        "metadata": {
            "ds_name": "news-value-labels",
            "ds_description": f"News value labeling: {request_count} requests",
        },
    }
    response = client.post(
        _api_url(base_url, "/batches"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
    )
    response.raise_for_status()
    return dict(response.json())


def _retrieve_batch(client: httpx.Client, base_url: str, api_key: str, batch_id: str) -> dict[str, Any]:
    response = client.get(
        _api_url(base_url, f"/batches/{batch_id}"),
        headers={"Authorization": f"Bearer {api_key}"},
    )
    response.raise_for_status()
    return dict(response.json())


def _download_file(client: httpx.Client, base_url: str, api_key: str, file_id: str, path: Path) -> None:
    response = client.get(
        _api_url(base_url, f"/files/{file_id}/content"),
        headers={"Authorization": f"Bearer {api_key}"},
    )
    response.raise_for_status()
    ensure_dir(path.parent)
    path.write_bytes(response.content)


def _extract_batch_content(record: dict[str, Any]) -> str:
    response = record.get("response")
    if not isinstance(response, dict):
        raise ValueError("Batch result row does not contain object field 'response'.")
    status_code = int(response.get("status_code", 0))
    if status_code != 200:
        raise ValueError(f"Batch result status_code is {status_code}.")
    body = response.get("body")
    if not isinstance(body, dict):
        raise ValueError("Batch result response does not contain object field 'body'.")
    return str(body["choices"][0]["message"]["content"])


def merge_aliyun_batch_results(
    result_path: str | Path,
    output_path: str | Path = "artifacts/labels/news_value_labels.jsonl",
    invalid_path: str | Path = "artifacts/labels/batches/invalid_results.jsonl",
) -> dict[str, int]:
    result_path = Path(result_path)
    output_path = Path(output_path)
    invalid_path = Path(invalid_path)
    ensure_dir(output_path.parent)
    ensure_dir(invalid_path.parent)
    cache = load_value_cache(output_path)
    stats = {"created": 0, "duplicates": 0, "invalid": 0, "processed": 0}

    with result_path.open("r", encoding="utf-8") as source, invalid_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as invalid_file:
        for raw_line in source:
            line = raw_line.strip()
            if not line:
                continue
            stats["processed"] += 1
            news_id = ""
            try:
                record = json.loads(line)
                news_id = str(record.get("custom_id", "")).strip()
                if not news_id:
                    raise ValueError("Batch result row is missing custom_id.")
                if news_id in cache:
                    stats["duplicates"] += 1
                    continue
                content = _extract_batch_content(record)
                parsed = json.loads(_strip_json_fences(content))
                label = validate_value_payload(news_id, parsed)
            except Exception as exc:
                stats["invalid"] += 1
                invalid_file.write(
                    json.dumps(
                        {
                            "custom_id": news_id,
                            "error": str(exc),
                            "line": line,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                )
                invalid_file.write("\n")
                continue
            append_jsonl(output_path, asdict(label))
            cache[news_id] = label
            stats["created"] += 1
    return stats


def label_values_aliyun_batch(
    data_dir: str | Path = "artifacts/data",
    output_path: str | Path = "artifacts/labels/news_value_labels.jsonl",
    max_news: int = 3000,
    batch_model: str = DEFAULT_ALIYUN_BATCH_MODEL,
    batch_base_url: str = DEFAULT_ALIYUN_BATCH_BASE_URL,
    completion_window: str = "24h",
    poll_interval: float = 60.0,
    submit_only: bool = False,
    batch_id: str | None = None,
    batch_run_dir: str | Path | None = None,
    timeout: float = 60.0,
    sample_path: str | Path | None = None,
) -> dict[str, Any]:
    run_dir = Path(batch_run_dir) if batch_run_dir is not None else _default_batch_run_dir()
    ensure_dir(run_dir)
    input_path = run_dir / "input.jsonl"
    result_path = run_dir / "result.jsonl"
    error_path = run_dir / "error.jsonl"
    invalid_path = run_dir / "invalid_results.jsonl"
    manifest_path = run_dir / "manifest.json"

    manifest = _load_manifest(manifest_path)
    manifest.update(
        {
            "prompt_version": PROMPT_VERSION,
            "model": batch_model,
            "base_url": batch_base_url,
            "endpoint": ALIYUN_BATCH_CHAT_ENDPOINT,
            "completion_window": completion_window,
            "input_path": str(input_path),
            "result_path": str(result_path),
            "error_path": str(error_path),
            "invalid_path": str(invalid_path),
            "label_output_path": str(output_path),
            "run_dir": str(run_dir),
            "sample_path": str(sample_path) if sample_path is not None else "",
        }
    )

    input_stats: dict[str, int] | None = None
    if not batch_id:
        input_stats = prepare_aliyun_batch_input(
            data_dir=data_dir,
            output_path=output_path,
            input_path=input_path,
            max_news=max_news,
            model=batch_model,
            sample_path=sample_path,
        )
        manifest.update(input_stats)
        manifest["created_at"] = datetime.now().isoformat(timespec="seconds")
        if input_stats["request_count"] == 0:
            manifest["status"] = "no_requests"
            _save_manifest(manifest_path, manifest)
            return {**input_stats, "status": "no_requests", "manifest_path": str(manifest_path)}

    api_key = _get_aliyun_api_key()
    with httpx.Client(timeout=timeout) as client:
        if not batch_id:
            input_file_id = _upload_batch_file(client, batch_base_url, api_key, input_path)
            batch = _create_batch_job(
                client,
                batch_base_url,
                api_key,
                input_file_id,
                completion_window,
                input_stats["request_count"] if input_stats is not None else 0,
            )
            batch_id = str(batch["id"])
            manifest.update(
                {
                    "input_file_id": input_file_id,
                    "batch_id": batch_id,
                    "status": batch.get("status", "submitted"),
                    "batch": batch,
                }
            )
            _save_manifest(manifest_path, manifest)
            if submit_only:
                return {
                    **input_stats,
                    "status": manifest["status"],
                    "batch_id": batch_id,
                    "manifest_path": str(manifest_path),
                }
        else:
            manifest["batch_id"] = batch_id
            _save_manifest(manifest_path, manifest)

        batch = _retrieve_batch(client, batch_base_url, api_key, batch_id)
        while batch.get("status") not in TERMINAL_BATCH_STATUSES:
            manifest.update({"status": batch.get("status"), "batch": batch})
            _save_manifest(manifest_path, manifest)
            time.sleep(poll_interval)
            batch = _retrieve_batch(client, batch_base_url, api_key, batch_id)

        status = str(batch.get("status"))
        manifest.update({"status": status, "batch": batch})
        _save_manifest(manifest_path, manifest)
        if status != "completed":
            return {
                "status": status,
                "batch_id": batch_id,
                "manifest_path": str(manifest_path),
                "request_counts": batch.get("request_counts", {}),
            }

        output_file_id = batch.get("output_file_id")
        if output_file_id:
            _download_file(client, batch_base_url, api_key, str(output_file_id), result_path)
            manifest["output_file_id"] = output_file_id
        error_file_id = batch.get("error_file_id")
        if error_file_id:
            _download_file(client, batch_base_url, api_key, str(error_file_id), error_path)
            manifest["error_file_id"] = error_file_id
        _save_manifest(manifest_path, manifest)

    merge_stats = {"created": 0, "duplicates": 0, "invalid": 0, "processed": 0}
    if result_path.exists():
        merge_stats = merge_aliyun_batch_results(result_path, output_path, invalid_path)
    manifest["merge_stats"] = merge_stats
    _save_manifest(manifest_path, manifest)
    return {
        **merge_stats,
        "status": status,
        "batch_id": batch_id,
        "manifest_path": str(manifest_path),
        "request_counts": batch.get("request_counts", {}),
    }


def label_values(
    data_dir: str | Path = "artifacts/data",
    output_path: str | Path = "artifacts/labels/news_value_labels.jsonl",
    max_news: int = 3000,
    sleep_seconds: float = 0.0,
    backend: str = "realtime",
    batch_model: str = DEFAULT_ALIYUN_BATCH_MODEL,
    batch_base_url: str = DEFAULT_ALIYUN_BATCH_BASE_URL,
    completion_window: str = "24h",
    poll_interval: float = 60.0,
    submit_only: bool = False,
    batch_id: str | None = None,
    batch_run_dir: str | Path | None = None,
    sample_path: str | Path | None = None,
) -> dict[str, Any]:
    if backend == "aliyun-batch":
        return label_values_aliyun_batch(
            data_dir=data_dir,
            output_path=output_path,
            max_news=max_news,
            batch_model=batch_model,
            batch_base_url=batch_base_url,
            completion_window=completion_window,
            poll_interval=poll_interval,
            submit_only=submit_only,
            batch_id=batch_id,
            batch_run_dir=batch_run_dir,
            sample_path=sample_path,
        )
    if backend != "realtime":
        raise ValueError(f"Unknown label backend: {backend}")

    data_dir = Path(data_dir)
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    news = load_news_records(data_dir / "news.jsonl")
    cache = load_value_cache(output_path)

    if sample_path is None:
        frequency = load_frequency(data_dir / "news_frequency.json")
        ordered_ids = _ordered_news_ids(news, frequency, max_news)
    else:
        ordered_ids = _sample_news_ids(sample_path, news, max_news)
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
