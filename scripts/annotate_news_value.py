from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import time
from pathlib import Path

from tqdm import tqdm

from src.data.preprocess import load_news_corpus
from src.features.news_value_annotator import NewsValueAnnotator
from src.utils.config import ExperimentConfig
from src.utils.env import load_project_env


VALUE_DIMENSIONS = ("conflict", "importance", "prominence", "proximity", "interest")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    load_project_env()

    parser = argparse.ArgumentParser(description="离线标注新闻价值五要素。")
    parser.add_argument("--provider", choices=["openai-compatible", "aliyun-batch"], default="aliyun-batch")
    parser.add_argument("--model", type=str, default=os.getenv("NEWS_VALUE_MODEL", "qwen-plus"))
    parser.add_argument(
        "--base-url",
        type=str,
        default=(
            os.getenv("NEWS_VALUE_API_BASE")
            or os.getenv("ALIYUN_BATCH_BASE_URL")
            or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ),
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=(
            os.getenv("NEWS_VALUE_API_KEY")
            or os.getenv("ALIYUN_BATCH_API_KEY")
            or os.getenv("DASHSCOPE_API_KEY")
            or os.getenv("MODELSCOPE_TOKEN")
        ),
    )
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--batch-endpoint", type=str, default=os.getenv("ALIYUN_BATCH_ENDPOINT", "/v1/chat/completions"))
    parser.add_argument("--completion-window", type=str, default=os.getenv("ALIYUN_BATCH_COMPLETION_WINDOW", "24h"))
    parser.add_argument("--poll-interval", type=float, default=float(os.getenv("ALIYUN_BATCH_POLL_INTERVAL", "60")))
    parser.add_argument("--batch-input-file", type=str, default=None, help="Batch 输入 JSONL 本地保存路径。")
    parser.add_argument("--batch-output-file", type=str, default=None, help="Batch 成功结果 JSONL 本地保存路径。")
    parser.add_argument("--batch-error-file", type=str, default=None, help="Batch 失败结果 JSONL 本地保存路径。")
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
    if args.single_mode and args.provider == "aliyun-batch":
        parser.error("aliyun-batch 不支持 --single-* 单条模式，请改用 --provider openai-compatible。")
    if args.poll_interval <= 0:
        parser.error("--poll-interval 必须大于 0。")

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
        batch_endpoint=args.batch_endpoint,
        completion_window=args.completion_window,
        poll_interval=args.poll_interval,
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

    existing_scores: dict[str, list[int]] = {}
    if config.news_value_file.exists() and not args.overwrite:
        existing_scores = json.loads(config.news_value_file.read_text(encoding="utf-8"))

    pending_items = [
        (news_id, article)
        for news_id, article in items
        if args.overwrite or news_id not in existing_scores
    ]
    if args.limit is not None:
        pending_items = pending_items[: args.limit]

    if not pending_items:
        print("No pending news items to annotate.")
        return

    run_started_at = datetime.now(timezone.utc)

    if args.provider == "aliyun-batch":
        batch_input_file = Path(args.batch_input_file) if args.batch_input_file else config.data_dir / "news_value_batch_input.jsonl"
        batch_output_file = Path(args.batch_output_file) if args.batch_output_file else config.data_dir / "news_value_batch_output.jsonl"
        batch_error_file = Path(args.batch_error_file) if args.batch_error_file else config.data_dir / "news_value_batch_error.jsonl"

        batch_result = annotator.annotate_batch(
            pending_items,
            input_file_path=batch_input_file,
            output_file_path=batch_output_file,
            error_file_path=batch_error_file,
        )

        existing_scores.update(batch_result.scores)
        config.news_value_file.write_text(json.dumps(existing_scores, ensure_ascii=False, indent=2), encoding="utf-8")

        failures_file = config.data_dir / "news_value_scores.failures.json"
        failures_file.write_text(json.dumps(batch_result.failures, ensure_ascii=False, indent=2), encoding="utf-8")

        run_finished_at = datetime.now(timezone.utc)
        meta = {
            "provider": args.provider,
            "model": args.model,
            "base_url": args.base_url,
            "batch_endpoint": args.batch_endpoint,
            "completion_window": args.completion_window,
            "batch_job_id": batch_result.batch_job_id,
            "batch_status": batch_result.status,
            "input_file_id": batch_result.input_file_id,
            "output_file_id": batch_result.output_file_id,
            "error_file_id": batch_result.error_file_id,
            "run_started_at": run_started_at.isoformat(),
            "run_finished_at": run_finished_at.isoformat(),
            "requested": len(pending_items),
            "succeeded": len(batch_result.scores),
            "failed": len(batch_result.failures),
            "saved_total": len(existing_scores),
        }
        meta_file = config.data_dir / "news_value_scores.meta.json"
        meta_file.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"Batch job {batch_result.batch_job_id} completed: {len(batch_result.scores)} succeeded, {len(batch_result.failures)} failed.")
        print(f"Saved {len(existing_scores)} news value entries to {config.news_value_file}")
        print(f"Meta saved to {meta_file}")
        return

    for news_id, article in tqdm(pending_items, desc="Annotating news values"):
        existing_scores[news_id] = annotator.annotate(article)
        if args.sleep > 0:
            time.sleep(args.sleep)

    config.news_value_file.write_text(json.dumps(existing_scores, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(existing_scores)} news value entries to {config.news_value_file}")


if __name__ == "__main__":
    main()