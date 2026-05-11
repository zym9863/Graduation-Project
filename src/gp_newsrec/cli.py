from __future__ import annotations

import argparse
import json
from pathlib import Path

from .constants import DEFAULT_SEED
from .cross_validation import CROSS_MODEL_EXAMPLE_IDS, analyze_cross_model_validation, prepare_cross_model_sample
from .features import extract_siglip_features
from .labels import label_values
from .mind import prepare_data
from .modeling import evaluate_many, train


def _print_json(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gpnews", description="Multimodal MIND news recommendation toolkit.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-data", help="Parse MIND-small and build train/dev caches.")
    prepare.add_argument("--train-dir", default="MINDsmall_train")
    prepare.add_argument("--dev-dir", default="MINDsmall_dev")
    prepare.add_argument("--image-dir", default="newData")
    prepare.add_argument("--output-dir", default="artifacts/data")
    prepare.add_argument("--negative-ratio", type=int, default=4)
    prepare.add_argument("--max-train-impressions", type=int)
    prepare.add_argument("--max-dev-impressions", type=int)
    prepare.add_argument("--seed", type=int, default=DEFAULT_SEED)

    siglip = subparsers.add_parser("extract-siglip", help="Precompute SigLIP text and image features.")
    siglip.add_argument("--data-dir", default="artifacts/data")
    siglip.add_argument("--image-dir", default="newData")
    siglip.add_argument("--output-dir", default="artifacts/features/siglip")
    siglip.add_argument("--model-name", default="google/siglip-base-patch16-224")
    siglip.add_argument("--batch-size", type=int, default=64)
    siglip.add_argument("--device", default="auto")
    siglip.add_argument("--max-news", type=int)

    value = subparsers.add_parser("label-values", help="Annotate news value dimensions with an LLM API.")
    value.add_argument("--data-dir", default="artifacts/data")
    value.add_argument("--output-path", default="artifacts/labels/news_value_labels.jsonl")
    value.add_argument("--max-news", type=int, default=3000)
    value.add_argument("--sleep-seconds", type=float, default=0.0)
    value.add_argument("--backend", choices=["realtime", "aliyun-batch"], default="realtime")
    value.add_argument("--batch-model", default="qwen3.5-flash")
    value.add_argument("--batch-base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    value.add_argument("--completion-window", default="24h")
    value.add_argument("--poll-interval", type=float, default=60.0)
    value.add_argument("--submit-only", action="store_true")
    value.add_argument("--batch-id")
    value.add_argument("--batch-run-dir")
    value.add_argument("--sample-path", help="Restrict labeling to news IDs listed in a JSONL sample file.")

    cross_sample = subparsers.add_parser(
        "prepare-cross-label-sample",
        help="Create the 300-item stratified sample for qwen3.5-plus cross-model validation.",
    )
    cross_sample.add_argument("--data-dir", default="artifacts/data")
    cross_sample.add_argument("--flash-label-path", default="artifacts/labels/news_value_labels.jsonl")
    cross_sample.add_argument("--output-path", default="artifacts/labels/cross_model_sample.jsonl")
    cross_sample.add_argument("--seed", type=int, default=DEFAULT_SEED)
    cross_sample.add_argument(
        "--anchor-news-id",
        action="append",
        dest="anchor_news_ids",
        help="News ID to force into the validation sample; can be passed multiple times.",
    )

    cross_analyze = subparsers.add_parser(
        "analyze-cross-labels",
        help="Compute agreement statistics and thesis figures for flash vs plus value labels.",
    )
    cross_analyze.add_argument("--sample-path", default="artifacts/labels/cross_model_sample.jsonl")
    cross_analyze.add_argument("--flash-label-path", default="artifacts/labels/news_value_labels.jsonl")
    cross_analyze.add_argument(
        "--plus-label-path",
        default="artifacts/labels/news_value_labels_qwen35_plus_sample.jsonl",
    )
    cross_analyze.add_argument("--output-dir", default="artifacts/thesis")
    cross_analyze.add_argument("--no-figures", action="store_true")
    cross_analyze.add_argument("--allow-partial", action="store_true")

    train_parser = subparsers.add_parser("train", help="Train one recommender experiment from a YAML config.")
    train_parser.add_argument("--config", required=True)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate trained checkpoints and write an ablation CSV.")
    evaluate.add_argument(
        "--configs",
        nargs="*",
        default=["configs/text.yaml", "configs/multimodal.yaml", "configs/value.yaml"],
    )
    evaluate.add_argument("--output-path", default="artifacts/reports/ablation.csv")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "prepare-data":
        _print_json(
            prepare_data(
                train_dir=args.train_dir,
                dev_dir=args.dev_dir,
                image_dir=args.image_dir,
                output_dir=args.output_dir,
                negative_ratio=args.negative_ratio,
                max_train_impressions=args.max_train_impressions,
                max_dev_impressions=args.max_dev_impressions,
                seed=args.seed,
            )
        )
    elif args.command == "extract-siglip":
        _print_json(
            extract_siglip_features(
                data_dir=args.data_dir,
                image_dir=args.image_dir,
                output_dir=args.output_dir,
                model_name=args.model_name,
                batch_size=args.batch_size,
                device=args.device,
                max_news=args.max_news,
            )
        )
    elif args.command == "label-values":
        _print_json(
            label_values(
                data_dir=args.data_dir,
                output_path=args.output_path,
                max_news=args.max_news,
                sleep_seconds=args.sleep_seconds,
                backend=args.backend,
                batch_model=args.batch_model,
                batch_base_url=args.batch_base_url,
                completion_window=args.completion_window,
                poll_interval=args.poll_interval,
                submit_only=args.submit_only,
                batch_id=args.batch_id,
                batch_run_dir=args.batch_run_dir,
                sample_path=args.sample_path,
            )
        )
    elif args.command == "prepare-cross-label-sample":
        _print_json(
            prepare_cross_model_sample(
                data_dir=args.data_dir,
                flash_label_path=args.flash_label_path,
                output_path=args.output_path,
                seed=args.seed,
                anchor_news_ids=args.anchor_news_ids
                if args.anchor_news_ids is not None
                else CROSS_MODEL_EXAMPLE_IDS,
            )
        )
    elif args.command == "analyze-cross-labels":
        _print_json(
            analyze_cross_model_validation(
                sample_path=args.sample_path,
                flash_label_path=args.flash_label_path,
                plus_label_path=args.plus_label_path,
                output_dir=args.output_dir,
                write_figures=not args.no_figures,
                strict=not args.allow_partial,
            )
        )
    elif args.command == "train":
        _print_json(train(Path(args.config)))
    elif args.command == "evaluate":
        _print_json(evaluate_many(args.configs, output_path=args.output_path))
    else:
        parser.error(f"Unknown command: {args.command}")
