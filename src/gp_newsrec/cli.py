from __future__ import annotations

import argparse
import json
from pathlib import Path

from .constants import DEFAULT_SEED
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
            )
        )
    elif args.command == "train":
        _print_json(train(Path(args.config)))
    elif args.command == "evaluate":
        _print_json(evaluate_many(args.configs, output_path=args.output_path))
    else:
        parser.error(f"Unknown command: {args.command}")
