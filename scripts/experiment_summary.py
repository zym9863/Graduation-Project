from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev

import torch

from src.utils.config import ExperimentConfig


_METRIC_KEYS = ("auc", "mrr", "ndcg@5", "ndcg@10")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总多个 checkpoint 的实验指标并生成对比表。")
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=None,
        help="checkpoint 文件路径列表；不传时会按 --glob-pattern 自动搜索。",
    )
    parser.add_argument(
        "--glob-pattern",
        type=str,
        default="data/processed/*.pt",
        help="自动搜索 checkpoint 的 glob 模式（相对项目根目录）。",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed/experiment_reports",
        help="结果输出目录。",
    )
    return parser.parse_args(argv)


def _resolve_checkpoints(args: argparse.Namespace, project_root: Path) -> list[Path]:
    if args.inputs:
        paths = [project_root / item for item in args.inputs]
    else:
        paths = sorted(project_root.glob(args.glob_pattern))
    return [path for path in paths if path.exists() and path.is_file()]


def _extract_record(path: Path) -> dict[str, object] | None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    metrics = checkpoint.get("metrics")
    if not isinstance(metrics, dict):
        return None

    config = checkpoint.get("config") if isinstance(checkpoint.get("config"), dict) else {}
    record: dict[str, object] = {
        "checkpoint": str(path),
        "fusion": checkpoint.get("fusion", "unknown"),
        "epoch": checkpoint.get("epoch"),
        "seed": config.get("seed"),
        "train_loss": checkpoint.get("train_loss"),
    }
    for metric_key in _METRIC_KEYS:
        record[metric_key] = metrics.get(metric_key)
    return record


def _safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _safe_std(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def _aggregate_by_fusion(records: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for record in records:
        fusion = str(record.get("fusion", "unknown"))
        grouped.setdefault(fusion, []).append(record)

    summary: list[dict[str, object]] = []
    for fusion, items in sorted(grouped.items()):
        row: dict[str, object] = {
            "fusion": fusion,
            "runs": len(items),
        }
        for metric_key in _METRIC_KEYS:
            values = [float(item[metric_key]) for item in items if isinstance(item.get(metric_key), (int, float))]
            row[f"{metric_key}_mean"] = _safe_mean(values)
            row[f"{metric_key}_std"] = _safe_std(values)
        summary.append(row)
    return summary


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = ExperimentConfig()

    checkpoints = _resolve_checkpoints(args, config.project_root)
    records = [record for path in checkpoints if (record := _extract_record(path)) is not None]
    summary = _aggregate_by_fusion(records)

    output_dir = config.project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    details_json = output_dir / "checkpoint_metrics.json"
    details_csv = output_dir / "checkpoint_metrics.csv"
    summary_json = output_dir / "fusion_summary.json"
    summary_csv = output_dir / "fusion_summary.csv"

    details_json.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(details_csv, records)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(summary_csv, summary)

    print(f"Discovered checkpoints: {len(checkpoints)}")
    print(f"Usable checkpoints(with metrics): {len(records)}")
    print(f"Detailed results: {details_csv}")
    print(f"Fusion summary: {summary_csv}")


if __name__ == "__main__":
    main()
