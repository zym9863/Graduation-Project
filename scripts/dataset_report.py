from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.analytics import build_dataset_statistics, export_statistics_csv, export_statistics_plots, render_markdown_report
from src.data.preprocess import load_news_corpus, parse_behaviors_file
from src.utils.config import ExperimentConfig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成数据集统计图表与报告。")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录，默认 data/processed/analytics")
    parser.add_argument("--top-n-subcategories", type=int, default=20, help="子类别 TopN 图表数量")
    parser.add_argument("--sample-size", type=int, default=3, help="报告中展示的样例条数")
    parser.add_argument("--news-limit", type=int, default=None, help="仅用于快速验证时限制新闻条数")
    parser.add_argument("--train-behavior-limit", type=int, default=None, help="仅用于快速验证时限制训练行为条数")
    parser.add_argument("--dev-behavior-limit", type=int, default=None, help="仅用于快速验证时限制开发行为条数")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = ExperimentConfig()
    config.ensure_directories()

    output_dir = Path(args.output_dir) if args.output_dir else config.analytics_dir
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    news = load_news_corpus(config.train_dir / "news.tsv", config.dev_dir / "news.tsv")
    if args.news_limit is not None:
        news = dict(list(news.items())[: args.news_limit])

    train_behaviors = parse_behaviors_file(config.train_dir / "behaviors.tsv")
    dev_behaviors = parse_behaviors_file(config.dev_dir / "behaviors.tsv")

    if args.train_behavior_limit is not None:
        train_behaviors = train_behaviors[: args.train_behavior_limit]
    if args.dev_behavior_limit is not None:
        dev_behaviors = dev_behaviors[: args.dev_behavior_limit]

    statistics = build_dataset_statistics(
        news_dict=news,
        train_behaviors=train_behaviors,
        dev_behaviors=dev_behaviors,
        top_n_subcategories=args.top_n_subcategories,
        sample_size=args.sample_size,
    )

    stats_json_path = output_dir / "dataset_statistics.json"
    stats_json_path.write_text(json.dumps(statistics, ensure_ascii=False, indent=2), encoding="utf-8")

    table_paths = export_statistics_csv(statistics, tables_dir)
    figure_paths = export_statistics_plots(statistics, figures_dir)

    report_path = output_dir / "data_report.md"
    render_markdown_report(statistics, report_path, figure_paths, table_paths)

    print(f"Saved statistics json: {stats_json_path}")
    print(f"Saved report: {report_path}")
    print(f"Saved figures to: {figures_dir}")
    print(f"Saved tables to: {tables_dir}")


if __name__ == "__main__":
    main()
