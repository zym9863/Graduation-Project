from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.data.feature_analytics import (
    build_feature_statistics,
    export_feature_statistics_csv,
    export_feature_statistics_plots,
    render_feature_markdown_report,
)
from src.data.preprocess import load_news_corpus
from src.utils.config import ExperimentConfig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成多模态特征维度与 t-SNE 分析报告。")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录，默认 data/processed/feature_analytics")
    parser.add_argument("--preview-rows", type=int, default=5, help="报告中展示的特征样本行数")
    parser.add_argument("--preview-dims", type=int, default=8, help="样本向量预览维度数")
    parser.add_argument("--news-limit", type=int, default=None, help="仅用于快速验证时限制新闻条数")
    parser.add_argument("--tsne-sample-size", type=int, default=3000, help="t-SNE 最大采样数")
    parser.add_argument("--tsne-perplexity", type=float, default=30.0, help="t-SNE perplexity")
    parser.add_argument("--tsne-random-state", type=int, default=42, help="t-SNE 随机种子")
    parser.add_argument("--tsne-pca-dim", type=int, default=50, help="t-SNE 前可选 PCA 维度")
    parser.add_argument("--top-categories", type=int, default=10, help="t-SNE 类别着色保留的 Top-K 类别")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = ExperimentConfig()
    config.ensure_directories()

    output_dir = Path(args.output_dir) if args.output_dir else config.feature_analytics_dir
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    news = load_news_corpus(config.train_dir / "news.tsv", config.dev_dir / "news.tsv")
    if args.news_limit is not None:
        news = dict(list(news.items())[: args.news_limit])

    siglip_features: dict[str, object] = {}
    if config.feature_file.exists():
        siglip_features = torch.load(config.feature_file, map_location="cpu", weights_only=False)

    news_value_scores: dict[str, object] = {}
    if config.news_value_file.exists():
        news_value_scores = json.loads(config.news_value_file.read_text(encoding="utf-8"))

    statistics = build_feature_statistics(
        news_dict=news,
        siglip_features=siglip_features,
        news_value_scores=news_value_scores,
        siglip_dim=config.siglip_dim,
        preview_rows=max(1, args.preview_rows),
        preview_dims=max(1, args.preview_dims),
        tsne_sample_size=max(0, args.tsne_sample_size),
        tsne_perplexity=max(2.0, args.tsne_perplexity),
        tsne_random_state=args.tsne_random_state,
        tsne_pca_dim=max(0, args.tsne_pca_dim),
        top_categories=max(1, args.top_categories),
    )

    stats_json_path = output_dir / "feature_statistics.json"
    stats_json_path.write_text(json.dumps(statistics, ensure_ascii=False, indent=2), encoding="utf-8")

    table_paths = export_feature_statistics_csv(statistics, tables_dir)
    figure_paths = export_feature_statistics_plots(statistics, figures_dir)

    report_path = output_dir / "data_report.md"
    render_feature_markdown_report(statistics, report_path, figure_paths, table_paths)

    print(f"Saved statistics json: {stats_json_path}")
    print(f"Saved report: {report_path}")
    print(f"Saved figures to: {figures_dir}")
    print(f"Saved tables to: {tables_dir}")


if __name__ == "__main__":
    main()
