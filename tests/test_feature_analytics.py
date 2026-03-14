from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.data.feature_analytics import (
    build_feature_statistics,
    export_feature_statistics_csv,
    export_feature_statistics_plots,
    render_feature_markdown_report,
)


def _mock_news(count: int = 12) -> dict[str, dict[str, str]]:
    categories = ["news", "sports", "finance"]
    news: dict[str, dict[str, str]] = {}
    for index in range(count):
        news_id = f"N{index + 1}"
        news[news_id] = {
            "category": categories[index % len(categories)],
            "subcategory": f"sub_{index % 4}",
            "title": f"Title {index + 1}",
            "abstract": f"Abstract {index + 1}",
        }
    return news


def _mock_siglip_features(news_ids: list[str], dim: int = 4) -> dict[str, dict[str, torch.Tensor]]:
    rng = np.random.default_rng(seed=123)
    features: dict[str, dict[str, torch.Tensor]] = {}
    for news_id in news_ids:
        features[news_id] = {
            "text_emb": torch.tensor(rng.normal(size=dim), dtype=torch.float32),
            "image_emb": torch.tensor(rng.normal(size=dim), dtype=torch.float32),
        }
    return features


def _mock_news_values(news_ids: list[str]) -> dict[str, list[float]]:
    scores: dict[str, list[float]] = {}
    for idx, news_id in enumerate(news_ids):
        base = float((idx % 5) + 1)
        scores[news_id] = [base, 5.0 - base * 0.2, 2.0 + (idx % 3), 1.0 + (idx % 4), 3.0 + (idx % 2)]
    return scores


def test_build_feature_statistics_contains_dimensions_and_tsne() -> None:
    news = _mock_news(15)
    news_ids = list(news.keys())
    siglip = _mock_siglip_features(news_ids, dim=4)
    values = _mock_news_values(news_ids)

    stats = build_feature_statistics(
        news_dict=news,
        siglip_features=siglip,
        news_value_scores=values,
        siglip_dim=4,
        preview_rows=4,
        preview_dims=3,
        tsne_sample_size=10,
        tsne_perplexity=5.0,
        tsne_random_state=7,
        tsne_pca_dim=2,
        top_categories=2,
    )

    assert stats["summary"]["text_dim_expected"] == 4
    assert stats["summary"]["image_dim_expected"] == 4
    assert stats["summary"]["fused_dim_expected"] == 8
    assert stats["summary"]["feature_coverage"] == 1.0
    assert len(stats["samples"]) == 4

    assert stats["tsne"]["sample_size"] == 10
    assert len(stats["tsne"]["text"]["points"]) == 10
    assert len(stats["tsne"]["image"]["points"]) == 10
    assert len(stats["tsne"]["fused"]["points"]) == 10
    assert len(stats["tsne"]["clustering_metrics"]) == 3


def test_export_feature_outputs_and_report(tmp_path: Path) -> None:
    news = _mock_news(12)
    news_ids = list(news.keys())
    siglip = _mock_siglip_features(news_ids, dim=4)
    values = _mock_news_values(news_ids)

    stats = build_feature_statistics(
        news_dict=news,
        siglip_features=siglip,
        news_value_scores=values,
        siglip_dim=4,
        preview_rows=5,
        preview_dims=4,
        tsne_sample_size=8,
        tsne_perplexity=4.0,
        tsne_random_state=11,
        tsne_pca_dim=2,
        top_categories=3,
    )

    figures_dir = tmp_path / "figures"
    tables_dir = tmp_path / "tables"
    report_path = tmp_path / "data_report.md"

    table_paths = export_feature_statistics_csv(stats, tables_dir)
    figure_paths = export_feature_statistics_plots(stats, figures_dir)
    render_feature_markdown_report(stats, report_path, figure_paths, table_paths)

    assert table_paths["dimension_summary"].exists()
    assert table_paths["norm_statistics"].exists()
    assert table_paths["clustering_metrics"].exists()
    assert table_paths["tsne_text_points"].exists()

    assert figure_paths["text_norm_distribution"].exists()
    assert figure_paths["tsne_text_by_category"].exists()
    assert figure_paths["tsne_fused_by_category"].exists()

    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")
    assert "# 多模态特征分析报告" in content
    assert "## 特征维度说明" in content
    assert "figures/tsne_text_by_category.png" in content
