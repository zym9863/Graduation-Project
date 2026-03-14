from __future__ import annotations

import csv
import math
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_VALUE_DIMENSIONS: tuple[str, ...] = (
    "conflict",
    "importance",
    "prominence",
    "proximity",
    "interest",
)


def build_feature_statistics(
    news_dict: dict[str, dict[str, Any]],
    siglip_features: dict[str, Any],
    news_value_scores: dict[str, Any],
    siglip_dim: int = 768,
    value_dimensions: tuple[str, ...] = DEFAULT_VALUE_DIMENSIONS,
    preview_rows: int = 5,
    preview_dims: int = 8,
    tsne_sample_size: int = 3000,
    tsne_perplexity: float = 30.0,
    tsne_random_state: int = 42,
    tsne_pca_dim: int = 50,
    top_categories: int = 10,
) -> dict[str, Any]:
    total_news = len(news_dict)
    feature_entries = len(siglip_features)
    value_entries = len(news_value_scores)

    matched_feature_ids = [news_id for news_id in news_dict if news_id in siglip_features]
    matched_value_ids = [news_id for news_id in news_dict if news_id in news_value_scores]
    matched_both_ids = [news_id for news_id in matched_feature_ids if news_id in news_value_scores]

    text_norms: list[float] = []
    image_norms: list[float] = []
    cosine_scores: list[float] = []
    sample_rows: list[dict[str, Any]] = []

    text_dim_counter: Counter[int] = Counter()
    image_dim_counter: Counter[int] = Counter()

    value_matrix_rows: list[np.ndarray] = []
    value_missing_count = 0

    tsne_ids: list[str] = []
    tsne_categories: list[str] = []
    tsne_text_vectors: list[np.ndarray] = []
    tsne_image_vectors: list[np.ndarray] = []

    for news_id in matched_feature_ids:
        article = news_dict.get(news_id, {})
        feature_entry = siglip_features.get(news_id, {})
        if not isinstance(feature_entry, dict):
            feature_entry = {}

        raw_text = _pick_feature(feature_entry, ("text_emb", "text", "title_emb"))
        raw_image = _pick_feature(feature_entry, ("image_emb", "image", "img_emb"))

        raw_text_dim = _raw_vector_dim(raw_text)
        raw_image_dim = _raw_vector_dim(raw_image)
        if raw_text_dim is not None:
            text_dim_counter[raw_text_dim] += 1
        if raw_image_dim is not None:
            image_dim_counter[raw_image_dim] += 1

        text_vector = _coerce_vector(raw_text, siglip_dim)
        image_vector = _coerce_vector(raw_image, siglip_dim)

        text_norm = float(np.linalg.norm(text_vector))
        image_norm = float(np.linalg.norm(image_vector))
        cosine = _safe_cosine_similarity(text_vector, image_vector)

        text_norms.append(text_norm)
        image_norms.append(image_norm)
        cosine_scores.append(cosine)

        value_raw = news_value_scores.get(news_id)
        if value_raw is None:
            value_missing_count += 1
            value_vector = np.zeros(len(value_dimensions), dtype=np.float32)
        else:
            value_vector = _coerce_vector(value_raw, len(value_dimensions))
        value_matrix_rows.append(value_vector)

        if len(sample_rows) < preview_rows:
            sample_rows.append(
                {
                    "news_id": news_id,
                    "category": _normalized(article.get("category")),
                    "subcategory": _normalized(article.get("subcategory")),
                    "text_norm": text_norm,
                    "image_norm": image_norm,
                    "text_image_cosine": cosine,
                    "text_preview": _preview_vector(text_vector, preview_dims),
                    "image_preview": _preview_vector(image_vector, preview_dims),
                }
            )

        tsne_ids.append(news_id)
        tsne_categories.append(_normalized(article.get("category")))
        tsne_text_vectors.append(text_vector)
        tsne_image_vectors.append(image_vector)

    value_matrix = np.stack(value_matrix_rows, axis=0) if value_matrix_rows else np.zeros((0, len(value_dimensions)), dtype=np.float32)
    value_stats_rows = _value_statistics_rows(value_matrix, value_dimensions)
    value_corr = _compute_correlation_matrix(value_matrix)

    tsne_summary = _build_tsne_summary(
        news_ids=tsne_ids,
        categories=tsne_categories,
        text_vectors=tsne_text_vectors,
        image_vectors=tsne_image_vectors,
        sample_size=tsne_sample_size,
        perplexity=tsne_perplexity,
        random_state=tsne_random_state,
        pca_dim=tsne_pca_dim,
        top_categories=top_categories,
    )

    text_dim_mode, text_dim_mode_count = _counter_mode(text_dim_counter)
    image_dim_mode, image_dim_mode_count = _counter_mode(image_dim_counter)

    summary = {
        "total_news": total_news,
        "feature_entries": feature_entries,
        "value_entries": value_entries,
        "matched_feature_entries": len(matched_feature_ids),
        "matched_value_entries": len(matched_value_ids),
        "matched_both_entries": len(matched_both_ids),
        "feature_coverage": _safe_divide(len(matched_feature_ids), total_news),
        "value_coverage": _safe_divide(len(matched_value_ids), total_news),
        "both_coverage": _safe_divide(len(matched_both_ids), total_news),
        "text_dim_expected": siglip_dim,
        "image_dim_expected": siglip_dim,
        "fused_dim_expected": siglip_dim * 2,
        "value_dim_expected": len(value_dimensions),
        "text_dim_mode": text_dim_mode,
        "text_dim_mode_count": text_dim_mode_count,
        "image_dim_mode": image_dim_mode,
        "image_dim_mode_count": image_dim_mode_count,
        "value_missing_count_in_feature_entries": value_missing_count,
    }

    feature_stats = {
        "text_norm": _summary_stats(text_norms),
        "image_norm": _summary_stats(image_norms),
        "text_image_cosine": _summary_stats(cosine_scores),
    }

    return {
        "summary": summary,
        "feature_stats": feature_stats,
        "value_dimensions": list(value_dimensions),
        "value_stats": value_stats_rows,
        "value_correlation": {
            "dimensions": list(value_dimensions),
            "matrix": value_corr.tolist(),
        },
        "samples": sample_rows,
        "tsne": tsne_summary,
    }


def export_feature_statistics_csv(statistics: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    exported: dict[str, Path] = {}

    summary = statistics["summary"]
    exported["dimension_summary"] = output_dir / "feature_dimension_summary.csv"
    _write_csv(
        exported["dimension_summary"],
        ["metric", "value"],
        [{"metric": key, "value": value} for key, value in summary.items()],
    )

    exported["norm_statistics"] = output_dir / "feature_norm_statistics.csv"
    norm_rows: list[dict[str, Any]] = []
    for metric_name, metric_values in statistics["feature_stats"].items():
        row = {"metric": metric_name}
        row.update(metric_values)
        norm_rows.append(row)
    _write_csv(
        exported["norm_statistics"],
        ["metric", "count", "min", "max", "mean", "std", "p50", "p90", "p95"],
        norm_rows,
    )

    exported["sample_preview"] = output_dir / "feature_sample_preview.csv"
    _write_csv(
        exported["sample_preview"],
        [
            "news_id",
            "category",
            "subcategory",
            "text_norm",
            "image_norm",
            "text_image_cosine",
            "text_preview",
            "image_preview",
        ],
        statistics["samples"],
    )

    exported["news_value_statistics"] = output_dir / "news_value_statistics.csv"
    _write_csv(
        exported["news_value_statistics"],
        ["dimension", "count", "min", "max", "mean", "std", "p50", "p90", "p95"],
        statistics["value_stats"],
    )

    exported["value_correlation_matrix"] = output_dir / "news_value_correlation_matrix.csv"
    corr_dimensions = statistics["value_correlation"]["dimensions"]
    corr_matrix = statistics["value_correlation"]["matrix"]
    corr_rows: list[dict[str, Any]] = []
    for row_idx, dim in enumerate(corr_dimensions):
        row: dict[str, Any] = {"dimension": dim}
        for col_idx, col_dim in enumerate(corr_dimensions):
            row[col_dim] = corr_matrix[row_idx][col_idx]
        corr_rows.append(row)
    _write_csv(exported["value_correlation_matrix"], ["dimension", *corr_dimensions], corr_rows)

    exported["clustering_metrics"] = output_dir / "clustering_metrics.csv"
    _write_csv(
        exported["clustering_metrics"],
        ["view", "point_count", "cluster_count", "silhouette", "davies_bouldin", "calinski_harabasz"],
        statistics["tsne"]["clustering_metrics"],
    )

    exported["tsne_text_points"] = output_dir / "tsne_text_points.csv"
    _write_csv(
        exported["tsne_text_points"],
        ["news_id", "category", "category_group", "x", "y"],
        statistics["tsne"]["text"]["points"],
    )

    exported["tsne_image_points"] = output_dir / "tsne_image_points.csv"
    _write_csv(
        exported["tsne_image_points"],
        ["news_id", "category", "category_group", "x", "y"],
        statistics["tsne"]["image"]["points"],
    )

    exported["tsne_fused_points"] = output_dir / "tsne_fused_points.csv"
    _write_csv(
        exported["tsne_fused_points"],
        ["news_id", "category", "category_group", "x", "y"],
        statistics["tsne"]["fused"]["points"],
    )

    return exported


def export_feature_statistics_plots(statistics: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plots: dict[str, Path] = {}

    plots["text_norm_distribution"] = output_dir / "text_norm_distribution.png"
    _plot_histogram(
        values=[row["value"] for row in _wrap_series(statistics["tsne"]["text"]["norms"])],
        title="Text Embedding L2 Norm Distribution",
        x_label="L2 Norm",
        y_label="Count",
        output_path=plots["text_norm_distribution"],
        color="#2563EB",
    )

    plots["image_norm_distribution"] = output_dir / "image_norm_distribution.png"
    _plot_histogram(
        values=[row["value"] for row in _wrap_series(statistics["tsne"]["image"]["norms"])],
        title="Image Embedding L2 Norm Distribution",
        x_label="L2 Norm",
        y_label="Count",
        output_path=plots["image_norm_distribution"],
        color="#059669",
    )

    plots["text_image_cosine_distribution"] = output_dir / "text_image_cosine_distribution.png"
    _plot_histogram(
        values=statistics["tsne"]["text_image_cosines"],
        title="Text-Image Cosine Similarity Distribution",
        x_label="Cosine Similarity",
        y_label="Count",
        output_path=plots["text_image_cosine_distribution"],
        color="#D97706",
    )

    plots["news_value_boxplot"] = output_dir / "news_value_boxplot.png"
    _plot_value_boxplot(
        value_stats=statistics["value_stats"],
        output_path=plots["news_value_boxplot"],
    )

    plots["news_value_correlation_heatmap"] = output_dir / "news_value_correlation_heatmap.png"
    _plot_correlation_heatmap(
        dimensions=statistics["value_correlation"]["dimensions"],
        matrix=np.array(statistics["value_correlation"]["matrix"], dtype=np.float32),
        output_path=plots["news_value_correlation_heatmap"],
        title="News Value Correlation Heatmap",
    )

    plots["tsne_text_by_category"] = output_dir / "tsne_text_by_category.png"
    _plot_tsne_points(
        points=statistics["tsne"]["text"]["points"],
        title="t-SNE Text Embeddings by Category",
        output_path=plots["tsne_text_by_category"],
    )

    plots["tsne_image_by_category"] = output_dir / "tsne_image_by_category.png"
    _plot_tsne_points(
        points=statistics["tsne"]["image"]["points"],
        title="t-SNE Image Embeddings by Category",
        output_path=plots["tsne_image_by_category"],
    )

    plots["tsne_fused_by_category"] = output_dir / "tsne_fused_by_category.png"
    _plot_tsne_points(
        points=statistics["tsne"]["fused"]["points"],
        title="t-SNE Fused Embeddings by Category",
        output_path=plots["tsne_fused_by_category"],
    )

    return plots


def render_feature_markdown_report(
    statistics: dict[str, Any],
    report_path: Path,
    figure_paths: dict[str, Path],
    table_paths: dict[str, Path],
) -> None:
    summary = statistics["summary"]

    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        "# 多模态特征分析报告",
        "",
        "本报告由 `uv run python main.py feature-report` 自动生成。",
        "",
        "## 特征维度说明",
        "",
        "| 指标 | 数值 |",
        "| --- | ---: |",
        f"| total_news | {summary['total_news']} |",
        f"| feature_entries | {summary['feature_entries']} |",
        f"| value_entries | {summary['value_entries']} |",
        f"| feature_coverage | {summary['feature_coverage']:.4f} |",
        f"| value_coverage | {summary['value_coverage']:.4f} |",
        f"| both_coverage | {summary['both_coverage']:.4f} |",
        f"| text_dim_expected | {summary['text_dim_expected']} |",
        f"| image_dim_expected | {summary['image_dim_expected']} |",
        f"| fused_dim_expected | {summary['fused_dim_expected']} |",
        f"| value_dim_expected | {summary['value_dim_expected']} |",
        f"| text_dim_mode | {summary['text_dim_mode']} |",
        f"| image_dim_mode | {summary['image_dim_mode']} |",
        "",
        "## 文本+图像特征展示",
        "",
    ]

    lines.extend(_render_sample_rows(statistics["samples"]))

    lines.extend([
        "",
        "## 新闻价值五维统计",
        "",
    ])

    lines.extend(_render_value_stats_rows(statistics["value_stats"]))

    lines.extend([
        "",
        "## t-SNE 可视化与聚类指标",
        "",
    ])

    lines.extend(_render_cluster_rows(statistics["tsne"]["clustering_metrics"]))

    lines.extend(["", "## 图表清单", ""])
    for key, path in figure_paths.items():
        relative = path.relative_to(report_path.parent).as_posix()
        lines.append(f"- {key}: [{path.name}]({relative})")

    lines.extend(["", "## 统计表清单", ""])
    for key, path in table_paths.items():
        relative = path.relative_to(report_path.parent).as_posix()
        lines.append(f"- {key}: [{path.name}]({relative})")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_tsne_summary(
    news_ids: list[str],
    categories: list[str],
    text_vectors: list[np.ndarray],
    image_vectors: list[np.ndarray],
    sample_size: int,
    perplexity: float,
    random_state: int,
    pca_dim: int,
    top_categories: int,
) -> dict[str, Any]:
    if not news_ids:
        empty_points: list[dict[str, Any]] = []
        empty_metrics = {
            "point_count": 0,
            "cluster_count": 0,
            "silhouette": None,
            "davies_bouldin": None,
            "calinski_harabasz": None,
        }
        return {
            "sample_size": 0,
            "perplexity": perplexity,
            "random_state": random_state,
            "top_categories": top_categories,
            "text_image_cosines": [],
            "text": {"points": empty_points, "norms": []},
            "image": {"points": empty_points, "norms": []},
            "fused": {"points": empty_points, "norms": []},
            "clustering_metrics": [
                {"view": "text", **empty_metrics},
                {"view": "image", **empty_metrics},
                {"view": "fused", **empty_metrics},
            ],
        }

    rng = np.random.default_rng(seed=random_state)
    selected_indices = np.arange(len(news_ids))
    if sample_size > 0 and len(news_ids) > sample_size:
        selected_indices = np.array(sorted(rng.choice(len(news_ids), size=sample_size, replace=False).tolist()))

    selected_ids = [news_ids[idx] for idx in selected_indices]
    selected_categories_raw = [categories[idx] for idx in selected_indices]
    selected_text = np.stack([text_vectors[idx] for idx in selected_indices], axis=0)
    selected_image = np.stack([image_vectors[idx] for idx in selected_indices], axis=0)
    selected_fused = np.concatenate([selected_text, selected_image], axis=1)

    selected_categories = _collapse_categories(selected_categories_raw, top_categories)

    text_points_2d = _reduce_to_tsne(selected_text, perplexity, random_state, pca_dim)
    image_points_2d = _reduce_to_tsne(selected_image, perplexity, random_state, pca_dim)
    fused_points_2d = _reduce_to_tsne(selected_fused, perplexity, random_state, pca_dim)

    text_norms = np.linalg.norm(selected_text, axis=1).tolist()
    image_norms = np.linalg.norm(selected_image, axis=1).tolist()
    fused_norms = np.linalg.norm(selected_fused, axis=1).tolist()

    text_image_cosines = [_safe_cosine_similarity(selected_text[i], selected_image[i]) for i in range(selected_text.shape[0])]

    text_points = _build_tsne_points(selected_ids, selected_categories_raw, selected_categories, text_points_2d)
    image_points = _build_tsne_points(selected_ids, selected_categories_raw, selected_categories, image_points_2d)
    fused_points = _build_tsne_points(selected_ids, selected_categories_raw, selected_categories, fused_points_2d)

    text_cluster = _compute_cluster_metrics(selected_text, random_state)
    image_cluster = _compute_cluster_metrics(selected_image, random_state)
    fused_cluster = _compute_cluster_metrics(selected_fused, random_state)

    return {
        "sample_size": int(selected_text.shape[0]),
        "perplexity": float(perplexity),
        "random_state": int(random_state),
        "top_categories": int(top_categories),
        "text_image_cosines": text_image_cosines,
        "text": {
            "points": text_points,
            "norms": text_norms,
        },
        "image": {
            "points": image_points,
            "norms": image_norms,
        },
        "fused": {
            "points": fused_points,
            "norms": fused_norms,
        },
        "clustering_metrics": [
            {"view": "text", **text_cluster},
            {"view": "image", **image_cluster},
            {"view": "fused", **fused_cluster},
        ],
    }


def _reduce_to_tsne(vectors: np.ndarray, perplexity: float, random_state: int, pca_dim: int) -> np.ndarray:
    if vectors.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if vectors.shape[0] == 1:
        return np.zeros((1, 2), dtype=np.float32)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(vectors)

    if pca_dim > 0:
        max_pca_dim = min(pca_dim, scaled.shape[1], scaled.shape[0] - 1)
        if max_pca_dim >= 2:
            scaled = PCA(n_components=max_pca_dim, random_state=random_state).fit_transform(scaled)

    effective_perplexity = max(2.0, min(float(perplexity), float(scaled.shape[0] - 1)))
    if scaled.shape[0] <= 3:
        return scaled[:, :2] if scaled.shape[1] >= 2 else np.pad(scaled, ((0, 0), (0, 2 - scaled.shape[1])))

    reducer = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
    )
    reduced = reducer.fit_transform(scaled)
    return reduced.astype(np.float32)


def _compute_cluster_metrics(vectors: np.ndarray, random_state: int) -> dict[str, Any]:
    point_count = int(vectors.shape[0])
    if point_count < 3:
        return {
            "point_count": point_count,
            "cluster_count": 0,
            "silhouette": None,
            "davies_bouldin": None,
            "calinski_harabasz": None,
        }

    cluster_count = min(8, max(2, int(round(math.sqrt(point_count)))))
    try:
        labels = KMeans(n_clusters=cluster_count, n_init=10, random_state=random_state).fit_predict(vectors)
        silhouette = float(silhouette_score(vectors, labels))
        davies_bouldin = float(davies_bouldin_score(vectors, labels))
        calinski_harabasz = float(calinski_harabasz_score(vectors, labels))
    except Exception:
        silhouette = None
        davies_bouldin = None
        calinski_harabasz = None

    return {
        "point_count": point_count,
        "cluster_count": cluster_count,
        "silhouette": silhouette,
        "davies_bouldin": davies_bouldin,
        "calinski_harabasz": calinski_harabasz,
    }


def _build_tsne_points(
    news_ids: list[str],
    categories: list[str],
    category_groups: list[str],
    points_2d: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, news_id in enumerate(news_ids):
        x_value = float(points_2d[idx, 0]) if points_2d.size else 0.0
        y_value = float(points_2d[idx, 1]) if points_2d.size else 0.0
        rows.append(
            {
                "news_id": news_id,
                "category": categories[idx],
                "category_group": category_groups[idx],
                "x": x_value,
                "y": y_value,
            }
        )
    return rows


def _collapse_categories(categories: list[str], top_categories: int) -> list[str]:
    if top_categories <= 0:
        return ["other"] * len(categories)

    counter = Counter(categories)
    top_names = {name for name, _ in counter.most_common(top_categories)}
    return [name if name in top_names else "other" for name in categories]


def _value_statistics_rows(value_matrix: np.ndarray, value_dimensions: tuple[str, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, name in enumerate(value_dimensions):
        values = value_matrix[:, idx].tolist() if value_matrix.size else []
        stats = _summary_stats(values)
        rows.append(
            {
                "dimension": name,
                "count": stats["count"],
                "min": stats["min"],
                "max": stats["max"],
                "mean": stats["mean"],
                "std": stats["std"],
                "p50": stats["p50"],
                "p90": stats["p90"],
                "p95": stats["p95"],
            }
        )
    return rows


def _compute_correlation_matrix(value_matrix: np.ndarray) -> np.ndarray:
    if value_matrix.shape[0] < 2:
        return np.eye(value_matrix.shape[1], dtype=np.float32)

    corr = np.corrcoef(value_matrix, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    return corr.astype(np.float32)


def _summary_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "count": 0.0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
        }

    arr = np.array(values, dtype=np.float32)
    return {
        "count": float(arr.size),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
    }


def _coerce_vector(value: Any, dim: int) -> np.ndarray:
    if value is None:
        return np.zeros(dim, dtype=np.float32)

    if isinstance(value, torch.Tensor):
        vector = value.detach().cpu().numpy().astype(np.float32).reshape(-1)
    else:
        vector = np.asarray(value, dtype=np.float32).reshape(-1)

    if vector.size == dim:
        return vector
    if vector.size > dim:
        return vector[:dim]

    padded = np.zeros(dim, dtype=np.float32)
    padded[: vector.size] = vector
    return padded


def _raw_vector_dim(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return int(value.numel())
    return int(np.asarray(value).size)


def _safe_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = float(np.linalg.norm(a))
    b_norm = float(np.linalg.norm(b))
    if a_norm <= 1e-12 or b_norm <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


def _pick_feature(entry: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in entry:
            return entry[key]
    return None


def _preview_vector(vector: np.ndarray, dims: int) -> str:
    if dims <= 0:
        return "[]"
    clipped = vector[:dims]
    values = ", ".join(f"{float(value):.4f}" for value in clipped)
    return f"[{values}]"


def _counter_mode(counter: Counter[int]) -> tuple[int, int]:
    if not counter:
        return 0, 0
    value, count = counter.most_common(1)[0]
    return int(value), int(count)


def _safe_divide(a: int, b: int) -> float:
    if b == 0:
        return 0.0
    return float(a / b)


def _wrap_series(values: list[float]) -> list[dict[str, float]]:
    return [{"value": value} for value in values]


def _plot_histogram(
    values: list[float],
    title: str,
    x_label: str,
    y_label: str,
    output_path: Path,
    color: str,
    bins: int = 40,
) -> None:
    plt.figure(figsize=(10, 5))
    if values:
        plt.hist(values, bins=min(bins, max(5, int(math.sqrt(len(values))))), color=color, alpha=0.9)
    else:
        plt.hist([0.0], bins=1, color=color, alpha=0.9)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_value_boxplot(value_stats: list[dict[str, Any]], output_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    labels = [row["dimension"] for row in value_stats]
    mins = [row["min"] for row in value_stats]
    p50s = [row["p50"] for row in value_stats]
    maxs = [row["max"] for row in value_stats]

    positions = np.arange(len(labels))
    plt.vlines(positions, mins, maxs, color="#6B7280", linewidth=2)
    plt.scatter(positions, p50s, color="#2563EB", s=60, label="median")
    plt.scatter(positions, mins, color="#10B981", s=25, label="min")
    plt.scatter(positions, maxs, color="#DC2626", s=25, label="max")
    plt.xticks(positions, labels, rotation=20)
    plt.ylabel("Score")
    plt.title("News Value Dimension Statistics")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_correlation_heatmap(dimensions: list[str], matrix: np.ndarray, output_path: Path, title: str) -> None:
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    plt.colorbar(label="Pearson Correlation")
    positions = np.arange(len(dimensions))
    plt.xticks(positions, dimensions, rotation=30, ha="right")
    plt.yticks(positions, dimensions)

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            plt.text(col_idx, row_idx, f"{matrix[row_idx, col_idx]:.2f}", ha="center", va="center", fontsize=8)

    plt.title(title)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_tsne_points(points: list[dict[str, Any]], title: str, output_path: Path) -> None:
    plt.figure(figsize=(9, 7))

    if not points:
        plt.text(0.5, 0.5, "No data", ha="center", va="center")
        plt.title(title)
        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
        plt.close()
        return

    grouped: dict[str, list[dict[str, Any]]] = {}
    for point in points:
        grouped.setdefault(point["category_group"], []).append(point)

    palette = plt.cm.tab20(np.linspace(0.0, 1.0, max(1, len(grouped))))
    for idx, (name, rows) in enumerate(sorted(grouped.items(), key=lambda item: item[0])):
        x_values = [row["x"] for row in rows]
        y_values = [row["y"] for row in rows]
        plt.scatter(x_values, y_values, s=14, alpha=0.75, color=palette[idx], label=name)

    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    if len(grouped) <= 20:
        plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def _render_sample_rows(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["无可展示特征样本。"]

    lines = [
        "| news_id | category | subcategory | text_norm | image_norm | cosine | text_preview | image_preview |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['news_id']} | {row['category']} | {row['subcategory']} | "
            f"{row['text_norm']:.4f} | {row['image_norm']:.4f} | {row['text_image_cosine']:.4f} | "
            f"{_escape_table(row['text_preview'])} | {_escape_table(row['image_preview'])} |"
        )
    return lines


def _render_value_stats_rows(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["无新闻价值统计。"]

    lines = [
        "| dimension | count | min | max | mean | std | p50 | p90 | p95 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['dimension']} | {int(row['count'])} | {row['min']:.4f} | {row['max']:.4f} | "
            f"{row['mean']:.4f} | {row['std']:.4f} | {row['p50']:.4f} | {row['p90']:.4f} | {row['p95']:.4f} |"
        )
    return lines


def _render_cluster_rows(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["无聚类指标。"]

    lines = [
        "| view | point_count | cluster_count | silhouette | davies_bouldin | calinski_harabasz |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['view']} | {row['point_count']} | {row['cluster_count']} | "
            f"{_fmt_optional_float(row['silhouette'])} | {_fmt_optional_float(row['davies_bouldin'])} | {_fmt_optional_float(row['calinski_harabasz'])} |"
        )
    return lines


def _fmt_optional_float(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.4f}"


def _write_csv(filepath: Path, headers: list[str], rows: list[dict[str, Any]]) -> None:
    with filepath.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _escape_table(text: str) -> str:
    return text.replace("|", "\\|")


def _normalized(value: Any) -> str:
    if value is None:
        return "unknown"
    text = str(value).strip()
    return text if text else "unknown"
