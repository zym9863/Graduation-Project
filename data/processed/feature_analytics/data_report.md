# 多模态特征分析报告

本报告由 `uv run python main.py feature-report` 自动生成。

## 特征维度说明

| 指标 | 数值 |
| --- | ---: |
| total_news | 65238 |
| feature_entries | 65238 |
| value_entries | 5 |
| feature_coverage | 1.0000 |
| value_coverage | 0.0001 |
| both_coverage | 0.0001 |
| text_dim_expected | 768 |
| image_dim_expected | 768 |
| fused_dim_expected | 1536 |
| value_dim_expected | 5 |
| text_dim_mode | 768 |
| image_dim_mode | 768 |

## 文本+图像特征展示

| news_id | category | subcategory | text_norm | image_norm | cosine | text_preview | image_preview |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| N55528 | lifestyle | lifestyleroyals | 20.0840 | 17.4655 | -0.0580 | [0.1659, -0.2263, -0.4420, -1.0151, -0.0667, -0.1274, 0.9077, 0.2586] | [0.0712, -0.4136, -0.3236, -0.0880, -0.3171, 0.5536, -0.3939, -0.2444] |
| N19639 | health | weightloss | 20.0000 | 17.4655 | -0.0724 | [-0.1359, -0.6186, 0.1402, 0.1802, 0.5261, -0.3312, -0.0357, -0.1235] | [0.0712, -0.4136, -0.3236, -0.0880, -0.3171, 0.5536, -0.3939, -0.2444] |
| N61837 | news | newsworld | 14.8441 | 17.4655 | -0.1010 | [0.0290, -0.1210, 0.1765, -0.7774, -0.6581, -0.2853, 0.4969, -0.6679] | [0.0712, -0.4136, -0.3236, -0.0880, -0.3171, 0.5536, -0.3939, -0.2444] |
| N53526 | health | voices | 17.4095 | 17.4655 | -0.0569 | [0.0434, 0.1184, -0.4732, -0.7414, -0.0652, -0.0194, 0.2435, -0.3109] | [0.0712, -0.4136, -0.3236, -0.0880, -0.3171, 0.5536, -0.3939, -0.2444] |
| N38324 | health | medical | 19.8534 | 17.4655 | -0.0451 | [-0.1336, -0.1499, 0.1669, -0.8569, -0.4086, -0.6970, -0.4377, -0.2347] | [0.0712, -0.4136, -0.3236, -0.0880, -0.3171, 0.5536, -0.3939, -0.2444] |

## 新闻价值五维统计

| dimension | count | min | max | mean | std | p50 | p90 | p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| conflict | 65238 | 0.0000 | 3.0000 | 0.0002 | 0.0263 | 0.0000 | 0.0000 | 0.0000 |
| importance | 65238 | 0.0000 | 4.0000 | 0.0003 | 0.0334 | 0.0000 | 0.0000 | 0.0000 |
| prominence | 65238 | 0.0000 | 4.0000 | 0.0003 | 0.0350 | 0.0000 | 0.0000 | 0.0000 |
| proximity | 65238 | 0.0000 | 3.0000 | 0.0002 | 0.0263 | 0.0000 | 0.0000 | 0.0000 |
| interest | 65238 | 0.0000 | 4.0000 | 0.0002 | 0.0282 | 0.0000 | 0.0000 | 0.0000 |

## t-SNE 可视化与聚类指标

| view | point_count | cluster_count | silhouette | davies_bouldin | calinski_harabasz |
| --- | ---: | ---: | ---: | ---: | ---: |
| text | 3000 | 8 | 0.0298 | 4.3809 | 71.7872 |
| image | 3000 | 8 | 0.9602 | 2.9478 | 904.6897 |
| fused | 3000 | 8 | 0.0314 | 4.0788 | 83.0467 |

## 图表清单

- text_norm_distribution: [text_norm_distribution.png](figures/text_norm_distribution.png)
- image_norm_distribution: [image_norm_distribution.png](figures/image_norm_distribution.png)
- text_image_cosine_distribution: [text_image_cosine_distribution.png](figures/text_image_cosine_distribution.png)
- news_value_boxplot: [news_value_boxplot.png](figures/news_value_boxplot.png)
- news_value_correlation_heatmap: [news_value_correlation_heatmap.png](figures/news_value_correlation_heatmap.png)
- tsne_text_by_category: [tsne_text_by_category.png](figures/tsne_text_by_category.png)
- tsne_image_by_category: [tsne_image_by_category.png](figures/tsne_image_by_category.png)
- tsne_fused_by_category: [tsne_fused_by_category.png](figures/tsne_fused_by_category.png)

## 统计表清单

- dimension_summary: [feature_dimension_summary.csv](tables/feature_dimension_summary.csv)
- norm_statistics: [feature_norm_statistics.csv](tables/feature_norm_statistics.csv)
- sample_preview: [feature_sample_preview.csv](tables/feature_sample_preview.csv)
- news_value_statistics: [news_value_statistics.csv](tables/news_value_statistics.csv)
- value_correlation_matrix: [news_value_correlation_matrix.csv](tables/news_value_correlation_matrix.csv)
- clustering_metrics: [clustering_metrics.csv](tables/clustering_metrics.csv)
- tsne_text_points: [tsne_text_points.csv](tables/tsne_text_points.csv)
- tsne_image_points: [tsne_image_points.csv](tables/tsne_image_points.csv)
- tsne_fused_points: [tsne_fused_points.csv](tables/tsne_fused_points.csv)
