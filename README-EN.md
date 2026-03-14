[English](./README-EN.md) | [中文](./README.md)

# Multimodal News Recommendation System Based on News Value Theory

This repository implements a multimodal news recommendation experimental pipeline based on MIND-small:

- SigLIP for offline extraction of news text and image features
- Offline annotation of news value five elements
- NRMS user encoder for click prediction
- Supports both concatenation fusion and gated fusion news encoding schemes

## Environment Setup

The project uses `uv` for dependency management.

```bash
uv sync
```

Common commands:

```bash
uv run python main.py preprocess
uv run python main.py dataset-report
uv run python main.py feature-report
uv run python main.py extract-features --batch-size 16
uv run python main.py annotate-news-value --provider heuristic
uv run python main.py train --epochs 3 --fusion concat
uv run python main.py evaluate --checkpoint data/processed/nrms_latest.pt
```

Single news value feature extraction (real API case):

Recommended: create a `.env` file in the project root:

```bash
NEWS_VALUE_API_BASE=https://api-inference.modelscope.cn/v1
NEWS_VALUE_API_KEY=<MODELSCOPE_TOKEN>
NEWS_VALUE_MODEL=ZhipuAI/GLM-5
```

You can also set environment variables temporarily (Windows CMD):

```bash
set NEWS_VALUE_API_BASE=https://api-inference.modelscope.cn/v1
set NEWS_VALUE_API_KEY=<MODELSCOPE_TOKEN>
set NEWS_VALUE_MODEL=ZhipuAI/GLM-5

uv run python main.py annotate-news-value --provider openai-compatible --single-title "Breaking: major policy released" --single-abstract "Authorities released a new policy today with broad industry impact." --single-category news --single-subcategory policy
```

The output includes:

- Input article content
- Five-dimensional value scores (`conflict`, `importance`, `prominence`, `proximity`, `interest`)
- Vector array, e.g. `[4, 4, 3, 3, 2]`

You can also run scripts directly:

```bash
uv run python scripts/train.py --epochs 3
```

## Data Conventions

- `MINDsmall_train/` and `MINDsmall_dev/` are the original MIND-small data
- `newData/` is the image directory aligned with news IDs, with filename format `{NewsID}.jpg`
- `data/processed/metadata.json` includes basic summary and detailed statistics fields
- `data/processed/analytics/` includes the generated dataset report, figures, and CSV tables
- `data/news_siglip_features.pt` contains offline image-text features
- `data/news_value_scores.json` contains news value five elements scores

If offline feature files do not exist, training and evaluation will use zero vectors as placeholders, only for pipeline verification, not representing the final experimental configuration.

## Dataset Report (Scale, Distribution, Examples)

Run the command below to generate the full data report (PNG + CSV + Markdown):

```bash
uv run python main.py dataset-report
```

Output directory:

- `data/processed/analytics/data_report.md` (report)
- `data/processed/analytics/figures/` (charts)
- `data/processed/analytics/tables/` (statistics tables)

## Multimodal Feature Report (Text + Image Dimensions, t-SNE)

Run the command below to generate a feature analysis report (JSON + CSV + PNG + Markdown):

```bash
uv run python main.py feature-report
```

Output directory:

- `data/processed/feature_analytics/feature_statistics.json` (statistics summary)
- `data/processed/feature_analytics/data_report.md` (report)
- `data/processed/feature_analytics/figures/` (figures)
- `data/processed/feature_analytics/tables/` (tables)

The report includes:

- Text and image feature dimension explanation (default 768 + 768, fused 1536)
- Sample-level text+image feature preview (norm, cosine similarity, vector preview)
- News value 5-dimension statistics and correlation heatmap
- t-SNE visualizations for text/image/fused features and clustering metrics

## Current Implementation Scope

- Original MIND data parsing and category mapping
- SigLIP feature extraction script
- News value scoring script, supporting heuristic mode and OpenAI-compatible interface
- NRMS main model, gated fusion, training and evaluation scripts
- Preprocessing and forward pass basic tests
