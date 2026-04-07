[English](./README-EN.md) | [中文](./README.md)

# Multimodal News Recommendation System Based on News Value Theory

This repository implements a multimodal news recommendation experimental pipeline based on MIND-small:

- SigLIP for offline extraction of news text and image features
- Offline annotation of news value five elements
- NRMS user encoder for click prediction
- Supports concatenation fusion, gated fusion, Cross-Modal Cross-Attention fusion, and modality-level ablation encoders

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
uv run python main.py annotate-news-value --provider openai-compatible
uv run python main.py train --epochs 3 --fusion concat
uv run python main.py train --epochs 3 --fusion cross_modal
uv run python main.py train --epochs 8 --fusion text_only --scheduler plateau --patience 3 --checkpoint data/processed/nrms_text_only.pt --eval-dev
uv run python main.py train --epochs 8 --fusion text_image --checkpoint data/processed/nrms_text_image.pt --eval-dev --seed 42
uv run python main.py train --epochs 8 --fusion text_value --checkpoint data/processed/nrms_text_value.pt --eval-dev --seed 42
uv run python main.py train --epochs 8 --fusion text_image_value --checkpoint data/processed/nrms_text_image_value.pt --eval-dev --seed 42
uv run python main.py train --epochs 3 --fusion text_only --checkpoint data/processed/nrms_text_only_quick.pt --eval-dev
uv run python main.py evaluate --checkpoint data/processed/nrms_latest.pt
uv run python main.py evaluate --checkpoint data/processed/nrms_text_only.pt --fusion text_only
uv run python main.py experiment-summary --glob-pattern "data/processed/*.pt"
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

uv run python main.py annotate-news-value --provider openai-compatible --single-title "突发：某地发布重大政策" --single-abstract "官方今天发布新规，影响多个行业。" --single-category news --single-subcategory policy
```

The output includes:

- Input article content
- Five-dimensional value scores (`conflict`, `importance`, `prominence`, `proximity`, `interest`)
- Vector array, e.g. `[4, 4, 3, 3, 2]`

You can also run scripts directly:

```bash
uv run python -m scripts.train --epochs 8 --fusion text_only --eval-dev --scheduler plateau --patience 3
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
- News value scoring script, supporting OpenAI-compatible interface
- NRMS main model, concat/gate/cross-modal fusion, modality ablation training and evaluation scripts
- Preprocessing and forward pass basic tests

## Comparison And Ablation Experiments

Recommended controlled setup:

- Fusion comparison: `concat`, `gate`, `cross_modal`
- Modality ablation: `text_only`, `text_image`, `text_value`, `text_image_value`

Example matrix (two seeds per setting):

```bash
uv run python main.py train --fusion concat --epochs 8 --eval-dev --seed 42 --checkpoint data/processed/concat_s42.pt
uv run python main.py train --fusion concat --epochs 8 --eval-dev --seed 2026 --checkpoint data/processed/concat_s2026.pt

uv run python main.py train --fusion gate --epochs 8 --eval-dev --seed 42 --checkpoint data/processed/gate_s42.pt
uv run python main.py train --fusion gate --epochs 8 --eval-dev --seed 2026 --checkpoint data/processed/gate_s2026.pt

uv run python main.py train --fusion cross_modal --epochs 8 --eval-dev --seed 42 --checkpoint data/processed/cross_modal_s42.pt
uv run python main.py train --fusion cross_modal --epochs 8 --eval-dev --seed 2026 --checkpoint data/processed/cross_modal_s2026.pt
```

Aggregate results:

```bash
uv run python main.py experiment-summary --glob-pattern "data/processed/*_s*.pt" --output-dir data/processed/experiment_reports
```

Template and tracking notes: `docs/baselines/fusion_ablation_experiments.md`

## Baseline: Text-Only Standard NRMS

Detailed template and results archive: `docs/baselines/nrms_text_only_baseline.md`
