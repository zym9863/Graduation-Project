[English](./README-EN.md) | [中文](./README.md)

# 基于新闻价值理论的多模态新闻推荐系统

这个仓库实现一套基于 MIND-small 的多模态新闻推荐实验管线：

- SigLIP 离线提取新闻文本和图片特征
- 新闻价值五要素离线标注
- NRMS 用户编码器进行点击预测
- 支持拼接融合、门控融合、Cross-Modal Cross-Attention 融合，以及模态级消融新闻编码方案

## 环境准备

项目使用 `uv` 管理依赖。

```bash
uv sync
```

## 当前阶段实验清单（你现在可直接跑）

你当前状态是：

- 已有 `data/news_siglip_features.pt`
- 新闻价值五要素离线标注尚未完成（`data/news_value_scores.json` 不完整或不存在）

在这个状态下，代码会对缺失的新闻价值向量自动补零，因此训练与评估可以正常运行。

### 1) 可直接做的数据与特征分析

```bash
uv run python main.py preprocess
uv run python main.py dataset-report
uv run python main.py feature-report
```

### 2) 当前最有意义的对比实验（不依赖五要素）

推荐先做这两组主实验：

- `text_only`：文本基线
- `text_image`：文本+图像

```bash
uv run python main.py train --fusion text_only --epochs 8 --eval-dev --scheduler plateau --scheduler-patience 1 --patience 2 --min-delta 0.0005 --seed 42 --checkpoint data/processed/text_only_s42.pt
uv run python main.py train --fusion text_only --epochs 8 --eval-dev --scheduler plateau --scheduler-patience 1 --patience 2 --min-delta 0.0005 --seed 2026 --checkpoint data/processed/text_only_s2026.pt

uv run python main.py train --fusion text_image --epochs 8 --eval-dev --scheduler plateau --scheduler-patience 1 --patience 2 --min-delta 0.0005 --seed 42 --checkpoint data/processed/text_image_s42.pt
uv run python main.py train --fusion text_image --epochs 8 --eval-dev --scheduler plateau --scheduler-patience 1 --patience 2 --min-delta 0.0005 --seed 2026 --checkpoint data/processed/text_image_s2026.pt
```

### 3) 可运行但仅用于流程验证的实验（价值通道当前为零向量）

以下融合策略现在都能跑，但因为新闻价值输入是零向量，暂时不建议用于得出“新闻价值贡献”的结论：

```bash
uv run python main.py train --fusion concat --epochs 8 --eval-dev --seed 42 --checkpoint data/processed/concat_s42.pt
uv run python main.py train --fusion concat --epochs 8 --eval-dev --seed 2026 --checkpoint data/processed/concat_s2026.pt

uv run python main.py train --fusion gate --epochs 8 --eval-dev --seed 42 --checkpoint data/processed/gate_s42.pt
uv run python main.py train --fusion gate --epochs 8 --eval-dev --seed 2026 --checkpoint data/processed/gate_s2026.pt

uv run python main.py train --fusion cross_modal --epochs 8 --eval-dev --seed 42 --checkpoint data/processed/cross_modal_s42.pt
uv run python main.py train --fusion cross_modal --epochs 8 --eval-dev --seed 2026 --checkpoint data/processed/cross_modal_s2026.pt

uv run python main.py train --fusion text_value --epochs 8 --eval-dev --seed 42 --checkpoint data/processed/text_value_s42.pt
uv run python main.py train --fusion text_value --epochs 8 --eval-dev --seed 2026 --checkpoint data/processed/text_value_s2026.pt

uv run python main.py train --fusion text_image_value --epochs 8 --eval-dev --seed 42 --checkpoint data/processed/text_image_value_s42.pt
uv run python main.py train --fusion text_image_value --epochs 8 --eval-dev --seed 2026 --checkpoint data/processed/text_image_value_s2026.pt
```

### 4) 评估与汇总

```bash
uv run python main.py evaluate --checkpoint data/processed/text_only_s42.pt --fusion text_only
uv run python main.py evaluate --checkpoint data/processed/text_image_s42.pt --fusion text_image

uv run python main.py experiment-summary --glob-pattern "data/processed/*_s*.pt" --output-dir data/processed/experiment_reports
```

### 5) 快速冒烟（可选）

```bash
uv run python main.py train --fusion text_only --epochs 1 --behavior-limit 200 --max-steps 50 --eval-dev --checkpoint data/processed/text_only_smoke.pt
```

新闻价值五要素离线标注（默认阿里云 Batch File）：

推荐在项目根目录创建 `.env`：

```bash
ALIYUN_BATCH_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
ALIYUN_BATCH_API_KEY=<DASHSCOPE_API_KEY>
NEWS_VALUE_MODEL=qwen-plus
ALIYUN_BATCH_ENDPOINT=/v1/chat/completions
ALIYUN_BATCH_COMPLETION_WINDOW=24h
ALIYUN_BATCH_POLL_INTERVAL=60
```

批量标注命令（会同步等待任务完成）：

```bash
uv run python main.py annotate-news-value --provider aliyun-batch --limit 500
```

单条新闻价值特征提取（仅支持 `openai-compatible`）：

```bash
set NEWS_VALUE_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
set NEWS_VALUE_API_KEY=<DASHSCOPE_API_KEY>
set NEWS_VALUE_MODEL=qwen-plus

uv run python main.py annotate-news-value --provider openai-compatible --single-title "突发：某地发布重大政策" --single-abstract "官方今天发布新规，影响多个行业。" --single-category news --single-subcategory policy
```

输出示例会包含：

- 输入新闻内容
- 五维价值打分（`conflict`、`importance`、`prominence`、`proximity`、`interest`）
- 向量数组，例如 `[4, 4, 3, 3, 2]`

批处理模式会额外产出：

- `data/news_value_scores.meta.json`（批任务元信息）
- `data/news_value_scores.failures.json`（失败条目与原因）

也可以直接运行脚本：

```bash
uv run python -m scripts.train --epochs 8 --fusion text_only --eval-dev --scheduler plateau --patience 3
```

## 数据约定

- `MINDsmall_train/` 和 `MINDsmall_dev/` 为原始 MIND-small 数据
- `newData/` 为与新闻 ID 对齐的图片目录，文件名格式为 `{NewsID}.jpg`
- `data/processed/metadata.json` 包含基础统计与详细统计字段
- `data/processed/analytics/` 包含自动生成的数据统计报告、图表和 CSV 表格
- `data/news_siglip_features.pt` 为离线图文特征
- `data/news_value_scores.json` 为新闻价值五要素打分

如果离线特征文件不存在，训练与评估会使用零向量占位，只用于流程验证，不代表最终实验配置。

## 数据集统计报告（规模、分布、示例）

使用以下命令可自动生成完整数据报告（PNG + CSV + Markdown）：

```bash
uv run python main.py dataset-report
```

输出目录：

- `data/processed/analytics/data_report.md`（报告）
- `data/processed/analytics/figures/`（图表）
- `data/processed/analytics/tables/`（统计表）

## 多模态特征报告（文本+图像维度、t-SNE）

使用以下命令可生成特征分析报告（JSON + CSV + PNG + Markdown）：

```bash
uv run python main.py feature-report
```

输出目录：

- `data/processed/feature_analytics/feature_statistics.json`（统计汇总）
- `data/processed/feature_analytics/data_report.md`（报告）
- `data/processed/feature_analytics/figures/`（图表）
- `data/processed/feature_analytics/tables/`（统计表）

报告包含：

- 文本特征与图像特征维度说明（默认 768 + 768，融合 1536）
- 文本+图像特征样本展示（范数、余弦相似度、向量预览）
- 新闻价值五维统计与相关性热力图
- 文本/图像/融合特征的 t-SNE 可视化与聚类指标

## 当前实现范围

- 原始 MIND 数据解析与类别映射
- SigLIP 特征提取脚本
- 新闻价值打分脚本，支持 `aliyun-batch`（默认）与 `openai-compatible`
- NRMS 主模型、拼接/门控/Cross-Modal 融合、模态级消融训练与评估脚本
- 预处理和前向过程基础测试

## 对比实验与消融实验

### 当前阶段（仅 SigLIP）建议口径

建议先固定训练口径（epoch、batch-size、scheduler、seed）后再进行对比：

- 主结论实验：`text_only` vs `text_image`
- 其余融合（`concat`、`gate`、`cross_modal`、`text_value`、`text_image_value`）可先做流程验证

### 五要素完成后再做完整版对比

当 `data/news_value_scores.json` 覆盖率足够后，重点进行以下完整实验：

- 融合策略对比：`concat`、`gate`、`cross_modal`
- 模态级消融：`text_only`、`text_image`、`text_value`、`text_image_value`

示例（每组双 seed）：

```bash
uv run python main.py train --fusion concat --epochs 8 --eval-dev --seed 42 --checkpoint data/processed/concat_s42.pt
uv run python main.py train --fusion concat --epochs 8 --eval-dev --seed 2026 --checkpoint data/processed/concat_s2026.pt

uv run python main.py train --fusion gate --epochs 8 --eval-dev --seed 42 --checkpoint data/processed/gate_s42.pt
uv run python main.py train --fusion gate --epochs 8 --eval-dev --seed 2026 --checkpoint data/processed/gate_s2026.pt

uv run python main.py train --fusion cross_modal --epochs 8 --eval-dev --seed 42 --checkpoint data/processed/cross_modal_s42.pt
uv run python main.py train --fusion cross_modal --epochs 8 --eval-dev --seed 2026 --checkpoint data/processed/cross_modal_s2026.pt
```

实验完成后可自动汇总：

```bash
uv run python main.py experiment-summary --glob-pattern "data/processed/*_s*.pt" --output-dir data/processed/experiment_reports
```

模板与记录建议见：`docs/baselines/fusion_ablation_experiments.md`

## 绝对底线：纯文本标准 NRMS

详细记录模板与结果归档见：`docs/baselines/nrms_text_only_baseline.md`
