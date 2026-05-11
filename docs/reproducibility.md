# 复现实验说明

## 默认设置

- 随机种子：`2026`
- 云端显卡：至少 NVIDIA T4 16GB
- 图文编码器：`google/siglip-base-patch16-224`
- 指标：AUC、MRR、nDCG@5、nDCG@10

## 推荐命令

```powershell
uv sync
uv run gpnews prepare-data
uv run gpnews extract-siglip --batch-size 64 --device auto
uv run gpnews label-values --max-news 3000
uv run gpnews train --config configs/text.yaml
uv run gpnews train --config configs/multimodal.yaml
uv run gpnews train --config configs/value.yaml
uv run gpnews evaluate
```

## 双模型标注可信性验证

```powershell
uv run gpnews prepare-cross-label-sample
uv run gpnews label-values --backend aliyun-batch --sample-path artifacts/labels/cross_model_sample.jsonl --output-path artifacts/labels/news_value_labels_qwen35_plus_sample.jsonl --batch-model qwen3.5-plus --submit-only
uv run gpnews label-values --backend aliyun-batch --sample-path artifacts/labels/cross_model_sample.jsonl --output-path artifacts/labels/news_value_labels_qwen35_plus_sample.jsonl --batch-model qwen3.5-plus --batch-id "batch_xxx" --batch-run-dir "artifacts/labels/batches/news-value-YYYYMMDD-HHMMSS"
uv run gpnews analyze-cross-labels
```

该流程使用已有 `qwen3.5-flash` 标签构造 300 条分层样本，再用 `qwen3.5-plus` 独立复核。复核结果只用于一致性统计和论文图表，不参与三组推荐模型训练。

## 小样本调试

```powershell
uv run gpnews prepare-data --max-train-impressions 200 --max-dev-impressions 50
uv run gpnews extract-siglip --max-news 500 --batch-size 32
```

小样本只用于验证流程，不作为论文最终结果。
