# 基于新闻价值理论的多模态新闻推荐

本项目是本科毕设工程实现：使用 MIND-small 新闻文本数据和 V-MIND 图像数据，基于 SigLIP 预计算图文特征，并调用 OpenAI-compatible 大语言模型 API 标注新闻价值，最终比较三组推荐模型：

1. 文本模态
2. 图文模态
3. 图文 + 新闻价值模态

默认随机种子为 `2026`。训练配置面向云端至少 T4 16GB 显卡；本地显卡建议只做小样本调试。

## 环境

```powershell
uv sync
```

数据目录保持为：

```text
MINDsmall_train/
MINDsmall_dev/
newData/
```

## 快速冒烟流程

```powershell
uv run gpnews prepare-data --max-train-impressions 200 --max-dev-impressions 50
uv run gpnews extract-siglip --max-news 500 --batch-size 32
uv run gpnews label-values --max-news 50
uv run gpnews train --config configs/text.yaml
uv run gpnews train --config configs/multimodal.yaml
uv run gpnews train --config configs/value.yaml
uv run gpnews evaluate
```

`label-values` 默认使用实时 OpenAI-compatible API，需要环境变量：

```powershell
$env:LLM_API_KEY="..."
$env:LLM_BASE_URL="https://api.openai.com/v1"
$env:LLM_MODEL="..."
```

也可以使用阿里云百炼 Batch File 批量标注。批处理默认使用中国内地兼容端点和 `qwen3.5-flash`，并在请求体中设置 `enable_thinking=false`，以保证 JSON Mode 输出更稳定。

提交批任务：

```powershell
$env:DASHSCOPE_API_KEY="..."
uv run gpnews label-values --backend aliyun-batch --max-news 3000 --submit-only
```

命令会在 `artifacts/labels/batches/news-value-YYYYMMDD-HHMMSS/` 下生成 `input.jsonl` 和 `manifest.json`。任务完成后，可用 manifest 中的 `batch_id` 恢复、下载并合并结果：

```powershell
uv run gpnews label-values --backend aliyun-batch --batch-id "batch_xxx" --batch-run-dir "artifacts/labels/batches/news-value-YYYYMMDD-HHMMSS"
```

如果不加 `--submit-only`，命令会按 `--poll-interval` 轮询直到任务结束。成功结果会追加到 `artifacts/labels/news_value_labels.jsonl`，行级失败写入 `error.jsonl`，JSON 或分数校验失败写入 `invalid_results.jsonl`。

项目不会读取或打印 `.env` 文件内容。

## 云端完整流程

```powershell
uv run gpnews prepare-data
uv run gpnews extract-siglip --batch-size 64 --device auto
uv run gpnews label-values --max-news 3000
uv run gpnews train --config configs/text.yaml
uv run gpnews train --config configs/multimodal.yaml
uv run gpnews train --config configs/value.yaml
uv run gpnews evaluate
```

主要输出：

- `artifacts/data/`：解析后的新闻、训练样本、验证 impression、新闻频次
- `artifacts/features/siglip/`：SigLIP 文本和图像特征
- `artifacts/labels/news_value_labels.jsonl`：新闻价值标注缓存
- `artifacts/models/`：三组模型 checkpoint 和指标
- `artifacts/reports/ablation.csv`：消融实验结果表

## 新闻价值维度

使用 5 个维度：重要性、显著性、冲突性、新奇性、人情味。

不使用时效性和接近性，因为 MIND-small 缺少可靠发布时间、地理位置和用户位置字段，强行标注会引入不可控噪声。
