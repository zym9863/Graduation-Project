# 新闻价值标注 Prompt

工程中的 prompt 由 `gp_newsrec.labels.build_value_prompt` 生成。核心要求如下：

- 只基于新闻类别、标题和摘要评分。
- 不评估时效性和接近性。
- 维度为重要性、显著性、冲突性、新奇性、人情味。
- 每维 `0-3` 分。
- 只输出 JSON。

输出格式：

```json
{
  "scores": {
    "importance": 0,
    "prominence": 0,
    "conflict": 0,
    "novelty": 0,
    "human_interest": 0
  },
  "reason": "不超过40个中文字的理由"
}
```

缓存文件为 `artifacts/labels/news_value_labels.jsonl`，以 `news_id + prompt_version` 去重。

## 阿里云 Batch File 标注

实时标注保留为默认行为。批量标注使用：

```powershell
$env:DASHSCOPE_API_KEY="..."
uv run gpnews label-values --backend aliyun-batch --max-news 3000 --submit-only
```

批任务默认模型为 `qwen3.5-flash`，端点为 `https://dashscope.aliyuncs.com/compatible-mode/v1`。输入文件为 UTF-8 JSONL，每行包含 `custom_id`、`method=POST`、`url=/v1/chat/completions` 和请求体。

`qwen3.5` 系列默认可能开启思考模式；本工程在批处理请求体中固定设置 `enable_thinking=false`，并使用 `response_format={"type":"json_object"}`，使输出更适合自动解析和校验。

任务完成后根据 `manifest.json` 中的 `batch_id` 恢复：

```powershell
uv run gpnews label-values --backend aliyun-batch --batch-id "batch_xxx" --batch-run-dir "artifacts/labels/batches/news-value-YYYYMMDD-HHMMSS"
```

成功解析的结果追加到主缓存；HTTP 行级失败保存在 `error.jsonl`；无法解析、缺少字段或分数不在 `0-3` 的结果保存在 `invalid_results.jsonl`。
