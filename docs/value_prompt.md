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
