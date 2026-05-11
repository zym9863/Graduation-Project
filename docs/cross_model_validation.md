# 双模型交叉标注可信性检验

本节用于回应“无法证明大模型标注的新闻价值可信”的质疑。验证目标不是证明 LLM 标注等于人工真值，而是证明同一新闻价值理论 prompt 在不同能力层级模型之间具有稳定一致性。

## 方法

主标注文件仍为 `artifacts/labels/news_value_labels.jsonl`，由 `qwen3.5-flash` 生成并用于推荐模型训练。复核标注单独保存为 `artifacts/labels/news_value_labels_qwen35_plus_sample.jsonl`，不参与训练，只用于可信性分析。

复核样本固定为 300 条，随机种子为 `2026`。抽样按 flash 总分分层：

| 分层 | flash 总分范围 | 样本数 |
| --- | --- | --- |
| low | 0-6 | 92 |
| medium | 7-9 | 136 |
| high | 10-15 | 72 |

每个分层内部再按新闻类别比例抽样，并强制包含 `N306`、`N31958`、`N5940`、`N11930`、`N6916` 作为论文样例候选。

## 复核命令

生成抽样文件：

```powershell
uv run gpnews prepare-cross-label-sample
```

提交 `qwen3.5-plus` 批量复核：

```powershell
$env:DASHSCOPE_API_KEY="..."
uv run gpnews label-values --backend aliyun-batch --sample-path artifacts/labels/cross_model_sample.jsonl --output-path artifacts/labels/news_value_labels_qwen35_plus_sample.jsonl --batch-model qwen3.5-plus --submit-only
```

批任务完成后，用 manifest 中的 `batch_id` 下载并合并：

```powershell
uv run gpnews label-values --backend aliyun-batch --sample-path artifacts/labels/cross_model_sample.jsonl --output-path artifacts/labels/news_value_labels_qwen35_plus_sample.jsonl --batch-model qwen3.5-plus --batch-id "batch_xxx" --batch-run-dir "artifacts/labels/batches/news-value-YYYYMMDD-HHMMSS"
```

生成一致性统计、样例表和论文图表：

```powershell
uv run gpnews analyze-cross-labels
```

## 输出

- `artifacts/labels/cross_model_sample.jsonl`：300 条分层复核样本。
- `artifacts/labels/news_value_labels_qwen35_plus_sample.jsonl`：plus 复核标签。
- `artifacts/thesis/tables/cross_model_agreement.csv`：一致性统计表。
- `artifacts/thesis/tables/cross_model_examples.csv`：高度一致和明显分歧样例。
- `artifacts/thesis/figures/cross_model_total_scatter.png`：总分散点图。
- `artifacts/thesis/figures/cross_model_confusion_matrices.png`：五维混淆矩阵。
- `artifacts/thesis/figures/cross_model_agreement_rates.png`：五维一致率柱状图。
- `artifacts/thesis/figures/cross_model_score_delta_distribution.png`：五维差异分布图。

## 论文表述

可将该验证称为“双模型交叉标注一致性检验”。写作时建议强调：`qwen3.5-flash` 承担大规模主标注，`qwen3.5-plus` 承担抽样复核；两者使用完全相同的新闻价值维度、评分规则和 JSON 输出约束。若总分散点集中在 `y=x` 附近、五维混淆矩阵对角线占优，且 `|差值|≤1` 一致率较高，则说明新闻价值标注在不同模型之间具有稳定一致性。
