# NRMS 纯文本标准绝对底线（text_only）

## 实验目标

给出纯文本标准 NRMS 的绝对底线指标，作为多模态模型对比基准。

## 口径定义

- 模型模式：`--fusion text_only`
- 输入特征：`text + category + subcategory`
- 排除特征：`image`、`news_value`
- 数据集：`MIND-small dev` 全量
- 指标：`AUC`、`MRR`、`nDCG@5`、`nDCG@10`

## 复现命令

```bash
uv sync
uv run python main.py train --fusion text_only --epochs 3 --checkpoint data/processed/nrms_text_only.pt --eval-dev
uv run python main.py evaluate --checkpoint data/processed/nrms_text_only.pt --fusion text_only
```

## 结果记录

- 运行日期：2026-03-16
- checkpoint：`data/processed/nrms_text_only.pt`
- 训练收敛摘要：`epoch=3 loss=1.2968`

| Split | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---:|---:|---:|---:|
| dev(full) | 0.6770738304 | 0.3799338632 | 0.3641303233 | 0.4262903963 |
