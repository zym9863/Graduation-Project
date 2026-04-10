# 融合对比与模态消融实验记录

## 1. 实验目标

- 融合策略对比：验证 `cross_modal` 是否优于 `concat` 和 `gate`
- 模态消融分析：量化 image 与 news_value 两种模态的独立贡献

## 2. 固定实验口径

- 数据：MIND-small train/dev
- 指标：AUC、MRR、nDCG@5、nDCG@10
- 训练轮数：建议 8（可按实际资源调整）
- 种子：`42`、`2026`
- 其余训练参数保持一致（batch size、scheduler、patience 等）

## 3. 融合策略对比矩阵

| 组别 | fusion | 输入特征 |
|---|---|---|
| C1 | concat | text + image + value + category + subcategory |
| C2 | gate | text + image + value + category + subcategory |
| C3 | cross_modal | text(query) + image(key/value) + value + category + subcategory |

### 运行命令模板

```bash
uv run python main.py train --fusion concat --epochs 8 --eval-dev --seed 42 --checkpoint data/processed/concat_s42.pt
uv run python main.py train --fusion concat --epochs 8 --eval-dev --seed 2026 --checkpoint data/processed/concat_s2026.pt

uv run python main.py train --fusion gate --epochs 8 --eval-dev --seed 42 --checkpoint data/processed/gate_s42.pt
uv run python main.py train --fusion gate --epochs 8 --eval-dev --seed 2026 --checkpoint data/processed/gate_s2026.pt

uv run python main.py train --fusion cross_modal --epochs 8 --eval-dev --seed 42 --checkpoint data/processed/cross_modal_s42.pt
uv run python main.py train --fusion cross_modal --epochs 8 --eval-dev --seed 2026 --checkpoint data/processed/cross_modal_s2026.pt
```

## 4. 模态级消融矩阵

| 组别 | fusion | 输入特征 |
|---|---|---|
| A1 | text_only | text + category + subcategory |
| A2 | text_image | text + image + category + subcategory |
| A3 | text_value | text + value + category + subcategory |
| A4 | text_image_value | text + image + value + category + subcategory |

### 运行命令模板

```bash
uv run python main.py train --fusion text_only --epochs 8 --eval-dev --seed 42 --checkpoint data/processed/text_only_s42.pt
uv run python main.py train --fusion text_only --epochs 8 --eval-dev --seed 2026 --checkpoint data/processed/text_only_s2026.pt

uv run python main.py train --fusion text_image --epochs 8 --eval-dev --seed 42 --checkpoint data/processed/text_image_s42.pt
uv run python main.py train --fusion text_image --epochs 8 --eval-dev --seed 2026 --checkpoint data/processed/text_image_s2026.pt

uv run python main.py train --fusion text_value --epochs 8 --eval-dev --seed 42 --checkpoint data/processed/text_value_s42.pt
uv run python main.py train --fusion text_value --epochs 8 --eval-dev --seed 2026 --checkpoint data/processed/text_value_s2026.pt

uv run python main.py train --fusion text_image_value --epochs 8 --eval-dev --seed 42 --checkpoint data/processed/text_image_value_s42.pt
uv run python main.py train --fusion text_image_value --epochs 8 --eval-dev --seed 2026 --checkpoint data/processed/text_image_value_s2026.pt
```

## 5. 结果汇总

执行：

```bash
uv run python main.py experiment-summary --glob-pattern "data/processed/*_s*.pt" --output-dir data/processed/experiment_reports
```

输出：

- `data/processed/experiment_reports/checkpoint_metrics.csv`
- `data/processed/experiment_reports/fusion_summary.csv`

## 6. 报告表模板

### 6.1 融合策略对比（mean ± std）

| Fusion | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---:|---:|---:|---:|
| concat | - | - | - | - |
| gate | - | - | - | - |
| cross_modal | - | - | - | - |

### 6.2 模态消融（mean ± std）

| Fusion | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---:|---:|---:|---:|
| text_only | 0.6806967062 ± 0.0049775044 | 0.3786821911 ± 0.0074919024 | 0.3643353261 ± 0.0067258647 | 0.4263792708 ± 0.0063694794 |
| text_image | - | - | - | - |
| text_value | - | - | - | - |
| text_image_value | - | - | - | - |

### 6.3 text_only 双种子明细

| Seed | Best Epoch | lr | loss | AUC | MRR | nDCG@5 | nDCG@10 | Checkpoint |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 42 | 7 | 5e-5 | 1.2279 | 0.6842163325 | 0.3839797641 | 0.3690912593 | 0.4308831524 | `data/processed/text_only_s42.pt` |
| 2026 | 6 | 1e-4 | 1.2520 | 0.6771770799 | 0.3733846181 | 0.3595793929 | 0.4218753891 | `data/processed/text_only_s2026.pt` |
