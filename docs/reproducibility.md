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

## 小样本调试

```powershell
uv run gpnews prepare-data --max-train-impressions 200 --max-dev-impressions 50
uv run gpnews extract-siglip --max-news 500 --batch-size 32
```

小样本只用于验证流程，不作为论文最终结果。
