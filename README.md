# 基于新闻价值理论的多模态新闻推荐系统

这个仓库实现一套基于 MIND-small 的多模态新闻推荐实验管线：

- SigLIP 离线提取新闻文本和图片特征
- 新闻价值五要素离线标注
- NRMS 用户编码器进行点击预测
- 支持拼接融合和门控融合两种新闻编码方案

## 环境准备

项目使用 `uv` 管理依赖。

```bash
uv sync
```

常用命令：

```bash
uv run python main.py preprocess
uv run python main.py extract-features --batch-size 16
uv run python main.py annotate-news-value --provider heuristic
uv run python main.py train --epochs 3 --fusion concat
uv run python main.py evaluate --checkpoint data/processed/nrms_latest.pt
```

也可以直接运行脚本：

```bash
uv run python scripts/train.py --epochs 3
```

## 数据约定

- `MINDsmall_train/` 和 `MINDsmall_dev/` 为原始 MIND-small 数据
- `newData/` 为与新闻 ID 对齐的图片目录，文件名格式为 `{NewsID}.jpg`
- `data/news_siglip_features.pt` 为离线图文特征
- `data/news_value_scores.json` 为新闻价值五要素打分

如果离线特征文件不存在，训练与评估会使用零向量占位，只用于流程验证，不代表最终实验配置。

## 当前实现范围

- 原始 MIND 数据解析与类别映射
- SigLIP 特征提取脚本
- 新闻价值打分脚本，支持启发式模式和 OpenAI 兼容接口
- NRMS 主模型、门控融合、训练与评估脚本
- 预处理和前向过程基础测试
