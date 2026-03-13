[English](./README-EN.md) | [中文](./README.md)

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
uv run python main.py dataset-report
uv run python main.py extract-features --batch-size 16
uv run python main.py annotate-news-value --provider heuristic
uv run python main.py train --epochs 3 --fusion concat
uv run python main.py evaluate --checkpoint data/processed/nrms_latest.pt
```

单条新闻价值特征提取（真实 API case）：

推荐在项目根目录创建 `.env`：

```bash
NEWS_VALUE_API_BASE=https://api-inference.modelscope.cn/v1
NEWS_VALUE_API_KEY=<MODELSCOPE_TOKEN>
NEWS_VALUE_MODEL=ZhipuAI/GLM-5
```

也支持临时设置环境变量（Windows CMD）：

```bash
set NEWS_VALUE_API_BASE=https://api-inference.modelscope.cn/v1
set NEWS_VALUE_API_KEY=<MODELSCOPE_TOKEN>
set NEWS_VALUE_MODEL=ZhipuAI/GLM-5

uv run python main.py annotate-news-value --provider openai-compatible --single-title "突发：某地发布重大政策" --single-abstract "官方今天发布新规，影响多个行业。" --single-category news --single-subcategory policy
```

输出示例会包含：

- 输入新闻内容
- 五维价值打分（`timeliness`、`importance`、`prominence`、`proximity`、`interest`）
- 向量数组，例如 `[4, 4, 3, 3, 2]`

也可以直接运行脚本：

```bash
uv run python scripts/train.py --epochs 3
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

## 当前实现范围

- 原始 MIND 数据解析与类别映射
- SigLIP 特征提取脚本
- 新闻价值打分脚本，支持启发式模式和 OpenAI 兼容接口
- NRMS 主模型、门控融合、训练与评估脚本
- 预处理和前向过程基础测试
