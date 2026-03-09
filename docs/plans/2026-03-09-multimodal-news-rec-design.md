# 基于新闻价值理论的多模态新闻推荐系统设计

## 概述

本项目构建一个结合 SigLIP 多模态特征提取与新闻价值理论的个性化新闻推荐系统。以 NRMS 为推荐骨架，用 SigLIP 统一编码文本和图像特征（解决双塔语义空间不对齐问题），用 LLM 标注新闻价值五要素得分作为显式特征，最终在 MIND-small 数据集上训练和评估。

## 数据集

| 数据 | 规模 |
|------|------|
| MIND-small 训练集新闻 | 51,282 条 |
| MIND-small 验证集新闻 | 42,416 条 |
| 去重唯一新闻 ID | 65,238 条 |
| V-MIND 图片匹配率 | 100% |
| 训练行为记录 | 156,965 条 |
| 验证行为记录 | 73,152 条 |
| 新闻类别 | 17 类 (news, sports, finance...) |

## 整体架构

```
点击预测 (Dot Product)
  score = user_repr · candidate_repr

用户编码器 (NRMS)                候选新闻表示
  Multi-Head Self-Attention       (同新闻编码器输出)
  over clicked news

新闻编码器
  SigLIP Text+Img (768×2)  +  新闻价值五要素 (5-d)  +  类别嵌入 (64×2)
  → Concat + Projection MLP → 256-d news repr

离线预处理
  1. SigLIP 编码 (title+abstract → text_emb, image → img_emb)
  2. LLM 标注新闻价值五要素得分 (5维)
  3. 类别/子类别 ID 映射
```

## 模块详细设计

### 1. SigLIP 特征提取（离线预计算）

- **模型**: `google/siglip-base-patch16-224`
- **文本输入**: `title + " [SEP] " + abstract`，截断至 64 tokens
- **图像输入**: `newData/{NewsID}.jpg`，resize 到 224×224
- **输出**: text_emb (768-d) + img_emb (768-d)，CLS token 或 mean pooling
- **存储**: `news_siglip_features.pt`，key 为 NewsID

### 2. 新闻价值五要素标注（离线）

- **工具**: DeepSeek/GPT API 批量标注
- **五要素**:
  - 时新性 (Timeliness): 事件的时效性和新鲜程度
  - 重要性 (Importance): 对公众利益的影响程度
  - 显著性 (Prominence): 涉及人物/事件的知名度
  - 接近性 (Proximity): 与读者的地理/心理距离
  - 趣味性 (Interest): 内容的吸引力和可读性
- **评分**: 每个要素 1-5 整数分
- **存储**: `news_value_scores.json`，格式 `{NewsID: [t, i, p, pr, in]}`

### 3. 新闻编码器

```
输入: siglip_text(768) + siglip_image(768) + news_value(5) + cat_emb(64) + subcat_emb(64)
     = 1669 维
→ Linear(1669, 256) + ReLU + Dropout(0.2)
→ news_repr (256-d)
```

### 4. 用户编码器 (NRMS)

```
输入: 用户历史点击新闻的 news_repr 序列 [max_history=50, 256]
→ Multi-Head Self-Attention (heads=16, dim=256)
→ Additive Attention (聚合为单向量)
→ user_repr (256-d)
```

### 5. 点击预测与训练

- **得分**: `score = dot(user_repr, candidate_repr)`
- **损失**: NLL Loss (impression 内 softmax)
- **评估指标**: AUC, MRR, nDCG@5, nDCG@10
- **优化器**: Adam, lr=1e-4
- **Batch size**: 64
- **Epochs**: 5-10

## 消融实验设计

### 方案 B: 门控融合 (Gated Fusion)

```
siglip_feat = concat(siglip_text, siglip_image)  # 1536-d
nv_feat = news_value_scores  # 5-d (投影到 256-d)

gate = sigmoid(W1 @ siglip_feat + W2 @ nv_feat)
fused = gate * proj(siglip_feat) + (1 - gate) * proj(nv_feat)
→ news_repr (256-d)
```

### 实验对比组

| 编号 | 模型变体 | 说明 |
|------|---------|------|
| 1 | NRMS-Text | 纯文本 baseline (SigLIP text only) |
| 2 | NRMS-MM | SigLIP 多模态，无新闻价值 |
| 3 | NRMS-MM-NV (主方案) | SigLIP + 五要素拼接融合 |
| 4 | NRMS-MM-NV-Gate | SigLIP + 五要素门控融合 |

## 技术栈

- Python 3.10+
- PyTorch 2.x
- transformers (SigLIP)
- 云 GPU 训练

## 项目结构

```
Graduation Project/
├── data/                     # 处理后的数据
│   ├── news_siglip_features.pt
│   ├── news_value_scores.json
│   └── processed/
├── src/
│   ├── data/                 # 数据加载与处理
│   │   ├── dataset.py
│   │   └── preprocess.py
│   ├── models/               # 模型定义
│   │   ├── news_encoder.py
│   │   ├── user_encoder.py
│   │   ├── nrms.py
│   │   └── gated_fusion.py
│   ├── features/             # 特征提取
│   │   ├── siglip_extractor.py
│   │   └── news_value_annotator.py
│   └── utils/
│       ├── metrics.py
│       └── config.py
├── scripts/
│   ├── extract_features.py   # 离线特征提取
│   ├── annotate_news_value.py# LLM标注
│   ├── train.py
│   └── evaluate.py
├── MINDsmall_train/          # 原始数据
├── MINDsmall_dev/
├── newData/                  # V-MIND 图片
└── docs/plans/
```
