# 设计文档：将新闻价值维度 timeliness 替换为 conflict

## 背景

项目使用新闻价值五要素对 MIND-small 数据集中的新闻进行标注，当前维度为：
`timeliness, importance, prominence, proximity, interest`。

**问题**：timeliness（时新性）依赖新闻发布时间与阅读时间的关系，纯文本标注无法可靠判断。启发式方法仅通过 "breaking"、"today" 等关键词猜测，LLM 标注同样缺乏时间上下文。

**决策**：用 **conflict（冲突性）** 替换 timeliness。理论依据来自 Galtung & Ruge (1965) 经典新闻价值理论，冲突性与现有其他四个维度正交性最好，且 LLM 从文本中判断冲突性的可操作性极高。

## proximity 维度重新定义

原 proximity（接近性）在传统新闻价值理论中指地理接近性，依赖读者位置信息，LLM 无法判断。本设计将其**重新定义为"心理接近性"**：事件与普通大众日常生活的关联程度。

这一重定义使得 proximity 可以从文本内容本身判断（如涉及物价、就业、教育、医疗等民生话题则接近性高），无需读者地理位置信息。原启发式中 `proximity = 3` 的硬编码也将更新为基于民生关键词的规则。

## 新五维度

替换后的五个维度：`conflict, importance, prominence, proximity, interest`

维度总数保持 5 不变，`news_value_dim` 配置无需修改。

## 评分标准（1-5 分整数）

### 1. conflict（冲突性）
新闻是否涉及对抗、争端、分歧或紧张关系。
- 1 分：无任何冲突元素
- 3 分：存在温和分歧或潜在矛盾
- 5 分：涉及激烈对抗、战争、诉讼等严重冲突

### 2. importance（重要性）
事件对社会、经济或公众生活的影响程度。
- 1 分：影响极小，仅涉及个别人
- 3 分：影响一定范围的群体或行业
- 5 分：影响国家或全球范围的重大事件

### 3. prominence（显著性）
涉及人物、机构或地点的知名度。
- 1 分：涉及普通个人或无名机构
- 3 分：涉及地区知名人物或机构
- 5 分：涉及国际级知名人物、领导人或机构

### 4. proximity（接近性）— 重新定义为心理接近性
事件与普通大众日常生活的心理关联程度。
- 1 分：与大多数人生活无关
- 3 分：与部分人群的日常生活相关
- 5 分：直接关系到每个人的切身利益

### 5. interest（趣味性）
内容的吸引力和可读性。
- 1 分：枯燥乏味，无吸引力
- 3 分：有一定趣味或话题性
- 5 分：极具吸引力，引人入胜

## 增强版 System Prompt

```
你是新闻价值标注器。请基于给定新闻的标题、摘要和类别信息，对以下五个维度分别打 1-5 分整数。

## 评分维度

1. conflict（冲突性）：新闻是否涉及对抗、争端、分歧或紧张关系。
   - 1 分：无任何冲突元素
   - 3 分：存在温和分歧或潜在矛盾
   - 5 分：涉及激烈对抗、战争、诉讼等严重冲突

2. importance（重要性）：事件对社会、经济或公众生活的影响程度。
   - 1 分：影响极小，仅涉及个别人
   - 3 分：影响一定范围的群体或行业
   - 5 分：影响国家或全球范围的重大事件

3. prominence（显著性）：涉及人物、机构或地点的知名度。
   - 1 分：涉及普通个人或无名机构
   - 3 分：涉及地区知名人物或机构
   - 5 分：涉及国际级知名人物、领导人或机构

4. proximity（接近性）：事件与普通大众日常生活的心理关联程度。
   - 1 分：与大多数人生活无关
   - 3 分：与部分人群的日常生活相关
   - 5 分：直接关系到每个人的切身利益

5. interest（趣味性）：内容的吸引力和可读性。
   - 1 分：枯燥乏味，无吸引力
   - 3 分：有一定趣味或话题性
   - 5 分：极具吸引力，引人入胜

## 输出格式

只返回 JSON 对象，格式如下：
{"conflict": <int>, "importance": <int>, "prominence": <int>, "proximity": <int>, "interest": <int>}
```

## 启发式规则更新

### conflict（新增，替换 timeliness）
- 冲突关键词列表：war, conflict, dispute, protest, lawsuit, attack, oppose, fight, debate, crisis, strike, clash, battle, sue, accuse, condemn, threat, sanction, ban, arrest
- 匹配到任一关键词 → 4 分，否则 → 2 分
- 默认值选择 2 而非 3 的理由：大多数新闻不涉及明显冲突，2 分更接近真实分布（"无明显冲突但不排除隐含分歧"）

### proximity（更新启发式逻辑）
- 原逻辑：硬编码为 3
- 新逻辑：使用民生关键词检测心理接近性
- 民生关键词列表：price, tax, job, employment, wage, salary, health, hospital, school, education, housing, rent, food, inflation, insurance, retirement, pension, traffic, commute
- 匹配到任一关键词 → 4 分，否则 → 3 分

### 其他维度（importance, prominence, interest）
- 保持原有启发式逻辑不变

## 代码变更清单

### 需要修改的文件

| 文件 | 变更内容 |
|------|---------|
| `src/features/news_value_annotator.py` | SYSTEM_PROMPT 替换为增强版；heuristic 中 timeliness→conflict（冲突关键词）、proximity 更新为民生关键词逻辑；parse 函数字段映射 `timeliness`/`时新性` → `conflict`/`冲突性` |
| `scripts/annotate_news_value.py` | VALUE_DIMENSIONS 元组中 timeliness → conflict |
| `tests/test_news_value_annotator.py` | 英文字段 `timeliness`→`conflict`，中文字段 `时新性`→`冲突性`；缺失字段测试用例更新；heuristic 期望值调整（conflict 默认 2 而非 4） |
| `tests/test_annotate_news_value.py` | 断言中 `"timeliness"` → `"conflict"`；heuristic 测试输入可添加冲突关键词以验证 4 分逻辑 |
| `README.md` | 五维度描述更新 |
| `README-EN.md` | 五维度描述更新 |

### 不需要修改的文件

- `src/utils/config.py` — `news_value_dim = 5` 不变
- `src/models/news_encoder.py` — 只消费数值向量
- `src/models/gated_fusion.py` — 只消费数值向量
- `src/data/dataset.py` — 只消费数值向量

### 数据影响

- 已有的 `data/news_value_scores.json` 需要重新生成（旧数据第一维是 timeliness 分数，语义已变）
