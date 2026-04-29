# 个性化新闻推荐模型构建与实验分析章节图表公式补充

本文档为《基于新闻价值理论的多模态特征提取及其在个性化新闻推荐中的应用》中“个性化新闻推荐模型构建与实验分析”章节提供可直接插入论文的图、表、公式和正文衔接文字。本文档独立于 `docs/thesis_chapter_supplements.md`，重点承接前文已经定义的文本特征、图像特征和新闻价值特征，说明推荐模型结构、训练评估流程与消融实验结果。

## 一、个性化推荐模型构建补充

建议放在“个性化新闻推荐模型构建”小节中。先说明三组实验只改变新闻输入特征，再放模型结构图，随后给出新闻编码、用户表示、候选新闻打分和训练目标公式。

### 图：个性化新闻推荐模型结构

![个性化新闻推荐模型结构](../artifacts/thesis/figures/recommendation_model_architecture.png)

**建议图题：** 图 4-1 个性化新闻推荐模型结构

**正文衔接句：** 如图 4-1 所示，本文构建的个性化新闻推荐模型采用统一的“新闻编码、用户建模、候选新闻打分”结构。候选新闻与用户历史点击新闻共享同一个 MLP 新闻编码器，历史新闻编码后通过 masked mean pooling 得到用户兴趣向量，最后使用用户向量与候选新闻向量的点积结果作为点击倾向得分。三组实验保持模型结构、训练样本、验证样本和评价指标一致，仅改变新闻输入向量的模态组成，从而对比文本、图文以及图文融合新闻价值特征的增量效果。

### 公式：新闻输入向量

设第 `n` 条新闻的文本特征为 `\mathbf{e}^{(t)}_n`，图像特征为 `\mathbf{e}^{(i)}_n`，新闻价值特征为 `\mathbf{v}_n`。三组实验对应的新闻输入向量 `\mathbf{z}_n` 定义如下：

```latex
\mathbf{z}^{T}_n = \mathbf{e}^{(t)}_n
```

```latex
\mathbf{z}^{TI}_n =
\left[
\mathbf{e}^{(t)}_n;
\mathbf{e}^{(i)}_n
\right]
```

```latex
\mathbf{z}^{TIV}_n =
\left[
\mathbf{e}^{(t)}_n;
\mathbf{e}^{(i)}_n;
\mathbf{v}_n
\right]
```

其中，`Text` 实验使用 `\mathbf{z}^{T}_n`，`Text+Image` 实验使用 `\mathbf{z}^{TI}_n`，`Text+Image+Value` 实验使用 `\mathbf{z}^{TIV}_n`。

### 公式：MLP 新闻编码器

推荐模型首先将新闻输入向量映射到统一的隐藏空间。设隐藏层维度为 `d_h`，本文实验中 `d_h=256`，新闻编码器为两层全连接网络：

```latex
\mathbf{h}_n =
W_2 \,
\operatorname{Dropout}
\left(
\operatorname{ReLU}
\left(
W_1 \mathbf{z}_n + \mathbf{b}_1
\right)
\right)
+ \mathbf{b}_2
```

其中，`\mathbf{h}_n` 为第 `n` 条新闻的隐藏表示。候选新闻和历史点击新闻共享同一组编码器参数，以保证两类新闻表示位于同一向量空间中。

### 公式：用户历史兴趣表示

设用户 `u` 的历史点击新闻序列为 `H_u = \{n_1,n_2,\ldots,n_L\}`，模型最多保留最近 `L=50` 条历史新闻。由于不同用户历史长度不同，引入 mask 变量 `m_i \in \{0,1\}` 标记第 `i` 个历史位置是否有效，则用户兴趣向量定义为：

```latex
\mathbf{p}_u =
\frac{
\sum_{i=1}^{L} m_i \mathbf{h}_{n_i}
}{
\max\left(\sum_{i=1}^{L} m_i, 1\right)
}
```

该公式对应 masked mean pooling。当用户历史不足最大长度时，padding 位置不会参与均值计算。

### 公式：用户与候选新闻打分

设候选新闻为 `c`，其编码结果为 `\mathbf{h}_c`。模型使用用户兴趣向量与候选新闻向量的缩放点积计算推荐得分：

```latex
r(u,c) =
\frac{
\mathbf{p}_u^\top \mathbf{h}_c
}{
\sqrt{d_h}
}
```

其中，`r(u,c)` 是未经过 Sigmoid 的 logit 分数。验证阶段对同一 impression 中的候选新闻按照该分数从高到低排序，并计算 AUC、MRR、nDCG@5 和 nDCG@10。

### 公式：训练目标函数

训练阶段将每个用户历史和候选新闻组成二分类样本，点击标签为 `y_i \in \{0,1\}`。模型使用 `BCEWithLogitsLoss` 进行优化，即直接以 logit `r_i` 作为输入：

```latex
\mathcal{L}
=
-\frac{1}{B}
\sum_{i=1}^{B}
\left[
y_i \log \sigma(r_i)
+
(1-y_i)\log\left(1-\sigma(r_i)\right)
\right]
```

其中，`B` 为 batch size，`\sigma(\cdot)` 为 Sigmoid 函数。实现中不需要在模型前向输出后手动添加 Sigmoid，损失函数内部会完成对应变换。

### 表：训练参数设置

**建议表题：** 表 4-1 三组推荐模型训练参数设置

| experiment | mode | epochs | batch_size | learning_rate | hidden_dim | dropout | max_history | eval_batch_size | device |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Text | text | 3 | 512 | 0.001 | 256 | 0.1 | 50 | 2048 | auto |
| Text+Image | multimodal | 3 | 384 | 0.001 | 256 | 0.1 | 50 | 2048 | auto |
| Text+Image+Value | value | 3 | 384 | 0.001 | 256 | 0.1 | 50 | 2048 | auto |

CSV 文件：`../artifacts/thesis/tables/training_config.csv`

**正文衔接句：** 表 4-1 展示了三组实验的训练参数。除 `Text` 模型由于输入维度较低而使用较大的 batch size 外，其余核心参数保持一致，包括训练轮数、学习率、隐藏层维度、dropout、最大历史长度和评估 batch size。这样可以尽量减少训练配置差异对结果比较的影响。

## 二、实验流程与评价指标补充

建议放在“实验设置与评价指标”小节中。先放训练与评价流程图，再给出四个排序指标的计算公式，说明模型评估基于 impression 内候选新闻排序结果。

### 图：训练与评价流程

![训练与评价流程](../artifacts/thesis/figures/training_evaluation_flow.png)

**建议图题：** 图 4-2 推荐模型训练与评价流程

**正文衔接句：** 如图 4-2 所示，实验首先读取预处理后的训练样本、验证 impression、SigLIP 图文特征和新闻价值标注缓存，然后分别训练 Text、Text+Image 和 Text+Image+Value 三组模型。每组模型训练完成后，在相同验证集上对候选新闻进行排序，并基于用户真实点击标签计算排序评价指标。由于验证集和评价逻辑保持一致，三组结果可以用于消融比较。

### 公式：AUC

对单个 impression，设点击新闻集合为 `P`，未点击新闻集合为 `N`，候选新闻得分为 `r`。AUC 衡量正样本得分高于负样本得分的概率：

```latex
\operatorname{AUC}
=
\frac{1}{|P||N|}
\sum_{p \in P}
\sum_{q \in N}
\left[
\mathbb{I}(r_p > r_q)
+
\frac{1}{2}\mathbb{I}(r_p = r_q)
\right]
```

若某个 impression 中缺少正样本或负样本，则该 impression 不参与 AUC 平均。

### 公式：MRR

MRR 关注第一个点击新闻在排序列表中的位置。设第 `j` 个 impression 中第一个点击新闻的排序位置为 `\operatorname{rank}_j`，验证集 impression 数为 `|\mathcal{Q}|`，则：

```latex
\operatorname{MRR}
=
\frac{1}{|\mathcal{Q}|}
\sum_{j=1}^{|\mathcal{Q}|}
\frac{1}{\operatorname{rank}_j}
```

MRR 对排序靠前的第一个正样本更敏感，能够反映模型是否能尽早把用户会点击的新闻排到前面。

### 公式：DCG 与 nDCG@K

nDCG@K 衡量前 `K` 个位置的整体排序质量。设排序后第 `i` 位新闻的相关性标签为 `rel_i`，则：

```latex
\operatorname{DCG}@K
=
\sum_{i=1}^{K}
\frac{2^{rel_i}-1}{\log_2(i+1)}
```

```latex
\operatorname{nDCG}@K
=
\frac{
\operatorname{DCG}@K
}{
\operatorname{IDCG}@K
}
```

其中，`\operatorname{IDCG}@K` 是理想排序下的 `DCG@K`。本文报告 `nDCG@5` 和 `nDCG@10`，分别衡量前 5 位和前 10 位推荐结果的排序质量。

### 公式：指标提升率

为了量化多模态特征和新闻价值特征相对于文本基线的提升，设文本基线指标为 `M_{\mathrm{Text}}`，对比模型指标为 `M_{\mathrm{model}}`，则绝对提升和相对提升定义为：

```latex
\Delta M
=
M_{\mathrm{model}}
-
M_{\mathrm{Text}}
```

```latex
G_{\mathrm{rel}}
=
\frac{
M_{\mathrm{model}} - M_{\mathrm{Text}}
}{
M_{\mathrm{Text}}
}
\times 100\%
```

## 三、消融实验结果分析补充

建议放在“实验结果与分析”小节中。先列三组模型的整体指标表，再放指标对比图和提升热力图，最后结合数值进行分析。

### 表：三组实验结果

**建议表题：** 表 4-2 三组推荐模型实验结果

| experiment | mode | AUC | MRR | nDCG@5 | nDCG@10 |
| --- | --- | ---: | ---: | ---: | ---: |
| Text | text | 0.551599 | 0.270629 | 0.250176 | 0.313733 |
| Text+Image | multimodal | 0.564296 | 0.265650 | 0.248802 | 0.315803 |
| Text+Image+Value | value | 0.592961 | 0.290900 | 0.273259 | 0.340781 |

CSV 文件：`../artifacts/thesis/tables/experiment_results.csv`

**正文衔接句：** 表 4-2 给出了三组实验在验证集上的排序指标。Text 模型作为文本基线，Text+Image 模型在文本特征基础上引入新闻图像特征，Text+Image+Value 模型进一步加入新闻价值向量和缺失标记。三组实验使用相同验证 impression，因此指标差异能够反映不同输入特征组合对推荐效果的影响。

### 图：实验指标对比

![三组实验指标对比](../artifacts/thesis/figures/experiment_metrics_comparison.png)

**建议图题：** 图 4-3 三组推荐模型评价指标对比

**正文衔接句：** 图 4-3 从 AUC、MRR、nDCG@5 和 nDCG@10 四个角度对比了三组模型。整体来看，Text+Image+Value 在四项指标上均达到最高值，说明新闻价值特征能够在图文语义特征之外提供额外的排序信息。

### 表：相对文本基线的指标提升

**建议表题：** 表 4-3 多模态与新闻价值特征相对文本基线的指标提升

| experiment | metric | baseline | value | absolute_gain | relative_gain_percent |
| --- | --- | ---: | ---: | ---: | ---: |
| Text+Image | AUC | 0.551599 | 0.564296 | 0.012697 | 2.30% |
| Text+Image | MRR | 0.270629 | 0.265650 | -0.004979 | -1.84% |
| Text+Image | nDCG@5 | 0.250176 | 0.248802 | -0.001374 | -0.55% |
| Text+Image | nDCG@10 | 0.313733 | 0.315803 | 0.002070 | 0.66% |
| Text+Image+Value | AUC | 0.551599 | 0.592961 | 0.041362 | 7.50% |
| Text+Image+Value | MRR | 0.270629 | 0.290900 | 0.020271 | 7.49% |
| Text+Image+Value | nDCG@5 | 0.250176 | 0.273259 | 0.023084 | 9.23% |
| Text+Image+Value | nDCG@10 | 0.313733 | 0.340781 | 0.027047 | 8.62% |

CSV 文件：`../artifacts/thesis/tables/metric_improvements.csv`

### 图：指标提升热力图

![指标提升热力图](../artifacts/thesis/figures/metric_improvement_heatmap.png)

**建议图题：** 图 4-4 多模态与新闻价值特征相对文本基线的指标提升

**正文衔接句：** 图 4-4 展示了 Text+Image 和 Text+Image+Value 相对于 Text 基线的指标变化。颜色越深表示相对提升越明显，负值则表示该指标相对文本基线有所下降。

### 结果分析正文

从表 4-2 和表 4-3 可以看出，仅加入图像特征后，Text+Image 模型的 AUC 从 `0.551599` 提升到 `0.564296`，绝对提升 `0.012697`，相对提升 `2.30%`；nDCG@10 从 `0.313733` 提升到 `0.315803`，相对提升 `0.66%`。这说明图像模态在一定程度上增强了模型区分点击新闻与未点击新闻的能力，尤其对较长候选列表上的整体排序具有正向作用。

不过，Text+Image 模型的 MRR 和 nDCG@5 相比 Text 基线略有下降，其中 MRR 下降 `1.84%`，nDCG@5 下降 `0.55%`。这表明单纯拼接图像特征并不必然改善所有排序位置，尤其是在前几位推荐结果上，视觉信息可能受到图片语义噪声、图文相关性差异以及简单特征拼接方式的影响。因此，本文不将 Text+Image 结果解释为全面提升，而将其视为图像模态对部分排序指标有帮助但仍存在不稳定性的消融结果。

进一步加入新闻价值特征后，Text+Image+Value 模型在四项指标上均优于 Text 基线。其中 AUC 达到 `0.592961`，相对提升 `7.50%`；MRR 达到 `0.290900`，相对提升 `7.49%`；nDCG@5 达到 `0.273259`，相对提升 `9.23%`；nDCG@10 达到 `0.340781`，相对提升 `8.62%`。相比单纯的图文特征，新闻价值向量为模型提供了重要性、显著性、冲突性、新奇性和人情味等传播学维度信息，使模型能够在语义相近的候选新闻之间获得额外排序依据。

综合来看，实验结果支持本文的核心假设：在个性化新闻推荐任务中，多模态图文特征能够带来一定信息增量，而基于新闻价值理论构建的结构化价值特征可以进一步改善排序效果。由于三组实验的训练样本、验证样本、模型主体结构和评价指标保持一致，Text+Image+Value 的整体提升可以较为直接地归因于新闻价值特征对图文语义表示的补充作用。同时，新闻价值标注覆盖范围仍有限，未标注新闻依赖零向量和 missing mask 表示，后续若扩大标注规模或改进价值特征融合方式，推荐效果仍有进一步提升空间。

## 四、符号说明

| 符号 | 含义 |
| --- | --- |
| `n` | 新闻编号 |
| `c` | 候选新闻 |
| `u` | 用户 |
| `\mathbf{e}^{(t)}_n` | 第 `n` 条新闻的文本特征 |
| `\mathbf{e}^{(i)}_n` | 第 `n` 条新闻的图像特征 |
| `\mathbf{v}_n` | 第 `n` 条新闻的新闻价值特征 |
| `\mathbf{z}_n` | 第 `n` 条新闻输入推荐模型的拼接特征 |
| `\mathbf{h}_n` | 第 `n` 条新闻经过 MLP 编码后的隐藏表示 |
| `H_u` | 用户 `u` 的历史点击新闻序列 |
| `m_i` | 历史序列第 `i` 个位置的有效 mask |
| `\mathbf{p}_u` | 用户 `u` 的兴趣表示 |
| `r(u,c)` | 用户 `u` 对候选新闻 `c` 的推荐得分 |
| `y_i` | 第 `i` 个训练样本的点击标签 |
| `B` | 训练 batch size |
| `d_h` | 新闻隐藏表示维度 |
| `K` | nDCG 指标中的截断排序位置 |
| `P` | 单个 impression 中点击新闻集合 |
| `N` | 单个 impression 中未点击新闻集合 |
| `M_{\mathrm{Text}}` | Text 基线模型的某项指标值 |
| `M_{\mathrm{model}}` | 对比模型的某项指标值 |
