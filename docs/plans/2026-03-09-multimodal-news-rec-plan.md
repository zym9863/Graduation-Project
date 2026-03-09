# 多模态新闻推荐系统 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 构建基于 SigLIP + 新闻价值五要素 + NRMS 的多模态新闻推荐系统，在 MIND-small 上训练评估。

**Architecture:** SigLIP 冻结离线提取对齐的文本/图像特征，LLM 标注新闻价值五要素得分，三者拼接后经 MLP 投影为新闻表示，送入 NRMS 用户编码器完成推荐。消融实验用门控融合替换拼接。

**Tech Stack:** Python 3.10+, PyTorch 2.x, transformers (SigLIP), PIL, tqdm

---

## 数据格式参考

**news.tsv** (tab-separated, 无表头):
```
列1: NewsID (e.g. N55528)
列2: Category (e.g. lifestyle) — 共17类
列3: SubCategory (e.g. lifestyleroyals) — 共264类
列4: Title
列5: Abstract
列6: URL
列7: Title Entities (JSON array)
列8: Abstract Entities (JSON array)
```

**behaviors.tsv** (tab-separated, 无表头):
```
列1: ImpressionID (数字)
列2: UserID (e.g. U13740)
列3: Time (e.g. 11/11/2019 9:05:58 AM)
列4: History (空格分隔的NewsID)
列5: Impressions (空格分隔的 NewsID-label, e.g. N55689-1 N35729-0)
```

**newData/**: `{NewsID}.jpg` 图片文件，65238张完整覆盖 MIND-small。

---

## Task 1: 项目脚手架与依赖

**Files:**
- Create: `src/__init__.py`
- Create: `src/utils/__init__.py`
- Create: `src/utils/config.py`
- Create: `src/data/__init__.py`
- Create: `src/models/__init__.py`
- Create: `src/features/__init__.py`
- Create: `scripts/`
- Create: `tests/__init__.py`
- Create: `requirements.txt`

**Step 1: 创建项目目录结构**

```bash
mkdir -p src/utils src/data src/models src/features scripts tests data/processed
touch src/__init__.py src/utils/__init__.py src/data/__init__.py src/models/__init__.py src/features/__init__.py tests/__init__.py
```

**Step 2: 创建 requirements.txt**

```
torch>=2.0.0
transformers>=4.37.0
Pillow>=10.0.0
tqdm>=4.65.0
numpy>=1.24.0
scikit-learn>=1.3.0
```

**Step 3: 创建 config.py**

```python
# src/utils/config.py
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # 路径
    project_root: Path = Path(__file__).resolve().parent.parent.parent
    train_dir: Path = field(default=None)
    dev_dir: Path = field(default=None)
    image_dir: Path = field(default=None)
    feature_dir: Path = field(default=None)

    # SigLIP
    siglip_model_name: str = "google/siglip-base-patch16-224"
    siglip_max_length: int = 64
    siglip_dim: int = 768

    # 新闻编码器
    news_repr_dim: int = 256
    category_emb_dim: int = 64
    num_categories: int = 17
    num_subcategories: int = 264
    news_value_dim: int = 5
    dropout: float = 0.2

    # 用户编码器
    num_attention_heads: int = 16
    max_history_len: int = 50

    # 训练
    batch_size: int = 64
    lr: float = 1e-4
    epochs: int = 10
    npratio: int = 4  # 负采样比例

    # 设备
    device: str = "cuda"

    def __post_init__(self):
        self.train_dir = self.project_root / "MINDsmall_train"
        self.dev_dir = self.project_root / "MINDsmall_dev"
        self.image_dir = self.project_root / "newData"
        self.feature_dir = self.project_root / "data"
```

**Step 4: 安装依赖并验证**

```bash
pip install -r requirements.txt
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

**Step 5: Commit**

```bash
git add -A
git commit -m "chore: project scaffolding with config and dependencies"
```

---

## Task 2: 数据解析模块

**Files:**
- Create: `src/data/preprocess.py`
- Create: `tests/test_preprocess.py`

**Step 1: 写测试**

```python
# tests/test_preprocess.py
import pytest
from src.data.preprocess import parse_news_file, parse_behaviors_file


def test_parse_news_returns_dict(tmp_path):
    tsv = tmp_path / "news.tsv"
    tsv.write_text(
        "N1\tnews\tnewsworld\tTitle One\tAbstract one.\thttps://ex.com\t[]\t[]\n"
        "N2\tsports\tbaseball\tTitle Two\tAbstract two.\thttps://ex.com\t[]\t[]\n"
    )
    news = parse_news_file(tsv)
    assert len(news) == 2
    assert news["N1"]["category"] == "news"
    assert news["N1"]["subcategory"] == "newsworld"
    assert news["N1"]["title"] == "Title One"
    assert news["N1"]["abstract"] == "Abstract one."


def test_parse_behaviors_returns_list(tmp_path):
    tsv = tmp_path / "behaviors.tsv"
    tsv.write_text(
        "1\tU1\t11/11/2019 9:00:00 AM\tN1 N2\tN3-1 N4-0 N5-0\n"
    )
    behaviors = parse_behaviors_file(tsv)
    assert len(behaviors) == 1
    b = behaviors[0]
    assert b["user_id"] == "U1"
    assert b["history"] == ["N1", "N2"]
    assert b["impressions"] == [("N3", 1), ("N4", 0), ("N5", 0)]
```

**Step 2: 运行测试确认失败**

```bash
python -m pytest tests/test_preprocess.py -v
```
Expected: FAIL — `ModuleNotFoundError`

**Step 3: 实现数据解析**

```python
# src/data/preprocess.py
"""MIND 数据集解析工具。"""
from pathlib import Path


def parse_news_file(filepath: Path) -> dict:
    """解析 news.tsv，返回 {NewsID: {category, subcategory, title, abstract}}。"""
    news = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue
            nid = parts[0]
            news[nid] = {
                "category": parts[1],
                "subcategory": parts[2],
                "title": parts[3],
                "abstract": parts[4] if len(parts) > 4 else "",
            }
    return news


def parse_behaviors_file(filepath: Path) -> list:
    """解析 behaviors.tsv，返回行为列表。"""
    behaviors = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue
            history = parts[3].split() if parts[3] else []
            impressions = []
            for imp in parts[4].split():
                nid, label = imp.rsplit("-", 1)
                impressions.append((nid, int(label)))
            behaviors.append({
                "impression_id": parts[0],
                "user_id": parts[1],
                "time": parts[2],
                "history": history,
                "impressions": impressions,
            })
    return behaviors


def build_category_maps(news_dict: dict) -> tuple[dict, dict]:
    """构建类别和子类别到ID的映射。"""
    categories = sorted(set(n["category"] for n in news_dict.values()))
    subcategories = sorted(set(n["subcategory"] for n in news_dict.values()))
    cat2id = {c: i + 1 for i, c in enumerate(categories)}  # 0 reserved for padding
    subcat2id = {s: i + 1 for i, s in enumerate(subcategories)}
    return cat2id, subcat2id
```

**Step 4: 运行测试确认通过**

```bash
python -m pytest tests/test_preprocess.py -v
```
Expected: 2 passed

**Step 5: 在真实数据上验证**

```bash
python -c "
from src.data.preprocess import parse_news_file, parse_behaviors_file, build_category_maps
from pathlib import Path
news = parse_news_file(Path('MINDsmall_train/news.tsv'))
print(f'Train news: {len(news)}')
beh = parse_behaviors_file(Path('MINDsmall_train/behaviors.tsv'))
print(f'Train behaviors: {len(beh)}')
cat2id, subcat2id = build_category_maps(news)
print(f'Categories: {len(cat2id)}, Subcategories: {len(subcat2id)}')
"
```
Expected: Train news: 51282, Train behaviors: 156965

**Step 6: Commit**

```bash
git add src/data/preprocess.py tests/test_preprocess.py
git commit -m "feat: MIND data parsing with category mapping"
```

---

## Task 3: SigLIP 特征提取脚本

**Files:**
- Create: `src/features/siglip_extractor.py`
- Create: `scripts/extract_features.py`

**Step 1: 实现 SigLIP 提取器**

```python
# src/features/siglip_extractor.py
"""SigLIP 离线特征提取。"""
import torch
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModel
from tqdm import tqdm


class SigLIPExtractor:
    def __init__(self, model_name: str = "google/siglip-base-patch16-224",
                 device: str = "cuda"):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()

    @torch.no_grad()
    def extract_batch(self, texts: list[str], image_paths: list[Path]
                      ) -> tuple[torch.Tensor, torch.Tensor]:
        """提取一批文本和图像的 SigLIP 嵌入。

        Returns:
            text_embeds: (batch, 768)
            image_embeds: (batch, 768)
        """
        images = []
        for p in image_paths:
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                img = Image.new("RGB", (224, 224))  # fallback
            images.append(img)

        inputs = self.processor(
            text=texts, images=images,
            padding="max_length", truncation=True, max_length=64,
            return_tensors="pt"
        ).to(self.device)

        outputs = self.model(**inputs)
        text_embeds = outputs.text_model_output.pooler_output   # (batch, 768)
        image_embeds = outputs.vision_model_output.pooler_output  # (batch, 768)
        return text_embeds.cpu(), image_embeds.cpu()

    def extract_all(self, news_dict: dict, image_dir: Path,
                    batch_size: int = 32) -> dict:
        """对所有新闻提取特征，返回 {nid: {'text': tensor, 'image': tensor}}。"""
        nids = list(news_dict.keys())
        features = {}

        for i in tqdm(range(0, len(nids), batch_size), desc="Extracting SigLIP"):
            batch_nids = nids[i:i + batch_size]
            texts = [
                news_dict[nid]["title"] + " [SEP] " + news_dict[nid]["abstract"]
                for nid in batch_nids
            ]
            image_paths = [image_dir / f"{nid}.jpg" for nid in batch_nids]
            text_embs, img_embs = self.extract_batch(texts, image_paths)

            for j, nid in enumerate(batch_nids):
                features[nid] = {
                    "text": text_embs[j],
                    "image": img_embs[j],
                }

        return features
```

**Step 2: 创建提取脚本**

```python
# scripts/extract_features.py
"""离线提取 SigLIP 特征并保存。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.data.preprocess import parse_news_file
from src.features.siglip_extractor import SigLIPExtractor
from src.utils.config import Config


def main():
    cfg = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 解析训练集+验证集新闻（取并集）
    train_news = parse_news_file(cfg.train_dir / "news.tsv")
    dev_news = parse_news_file(cfg.dev_dir / "news.tsv")
    all_news = {**train_news, **dev_news}
    print(f"Total unique news: {len(all_news)}")

    # 提取特征
    extractor = SigLIPExtractor(cfg.siglip_model_name, device=device)
    features = extractor.extract_all(all_news, cfg.image_dir, batch_size=32)

    # 保存
    save_path = cfg.feature_dir / "news_siglip_features.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(features, save_path)
    print(f"Saved {len(features)} news features to {save_path}")


if __name__ == "__main__":
    main()
```

**Step 3: 用少量数据冒烟测试**

```bash
python -c "
import torch
from src.features.siglip_extractor import SigLIPExtractor
from pathlib import Path
ext = SigLIPExtractor(device='cpu')
news = {'N55528': {'title': 'Test title', 'abstract': 'Test abstract'}}
feats = ext.extract_all(news, Path('newData'), batch_size=1)
print(feats['N55528']['text'].shape, feats['N55528']['image'].shape)
"
```
Expected: `torch.Size([768]) torch.Size([768])`

**Step 4: Commit**

```bash
git add src/features/siglip_extractor.py scripts/extract_features.py
git commit -m "feat: SigLIP offline feature extraction"
```

**Step 5: 在云 GPU 上运行完整提取**

```bash
python scripts/extract_features.py
```
Expected: 处理 65238 条新闻，保存至 `data/news_siglip_features.pt`

**Step 6: Commit 数据**（将 .pt 加入 .gitignore）

```bash
echo "data/*.pt" >> .gitignore
echo "data/*.json" >> .gitignore
git add .gitignore
git commit -m "chore: ignore large data files"
```

---

## Task 4: LLM 新闻价值标注脚本

**Files:**
- Create: `src/features/news_value_annotator.py`
- Create: `scripts/annotate_news_value.py`

**Step 1: 实现标注器**

```python
# src/features/news_value_annotator.py
"""用 LLM 批量标注新闻价值五要素得分。"""
import json
import time
from pathlib import Path
from openai import OpenAI


SYSTEM_PROMPT = """You are a journalism expert. Rate the following news article on 5 news value dimensions.
Return ONLY a JSON object with exactly these keys and integer values from 1-5:
{"timeliness": N, "importance": N, "prominence": N, "proximity": N, "interest": N}

Definitions:
- timeliness: How time-sensitive or fresh is this news?
- importance: How much does this affect public interest?
- prominence: How well-known are the people/events involved?
- proximity: How close is this to the average reader (geographically/psychologically)?
- interest: How engaging or readable is this content?"""


def annotate_single(client: OpenAI, title: str, abstract: str,
                    model: str = "deepseek-chat") -> list[int]:
    """对单条新闻标注五要素得分，返回 [t, i, p, pr, in]。"""
    user_msg = f"Title: {title}\nAbstract: {abstract}"
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.1,
        max_tokens=100,
    )
    text = resp.choices[0].message.content.strip()
    # 提取 JSON
    if "{" in text:
        text = text[text.index("{"):text.rindex("}") + 1]
    scores = json.loads(text)
    return [
        scores["timeliness"],
        scores["importance"],
        scores["prominence"],
        scores["proximity"],
        scores["interest"],
    ]


def annotate_batch(client: OpenAI, news_dict: dict, output_path: Path,
                   model: str = "deepseek-chat",
                   batch_delay: float = 0.5):
    """批量标注所有新闻，支持断点续传。"""
    # 加载已有结果
    if output_path.exists():
        with open(output_path, "r") as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing annotations")
    else:
        results = {}

    nids = [nid for nid in news_dict if nid not in results]
    print(f"Remaining to annotate: {len(nids)}")

    for i, nid in enumerate(nids):
        news = news_dict[nid]
        try:
            scores = annotate_single(client, news["title"], news["abstract"],
                                     model=model)
            results[nid] = scores
        except Exception as e:
            print(f"Error on {nid}: {e}")
            results[nid] = [3, 3, 3, 3, 3]  # fallback to neutral

        # 定期保存
        if (i + 1) % 100 == 0:
            with open(output_path, "w") as f:
                json.dump(results, f)
            print(f"Progress: {i + 1}/{len(nids)}")

        time.sleep(batch_delay)

    # 最终保存
    with open(output_path, "w") as f:
        json.dump(results, f)
    print(f"Done. Total annotations: {len(results)}")
```

**Step 2: 创建标注脚本**

```python
# scripts/annotate_news_value.py
"""批量调用 LLM 标注新闻价值得分。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI
from src.data.preprocess import parse_news_file
from src.features.news_value_annotator import annotate_batch
from src.utils.config import Config


def main():
    cfg = Config()

    # 使用 DeepSeek API (兼容 OpenAI SDK)
    client = OpenAI(
        api_key="YOUR_API_KEY",  # 替换为实际 key
        base_url="https://api.deepseek.com",
    )

    # 解析所有新闻
    train_news = parse_news_file(cfg.train_dir / "news.tsv")
    dev_news = parse_news_file(cfg.dev_dir / "news.tsv")
    all_news = {**train_news, **dev_news}
    print(f"Total news to annotate: {len(all_news)}")

    # 批量标注
    output_path = cfg.feature_dir / "news_value_scores.json"
    annotate_batch(client, all_news, output_path,
                   model="deepseek-chat", batch_delay=0.3)


if __name__ == "__main__":
    main()
```

**Step 3: 用少量数据测试**

```bash
python -c "
import json
from openai import OpenAI
from src.features.news_value_annotator import annotate_single
client = OpenAI(api_key='YOUR_KEY', base_url='https://api.deepseek.com')
scores = annotate_single(client, 'Breaking: Major earthquake hits Tokyo', 'A 7.2 magnitude earthquake struck central Tokyo today.')
print(scores)  # Expected: [5, 5, 4, 3, 4] or similar
"
```

**Step 4: Commit**

```bash
git add src/features/news_value_annotator.py scripts/annotate_news_value.py
git commit -m "feat: LLM-based news value annotation with resume support"
```

**Step 5: 运行完整标注**（需要 API key，约 65K 调用）

```bash
python scripts/annotate_news_value.py
```

---

## Task 5: Dataset 与 DataLoader

**Files:**
- Create: `src/data/dataset.py`
- Create: `tests/test_dataset.py`

**Step 1: 写测试**

```python
# tests/test_dataset.py
import pytest
import torch
from src.data.dataset import MINDDataset


def test_dataset_getitem_shapes():
    """验证 Dataset 输出的张量形状正确。"""
    # 构造 mock 数据
    news_features = {
        "N1": {"text": torch.randn(768), "image": torch.randn(768)},
        "N2": {"text": torch.randn(768), "image": torch.randn(768)},
        "N3": {"text": torch.randn(768), "image": torch.randn(768)},
    }
    news_value = {"N1": [3, 4, 2, 3, 5], "N2": [1, 2, 3, 4, 5], "N3": [5, 4, 3, 2, 1]}
    news_info = {
        "N1": {"category": "news", "subcategory": "newsworld"},
        "N2": {"category": "sports", "subcategory": "baseball"},
        "N3": {"category": "news", "subcategory": "newsus"},
    }
    cat2id = {"news": 1, "sports": 2}
    subcat2id = {"newsworld": 1, "baseball": 2, "newsus": 3}
    behaviors = [
        {"user_id": "U1", "history": ["N1"], "impressions": [("N2", 1), ("N3", 0)]}
    ]

    ds = MINDDataset(
        behaviors=behaviors,
        news_features=news_features,
        news_value=news_value,
        news_info=news_info,
        cat2id=cat2id,
        subcat2id=subcat2id,
        max_history=50,
        npratio=1,
    )

    sample = ds[0]
    # history: (max_history, feature_dim)
    assert sample["history_features"].shape == (50, 768 * 2 + 5 + 2)
    assert sample["history_mask"].shape == (50,)
    # candidates: (1+npratio, feature_dim)
    assert sample["candidate_features"].shape[0] == 2  # 1 pos + 1 neg
    assert sample["labels"].shape == (2,)
```

**Step 2: 运行测试确认失败**

```bash
python -m pytest tests/test_dataset.py -v
```

**Step 3: 实现 Dataset**

```python
# src/data/dataset.py
"""MIND 推荐数据集。"""
import random
import torch
from torch.utils.data import Dataset


class MINDDataset(Dataset):
    def __init__(self, behaviors: list, news_features: dict,
                 news_value: dict, news_info: dict,
                 cat2id: dict, subcat2id: dict,
                 max_history: int = 50, npratio: int = 4):
        self.behaviors = behaviors
        self.news_features = news_features
        self.news_value = news_value
        self.news_info = news_info
        self.cat2id = cat2id
        self.subcat2id = subcat2id
        self.max_history = max_history
        self.npratio = npratio

    def _get_news_vector(self, nid: str) -> torch.Tensor:
        """获取单条新闻的完整特征向量。"""
        if nid in self.news_features:
            text_emb = self.news_features[nid]["text"]   # (768,)
            img_emb = self.news_features[nid]["image"]   # (768,)
        else:
            text_emb = torch.zeros(768)
            img_emb = torch.zeros(768)

        nv = torch.tensor(self.news_value.get(nid, [3, 3, 3, 3, 3]),
                          dtype=torch.float32)  # (5,)

        info = self.news_info.get(nid, {"category": "", "subcategory": ""})
        cat_id = self.cat2id.get(info["category"], 0)
        subcat_id = self.subcat2id.get(info["subcategory"], 0)
        cat_vec = torch.tensor([cat_id, subcat_id], dtype=torch.float32)  # (2,)

        return torch.cat([text_emb, img_emb, nv, cat_vec])  # (1543,)

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx):
        beh = self.behaviors[idx]
        history = beh["history"]

        # 历史点击特征 (padding/truncation)
        hist_vecs = []
        for nid in history[-self.max_history:]:
            hist_vecs.append(self._get_news_vector(nid))

        # Padding
        feat_dim = 768 * 2 + 5 + 2  # 1543
        mask = torch.zeros(self.max_history)
        if hist_vecs:
            hist_tensor = torch.stack(hist_vecs)
            pad_len = self.max_history - len(hist_vecs)
            if pad_len > 0:
                padding = torch.zeros(pad_len, feat_dim)
                hist_tensor = torch.cat([hist_tensor, padding], dim=0)
            mask[:len(hist_vecs)] = 1.0
        else:
            hist_tensor = torch.zeros(self.max_history, feat_dim)

        # 候选新闻采样
        pos = [(nid, label) for nid, label in beh["impressions"] if label == 1]
        neg = [(nid, label) for nid, label in beh["impressions"] if label == 0]

        if not pos:
            pos = [beh["impressions"][0]]
        if not neg:
            neg = [beh["impressions"][-1]]

        # 随机选一个正样本 + npratio 个负样本
        p = random.choice(pos)
        ns = random.choices(neg, k=self.npratio) if len(neg) >= self.npratio else neg * self.npratio
        ns = ns[:self.npratio]

        candidates = [p] + ns
        cand_vecs = torch.stack([self._get_news_vector(nid) for nid, _ in candidates])
        labels = torch.tensor([label for _, label in candidates], dtype=torch.float32)

        return {
            "history_features": hist_tensor,      # (max_history, feat_dim)
            "history_mask": mask,                  # (max_history,)
            "candidate_features": cand_vecs,       # (1+npratio, feat_dim)
            "labels": labels,                      # (1+npratio,)
        }


class MINDEvalDataset(Dataset):
    """评估用 Dataset，保留完整 impression 列表。"""

    def __init__(self, behaviors: list, news_features: dict,
                 news_value: dict, news_info: dict,
                 cat2id: dict, subcat2id: dict,
                 max_history: int = 50):
        self.behaviors = behaviors
        self.news_features = news_features
        self.news_value = news_value
        self.news_info = news_info
        self.cat2id = cat2id
        self.subcat2id = subcat2id
        self.max_history = max_history

    def _get_news_vector(self, nid: str) -> torch.Tensor:
        if nid in self.news_features:
            text_emb = self.news_features[nid]["text"]
            img_emb = self.news_features[nid]["image"]
        else:
            text_emb = torch.zeros(768)
            img_emb = torch.zeros(768)
        nv = torch.tensor(self.news_value.get(nid, [3, 3, 3, 3, 3]),
                          dtype=torch.float32)
        info = self.news_info.get(nid, {"category": "", "subcategory": ""})
        cat_id = self.cat2id.get(info["category"], 0)
        subcat_id = self.subcat2id.get(info["subcategory"], 0)
        cat_vec = torch.tensor([cat_id, subcat_id], dtype=torch.float32)
        return torch.cat([text_emb, img_emb, nv, cat_vec])

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx):
        beh = self.behaviors[idx]
        history = beh["history"]

        feat_dim = 768 * 2 + 5 + 2
        hist_vecs = []
        for nid in history[-self.max_history:]:
            hist_vecs.append(self._get_news_vector(nid))

        mask = torch.zeros(self.max_history)
        if hist_vecs:
            hist_tensor = torch.stack(hist_vecs)
            pad_len = self.max_history - len(hist_vecs)
            if pad_len > 0:
                hist_tensor = torch.cat(
                    [hist_tensor, torch.zeros(pad_len, feat_dim)], dim=0)
            mask[:len(hist_vecs)] = 1.0
        else:
            hist_tensor = torch.zeros(self.max_history, feat_dim)

        # 评估时保留全部 impression
        cand_vecs = torch.stack(
            [self._get_news_vector(nid) for nid, _ in beh["impressions"]])
        labels = torch.tensor(
            [label for _, label in beh["impressions"]], dtype=torch.float32)

        return {
            "history_features": hist_tensor,
            "history_mask": mask,
            "candidate_features": cand_vecs,
            "labels": labels,
        }
```

**Step 4: 运行测试确认通过**

```bash
python -m pytest tests/test_dataset.py -v
```

**Step 5: Commit**

```bash
git add src/data/dataset.py tests/test_dataset.py
git commit -m "feat: MIND dataset with SigLIP features and news value scores"
```

---

## Task 6: 新闻编码器

**Files:**
- Create: `src/models/news_encoder.py`
- Create: `tests/test_models.py`

**Step 1: 写形状测试**

```python
# tests/test_models.py
import torch
import pytest
from src.models.news_encoder import NewsEncoder


def test_news_encoder_output_shape():
    encoder = NewsEncoder(
        siglip_dim=768,
        news_value_dim=5,
        num_categories=17,
        num_subcategories=264,
        category_emb_dim=64,
        output_dim=256,
        dropout=0.2,
    )
    # 模拟输入: (batch=4, feat_dim=1543)
    x = torch.randn(4, 768 * 2 + 5 + 2)
    out = encoder(x)
    assert out.shape == (4, 256)
```

**Step 2: 运行测试确认失败**

**Step 3: 实现新闻编码器**

```python
# src/models/news_encoder.py
"""新闻编码器：SigLIP特征 + 新闻价值 + 类别 → 新闻表示。"""
import torch
import torch.nn as nn


class NewsEncoder(nn.Module):
    def __init__(self, siglip_dim: int = 768, news_value_dim: int = 5,
                 num_categories: int = 17, num_subcategories: int = 264,
                 category_emb_dim: int = 64, output_dim: int = 256,
                 dropout: float = 0.2):
        super().__init__()
        self.siglip_dim = siglip_dim
        self.news_value_dim = news_value_dim

        # 类别嵌入
        self.cat_embedding = nn.Embedding(num_categories + 1, category_emb_dim,
                                          padding_idx=0)
        self.subcat_embedding = nn.Embedding(num_subcategories + 1, category_emb_dim,
                                              padding_idx=0)

        # 投影层: siglip_text(768) + siglip_image(768) + nv(5) + cat(64) + subcat(64)
        input_dim = siglip_dim * 2 + news_value_dim + category_emb_dim * 2
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, ..., 768*2 + 5 + 2) 拼接的原始特征
               最后2维是 [cat_id, subcat_id]
        Returns:
            (batch, ..., output_dim)
        """
        original_shape = x.shape[:-1]
        x_flat = x.reshape(-1, x.shape[-1])

        siglip_feat = x_flat[:, :self.siglip_dim * 2]          # (N, 1536)
        nv_feat = x_flat[:, self.siglip_dim * 2:
                           self.siglip_dim * 2 + self.news_value_dim]  # (N, 5)
        cat_ids = x_flat[:, -2].long()                          # (N,)
        subcat_ids = x_flat[:, -1].long()                       # (N,)

        cat_emb = self.cat_embedding(cat_ids)                   # (N, 64)
        subcat_emb = self.subcat_embedding(subcat_ids)          # (N, 64)

        combined = torch.cat([siglip_feat, nv_feat, cat_emb, subcat_emb], dim=-1)
        projected = self.projection(combined)                    # (N, 256)

        return projected.reshape(*original_shape, -1)
```

**Step 4: 运行测试确认通过**

```bash
python -m pytest tests/test_models.py::test_news_encoder_output_shape -v
```

**Step 5: Commit**

```bash
git add src/models/news_encoder.py tests/test_models.py
git commit -m "feat: news encoder with SigLIP + news value + category embeddings"
```

---

## Task 7: 用户编码器

**Files:**
- Modify: `src/models/` — 创建 `user_encoder.py`
- Modify: `tests/test_models.py` — 添加测试

**Step 1: 写测试**

```python
# 追加到 tests/test_models.py
from src.models.user_encoder import UserEncoder


def test_user_encoder_output_shape():
    encoder = UserEncoder(input_dim=256, num_heads=16)
    history = torch.randn(4, 50, 256)  # batch=4, history=50
    mask = torch.ones(4, 50)
    mask[:, 30:] = 0  # 后20个为padding
    out = encoder(history, mask)
    assert out.shape == (4, 256)
```

**Step 2: 实现用户编码器**

```python
# src/models/user_encoder.py
"""NRMS 用户编码器：Multi-Head Self-Attention + Additive Attention。"""
import torch
import torch.nn as nn


class AdditiveAttention(nn.Module):
    """加性注意力，将序列聚合为单向量。"""
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.query = nn.Parameter(torch.randn(dim))

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, dim)
            mask: (batch, seq_len) — 1 for valid, 0 for padding
        Returns:
            (batch, dim)
        """
        scores = torch.tanh(self.proj(x)) @ self.query  # (batch, seq_len)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # (batch, seq_len, 1)
        return (x * weights).sum(dim=1)  # (batch, dim)


class UserEncoder(nn.Module):
    def __init__(self, input_dim: int = 256, num_heads: int = 16):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=input_dim, num_heads=num_heads, batch_first=True)
        self.additive_attn = AdditiveAttention(input_dim)

    def forward(self, history: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            history: (batch, max_history, dim)
            mask: (batch, max_history)
        Returns:
            user_repr: (batch, dim)
        """
        # Self-attention
        key_padding_mask = (mask == 0) if mask is not None else None
        attn_out, _ = self.self_attn(history, history, history,
                                      key_padding_mask=key_padding_mask)
        # Additive attention to aggregate
        user_repr = self.additive_attn(attn_out, mask)
        return user_repr
```

**Step 3: 运行测试**

```bash
python -m pytest tests/test_models.py -v
```

**Step 4: Commit**

```bash
git add src/models/user_encoder.py tests/test_models.py
git commit -m "feat: NRMS user encoder with self-attention and additive attention"
```

---

## Task 8: NRMS 模型组装

**Files:**
- Create: `src/models/nrms.py`
- Modify: `tests/test_models.py`

**Step 1: 写测试**

```python
# 追加到 tests/test_models.py
from src.models.nrms import NRMS


def test_nrms_forward_shape():
    model = NRMS(
        siglip_dim=768,
        news_value_dim=5,
        num_categories=17,
        num_subcategories=264,
    )
    feat_dim = 768 * 2 + 5 + 2  # 1543
    history = torch.randn(2, 50, feat_dim)
    mask = torch.ones(2, 50)
    candidates = torch.randn(2, 5, feat_dim)  # 1 pos + 4 neg

    scores = model(history, mask, candidates)
    assert scores.shape == (2, 5)
```

**Step 2: 实现 NRMS 模型**

```python
# src/models/nrms.py
"""NRMS 多模态新闻推荐模型。"""
import torch
import torch.nn as nn
from src.models.news_encoder import NewsEncoder
from src.models.user_encoder import UserEncoder


class NRMS(nn.Module):
    def __init__(self, siglip_dim: int = 768, news_value_dim: int = 5,
                 num_categories: int = 17, num_subcategories: int = 264,
                 category_emb_dim: int = 64, news_repr_dim: int = 256,
                 num_heads: int = 16, dropout: float = 0.2):
        super().__init__()
        self.news_encoder = NewsEncoder(
            siglip_dim=siglip_dim,
            news_value_dim=news_value_dim,
            num_categories=num_categories,
            num_subcategories=num_subcategories,
            category_emb_dim=category_emb_dim,
            output_dim=news_repr_dim,
            dropout=dropout,
        )
        self.user_encoder = UserEncoder(
            input_dim=news_repr_dim,
            num_heads=num_heads,
        )

    def forward(self, history_features: torch.Tensor,
                history_mask: torch.Tensor,
                candidate_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            history_features: (batch, max_history, raw_feat_dim)
            history_mask: (batch, max_history)
            candidate_features: (batch, num_candidates, raw_feat_dim)
        Returns:
            scores: (batch, num_candidates)
        """
        # 编码历史新闻
        history_repr = self.news_encoder(history_features)  # (B, H, 256)
        # 编码候选新闻
        candidate_repr = self.news_encoder(candidate_features)  # (B, C, 256)
        # 用户表示
        user_repr = self.user_encoder(history_repr, history_mask)  # (B, 256)
        # 点击得分
        scores = torch.bmm(candidate_repr,
                           user_repr.unsqueeze(-1)).squeeze(-1)  # (B, C)
        return scores
```

**Step 3: 运行测试**

```bash
python -m pytest tests/test_models.py -v
```

**Step 4: Commit**

```bash
git add src/models/nrms.py tests/test_models.py
git commit -m "feat: NRMS model assembling news and user encoders"
```

---

## Task 9: 评估指标

**Files:**
- Create: `src/utils/metrics.py`
- Modify: `tests/test_models.py`

**Step 1: 写测试**

```python
# 追加到 tests/test_models.py（或创建 tests/test_metrics.py）
from src.utils.metrics import calc_auc, calc_mrr, calc_ndcg


def test_metrics_perfect_ranking():
    # 完美排序: 正样本排第一
    y_true = [1, 0, 0, 0, 0]
    y_score = [0.9, 0.2, 0.1, 0.05, 0.01]
    assert calc_auc(y_true, y_score) == 1.0
    assert calc_mrr(y_true, y_score) == 1.0
    assert calc_ndcg(y_true, y_score, k=5) == 1.0
```

**Step 2: 实现指标**

```python
# src/utils/metrics.py
"""MIND 标准评估指标: AUC, MRR, nDCG@k。"""
import numpy as np
from sklearn.metrics import roc_auc_score


def calc_auc(y_true: list, y_score: list) -> float:
    try:
        return roc_auc_score(y_true, y_score)
    except ValueError:
        return 0.5


def calc_mrr(y_true: list, y_score: list) -> float:
    order = np.argsort(-np.array(y_score))
    y_sorted = np.array(y_true)[order]
    for i, val in enumerate(y_sorted):
        if val == 1:
            return 1.0 / (i + 1)
    return 0.0


def calc_ndcg(y_true: list, y_score: list, k: int = 5) -> float:
    order = np.argsort(-np.array(y_score))
    y_sorted = np.array(y_true)[order][:k]
    dcg = sum(y_sorted[i] / np.log2(i + 2) for i in range(len(y_sorted)))
    ideal = sorted(y_true, reverse=True)[:k]
    idcg = sum(ideal[i] / np.log2(i + 2) for i in range(len(ideal)))
    return dcg / idcg if idcg > 0 else 0.0
```

**Step 3: 运行测试**

```bash
python -m pytest tests/ -v
```

**Step 4: Commit**

```bash
git add src/utils/metrics.py tests/
git commit -m "feat: evaluation metrics (AUC, MRR, nDCG)"
```

---

## Task 10: 训练脚本

**Files:**
- Create: `scripts/train.py`

**Step 1: 实现训练循环**

```python
# scripts/train.py
"""模型训练脚本。"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.config import Config
from src.data.preprocess import parse_news_file, parse_behaviors_file, build_category_maps
from src.data.dataset import MINDDataset
from src.models.nrms import NRMS


def train():
    cfg = Config()
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 加载数据
    print("Loading data...")
    train_news = parse_news_file(cfg.train_dir / "news.tsv")
    dev_news = parse_news_file(cfg.dev_dir / "news.tsv")
    all_news = {**train_news, **dev_news}
    cat2id, subcat2id = build_category_maps(all_news)

    train_behaviors = parse_behaviors_file(cfg.train_dir / "behaviors.tsv")
    print(f"Train behaviors: {len(train_behaviors)}")

    # 加载预计算特征
    print("Loading precomputed features...")
    news_features = torch.load(cfg.feature_dir / "news_siglip_features.pt",
                               weights_only=False)
    with open(cfg.feature_dir / "news_value_scores.json", "r") as f:
        news_value = json.load(f)
    print(f"SigLIP features: {len(news_features)}, NV scores: {len(news_value)}")

    # Dataset
    train_ds = MINDDataset(
        behaviors=train_behaviors,
        news_features=news_features,
        news_value=news_value,
        news_info=all_news,
        cat2id=cat2id,
        subcat2id=subcat2id,
        max_history=cfg.max_history_len,
        npratio=cfg.npratio,
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)

    # 模型
    model = NRMS(
        siglip_dim=cfg.siglip_dim,
        news_value_dim=cfg.news_value_dim,
        num_categories=len(cat2id) + 1,
        num_subcategories=len(subcat2id) + 1,
        category_emb_dim=cfg.category_emb_dim,
        news_repr_dim=cfg.news_repr_dim,
        num_heads=cfg.num_attention_heads,
        dropout=cfg.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    # 训练
    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}"):
            history = batch["history_features"].to(device)
            mask = batch["history_mask"].to(device)
            candidates = batch["candidate_features"].to(device)
            labels = batch["labels"].to(device)

            scores = model(history, mask, candidates)
            # labels 中 index 0 是正样本
            target = torch.zeros(scores.shape[0], dtype=torch.long, device=device)
            loss = criterion(scores, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}")

        # 保存 checkpoint
        ckpt_path = cfg.feature_dir / "checkpoints"
        ckpt_path.mkdir(exist_ok=True)
        torch.save(model.state_dict(), ckpt_path / f"model_epoch{epoch+1}.pt")

    print("Training complete!")


if __name__ == "__main__":
    train()
```

**Step 2: 用 mock 数据冒烟测试**

```bash
python -c "
import torch
from src.models.nrms import NRMS
model = NRMS()
feat_dim = 768 * 2 + 5 + 2
h = torch.randn(2, 50, feat_dim)
m = torch.ones(2, 50)
c = torch.randn(2, 5, feat_dim)
scores = model(h, m, c)
loss = torch.nn.CrossEntropyLoss()(scores, torch.zeros(2, dtype=torch.long))
loss.backward()
print(f'Loss: {loss.item():.4f} — backward OK')
"
```

**Step 3: Commit**

```bash
git add scripts/train.py
git commit -m "feat: training script with NLL loss"
```

---

## Task 11: 评估脚本

**Files:**
- Create: `scripts/evaluate.py`

**Step 1: 实现评估**

```python
# scripts/evaluate.py
"""模型评估脚本。"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.config import Config
from src.data.preprocess import parse_news_file, parse_behaviors_file, build_category_maps
from src.data.dataset import MINDEvalDataset
from src.models.nrms import NRMS
from src.utils.metrics import calc_auc, calc_mrr, calc_ndcg


def evaluate(checkpoint_path: str = None):
    cfg = Config()
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # 加载数据
    train_news = parse_news_file(cfg.train_dir / "news.tsv")
    dev_news = parse_news_file(cfg.dev_dir / "news.tsv")
    all_news = {**train_news, **dev_news}
    cat2id, subcat2id = build_category_maps(all_news)

    dev_behaviors = parse_behaviors_file(cfg.dev_dir / "behaviors.tsv")

    news_features = torch.load(cfg.feature_dir / "news_siglip_features.pt",
                               weights_only=False)
    with open(cfg.feature_dir / "news_value_scores.json", "r") as f:
        news_value = json.load(f)

    eval_ds = MINDEvalDataset(
        behaviors=dev_behaviors,
        news_features=news_features,
        news_value=news_value,
        news_info=all_news,
        cat2id=cat2id,
        subcat2id=subcat2id,
        max_history=cfg.max_history_len,
    )
    # 评估时 batch_size=1 因为 impression 长度不一
    eval_loader = DataLoader(eval_ds, batch_size=1, shuffle=False)

    # 模型
    model = NRMS(
        siglip_dim=cfg.siglip_dim,
        news_value_dim=cfg.news_value_dim,
        num_categories=len(cat2id) + 1,
        num_subcategories=len(subcat2id) + 1,
    ).to(device)

    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    aucs, mrrs, ndcg5s, ndcg10s = [], [], [], []

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            history = batch["history_features"].to(device)
            mask = batch["history_mask"].to(device)
            candidates = batch["candidate_features"].to(device)
            labels = batch["labels"]

            scores = model(history, mask, candidates)
            scores = scores.squeeze(0).cpu().numpy()
            labels = labels.squeeze(0).numpy()

            y_true = labels.tolist()
            y_score = scores.tolist()

            aucs.append(calc_auc(y_true, y_score))
            mrrs.append(calc_mrr(y_true, y_score))
            ndcg5s.append(calc_ndcg(y_true, y_score, k=5))
            ndcg10s.append(calc_ndcg(y_true, y_score, k=10))

    print(f"\nResults:")
    print(f"  AUC:     {np.mean(aucs):.4f}")
    print(f"  MRR:     {np.mean(mrrs):.4f}")
    print(f"  nDCG@5:  {np.mean(ndcg5s):.4f}")
    print(f"  nDCG@10: {np.mean(ndcg10s):.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()
    evaluate(args.checkpoint)
```

**Step 2: Commit**

```bash
git add scripts/evaluate.py
git commit -m "feat: evaluation script with AUC/MRR/nDCG metrics"
```

---

## Task 12: 门控融合变体（消融实验）

**Files:**
- Create: `src/models/gated_fusion.py`
- Create: `src/models/nrms_gated.py`

**Step 1: 实现门控融合新闻编码器**

```python
# src/models/gated_fusion.py
"""门控融合新闻编码器（消融实验用）。"""
import torch
import torch.nn as nn


class GatedNewsEncoder(nn.Module):
    def __init__(self, siglip_dim: int = 768, news_value_dim: int = 5,
                 num_categories: int = 17, num_subcategories: int = 264,
                 category_emb_dim: int = 64, output_dim: int = 256,
                 dropout: float = 0.2):
        super().__init__()
        self.siglip_dim = siglip_dim
        self.news_value_dim = news_value_dim

        self.cat_embedding = nn.Embedding(num_categories + 1, category_emb_dim,
                                          padding_idx=0)
        self.subcat_embedding = nn.Embedding(num_subcategories + 1, category_emb_dim,
                                              padding_idx=0)

        # SigLIP 投影
        mm_input_dim = siglip_dim * 2 + category_emb_dim * 2
        self.mm_proj = nn.Linear(mm_input_dim, output_dim)

        # 新闻价值投影
        self.nv_proj = nn.Linear(news_value_dim, output_dim)

        # 门控
        self.gate_mm = nn.Linear(output_dim, output_dim)
        self.gate_nv = nn.Linear(output_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape[:-1]
        x_flat = x.reshape(-1, x.shape[-1])

        siglip_feat = x_flat[:, :self.siglip_dim * 2]
        nv_feat = x_flat[:, self.siglip_dim * 2:
                           self.siglip_dim * 2 + self.news_value_dim]
        cat_ids = x_flat[:, -2].long()
        subcat_ids = x_flat[:, -1].long()

        cat_emb = self.cat_embedding(cat_ids)
        subcat_emb = self.subcat_embedding(subcat_ids)

        # 多模态特征
        mm_combined = torch.cat([siglip_feat, cat_emb, subcat_emb], dim=-1)
        mm_repr = torch.relu(self.mm_proj(mm_combined))  # (N, 256)

        # 新闻价值特征
        nv_repr = torch.relu(self.nv_proj(nv_feat))  # (N, 256)

        # 门控融合
        gate = torch.sigmoid(self.gate_mm(mm_repr) + self.gate_nv(nv_repr))
        fused = gate * mm_repr + (1 - gate) * nv_repr  # (N, 256)
        fused = self.dropout(fused)

        return fused.reshape(*original_shape, -1)
```

**Step 2: 创建门控 NRMS 变体**

```python
# src/models/nrms_gated.py
"""NRMS 门控融合变体。"""
from src.models.nrms import NRMS
from src.models.gated_fusion import GatedNewsEncoder


class NRMSGated(NRMS):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 替换新闻编码器
        self.news_encoder = GatedNewsEncoder(
            siglip_dim=kwargs.get("siglip_dim", 768),
            news_value_dim=kwargs.get("news_value_dim", 5),
            num_categories=kwargs.get("num_categories", 17),
            num_subcategories=kwargs.get("num_subcategories", 264),
            category_emb_dim=kwargs.get("category_emb_dim", 64),
            output_dim=kwargs.get("news_repr_dim", 256),
            dropout=kwargs.get("dropout", 0.2),
        )
```

**Step 3: 添加测试**

```python
# 追加到 tests/test_models.py
from src.models.nrms_gated import NRMSGated


def test_nrms_gated_forward_shape():
    model = NRMSGated()
    feat_dim = 768 * 2 + 5 + 2
    h = torch.randn(2, 50, feat_dim)
    m = torch.ones(2, 50)
    c = torch.randn(2, 5, feat_dim)
    scores = model(h, m, c)
    assert scores.shape == (2, 5)
```

**Step 4: 运行全部测试**

```bash
python -m pytest tests/ -v
```

**Step 5: Commit**

```bash
git add src/models/gated_fusion.py src/models/nrms_gated.py tests/test_models.py
git commit -m "feat: gated fusion variant for ablation study"
```

---

## Task 13: 消融实验配置脚本

**Files:**
- Create: `scripts/run_experiments.py`

**Step 1: 创建实验运行器**

```python
# scripts/run_experiments.py
"""运行所有消融实验变体。

Usage:
    python scripts/run_experiments.py --variant text_only
    python scripts/run_experiments.py --variant multimodal
    python scripts/run_experiments.py --variant mm_nv (主方案)
    python scripts/run_experiments.py --variant mm_nv_gate
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from src.utils.config import Config
from src.data.preprocess import parse_news_file, parse_behaviors_file, build_category_maps
from src.data.dataset import MINDDataset, MINDEvalDataset
from src.models.nrms import NRMS
from src.models.nrms_gated import NRMSGated
from src.utils.metrics import calc_auc, calc_mrr, calc_ndcg


VARIANTS = {
    "text_only": "NRMS-Text: SigLIP text only, no image, no NV",
    "multimodal": "NRMS-MM: SigLIP text + image, no NV",
    "mm_nv": "NRMS-MM-NV: SigLIP + news value (concat)",
    "mm_nv_gate": "NRMS-MM-NV-Gate: SigLIP + news value (gated)",
}


def mask_features(batch, variant: str):
    """根据变体屏蔽对应特征。"""
    if variant == "text_only":
        # 将图像特征清零 (位置 768:1536)
        batch["history_features"][:, :, 768:1536] = 0
        batch["candidate_features"][:, :, 768:1536] = 0
        # 将新闻价值清零 (位置 1536:1541)
        batch["history_features"][:, :, 1536:1541] = 0
        batch["candidate_features"][:, :, 1536:1541] = 0
    elif variant == "multimodal":
        # 将新闻价值清零
        batch["history_features"][:, :, 1536:1541] = 0
        batch["candidate_features"][:, :, 1536:1541] = 0
    return batch


def run_experiment(variant: str):
    print(f"\n{'='*60}")
    print(f"Running: {VARIANTS[variant]}")
    print(f"{'='*60}")

    cfg = Config()
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # 数据加载
    train_news = parse_news_file(cfg.train_dir / "news.tsv")
    dev_news = parse_news_file(cfg.dev_dir / "news.tsv")
    all_news = {**train_news, **dev_news}
    cat2id, subcat2id = build_category_maps(all_news)

    train_behaviors = parse_behaviors_file(cfg.train_dir / "behaviors.tsv")
    dev_behaviors = parse_behaviors_file(cfg.dev_dir / "behaviors.tsv")

    news_features = torch.load(cfg.feature_dir / "news_siglip_features.pt",
                               weights_only=False)
    with open(cfg.feature_dir / "news_value_scores.json", "r") as f:
        news_value = json.load(f)

    train_ds = MINDDataset(train_behaviors, news_features, news_value,
                           all_news, cat2id, subcat2id,
                           cfg.max_history_len, cfg.npratio)
    eval_ds = MINDEvalDataset(dev_behaviors, news_features, news_value,
                              all_news, cat2id, subcat2id, cfg.max_history_len)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    eval_loader = DataLoader(eval_ds, batch_size=1, shuffle=False)

    # 模型选择
    model_kwargs = dict(
        siglip_dim=cfg.siglip_dim, news_value_dim=cfg.news_value_dim,
        num_categories=len(cat2id) + 1, num_subcategories=len(subcat2id) + 1,
    )
    if variant == "mm_nv_gate":
        model = NRMSGated(**model_kwargs).to(device)
    else:
        model = NRMS(**model_kwargs).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    # 训练
    best_auc = 0
    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            batch = mask_features(batch, variant)
            history = batch["history_features"].to(device)
            mask = batch["history_mask"].to(device)
            candidates = batch["candidate_features"].to(device)

            scores = model(history, mask, candidates)
            target = torch.zeros(scores.shape[0], dtype=torch.long, device=device)
            loss = criterion(scores, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # 评估
        model.eval()
        aucs, mrrs, ndcg5s, ndcg10s = [], [], [], []
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Eval"):
                batch = mask_features(batch, variant)
                history = batch["history_features"].to(device)
                mask_h = batch["history_mask"].to(device)
                candidates = batch["candidate_features"].to(device)
                labels = batch["labels"].squeeze(0).numpy()

                scores = model(history, mask_h, candidates)
                scores = scores.squeeze(0).cpu().numpy()
                y_true, y_score = labels.tolist(), scores.tolist()

                aucs.append(calc_auc(y_true, y_score))
                mrrs.append(calc_mrr(y_true, y_score))
                ndcg5s.append(calc_ndcg(y_true, y_score, k=5))
                ndcg10s.append(calc_ndcg(y_true, y_score, k=10))

        auc_val = np.mean(aucs)
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f} "
              f"AUC={auc_val:.4f} MRR={np.mean(mrrs):.4f} "
              f"nDCG@5={np.mean(ndcg5s):.4f} nDCG@10={np.mean(ndcg10s):.4f}")

        if auc_val > best_auc:
            best_auc = auc_val
            ckpt_dir = cfg.feature_dir / "checkpoints"
            ckpt_dir.mkdir(exist_ok=True)
            torch.save(model.state_dict(), ckpt_dir / f"best_{variant}.pt")

    print(f"\nBest AUC for {variant}: {best_auc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=list(VARIANTS.keys()),
                        required=True)
    args = parser.parse_args()
    run_experiment(args.variant)
```

**Step 2: Commit**

```bash
git add scripts/run_experiments.py
git commit -m "feat: ablation experiment runner with 4 model variants"
```

---

## 执行顺序总结

```
Task 1:  项目脚手架        → 立即可做
Task 2:  数据解析          → 立即可做
Task 3:  SigLIP 特征提取   → 需要 GPU，运行时间较长
Task 4:  LLM 标注          → 需要 API key，可与 Task 3 并行
Task 5:  Dataset           → 依赖 Task 2
Task 6:  新闻编码器        → 立即可做
Task 7:  用户编码器        → 立即可做
Task 8:  NRMS 组装         → 依赖 Task 6, 7
Task 9:  评估指标          → 立即可做
Task 10: 训练脚本          → 依赖 Task 5, 8, 9
Task 11: 评估脚本          → 依赖 Task 10
Task 12: 门控融合          → 依赖 Task 8
Task 13: 实验运行器        → 依赖 Task 10, 12
```

**并行机会**: Task 3 与 Task 4 可同时进行；Task 6, 7, 9 互不依赖可并行开发。
