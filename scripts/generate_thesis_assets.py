from __future__ import annotations

import csv
import html
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "artifacts" / "data"
FEATURE_DIR = ROOT / "artifacts" / "features" / "siglip"
LABEL_PATH = ROOT / "artifacts" / "labels" / "news_value_labels.jsonl"
BATCH_DIR = ROOT / "artifacts" / "labels" / "batches" / "news-value-20260425-181001"
OUTPUT_DIR = ROOT / "artifacts" / "thesis"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
DOC_PATH = ROOT / "docs" / "thesis_chapter_supplements.md"

VALUE_DIMENSIONS = [
    ("importance", "重要性"),
    ("prominence", "显著性"),
    ("conflict", "冲突性"),
    ("novelty", "新奇性"),
    ("human_interest", "人情味"),
]

EXPERIMENTS = [
    ("Text", "text"),
    ("Text+Image", "multimodal"),
    ("Text+Image+Value", "value"),
]

METRICS = [
    ("auc", "AUC"),
    ("mrr", "MRR"),
    ("ndcg5", "nDCG@5"),
    ("ndcg10", "nDCG@10"),
]

PALETTE = {
    "ink": "#1f2933",
    "muted": "#5b6472",
    "line": "#8f9aa8",
    "grid": "#e6e9ee",
    "paper": "#ffffff",
    "soft": "#f7f8fa",
    "blue": "#2f5f9f",
    "green": "#2f7f6f",
    "amber": "#b7791f",
    "red": "#b44a4a",
    "violet": "#6f5aa7",
    "cyan": "#2c7a9a",
    "gray": "#6b7280",
    "light_blue": "#eaf1fb",
    "light_green": "#e8f3ee",
    "light_amber": "#f8efdf",
    "light_red": "#f6e8e8",
    "light_violet": "#eeeafb",
    "light_cyan": "#e6f2f5",
}


def ensure_dirs() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                yield json.loads(line)


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as file:
        return sum(1 for line in file if line.strip())


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


FONT_PATHS = [
    Path("C:/Windows/Fonts/msyh.ttc"),
    Path("C:/Windows/Fonts/msyhbd.ttc"),
    Path("C:/Windows/Fonts/simhei.ttf"),
    Path("C:/Windows/Fonts/simsun.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/System/Library/Fonts/PingFang.ttc"),
]


def choose_font_family() -> str:
    for path in FONT_PATHS:
        if path.exists():
            try:
                font_manager.fontManager.addfont(str(path))
                return font_manager.FontProperties(fname=str(path)).get_name()
            except RuntimeError:
                continue
    for family in ["Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC", "PingFang SC"]:
        if any(font.name == family for font in font_manager.fontManager.ttflist):
            return family
    return "DejaVu Sans"


FONT_FAMILY = choose_font_family()
GRAPHVIZ_FONT = FONT_FAMILY


def configure_matplotlib_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [FONT_FAMILY, "Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "axes.edgecolor": PALETTE["line"],
            "axes.labelcolor": PALETTE["ink"],
            "xtick.color": PALETTE["muted"],
            "ytick.color": PALETTE["muted"],
            "text.color": PALETTE["ink"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


configure_matplotlib_style()


def save_figure(fig, stem: str) -> Path:
    png_path = FIGURE_DIR / f"{stem}.png"
    pdf_path = FIGURE_DIR / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.08, facecolor=PALETTE["paper"])
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.08, facecolor=PALETTE["paper"])
    plt.close(fig)
    return png_path


def require_graphviz() -> None:
    try:
        import graphviz  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("缺少 Python graphviz 包。请先运行 `uv sync` 或 `uv run python scripts/generate_thesis_assets.py`。") from exc
    if shutil.which("dot") is None:
        raise RuntimeError(
            "未检测到 Graphviz 的 `dot` 命令。请先安装 Graphviz CLI 并确保 `dot` 已加入 PATH；"
            "Windows 可使用 `winget install Graphviz.Graphviz`，安装后重新打开终端再运行脚本。"
        )


def make_graph(name: str, rankdir: str = "LR", nodesep: str = "0.55", ranksep: str = "0.80"):
    require_graphviz()
    from graphviz import Digraph

    graph = Digraph(name=name, engine="dot")
    graph.attr(
        bgcolor=PALETTE["paper"],
        color=PALETTE["line"],
        fontname=GRAPHVIZ_FONT,
        margin="0.04",
        nodesep=nodesep,
        outputorder="edgesfirst",
        pad="0.18",
        rankdir=rankdir,
        ranksep=ranksep,
        splines="ortho",
    )
    graph.attr(
        "node",
        color=PALETTE["line"],
        fillcolor=PALETTE["soft"],
        fontname=GRAPHVIZ_FONT,
        fontsize="18",
        margin="0.18,0.12",
        penwidth="1.6",
        shape="box",
        style="rounded,filled",
    )
    graph.attr("edge", arrowsize="0.75", color=PALETTE["gray"], fontname=GRAPHVIZ_FONT, penwidth="1.5")
    return graph


def graph_label(title: str, body: str = "", prefix: str = "") -> str:
    heading = f"{html.escape(prefix)} {html.escape(title)}".strip()
    rows = [
        '<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="4">',
        f'<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="22"><B>{heading}</B></FONT></TD></TR>',
    ]
    if body:
        lines = '<BR ALIGN="LEFT"/>'.join(html.escape(line) for line in str(body).split("\n"))
        rows.append(f'<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="16" COLOR="{PALETTE["muted"]}">{lines}</FONT></TD></TR>')
    rows.append("</TABLE>")
    return "<" + "".join(rows) + ">"


def add_graph_node(
    graph,
    node_id: str,
    title: str,
    body: str = "",
    fill: str = PALETTE["soft"],
    color: str = PALETTE["line"],
    prefix: str = "",
) -> None:
    graph.node(node_id, label=graph_label(title, body, prefix), fillcolor=fill, color=color)


def save_graph(graph, stem: str) -> Path:
    png_path = FIGURE_DIR / f"{stem}.png"
    pdf_path = FIGURE_DIR / f"{stem}.pdf"
    png_path.write_bytes(graph.pipe(format="png"))
    pdf_path.write_bytes(graph.pipe(format="pdf"))
    return png_path


def style_axis(ax, xgrid: bool = True, ygrid: bool = False) -> None:
    ax.set_facecolor(PALETTE["paper"])
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(PALETTE["line"])
    ax.spines["bottom"].set_color(PALETTE["line"])
    if xgrid:
        ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.8)
    if ygrid:
        ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def format_percent(value: float) -> str:
    return f"{value:.1f}%"


def collect_stats() -> dict:
    meta = load_json(FEATURE_DIR / "meta.json")
    manifest_path = BATCH_DIR / "manifest.json"
    manifest = load_json(manifest_path) if manifest_path.exists() else {}
    news_records = list(read_jsonl(DATA_DIR / "news.jsonl"))
    labels = list(read_jsonl(LABEL_PATH))
    category_counts = Counter(record.get("category", "") for record in news_records)
    abstract_missing = sum(1 for record in news_records if not record.get("abstract"))
    image_aligned = sum(1 for record in news_records if record.get("image_path"))

    value_dist = {name: Counter() for name, _ in VALUE_DIMENSIONS}
    value_totals: list[int] = []
    for record in labels:
        scores = record.get("scores", {})
        total = 0
        for name, _ in VALUE_DIMENSIONS:
            value = int(scores.get(name, 0))
            value_dist[name][value] += 1
            total += value
        value_totals.append(total)

    return {
        "news_records": news_records,
        "labels": labels,
        "category_counts": category_counts,
        "abstract_missing": abstract_missing,
        "image_aligned": image_aligned,
        "newdata_files": len(list((ROOT / "newData").glob("*.jpg"))),
        "train_samples": count_lines(DATA_DIR / "train_samples.jsonl"),
        "dev_impressions": count_lines(DATA_DIR / "dev_impressions.jsonl"),
        "feature_meta": meta,
        "manifest": manifest,
        "value_dist": value_dist,
        "value_totals": value_totals,
    }


def write_tables(stats: dict) -> dict[str, Path]:
    news_count = len(stats["news_records"])
    label_count = len(stats["labels"])
    meta = stats["feature_meta"]
    manifest = stats["manifest"]
    text_dim = int(meta["text_dim"])
    image_dim = int(meta["image_dim"])
    value_dim = len(VALUE_DIMENSIONS) + 1
    coverage = label_count / news_count * 100

    category_rows = []
    for category, count in stats["category_counts"].most_common():
        category_rows.append(
            {
                "category": category,
                "count": count,
                "percentage": f"{count / news_count * 100:.2f}%",
            }
        )
    write_csv(
        TABLE_DIR / "category_distribution.csv",
        ["category", "count", "percentage"],
        category_rows,
    )

    data_rows = [
        {"item": "新闻总数", "value": news_count, "description": "MIND-small train/dev 合并去重后的新闻条目数"},
        {"item": "图片对齐新闻数", "value": stats["image_aligned"], "description": "news_id 与 newData/{news_id}.jpg 成功匹配"},
        {"item": "图像文件目录规模", "value": stats["newdata_files"], "description": "newData 目录中的 JPG 文件数"},
        {"item": "训练样本数", "value": stats["train_samples"], "description": "由点击正样本与负采样候选新闻构成"},
        {"item": "验证 impression 数", "value": stats["dev_impressions"], "description": "保留候选新闻列表用于排序评价"},
        {"item": "摘要缺失新闻数", "value": stats["abstract_missing"], "description": "摘要为空时使用类别、子类和标题构造文本输入"},
        {"item": "新闻价值标注缓存", "value": label_count, "description": "成功解析并写入 JSONL 的新闻价值标注数"},
    ]
    write_csv(TABLE_DIR / "data_statistics.csv", ["item", "value", "description"], data_rows)

    feature_rows = [
        {"mode": "Text", "text_dim": text_dim, "image_dim": 0, "value_dim": 0, "total_dim": text_dim},
        {
            "mode": "Text+Image",
            "text_dim": text_dim,
            "image_dim": image_dim,
            "value_dim": 0,
            "total_dim": text_dim + image_dim,
        },
        {
            "mode": "Text+Image+Value",
            "text_dim": text_dim,
            "image_dim": image_dim,
            "value_dim": value_dim,
            "total_dim": text_dim + image_dim + value_dim,
        },
    ]
    write_csv(
        TABLE_DIR / "feature_dimensions.csv",
        ["mode", "text_dim", "image_dim", "value_dim", "total_dim"],
        feature_rows,
    )

    request_counts = manifest.get("batch", {}).get("request_counts", {})
    value_stats_rows = [
        {"item": "目标标注数", "value": manifest.get("target", 3000), "description": "按新闻曝光/点击频次排序选取高频新闻"},
        {"item": "成功标注数", "value": label_count, "description": "通过 JSON 解析和分数范围校验后进入缓存"},
        {"item": "批处理完成数", "value": request_counts.get("completed", label_count), "description": "阿里云 Batch File 返回成功的请求数"},
        {"item": "批处理失败数", "value": request_counts.get("failed", 0), "description": "该批次存在 1 条内容安全检查失败"},
        {"item": "无效解析数", "value": manifest.get("merge_stats", {}).get("invalid", 0), "description": "JSON 或 0-3 分数校验失败数"},
        {"item": "全量新闻覆盖率", "value": f"{coverage:.2f}%", "description": "成功标注新闻数 / 全量新闻数"},
        {"item": "标注模型", "value": manifest.get("model", "qwen3.5-flash"), "description": "OpenAI-compatible JSON Mode 调用"},
    ]
    write_csv(
        TABLE_DIR / "news_value_labeling_statistics.csv",
        ["item", "value", "description"],
        value_stats_rows,
    )

    value_rows = []
    for name, cn in VALUE_DIMENSIONS:
        counter = stats["value_dist"][name]
        total = sum(counter.values())
        mean = sum(score * count for score, count in counter.items()) / max(total, 1)
        value_rows.append(
            {
                "dimension": name,
                "dimension_cn": cn,
                "score_0": counter.get(0, 0),
                "score_1": counter.get(1, 0),
                "score_2": counter.get(2, 0),
                "score_3": counter.get(3, 0),
                "mean": f"{mean:.3f}",
            }
        )
    write_csv(
        TABLE_DIR / "value_dimension_distribution.csv",
        ["dimension", "dimension_cn", "score_0", "score_1", "score_2", "score_3", "mean"],
        value_rows,
    )

    model_rows = [
        {"experiment": "Text", "input": "文本特征", "formula": "z=[e_t]", "total_dim": text_dim},
        {"experiment": "Text+Image", "input": "文本特征 + 图像特征", "formula": "z=[e_t;e_i]", "total_dim": text_dim + image_dim},
        {
            "experiment": "Text+Image+Value",
            "input": "文本特征 + 图像特征 + 新闻价值向量",
            "formula": "z=[e_t;e_i;v]",
            "total_dim": text_dim + image_dim + value_dim,
        },
    ]
    write_csv(
        TABLE_DIR / "model_input_comparison.csv",
        ["experiment", "input", "formula", "total_dim"],
        model_rows,
    )

    return {
        "category": TABLE_DIR / "category_distribution.csv",
        "data": TABLE_DIR / "data_statistics.csv",
        "feature": TABLE_DIR / "feature_dimensions.csv",
        "value_stats": TABLE_DIR / "news_value_labeling_statistics.csv",
        "value_dist": TABLE_DIR / "value_dimension_distribution.csv",
        "model_inputs": TABLE_DIR / "model_input_comparison.csv",
    }


def draw_technical_route(stats: dict) -> Path:
    meta = stats["feature_meta"]
    graph = make_graph("technical_route", rankdir="LR", nodesep="0.55", ranksep="0.70")
    add_graph_node(graph, "s1", "研究对象", "多模态新闻推荐任务\nMIND-small + V-MIND\n新闻文本与图像按 ID 对齐", PALETTE["light_blue"], PALETTE["blue"], "01")
    add_graph_node(
        graph,
        "s2",
        "数据准备",
        f"新闻去重与主表构建\n训练样本 {stats['train_samples']:,}\n验证 impression {stats['dev_impressions']:,}",
        PALETTE["light_green"],
        PALETTE["green"],
        "02",
    )
    add_graph_node(
        graph,
        "s3",
        "特征提取",
        f"SigLIP 文本编码 {meta['text_dim']} 维\nSigLIP 图像编码 {meta['image_dim']} 维\nL2 归一化后缓存",
        PALETTE["light_cyan"],
        PALETTE["cyan"],
        "03",
    )
    add_graph_node(
        graph,
        "s4",
        "价值量化",
        f"五维新闻价值评分\n成功标注 {len(stats['labels']):,}\n5 个分数 + missing mask",
        PALETTE["light_amber"],
        PALETTE["amber"],
        "04",
    )
    add_graph_node(graph, "s5", "推荐建模", "共享 MLP 新闻编码器\n历史点击平均池化\n候选新闻点积打分", PALETTE["light_violet"], PALETTE["violet"], "05")
    add_graph_node(graph, "s6", "实验分析", "Text / Text+Image / TIV\nAUC、MRR、nDCG\n对比与消融验证", PALETTE["light_red"], PALETTE["red"], "06")
    for start, end in [("s1", "s2"), ("s2", "s3"), ("s3", "s4"), ("s4", "s5"), ("s5", "s6")]:
        graph.edge(start, end)
    return save_graph(graph, "technical_route")


def draw_data_processing_flow(stats: dict) -> Path:
    graph = make_graph("data_processing_flow", rankdir="LR", nodesep="0.45", ranksep="0.70")
    add_graph_node(graph, "mind", "MIND-small 文本数据", "news.tsv：类别、子类、标题、摘要\nbehaviors.tsv：历史点击与候选曝光", PALETTE["light_blue"], PALETTE["blue"])
    add_graph_node(graph, "vmind", "V-MIND 图像数据", "newData/{news_id}.jpg\n按新闻 ID 对齐图像", PALETTE["light_green"], PALETTE["green"])
    add_graph_node(graph, "master", "新闻主表构建", f"合并去重 {len(stats['news_records']):,} 条新闻\n写入 news.jsonl 并保留关键字段", PALETTE["light_cyan"], PALETTE["cyan"])
    add_graph_node(graph, "freq", "频次统计", "统计历史点击与候选曝光\n支撑高频新闻价值标注抽样", PALETTE["light_amber"], PALETTE["amber"])
    add_graph_node(graph, "train", "训练样本生成", f"正样本 + 负采样\n{stats['train_samples']:,} 条训练样本", PALETTE["light_violet"], PALETTE["violet"])
    add_graph_node(graph, "dev", "验证样本生成", f"保留 impression 内候选排序\n{stats['dev_impressions']:,} 个验证组", PALETTE["light_red"], PALETTE["red"])
    add_graph_node(graph, "outputs", "中间数据产物", "news.jsonl\ntrain_samples.jsonl\ndev_impressions.jsonl\nnews_frequency.json", PALETTE["soft"], PALETTE["gray"])
    graph.edge("mind", "master")
    graph.edge("vmind", "master")
    graph.edge("master", "freq")
    graph.edge("master", "train")
    graph.edge("master", "dev")
    graph.edge("freq", "outputs")
    graph.edge("train", "outputs")
    graph.edge("dev", "outputs")
    return save_graph(graph, "data_processing_flow")


def draw_category_distribution(stats: dict) -> Path:
    counts = stats["category_counts"]
    news_count = len(stats["news_records"])
    top = counts.most_common(10)
    other = news_count - sum(count for _, count in top)
    rows = top + [("other", other)]
    labels = [category for category, _ in rows]
    values = [count for _, count in rows]
    y = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    colors = [PALETTE["blue"] if idx < 10 else PALETTE["gray"] for idx in range(len(rows))]
    bars = ax.barh(y, values, color=colors, edgecolor=PALETTE["ink"], linewidth=0.4)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("新闻数量")
    ax.set_title("新闻类别分布（Top 10 + other）", fontsize=15, weight="bold", pad=12)
    style_axis(ax, xgrid=True)
    ax.set_xlim(0, max(values) * 1.22)
    for bar, count in zip(bars, values, strict=True):
        pct = count / news_count * 100
        ax.text(bar.get_width() + max(values) * 0.015, bar.get_y() + bar.get_height() / 2, f"{count:,}  {format_percent(pct)}", va="center", fontsize=9)
    ax.text(0, 1.02, f"合计：{news_count:,} 条新闻", transform=ax.transAxes, fontsize=9, color=PALETTE["muted"])
    fig.tight_layout()
    return save_figure(fig, "category_distribution")


def draw_feature_extraction_flow(stats: dict) -> Path:
    meta = stats["feature_meta"]
    graph = make_graph("feature_extraction_flow", rankdir="LR", nodesep="0.45", ranksep="0.70")
    add_graph_node(graph, "text", "文本字段拼接", "category [SEP] subcategory\n[SEP] title [SEP] abstract", PALETTE["light_blue"], PALETTE["blue"])
    add_graph_node(graph, "image", "新闻图片输入", "newData/{news_id}.jpg\nRGB 图像", PALETTE["light_green"], PALETTE["green"])
    add_graph_node(graph, "text_encoder", "SigLIP 文本编码器", f"{meta['model_name']}\n输出 {meta['text_dim']} 维文本向量", PALETTE["light_blue"], PALETTE["blue"])
    add_graph_node(graph, "image_encoder", "SigLIP 图像编码器", f"{meta['model_name']}\n输出 {meta['image_dim']} 维图像向量", PALETTE["light_green"], PALETTE["green"])
    add_graph_node(graph, "text_norm", "L2 归一化", "得到 e_t\n统一向量尺度", PALETTE["light_cyan"], PALETTE["cyan"])
    add_graph_node(graph, "image_norm", "L2 归一化", "得到 e_i\n统一向量尺度", PALETTE["light_cyan"], PALETTE["cyan"])
    add_graph_node(graph, "concat", "特征拼接", "Text: 768\nT+I: 1536\nT+I+V: 1542", PALETTE["light_amber"], PALETTE["amber"])
    graph.edge("text", "text_encoder")
    graph.edge("image", "image_encoder")
    graph.edge("text_encoder", "text_norm")
    graph.edge("image_encoder", "image_norm")
    graph.edge("text_norm", "concat")
    graph.edge("image_norm", "concat")
    return save_graph(graph, "feature_extraction_flow")


def draw_news_value_labeling_flow(stats: dict) -> Path:
    manifest = stats["manifest"]
    label_count = len(stats["labels"])
    request_counts = manifest.get("batch", {}).get("request_counts", {})
    completed = request_counts.get("completed", label_count)
    failed = request_counts.get("failed", 0)

    graph = make_graph("news_value_labeling_flow", rankdir="LR", nodesep="0.45", ranksep="0.70")
    add_graph_node(graph, "freq", "新闻频次排序", "基于历史点击和候选曝光\n优先选择高频新闻", PALETTE["light_blue"], PALETTE["blue"])
    add_graph_node(graph, "prompt", "Prompt 构造", "只使用类别、标题、摘要\n不评价时效性和接近性", PALETTE["light_green"], PALETTE["green"])
    add_graph_node(graph, "llm", "LLM 批量评分", f"{manifest.get('model', 'qwen3.5-flash')}\n五维度均为 0-3 分", PALETTE["light_violet"], PALETTE["violet"])
    add_graph_node(graph, "validate", "JSON 校验", "字段完整性、整数分数\n范围必须为 0-3", PALETTE["light_amber"], PALETTE["amber"])
    add_graph_node(graph, "cache", "成功缓存", f"{label_count:,} 条写入\nnews_value_labels.jsonl", PALETTE["light_green"], PALETTE["green"])
    add_graph_node(graph, "missing", "缺失处理", "未标注新闻使用零向量\nmissing mask = 1", PALETTE["light_red"], PALETTE["red"])
    add_graph_node(graph, "vector", "价值向量", "5 个价值分数 + mask\n共 6 维", PALETTE["light_cyan"], PALETTE["cyan"])
    add_graph_node(graph, "summary", "批处理统计", f"目标 3,000 条\n完成 {completed:,} 条\n失败 {failed:,} 条", PALETTE["soft"], PALETTE["gray"])
    for start, end in [("freq", "prompt"), ("prompt", "llm"), ("llm", "validate"), ("validate", "cache"), ("cache", "vector"), ("missing", "vector")]:
        graph.edge(start, end)
    graph.edge("validate", "missing")
    graph.edge("llm", "summary", style="dashed")
    return save_graph(graph, "news_value_labeling_flow")


def draw_value_dimension_distribution(stats: dict) -> Path:
    label_count = len(stats["labels"])
    dimensions = [cn for _, cn in VALUE_DIMENSIONS]
    matrix = np.array([[stats["value_dist"][name].get(score, 0) for score in range(4)] for name, _ in VALUE_DIMENSIONS])
    x = np.arange(len(dimensions))
    width = 0.18
    colors = [PALETTE["blue"], PALETTE["green"], PALETTE["amber"], PALETTE["red"]]

    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    for score in range(4):
        offset = (score - 1.5) * width
        bars = ax.bar(x + offset, matrix[:, score], width=width, label=f"{score} 分", color=colors[score], edgecolor=PALETTE["ink"], linewidth=0.35)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + matrix.max() * 0.012, f"{int(bar.get_height())}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, dimensions)
    ax.set_ylabel("新闻数量")
    ax.set_title("五维新闻价值评分分布", fontsize=15, weight="bold", pad=12)
    ax.legend(frameon=False, ncols=4, loc="upper right")
    style_axis(ax, xgrid=False, ygrid=True)
    ax.set_ylim(0, matrix.max() * 1.16)
    ax.text(0, 1.02, f"样本量：{label_count:,} 条成功标注新闻", transform=ax.transAxes, fontsize=9, color=PALETTE["muted"])
    fig.tight_layout()
    return save_figure(fig, "value_dimension_distribution")


def draw_model_input_comparison(stats: dict) -> Path:
    meta = stats["feature_meta"]
    text_dim = int(meta["text_dim"])
    image_dim = int(meta["image_dim"])
    value_dim = len(VALUE_DIMENSIONS) + 1
    rows = [
        ("Text", [("文本", text_dim, PALETTE["blue"])]),
        ("Text+Image", [("文本", text_dim, PALETTE["blue"]), ("图像", image_dim, PALETTE["green"])]),
        ("Text+Image+Value", [("文本", text_dim, PALETTE["blue"]), ("图像", image_dim, PALETTE["green"]), ("价值+mask", value_dim, PALETTE["amber"])]),
    ]
    max_total = text_dim + image_dim + value_dim
    y = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    for idx, (_, parts) in enumerate(rows):
        left = 0
        for part_name, dim, color in parts:
            ax.barh(idx, dim, left=left, height=0.48, color=color, edgecolor=PALETTE["ink"], linewidth=0.35)
            if dim >= 120:
                ax.text(left + dim / 2, idx, f"{part_name}\n{dim}", ha="center", va="center", fontsize=9, color="white", weight="bold")
            else:
                ax.annotate(
                    f"{part_name} {dim}",
                    xy=(left + dim / 2, idx),
                    xytext=(max_total * 0.97, idx - 0.33),
                    arrowprops={"arrowstyle": "-", "color": PALETTE["amber"], "lw": 1.0},
                    ha="right",
                    va="center",
                    fontsize=8,
                    color=PALETTE["amber"],
                )
            left += dim
        ax.text(left + max_total * 0.035, idx, f"{left} 维", va="center", fontsize=10, color=PALETTE["muted"])

    ax.set_yticks(y, [name for name, _ in rows])
    ax.invert_yaxis()
    ax.set_xlabel("输入特征维度")
    ax.set_title("三组推荐实验输入特征对比", fontsize=15, weight="bold", pad=12)
    ax.set_xlim(0, max_total * 1.18)
    style_axis(ax, xgrid=True)
    ax.text(
        0,
        -0.20,
        "颜色说明：蓝色为文本特征，绿色为图像特征，棕色为新闻价值向量；新闻价值向量由 5 个归一化价值分数和 1 个 missing mask 组成。",
        transform=ax.transAxes,
        fontsize=9,
        color=PALETTE["muted"],
    )
    fig.tight_layout()
    return save_figure(fig, "model_input_comparison")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def as_float(value: str) -> float:
    return float(str(value).replace("%", "").strip())


def blend_hex(light: str, dark: str, ratio: float) -> str:
    ratio = max(0.0, min(1.0, ratio))
    light_rgb = tuple(int(light.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
    dark_rgb = tuple(int(dark.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
    rgb = tuple(round(l + (d - l) * ratio) for l, d in zip(light_rgb, dark_rgb, strict=True))
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def load_experiment_results() -> list[dict[str, str]]:
    path = TABLE_DIR / "experiment_results.csv"
    if path.exists():
        return read_csv_rows(path)

    fallback = ROOT / "artifacts" / "reports" / "ablation.csv"
    rows = []
    name_map = {mode: experiment for experiment, mode in EXPERIMENTS}
    for row in read_csv_rows(fallback):
        mode = row["mode"]
        rows.append(
            {
                "experiment": name_map.get(mode, row["experiment"]),
                "mode": mode,
                "auc": f"{as_float(row['auc']):.6f}",
                "mrr": f"{as_float(row['mrr']):.6f}",
                "ndcg5": f"{as_float(row['ndcg5']):.6f}",
                "ndcg10": f"{as_float(row['ndcg10']):.6f}",
            }
        )
    return rows


def load_metric_improvements() -> list[dict[str, str]]:
    path = TABLE_DIR / "metric_improvements.csv"
    if path.exists():
        return read_csv_rows(path)

    rows = load_experiment_results()
    by_experiment = {row["experiment"]: row for row in rows}
    baseline = by_experiment["Text"]
    improvements = []
    for experiment in ["Text+Image", "Text+Image+Value"]:
        row = by_experiment[experiment]
        for key, label in METRICS:
            base_value = as_float(baseline[key])
            value = as_float(row[key])
            gain = value - base_value
            rel_gain = gain / base_value * 100 if base_value else 0.0
            improvements.append(
                {
                    "experiment": experiment,
                    "metric": label.replace("nDCG", "NDCG"),
                    "baseline": f"{base_value:.6f}",
                    "value": f"{value:.6f}",
                    "absolute_gain": f"{gain:.6f}",
                    "relative_gain_percent": f"{rel_gain:.2f}%",
                }
            )
    return improvements


def draw_recommendation_model_architecture(stats: dict) -> Path:
    meta = stats["feature_meta"]
    text_dim = int(meta["text_dim"])
    image_dim = int(meta["image_dim"])
    value_dim = len(VALUE_DIMENSIONS) + 1
    graph = make_graph("recommendation_model_architecture", rankdir="LR", nodesep="0.45", ranksep="0.75")
    add_graph_node(graph, "candidate", "候选新闻输入", f"z_c：文本/图像/价值特征\nT={text_dim}, TI={text_dim + image_dim}, TIV={text_dim + image_dim + value_dim}", PALETTE["light_blue"], PALETTE["blue"])
    add_graph_node(graph, "history", "历史点击序列", "H_u = {n_1, ..., n_L}\n读取同一特征空间下新闻向量", PALETTE["light_green"], PALETTE["green"])
    add_graph_node(graph, "encoder", "共享 MLP 新闻编码器", "候选新闻与历史新闻共用参数\n输出 256 维隐藏表示", PALETTE["soft"], PALETTE["gray"])
    add_graph_node(graph, "cand_repr", "候选新闻表示", "h_c", PALETTE["light_blue"], PALETTE["blue"])
    add_graph_node(graph, "user_repr", "用户兴趣表示", "历史新闻表示平均池化\n得到 p_u", PALETTE["light_green"], PALETTE["green"])
    add_graph_node(graph, "score", "点积打分", "r(u,c) = p_u · h_c", PALETTE["light_amber"], PALETTE["amber"])
    graph.edge("candidate", "encoder")
    graph.edge("history", "encoder")
    graph.edge("encoder", "cand_repr")
    graph.edge("encoder", "user_repr")
    graph.edge("cand_repr", "score")
    graph.edge("user_repr", "score")
    return save_graph(graph, "recommendation_model_architecture")


def draw_training_evaluation_flow(stats: dict) -> Path:
    graph = make_graph("training_evaluation_flow", rankdir="LR", nodesep="0.45", ranksep="0.70")
    add_graph_node(graph, "input", "实验输入", f"训练样本 {stats['train_samples']:,}\n验证 impression {stats['dev_impressions']:,}\nSigLIP 特征与价值标注", PALETTE["light_blue"], PALETTE["blue"])
    add_graph_node(graph, "text", "Text", "仅使用文本特征", PALETTE["light_blue"], PALETTE["blue"])
    add_graph_node(graph, "ti", "Text+Image", "文本特征 + 图像特征", PALETTE["light_green"], PALETTE["green"])
    add_graph_node(graph, "tiv", "Text+Image+Value", "图文特征 + 新闻价值向量", PALETTE["light_amber"], PALETTE["amber"])
    add_graph_node(graph, "sort", "验证集排序", "同一 impression 内\n按模型得分排序", PALETTE["soft"], PALETTE["gray"])
    add_graph_node(graph, "metrics", "指标计算", "AUC、MRR\nnDCG@5、nDCG@10", PALETTE["light_cyan"], PALETTE["cyan"])
    add_graph_node(graph, "compare", "消融对比", "三组实验使用同一验证集\n归因图像与新闻价值增益", PALETTE["light_violet"], PALETTE["violet"])
    for exp in ["text", "ti", "tiv"]:
        graph.edge("input", exp)
        graph.edge(exp, "sort")
    graph.edge("sort", "metrics")
    graph.edge("metrics", "compare")
    return save_graph(graph, "training_evaluation_flow")


def draw_experiment_metrics_comparison(stats: dict) -> Path:
    rows = load_experiment_results()
    by_experiment = {row["experiment"]: row for row in rows}
    colors = {
        "Text": PALETTE["blue"],
        "Text+Image": PALETTE["green"],
        "Text+Image+Value": PALETTE["amber"],
    }
    x = np.arange(len(METRICS))
    width = 0.22

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    for exp_idx, (experiment, _) in enumerate(EXPERIMENTS):
        values = [as_float(by_experiment[experiment][metric_key]) for metric_key, _ in METRICS]
        bars = ax.bar(x + (exp_idx - 1) * width, values, width=width, label=experiment, color=colors[experiment], edgecolor=PALETTE["ink"], linewidth=0.35)
        for bar, value in zip(bars, values, strict=True):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.009, f"{value:.3f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x, [label for _, label in METRICS])
    ax.set_ylabel("指标值")
    ax.set_ylim(0, 0.65)
    ax.set_title("三组实验指标对比", fontsize=15, weight="bold", pad=12)
    ax.legend(frameon=False, loc="upper right")
    style_axis(ax, xgrid=False, ygrid=True)
    fig.tight_layout()
    return save_figure(fig, "experiment_metrics_comparison")


def draw_metric_improvement_heatmap(stats: dict) -> Path:
    rows = load_metric_improvements()
    row_names = ["Text+Image", "Text+Image+Value"]
    metric_names = [label.replace("nDCG", "NDCG") for _, label in METRICS]
    values = {(row["experiment"], row["metric"]): row for row in rows}
    gain_matrix = np.array([[as_float(values[(experiment, metric)]["absolute_gain"]) for metric in metric_names] for experiment in row_names])
    max_abs = max(abs(gain_matrix.min()), abs(gain_matrix.max()), 0.001)
    cmap = LinearSegmentedColormap.from_list("gain_cmap", [PALETTE["red"], "#ffffff", PALETTE["green"]])
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)

    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    image = ax.imshow(gain_matrix, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(np.arange(len(metric_names)), metric_names)
    ax.set_yticks(np.arange(len(row_names)), row_names)
    ax.set_title("相对 Text 基线的指标增益", fontsize=15, weight="bold", pad=12)
    for row_idx, experiment in enumerate(row_names):
        for col_idx, metric in enumerate(metric_names):
            row = values[(experiment, metric)]
            gain = as_float(row["absolute_gain"])
            sign = "+" if gain >= 0 else ""
            ax.text(col_idx, row_idx, f"{sign}{gain:.6f}\n{row['relative_gain_percent']}", ha="center", va="center", fontsize=9, color=PALETTE["ink"])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(metric_names), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_names), 1), minor=True)
    ax.tick_params(which="minor", bottom=False, left=False, labelbottom=False, labelleft=False)
    ax.grid(which="minor", color=PALETTE["paper"], linewidth=2)
    cbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.04)
    cbar.set_label("绝对增益", rotation=270, labelpad=14)
    fig.text(0.02, 0.02, "单元格第一行为绝对增益，第二行为相对增益。", fontsize=9, color=PALETTE["muted"])
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return save_figure(fig, "metric_improvement_heatmap")


def markdown_table(rows: Sequence[dict], columns: Sequence[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join([header, divider, *body])


def rel(path: Path) -> str:
    if path.is_relative_to(ROOT):
        return (".." / path.relative_to(ROOT)).as_posix()
    return path.as_posix()


def write_markdown(stats: dict, figure_paths: dict[str, Path], table_paths: dict[str, Path]) -> None:
    news_count = len(stats["news_records"])
    label_count = len(stats["labels"])
    meta = stats["feature_meta"]
    text_dim = int(meta["text_dim"])
    image_dim = int(meta["image_dim"])
    value_dim = len(VALUE_DIMENSIONS) + 1
    data_rows = read_csv_rows(table_paths["data"])
    feature_rows = read_csv_rows(table_paths["feature"])
    value_stat_rows = read_csv_rows(table_paths["value_stats"])
    value_dist_rows = read_csv_rows(table_paths["value_dist"])
    category_rows = read_csv_rows(table_paths["category"])[:12]
    model_rows = read_csv_rows(table_paths["model_inputs"])
    experiment_rows = load_experiment_results()
    improvement_rows = load_metric_improvements()

    content = f"""# 论文图表与公式补充素材

本文档为《基于新闻价值理论的多模态特征提取及其在个性化新闻推荐中的应用》提供可直接插入论文的图表、表格、公式和衔接文字。图表均由现有实验产物生成，避免手工统计误差。

## 生成文件总览

- 图表目录：`artifacts/thesis/figures/`
- 表格目录：`artifacts/thesis/tables/`
- 本脚本生成 11 张正式图示；每张图均同步保存 PNG 与 PDF，论文正文引用 PNG，同名 PDF 可用于矢量排版归档。
- 核心统计：新闻 `{news_count:,}` 条，训练样本 `{stats['train_samples']:,}` 条，验证 impression `{stats['dev_impressions']:,}` 个，新闻价值成功标注 `{label_count:,}` 条。

## 一、技术路线总览

### 建议插入位置

建议放在研究方法章节开头，用于承接研究目标、数据基础、特征构建、推荐建模和实验验证之间的关系。

### 图：技术路线图

![技术路线图]({rel(figure_paths['technical_route'])})

**建议图题：** 图 2-1 技术路线图

**正文衔接句：** 本研究按照“数据准备—特征提取—新闻价值量化—推荐建模—实验评估”的路径展开。首先完成新闻文本、图像和用户行为数据的结构化处理，再基于 SigLIP 提取图文特征，并将新闻价值维度作为可解释增强信号注入推荐模型，最终通过三组消融实验验证图像模态和新闻价值特征的增量贡献。

## 二、数据采集章节补充

### 建议插入位置

建议放在“数据来源与数据预处理”小节末尾。在介绍 MIND-small 与图像数据来源后，先放数据处理流程图，再放数据规模统计表和类别分布图。

### 图：数据采集与预处理流程

![数据采集与预处理流程]({rel(figure_paths['data_flow'])})

**建议图题：** 图 3-1 数据采集与预处理流程

**正文衔接句：** 本研究首先将 MIND-small 中的新闻文本、用户历史点击和候选曝光记录解析为结构化数据，并通过新闻 ID 与 `newData/{{news_id}}.jpg` 图像文件进行对齐。预处理后的数据进一步拆分为训练样本、验证 impression 和新闻频次统计文件，为后续多模态特征提取与新闻价值标注提供统一输入。

### 表：数据规模统计

**建议表题：** 表 3-1 数据采集与预处理结果统计

{markdown_table(data_rows, ['item', 'value', 'description'])}

CSV 文件：`{rel(table_paths['data'])}`

### 图：新闻类别分布

![新闻类别分布]({rel(figure_paths['category'])})

**建议图题：** 图 3-2 新闻类别分布

**正文衔接句：** 从类别分布看，数据集中 `news` 与 `sports` 类新闻占比较高，说明样本包含较多公共事件和体育资讯。该分布特点会影响用户兴趣建模，因此后续模型在相同训练与验证数据上进行消融对比，以减少数据划分差异造成的影响。

### 表：主要类别分布

{markdown_table(category_rows, ['category', 'count', 'percentage'])}

CSV 文件：`{rel(table_paths['category'])}`

## 三、特征提取章节补充

### 建议插入位置

建议放在“多模态特征构建”小节中。先说明文本字段拼接和图像输入，再放多模态特征提取结构图，随后列出公式和特征维度表。

### 图：多模态特征提取结构

![多模态特征提取结构]({rel(figure_paths['feature_flow'])})

**建议图题：** 图 3-3 多模态特征提取结构

**正文衔接句：** 本研究使用 SigLIP 作为统一的图文编码器。文本侧将新闻类别、子类别、标题和摘要拼接为输入序列，图像侧使用与新闻 ID 对齐的新闻图片。为降低推荐模型训练成本，图文向量在训练前预计算并缓存，推荐模型阶段只读取固定维度特征。

### 公式：文本构造与图文编码

设第 `n` 条新闻的类别、子类别、标题、摘要分别为 `c_n`、`sc_n`、`t_n`、`a_n`，对应图片为 `I_n`。文本输入定义为：

```latex
x_n = [c_n; sc_n; t_n; a_n]
```

SigLIP 文本编码器和图像编码器分别记为 `f_t(·)` 与 `f_i(·)`，编码后进行 L2 归一化：

```latex
\\mathbf{{e}}^{{(t)}}_n = \\frac{{f_t(x_n)}}{{\\|f_t(x_n)\\|_2}}, \\quad
\\mathbf{{e}}^{{(i)}}_n = \\frac{{f_i(I_n)}}{{\\|f_i(I_n)\\|_2}}
```

图文模态特征拼接为：

```latex
\\mathbf{{z}}^{{TI}}_n = [\\mathbf{{e}}^{{(t)}}_n; \\mathbf{{e}}^{{(i)}}_n]
```

### 表：特征维度设置

**建议表题：** 表 3-2 不同实验设置下的输入特征维度

{markdown_table(feature_rows, ['mode', 'text_dim', 'image_dim', 'value_dim', 'total_dim'])}

CSV 文件：`{rel(table_paths['feature'])}`

### 图：三组实验输入对比

![三组推荐实验输入特征对比]({rel(figure_paths['model_inputs'])})

**建议图题：** 图 3-4 三组推荐实验输入特征对比

**正文衔接句：** 三组实验的训练样本、验证样本和推荐模型结构保持一致，仅改变新闻输入特征。这样可以将性能变化主要归因于图像模态和新闻价值特征的增量贡献。

### 表：三组实验输入对比

{markdown_table(model_rows, ['experiment', 'input', 'formula', 'total_dim'])}

CSV 文件：`{rel(table_paths['model_inputs'])}`

## 四、新闻价值量化章节补充

### 建议插入位置

建议放在“新闻价值维度定义与自动标注”小节中。先说明采用五个维度，再放标注流程图，随后给出价值向量公式、标注统计表和五维分布图。

### 图：新闻价值量化流程

![新闻价值量化流程]({rel(figure_paths['value_flow'])})

**建议图题：** 图 3-5 新闻价值量化流程

**正文衔接句：** 由于 MIND-small 缺少可靠发布时间、地理位置和用户位置字段，本研究不对“时效性”和“接近性”进行自动评分，而选取重要性、显著性、冲突性、新奇性和人情味五个维度进行量化。标注时按新闻频次优先选择高频新闻，并通过 JSON 校验保证分数可被程序稳定解析。

### 公式：新闻价值向量与缺失标记

设第 `n` 条新闻在第 `j` 个新闻价值维度上的原始评分为 `s_{{n,j}}`，其中 `s_{{n,j}} \\in {{0,1,2,3}}`，五维总分为：

```latex
S_n = \\sum_{{j=1}}^5 s_{{n,j}}
```

为与神经网络输入尺度保持一致，将每个维度除以 3 归一化，并加入缺失标记 `m_n`：

```latex
\\mathbf{{v}}_n =
\\left[
\\frac{{s_{{n,1}}}}3,
\\frac{{s_{{n,2}}}}3,
\\frac{{s_{{n,3}}}}3,
\\frac{{s_{{n,4}}}}3,
\\frac{{s_{{n,5}}}}3,
m_n
\\right]
```

其中，已标注新闻令 `m_n=0`；未标注新闻的五个价值分数置为 0，并令 `m_n=1`。最终图文价值融合特征为：

```latex
\\mathbf{{z}}^{{TIV}}_n = [\\mathbf{{e}}^{{(t)}}_n; \\mathbf{{e}}^{{(i)}}_n; \\mathbf{{v}}_n]
```

### 表：新闻价值标注统计

**建议表题：** 表 3-3 新闻价值标注结果统计

{markdown_table(value_stat_rows, ['item', 'value', 'description'])}

CSV 文件：`{rel(table_paths['value_stats'])}`

### 图：五维新闻价值评分分布

![五维新闻价值评分分布]({rel(figure_paths['value_dist'])})

**建议图题：** 图 3-6 五维新闻价值评分分布

**正文衔接句：** 五个新闻价值维度的评分分布存在差异。例如，人情味和显著性中高分样本较多，而冲突性中 0 分样本较多。这表明新闻价值特征并非简单重复文本类别信息，而是从新闻传播学视角为推荐模型提供额外的可解释信号。

### 表：五维新闻价值评分分布

{markdown_table(value_dist_rows, ['dimension_cn', 'score_0', 'score_1', 'score_2', 'score_3', 'mean'])}

CSV 文件：`{rel(table_paths['value_dist'])}`

## 五、推荐建模与实验评价章节补充

### 建议插入位置

建议放在推荐模型结构和实验设计章节中。先展示模型结构，再展示训练评价流程，最后给出指标对比图与增益热力图。

### 图：个性化新闻推荐模型结构

![个性化新闻推荐模型结构]({rel(figure_paths['recommendation_model'])})

**建议图题：** 图 4-1 个性化新闻推荐模型结构

**正文衔接句：** 推荐模型对候选新闻和用户历史点击新闻使用共享 MLP 编码器，将不同实验设置下的新闻输入特征映射到统一隐藏空间。用户兴趣表示由历史点击新闻表示平均池化得到，并与候选新闻表示进行点积打分。

### 图：推荐模型训练与评价流程

![推荐模型训练与评价流程]({rel(figure_paths['training_evaluation'])})

**建议图题：** 图 4-2 推荐模型训练与评价流程

**正文衔接句：** 三组实验使用相同的训练样本、验证 impression 和评价指标，仅改变新闻输入特征，从而将性能差异主要归因于图像特征和新闻价值特征的引入。

### 图：三组实验指标对比

![三组实验指标对比]({rel(figure_paths['experiment_metrics'])})

**建议图题：** 图 4-3 三组实验指标对比

### 表：三组实验指标

{markdown_table(experiment_rows, ['experiment', 'mode', 'auc', 'mrr', 'ndcg5', 'ndcg10'])}

### 图：相对文本基线的指标增益

![相对文本基线的指标增益]({rel(figure_paths['metric_improvement'])})

**建议图题：** 图 4-4 相对文本基线的指标增益

**正文衔接句：** 与仅使用文本特征相比，单独加入图像特征在部分排序指标上存在波动，而进一步加入新闻价值特征后，各项指标均取得提升，说明新闻价值信号能够为图文推荐特征提供额外的解释性补充。

### 表：相对文本基线的指标增益

{markdown_table(improvement_rows, ['experiment', 'metric', 'baseline', 'value', 'absolute_gain', 'relative_gain_percent'])}

## 六、符号统一建议

| 符号 | 含义 |
| --- | --- |
| `n` | 新闻编号 |
| `x_n` | 第 `n` 条新闻的文本输入 |
| `I_n` | 第 `n` 条新闻的图片 |
| `\\mathbf{{e}}^{{(t)}}_n` | SigLIP 文本特征 |
| `\\mathbf{{e}}^{{(i)}}_n` | SigLIP 图像特征 |
| `s_{{n,j}}` | 第 `j` 个新闻价值维度的 0-3 原始分数 |
| `S_n` | 五维新闻价值总分 |
| `m_n` | 新闻价值缺失标记 |
| `\\mathbf{{v}}_n` | 新闻价值向量 |
| `\\mathbf{{z}}^{{TI}}_n` | 图文融合特征 |
| `\\mathbf{{z}}^{{TIV}}_n` | 图文与新闻价值融合特征 |
"""
    DOC_PATH.write_text(content, encoding="utf-8", newline="\n")


def main() -> None:
    ensure_dirs()
    stats = collect_stats()
    table_paths = write_tables(stats)
    figure_paths = {
        "technical_route": draw_technical_route(stats),
        "data_flow": draw_data_processing_flow(stats),
        "category": draw_category_distribution(stats),
        "feature_flow": draw_feature_extraction_flow(stats),
        "value_flow": draw_news_value_labeling_flow(stats),
        "value_dist": draw_value_dimension_distribution(stats),
        "model_inputs": draw_model_input_comparison(stats),
        "recommendation_model": draw_recommendation_model_architecture(stats),
        "training_evaluation": draw_training_evaluation_flow(stats),
        "experiment_metrics": draw_experiment_metrics_comparison(stats),
        "metric_improvement": draw_metric_improvement_heatmap(stats),
    }
    write_markdown(stats, figure_paths, table_paths)
    print(f"Wrote {len(figure_paths)} figures to {FIGURE_DIR}")
    print(f"Wrote {len(table_paths)} tables to {TABLE_DIR}")
    print(f"Wrote supplement markdown to {DOC_PATH}")


if __name__ == "__main__":
    main()
