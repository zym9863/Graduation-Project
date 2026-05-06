from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageDraw, ImageFont


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
    "ink": "#222222",
    "muted": "#5f6875",
    "line": "#b8c0cc",
    "grid": "#e4e7eb",
    "paper": "#ffffff",
    "soft": "#ffffff",
    "blue": "#6f8fbe",
    "green": "#79a88d",
    "amber": "#d1a05f",
    "red": "#c98b8b",
    "violet": "#9887b5",
    "cyan": "#75a7b5",
    "gray": "#8a93a0",
    "light_blue": "#e9eef7",
    "light_green": "#eaf3ee",
    "light_amber": "#f5eee2",
    "light_red": "#f5eaea",
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


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/simhei.ttf" if bold else "C:/Windows/Fonts/simsun.ttc",
        "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/PingFang.ttc",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def wrap_by_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    lines: list[str] = []
    for paragraph in str(text).split("\n"):
        current = ""
        for char in paragraph:
            test = current + char
            if current and text_size(draw, test, font)[0] > max_width:
                lines.append(current)
                current = char
            else:
                current = test
        if current:
            lines.append(current)
    return lines or [""]


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: str = PALETTE["ink"],
    max_width: int | None = None,
    line_gap: int = 8,
) -> None:
    x0, y0, x1, y1 = box
    width = x1 - x0
    max_width = max_width or width - 34
    lines = wrap_by_width(draw, text, font, max_width)
    line_heights = [text_size(draw, line, font)[1] for line in lines]
    total_height = sum(line_heights) + line_gap * (len(lines) - 1)
    y = y0 + ((y1 - y0) - total_height) / 2
    for line, line_height in zip(lines, line_heights, strict=True):
        line_width = text_size(draw, line, font)[0]
        draw.text((x0 + (width - line_width) / 2, y), line, font=font, fill=fill)
        y += line_height + line_gap


def draw_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    subtitle: str = "",
    fill: str = "#ffffff",
    outline: str = PALETTE["line"],
    accent: str = PALETTE["blue"],
) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=4, fill=fill, outline=outline, width=2)
    draw.line((x0, y0, x0, y1), fill=accent, width=5)
    title_font = load_font(24, bold=True)
    body_font = load_font(19)
    if subtitle:
        draw.text((x0 + 26, y0 + 24), title, font=title_font, fill=PALETTE["ink"])
        lines = wrap_by_width(draw, subtitle, body_font, x1 - x0 - 54)
        y = y0 + 66
        for line in lines[:3]:
            draw.text((x0 + 24, y), line, font=body_font, fill=PALETTE["muted"])
            y += 30
    else:
        draw_centered_text(draw, box, title, title_font)


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: str = PALETTE["gray"],
    width: int = 3,
) -> None:
    draw.line((start, end), fill=color, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    arrow_len = 14
    arrow_angle = math.pi / 8
    points = [
        end,
        (
            end[0] - arrow_len * math.cos(angle - arrow_angle),
            end[1] - arrow_len * math.sin(angle - arrow_angle),
        ),
        (
            end[0] - arrow_len * math.cos(angle + arrow_angle),
            end[1] - arrow_len * math.sin(angle + arrow_angle),
        ),
    ]
    draw.polygon(points, fill=color)


def make_canvas(width: int, height: int, title: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (width, height), PALETTE["paper"])
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width, height), fill=PALETTE["paper"])
    return image, draw


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


def draw_data_processing_flow(stats: dict) -> Path:
    image, draw = make_canvas(1800, 1000, "数据采集与预处理流程")
    boxes = [
        ((70, 160, 410, 330), "MIND-small 文本数据", "news.tsv：类别、子类、标题、摘要\nbehaviors.tsv：用户历史与曝光点击", PALETTE["blue"]),
        ((520, 160, 860, 330), "V-MIND 图像数据", "newData/{news_id}.jpg\n以新闻 ID 完成图片对齐", PALETTE["green"]),
        ((970, 160, 1310, 330), "新闻主表构建", f"合并去重 {len(stats['news_records']):,} 条新闻\n写入 news.jsonl", PALETTE["cyan"]),
        ((1420, 160, 1760, 330), "频次统计", "统计历史点击与候选曝光\n用于高频新闻价值标注", PALETTE["amber"]),
        ((300, 590, 640, 760), "训练样本生成", f"正样本 + 负采样\n{stats['train_samples']:,} 条训练样本", PALETTE["violet"]),
        ((770, 590, 1110, 760), "验证样本生成", f"保留 impression 排序列表\n{stats['dev_impressions']:,} 个验证组", PALETTE["red"]),
        ((1210, 590, 1720, 760), "中间数据产物", "news.jsonl、train_samples.jsonl\ndev_impressions.jsonl\nnews_frequency.json", PALETTE["blue"]),
    ]
    for box, title, subtitle, color in boxes:
        draw_box(draw, box, title, subtitle, accent=color)
    draw_arrow(draw, (410, 245), (520, 245))
    draw_arrow(draw, (860, 245), (970, 245))
    draw_arrow(draw, (1310, 245), (1420, 245))
    draw_arrow(draw, (1140, 330), (520, 590))
    draw_arrow(draw, (1140, 330), (940, 590))
    draw_arrow(draw, (640, 675), (770, 675), color=PALETTE["line"])
    draw_arrow(draw, (1110, 675), (1210, 675))
    footer_font = load_font(20)
    draw.text(
        (70, 905),
        "说明：工程中要求新闻文本与图片全部完成 ID 对齐；若缺少对应图片，数据准备阶段会直接报错。",
        font=footer_font,
        fill=PALETTE["muted"],
    )
    path = FIGURE_DIR / "data_processing_flow.png"
    image.save(path, quality=95)
    return path


def draw_category_distribution(stats: dict) -> Path:
    width, height = 1800, 1050
    image, draw = make_canvas(width, height, "新闻类别分布")
    counts = stats["category_counts"]
    news_count = len(stats["news_records"])
    top = counts.most_common(10)
    other = news_count - sum(count for _, count in top)
    rows = top + [("other", other)]
    max_count = max(count for _, count in rows)
    left, top_y = 280, 150
    bar_width, row_height = 1180, 68
    colors = [
        PALETTE["blue"],
        PALETTE["green"],
        PALETTE["amber"],
        PALETTE["red"],
        PALETTE["violet"],
        PALETTE["cyan"],
    ]
    label_font = load_font(24, bold=True)
    body_font = load_font(22)
    for idx, (category, count) in enumerate(rows):
        y = top_y + idx * row_height
        draw.text((70, y + 12), category, font=label_font, fill=PALETTE["ink"])
        draw.rounded_rectangle((left, y + 10, left + bar_width, y + 46), radius=3, fill=PALETTE["grid"])
        fill_width = int(bar_width * count / max_count)
        draw.rounded_rectangle(
            (left, y + 10, left + fill_width, y + 46),
            radius=3,
            fill=colors[idx % len(colors)],
        )
        pct = count / news_count * 100
        draw.text((left + bar_width + 35, y + 9), f"{count:,}  ({pct:.1f}%)", font=body_font, fill=PALETTE["muted"])
    draw.text(
        (70, 930),
        f"合计：{news_count:,} 条新闻；图中展示前 10 类，其余类别合并为 other。",
        font=body_font,
        fill=PALETTE["muted"],
    )
    path = FIGURE_DIR / "category_distribution.png"
    image.save(path, quality=95)
    return path


def draw_feature_extraction_flow(stats: dict) -> Path:
    image, draw = make_canvas(1800, 740, "多模态特征提取结构")
    meta = stats["feature_meta"]
    boxes = [
        ((70, 160, 480, 330), "文本字段拼接", "category [SEP] subcategory\n[SEP] title [SEP] abstract", PALETTE["blue"]),
        ((70, 460, 480, 630), "新闻图片输入", "newData/{news_id}.jpg\nRGB 图像", PALETTE["green"]),
        ((610, 160, 1010, 330), "SigLIP 文本编码器", f"{meta['model_name']}\n输出 {meta['text_dim']} 维文本向量", PALETTE["blue"]),
        ((610, 460, 1010, 630), "SigLIP 图像编码器", f"{meta['model_name']}\n输出 {meta['image_dim']} 维图像向量", PALETTE["green"]),
        ((1130, 160, 1450, 330), "L2 归一化", "得到 e_t\n用于统一向量尺度", PALETTE["cyan"]),
        ((1130, 460, 1450, 630), "L2 归一化", "得到 e_i\n用于统一向量尺度", PALETTE["cyan"]),
        ((1540, 310, 1760, 500), "特征拼接", "Text: 768\nT+I: 1536\nT+I+V: 1542", PALETTE["amber"]),
    ]
    for box, title, subtitle, color in boxes:
        draw_box(draw, box, title, subtitle, accent=color)
    draw_arrow(draw, (480, 245), (610, 245))
    draw_arrow(draw, (480, 545), (610, 545))
    draw_arrow(draw, (1010, 245), (1130, 245))
    draw_arrow(draw, (1010, 545), (1130, 545))
    draw_arrow(draw, (1450, 245), (1540, 380))
    draw_arrow(draw, (1450, 545), (1540, 430))
    path = FIGURE_DIR / "feature_extraction_flow.png"
    image.save(path, quality=95)
    return path


def draw_news_value_labeling_flow(stats: dict) -> Path:
    image, draw = make_canvas(1800, 1000, "新闻价值量化流程")
    manifest = stats["manifest"]
    label_count = len(stats["labels"])
    request_counts = manifest.get("batch", {}).get("request_counts", {})
    boxes = [
        ((70, 180, 390, 350), "新闻频次排序", "基于历史点击和候选曝光\n优先选择高频新闻", PALETTE["blue"]),
        ((500, 180, 820, 350), "Prompt 构造", "只使用类别、标题、摘要\n不评价时效性和接近性", PALETTE["green"]),
        ((930, 180, 1250, 350), "LLM 批量评分", f"{manifest.get('model', 'qwen3.5-flash')}\n五维度均为 0-3 分", PALETTE["violet"]),
        ((1360, 180, 1680, 350), "JSON 校验", "字段完整性、整数分数\n范围必须为 0-3", PALETTE["amber"]),
        ((330, 600, 650, 770), "成功缓存", f"{label_count:,} 条写入\nnews_value_labels.jsonl", PALETTE["green"]),
        ((760, 600, 1080, 770), "缺失处理", "未标注新闻使用零向量\nmissing mask = 1", PALETTE["red"]),
        ((1190, 600, 1510, 770), "价值向量", "5 个价值分数 + mask\n共 6 维", PALETTE["cyan"]),
    ]
    for box, title, subtitle, color in boxes:
        draw_box(draw, box, title, subtitle, accent=color)
    draw_arrow(draw, (390, 265), (500, 265))
    draw_arrow(draw, (820, 265), (930, 265))
    draw_arrow(draw, (1250, 265), (1360, 265))
    draw_arrow(draw, (1520, 350), (520, 600))
    draw_arrow(draw, (650, 685), (760, 685))
    draw_arrow(draw, (1080, 685), (1190, 685))
    body_font = load_font(22)
    completed = request_counts.get("completed", label_count)
    failed = request_counts.get("failed", 0)
    draw.text(
        (70, 900),
        f"批处理统计：目标 3,000 条，完成 {completed:,} 条，失败 {failed:,} 条；解析无效结果 0 条。",
        font=body_font,
        fill=PALETTE["muted"],
    )
    path = FIGURE_DIR / "news_value_labeling_flow.png"
    image.save(path, quality=95)
    return path


def draw_value_dimension_distribution(stats: dict) -> Path:
    width, height = 1800, 1050
    image, draw = make_canvas(width, height, "五维新闻价值评分分布")
    chart = (160, 180, 1640, 850)
    x0, y0, x1, y1 = chart
    draw.rectangle(chart, fill="#ffffff", outline=PALETTE["line"], width=2)
    label_count = len(stats["labels"])
    max_count = max(
        stats["value_dist"][name].get(score, 0)
        for name, _ in VALUE_DIMENSIONS
        for score in range(4)
    )
    grid_font = load_font(18)
    for step in range(0, 5):
        value = max_count * step / 4
        y = y1 - int((y1 - y0 - 80) * step / 4) - 40
        draw.line((x0 + 70, y, x1 - 30, y), fill=PALETTE["grid"], width=1)
        draw.text((x0 + 15, y - 12), f"{int(value)}", font=grid_font, fill=PALETTE["muted"])
    colors = [PALETTE["blue"], PALETTE["green"], PALETTE["amber"], PALETTE["red"]]
    group_width = (x1 - x0 - 150) / len(VALUE_DIMENSIONS)
    bar_width = 42
    axis_bottom = y1 - 40
    for group_idx, (name, cn) in enumerate(VALUE_DIMENSIONS):
        group_x = x0 + 95 + group_idx * group_width
        for score in range(4):
            count = stats["value_dist"][name].get(score, 0)
            bar_h = int((axis_bottom - y0 - 45) * count / max_count)
            bx0 = int(group_x + score * (bar_width + 12))
            bx1 = bx0 + bar_width
            by0 = axis_bottom - bar_h
            draw.rounded_rectangle((bx0, by0, bx1, axis_bottom), radius=2, fill=colors[score])
            draw.text((bx0 - 5, by0 - 26), str(count), font=grid_font, fill=PALETTE["muted"])
        draw.text((int(group_x), axis_bottom + 24), cn, font=load_font(22, bold=True), fill=PALETTE["ink"])
    legend_x = 1220
    for score, color in enumerate(colors):
        y = 900 + score * 34
        draw.rounded_rectangle((legend_x, y, legend_x + 28, y + 20), radius=2, fill=color)
        draw.text((legend_x + 42, y - 3), f"{score} 分", font=load_font(20), fill=PALETTE["ink"])
    draw.text((160, 912), f"样本量：{label_count:,} 条成功标注新闻。", font=load_font(22), fill=PALETTE["muted"])
    path = FIGURE_DIR / "value_dimension_distribution.png"
    image.save(path, quality=95)
    return path


def draw_model_input_comparison(stats: dict) -> Path:
    image, draw = make_canvas(1800, 900, "三组推荐实验输入特征对比")
    meta = stats["feature_meta"]
    text_dim = int(meta["text_dim"])
    image_dim = int(meta["image_dim"])
    value_dim = len(VALUE_DIMENSIONS) + 1
    rows = [
        ("Text", [("文本", text_dim, PALETTE["blue"])]),
        ("Text+Image", [("文本", text_dim, PALETTE["blue"]), ("图像", image_dim, PALETTE["green"])]),
        (
            "Text+Image+Value",
            [("文本", text_dim, PALETTE["blue"]), ("图像", image_dim, PALETTE["green"]), ("价值+mask", value_dim, PALETTE["amber"])],
        ),
    ]
    max_total = text_dim + image_dim + value_dim
    label_font = load_font(28, bold=True)
    body_font = load_font(22)
    start_y = 180
    bar_x = 420
    bar_width = 1050
    for idx, (name, parts) in enumerate(rows):
        y = start_y + idx * 180
        total = sum(dim for _, dim, _ in parts)
        draw.text((80, y + 38), name, font=label_font, fill=PALETTE["ink"])
        cursor = bar_x
        for part_name, dim, color in parts:
            part_width = max(8, int(bar_width * dim / max_total))
            draw.rounded_rectangle((cursor, y + 28, cursor + part_width, y + 88), radius=3, fill=color)
            if part_width > 110:
                draw_centered_text(
                    draw,
                    (cursor, y + 28, cursor + part_width, y + 88),
                    f"{part_name} {dim}",
                    body_font,
                    fill="#ffffff",
                )
            cursor += part_width
        draw.text((bar_x + bar_width + 50, y + 42), f"{total} 维", font=label_font, fill=PALETTE["muted"])
    draw.text(
        (80, 760),
        "说明：新闻价值向量由 5 个归一化价值分数和 1 个 missing mask 组成，因此为 6 维。",
        font=body_font,
        fill=PALETTE["muted"],
    )
    path = FIGURE_DIR / "model_input_comparison.png"
    image.save(path, quality=95)
    return path


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


def draw_polyline_arrow(draw: ImageDraw.ImageDraw, points: Sequence[tuple[int, int]]) -> None:
    if len(points) < 2:
        return
    for start, end in zip(points[:-2], points[1:-1], strict=False):
        draw.line((start, end), fill=PALETTE["gray"], width=3)
    draw_arrow(draw, points[-2], points[-1])


def draw_recommendation_model_architecture(stats: dict) -> Path:
    image, draw = make_canvas(1800, 760, "个性化新闻推荐模型结构")
    boxes = [
        ((70, 90, 390, 230), "候选新闻输入", "z_c：文本/图像/价值特征", PALETTE["blue"]),
        ((70, 430, 390, 570), "历史点击序列", "H_u = {n_1,...,n_L}", PALETTE["green"]),
        ((560, 245, 890, 415), "共享 MLP 新闻编码器", "候选新闻与历史新闻共用参数\n输出 256 维隐藏表示", PALETTE["gray"]),
        ((1060, 90, 1380, 230), "候选新闻表示", "h_c", PALETTE["blue"]),
        ((1060, 430, 1380, 570), "用户兴趣表示", "历史新闻表示平均池化\n得到 p_u", PALETTE["green"]),
        ((1510, 250, 1730, 410), "点积打分", "r(u,c)", PALETTE["amber"]),
    ]
    for box, title, subtitle, color in boxes:
        draw_box(draw, box, title, subtitle, accent=color)

    draw_polyline_arrow(draw, [(390, 160), (475, 160), (475, 300), (560, 300)])
    draw_polyline_arrow(draw, [(390, 500), (475, 500), (475, 360), (560, 360)])
    draw_polyline_arrow(draw, [(890, 300), (980, 300), (980, 160), (1060, 160)])
    draw_polyline_arrow(draw, [(890, 360), (980, 360), (980, 500), (1060, 500)])
    draw_polyline_arrow(draw, [(1380, 160), (1450, 160), (1450, 305), (1510, 305)])
    draw_polyline_arrow(draw, [(1380, 500), (1450, 500), (1450, 355), (1510, 355)])

    note_font = load_font(20)
    draw.text(
        (70, 675),
        f"输入维度：Text={stats['feature_meta']['text_dim']}，Text+Image={int(stats['feature_meta']['text_dim']) + int(stats['feature_meta']['image_dim'])}，Text+Image+Value={int(stats['feature_meta']['text_dim']) + int(stats['feature_meta']['image_dim']) + len(VALUE_DIMENSIONS) + 1}。",
        font=note_font,
        fill=PALETTE["muted"],
    )
    path = FIGURE_DIR / "recommendation_model_architecture.png"
    image.save(path, quality=95)
    return path


def draw_training_evaluation_flow(stats: dict) -> Path:
    image, draw = make_canvas(1800, 760, "推荐模型训练与评价流程")
    boxes = [
        ((70, 120, 360, 290), "实验输入", "训练样本、验证集\nSigLIP 特征、价值标注", PALETTE["blue"]),
        ((510, 70, 820, 210), "Text", "仅使用文本特征", PALETTE["blue"]),
        ((510, 270, 820, 410), "Text+Image", "文本特征 + 图像特征", PALETTE["green"]),
        ((510, 470, 820, 610), "Text+Image+Value", "图文特征 + 新闻价值向量", PALETTE["amber"]),
        ((990, 200, 1290, 360), "验证集排序", "同一 impression 内\n按模型得分排序", PALETTE["gray"]),
        ((1450, 200, 1720, 360), "指标计算", "AUC、MRR\nnDCG@5、nDCG@10", PALETTE["cyan"]),
    ]
    for box, title, subtitle, color in boxes:
        draw_box(draw, box, title, subtitle, accent=color)

    draw_polyline_arrow(draw, [(360, 205), (435, 205), (435, 140), (510, 140)])
    draw_polyline_arrow(draw, [(360, 205), (435, 205), (435, 340), (510, 340)])
    draw_polyline_arrow(draw, [(360, 205), (435, 205), (435, 540), (510, 540)])
    draw_polyline_arrow(draw, [(820, 140), (905, 140), (905, 250), (990, 250)])
    draw_polyline_arrow(draw, [(820, 340), (990, 280)])
    draw_polyline_arrow(draw, [(820, 540), (905, 540), (905, 315), (990, 315)])
    draw_arrow(draw, (1290, 280), (1450, 280))

    note_font = load_font(20)
    draw.text(
        (70, 690),
        f"三组实验使用相同验证集，共 {stats['dev_impressions']:,} 个 impression；图中只展示评价流程，具体训练参数见表 4-1。",
        font=note_font,
        fill=PALETTE["muted"],
    )
    path = FIGURE_DIR / "training_evaluation_flow.png"
    image.save(path, quality=95)
    return path


def draw_experiment_metrics_comparison(stats: dict) -> Path:
    rows = load_experiment_results()
    by_experiment = {row["experiment"]: row for row in rows}
    image, draw = make_canvas(1800, 920, "三组实验指标对比")

    chart = (150, 90, 1420, 730)
    x0, y0, x1, y1 = chart
    axis_font = load_font(21)
    label_font = load_font(25, bold=True)
    value_font = load_font(20)
    y_max = 0.65

    for step in range(0, 8):
        value = step * 0.1
        y = y1 - int((y1 - y0) * value / y_max)
        draw.line((x0, y, x1, y), fill=PALETTE["grid"], width=1)
        draw.text((70, y - 13), f"{value:.1f}", font=axis_font, fill=PALETTE["muted"])
    draw.line((x0, y0, x0, y1), fill=PALETTE["ink"], width=2)
    draw.line((x0, y1, x1, y1), fill=PALETTE["ink"], width=2)

    colors = {
        "Text": PALETTE["blue"],
        "Text+Image": PALETTE["green"],
        "Text+Image+Value": PALETTE["amber"],
    }
    group_width = (x1 - x0) / len(METRICS)
    bar_width = 70
    bar_gap = 24
    total_bar_width = bar_width * len(EXPERIMENTS) + bar_gap * (len(EXPERIMENTS) - 1)
    for metric_idx, (metric_key_name, metric_label) in enumerate(METRICS):
        group_center = x0 + group_width * metric_idx + group_width / 2
        start_x = int(group_center - total_bar_width / 2)
        for exp_idx, (experiment, _) in enumerate(EXPERIMENTS):
            value = as_float(by_experiment[experiment][metric_key_name])
            bx0 = start_x + exp_idx * (bar_width + bar_gap)
            bx1 = bx0 + bar_width
            by0 = y1 - int((y1 - y0) * value / y_max)
            draw.rounded_rectangle((bx0, by0, bx1, y1), radius=3, fill=colors[experiment], outline=PALETTE["ink"], width=1)
            draw.text((bx0 - 6, by0 - 30), f"{value:.3f}", font=value_font, fill=PALETTE["ink"])
        metric_width = text_size(draw, metric_label, label_font)[0]
        draw.text((group_center - metric_width / 2, y1 + 32), metric_label, font=label_font, fill=PALETTE["ink"])

    legend_x, legend_y = 1500, 140
    for idx, (experiment, _) in enumerate(EXPERIMENTS):
        y = legend_y + idx * 58
        draw.rounded_rectangle((legend_x, y, legend_x + 34, y + 22), radius=2, fill=colors[experiment], outline=PALETTE["ink"], width=1)
        draw.text((legend_x + 52, y - 3), experiment, font=axis_font, fill=PALETTE["ink"])

    path = FIGURE_DIR / "experiment_metrics_comparison.png"
    image.save(path, quality=95)
    return path


def draw_metric_improvement_heatmap(stats: dict) -> Path:
    rows = load_metric_improvements()
    row_names = ["Text+Image", "Text+Image+Value"]
    metric_names = [label.replace("nDCG", "NDCG") for _, label in METRICS]
    values = {(row["experiment"], row["metric"]): row for row in rows}
    gains = [as_float(row["absolute_gain"]) for row in rows]
    max_pos = max([gain for gain in gains if gain > 0], default=1.0)
    max_neg = max([abs(gain) for gain in gains if gain < 0], default=1.0)

    image, draw = make_canvas(1800, 780, "相对文本基线的指标增益")
    x0, y0 = 420, 140
    cell_w, cell_h = 260, 150
    header_font = load_font(28, bold=True)
    label_font = load_font(24, bold=True)
    gain_font = load_font(26, bold=True)
    rel_font = load_font(21)

    for col_idx, metric in enumerate(metric_names):
        x = x0 + col_idx * cell_w
        metric_width = text_size(draw, metric, header_font)[0]
        draw.text((x + (cell_w - metric_width) / 2, 80), metric, font=header_font, fill=PALETTE["ink"])

    for row_idx, experiment in enumerate(row_names):
        y = y0 + row_idx * cell_h
        draw.text((80, y + 55), experiment, font=label_font, fill=PALETTE["ink"])
        for col_idx, metric in enumerate(metric_names):
            row = values[(experiment, metric)]
            gain = as_float(row["absolute_gain"])
            if gain >= 0:
                ratio = 0.15 + 0.65 * (gain / max_pos)
                fill = blend_hex(PALETTE["light_green"], PALETTE["green"], ratio)
            else:
                ratio = 0.15 + 0.65 * (abs(gain) / max_neg)
                fill = blend_hex(PALETTE["light_red"], PALETTE["red"], ratio)
            x = x0 + col_idx * cell_w
            draw.rounded_rectangle((x, y, x + cell_w - 18, y + cell_h - 22), radius=3, fill=fill, outline="#ffffff", width=3)
            sign = "+" if gain >= 0 else ""
            gain_text = f"{sign}{gain:.6f}"
            rel_text = row["relative_gain_percent"]
            gain_width = text_size(draw, gain_text, gain_font)[0]
            rel_width = text_size(draw, rel_text, rel_font)[0]
            draw.text((x + (cell_w - 18 - gain_width) / 2, y + 42), gain_text, font=gain_font, fill=PALETTE["ink"])
            draw.text((x + (cell_w - 18 - rel_width) / 2, y + 92), rel_text, font=rel_font, fill=PALETTE["ink"])

    note_font = load_font(20)
    draw.text(
        (80, 610),
        "单元格上方为绝对增益，下方为相对增益；正负号表示相对于 Text 基线的提升或下降。",
        font=note_font,
        fill=PALETTE["muted"],
    )
    path = FIGURE_DIR / "metric_improvement_heatmap.png"
    image.save(path, quality=95)
    return path


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

    content = f"""# 论文图表与公式补充素材

本文档为《基于新闻价值理论的多模态特征提取及其在个性化新闻推荐中的应用》中“数据采集”“特征提取”“新闻价值量化”三节提供可直接插入论文的图表、表格、公式和衔接文字。图表均由现有实验产物生成，避免手工统计误差。

## 生成文件总览

- 图表目录：`artifacts/thesis/figures/`
- 表格目录：`artifacts/thesis/tables/`
- 核心统计：新闻 `{news_count:,}` 条，训练样本 `{stats['train_samples']:,}` 条，验证 impression `{stats['dev_impressions']:,}` 个，新闻价值成功标注 `{label_count:,}` 条。

## 一、数据采集章节补充

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

## 二、特征提取章节补充

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

## 三、新闻价值量化章节补充

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

## 四、符号统一建议

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
