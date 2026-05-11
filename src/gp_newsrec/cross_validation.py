from __future__ import annotations

import csv
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from .constants import DEFAULT_SEED, PROMPT_VERSION, VALUE_DIMENSIONS, VALUE_DIMENSION_CN
from .io import ensure_dir, read_jsonl, write_jsonl
from .labels import ValueLabel, load_value_cache


CROSS_MODEL_SAMPLE_SIZE = 300
CROSS_MODEL_BANDS: tuple[tuple[str, int, int, int], ...] = (
    ("low", 0, 6, 92),
    ("medium", 7, 9, 136),
    ("high", 10, 15, 72),
)
CROSS_MODEL_EXAMPLE_IDS: tuple[str, ...] = ("N306", "N31958", "N5940", "N11930", "N6916")

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
}

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


def configure_matplotlib_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [choose_font_family(), "Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"],
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


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_news(path: Path) -> dict[str, dict[str, Any]]:
    return {str(record["news_id"]): record for record in read_jsonl(path)}


def score_band(total: int, bands: Sequence[tuple[str, int, int, int]] = CROSS_MODEL_BANDS) -> str | None:
    for name, lower, upper, _ in bands:
        if lower <= total <= upper:
            return name
    return None


def _allocate_category_quotas(groups: dict[str, list[dict[str, Any]]], target: int) -> dict[str, int]:
    total = sum(len(items) for items in groups.values())
    if target > total:
        raise ValueError(f"Cannot sample {target} records from only {total} candidates.")
    quotas: dict[str, int] = {}
    remainders: list[tuple[float, int, str]] = []
    for category, items in groups.items():
        raw = len(items) / total * target if total else 0.0
        base = min(len(items), int(math.floor(raw)))
        quotas[category] = base
        remainders.append((raw - base, len(items), category))

    remaining = target - sum(quotas.values())
    order = sorted(remainders, key=lambda item: (-item[0], -item[1], item[2]))
    while remaining > 0:
        progressed = False
        for _, _, category in order:
            if quotas[category] < len(groups[category]):
                quotas[category] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            raise ValueError("Could not allocate category quotas with the available candidates.")
    return quotas


def _sample_band(
    candidates: list[dict[str, Any]],
    target: int,
    rng: random.Random,
    anchor_ids: set[str],
) -> list[dict[str, Any]]:
    anchors = sorted((row for row in candidates if row["news_id"] in anchor_ids), key=lambda row: row["news_id"])
    if len(anchors) > target:
        raise ValueError(f"Anchor news count {len(anchors)} exceeds band target {target}.")
    remaining_target = target - len(anchors)
    if remaining_target == 0:
        return anchors

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        if row["news_id"] not in anchor_ids:
            groups[str(row.get("category", ""))].append(row)
    quotas = _allocate_category_quotas(groups, remaining_target)
    selected = list(anchors)
    for category in sorted(groups):
        pool = list(groups[category])
        rng.shuffle(pool)
        selected.extend(pool[: quotas[category]])
    return selected


def prepare_cross_model_sample(
    data_dir: str | Path = "artifacts/data",
    flash_label_path: str | Path = "artifacts/labels/news_value_labels.jsonl",
    output_path: str | Path = "artifacts/labels/cross_model_sample.jsonl",
    seed: int = DEFAULT_SEED,
    bands: Sequence[tuple[str, int, int, int]] = CROSS_MODEL_BANDS,
    anchor_news_ids: Iterable[str] = CROSS_MODEL_EXAMPLE_IDS,
) -> dict[str, Any]:
    data_dir = Path(data_dir)
    flash_label_path = Path(flash_label_path)
    output_path = Path(output_path)
    news = _read_news(data_dir / "news.jsonl")
    flash_labels = load_value_cache(flash_label_path)
    anchor_ids = {str(news_id) for news_id in anchor_news_ids}
    candidates_by_band: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for news_id, label in flash_labels.items():
        record = news.get(news_id)
        band = score_band(label.total, bands)
        if record is None or band is None:
            continue
        candidates_by_band[band].append(
            {
                "news_id": news_id,
                "category": str(record.get("category", "")),
                "subcategory": str(record.get("subcategory", "")),
                "title": str(record.get("title", "")),
                "abstract": str(record.get("abstract", "")),
                "score_band": band,
                "prompt_version": label.prompt_version,
                "flash_scores": label.scores,
                "flash_total": label.total,
                "flash_reason": label.reason,
            }
        )

    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    band_stats: dict[str, dict[str, Any]] = {}
    for band_name, _, _, target in bands:
        candidates = candidates_by_band.get(band_name, [])
        band_selected = _sample_band(candidates, target, rng, anchor_ids)
        selected.extend(band_selected)
        band_stats[band_name] = {
            "target": target,
            "available": len(candidates),
            "selected": len(band_selected),
            "categories": dict(Counter(str(row.get("category", "")) for row in band_selected)),
        }

    band_order = {name: idx for idx, (name, _, _, _) in enumerate(bands)}
    selected.sort(key=lambda row: (band_order.get(str(row["score_band"]), 99), row["category"], row["news_id"]))
    if len({row["news_id"] for row in selected}) != len(selected):
        raise ValueError("Cross-model sample contains duplicate news IDs.")
    expected = sum(target for _, _, _, target in bands)
    if len(selected) != expected:
        raise ValueError(f"Cross-model sample has {len(selected)} rows; expected {expected}.")

    write_jsonl(output_path, selected)
    return {
        "status": "created",
        "sample_path": str(output_path),
        "sample_size": len(selected),
        "seed": seed,
        "prompt_version": PROMPT_VERSION,
        "band_stats": band_stats,
        "anchors_included": sorted(anchor_ids.intersection(row["news_id"] for row in selected)),
    }


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) < 2:
        return 1.0
    left_mean = _mean(left)
    right_mean = _mean(right)
    num = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right, strict=True))
    left_den = math.sqrt(sum((x - left_mean) ** 2 for x in left))
    right_den = math.sqrt(sum((y - right_mean) ** 2 for y in right))
    den = left_den * right_den
    if den == 0:
        return 1.0 if all(x == y for x, y in zip(left, right, strict=True)) else 0.0
    return num / den


def _average_ranks(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    idx = 0
    while idx < len(indexed):
        end = idx + 1
        while end < len(indexed) and indexed[end][1] == indexed[idx][1]:
            end += 1
        rank = (idx + 1 + end) / 2.0
        for original_idx, _ in indexed[idx:end]:
            ranks[original_idx] = rank
        idx = end
    return ranks


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    return _pearson(_average_ranks(left), _average_ranks(right))


def _quadratic_weighted_kappa(left: Sequence[int], right: Sequence[int], min_score: int, max_score: int) -> float:
    scores = list(range(min_score, max_score + 1))
    size = len(scores)
    index = {score: idx for idx, score in enumerate(scores)}
    observed = np.zeros((size, size), dtype="float64")
    for flash_score, plus_score in zip(left, right, strict=True):
        if flash_score in index and plus_score in index:
            observed[index[flash_score], index[plus_score]] += 1.0
    total = observed.sum()
    if total == 0:
        return 0.0
    flash_hist = observed.sum(axis=1)
    plus_hist = observed.sum(axis=0)
    expected = np.outer(flash_hist, plus_hist) / total
    weights = np.zeros((size, size), dtype="float64")
    denominator = (size - 1) ** 2
    for row in range(size):
        for col in range(size):
            weights[row, col] = ((row - col) ** 2) / denominator if denominator else 0.0
    weighted_observed = float((weights * observed).sum())
    weighted_expected = float((weights * expected).sum())
    if weighted_expected == 0:
        return 1.0 if weighted_observed == 0 else 0.0
    return 1.0 - weighted_observed / weighted_expected


def _score_text(scores: dict[str, int]) -> str:
    return "；".join(f"{VALUE_DIMENSION_CN[dim]}={scores[dim]}" for dim in VALUE_DIMENSIONS)


def _format_float(value: float) -> str:
    return f"{value:.6f}"


def _load_pairs(
    sample_path: Path,
    flash_label_path: Path,
    plus_label_path: Path,
) -> list[dict[str, Any]]:
    sample_rows = list(read_jsonl(sample_path))
    flash_labels = load_value_cache(flash_label_path)
    plus_labels = load_value_cache(plus_label_path)
    pairs = []
    for row in sample_rows:
        news_id = str(row["news_id"])
        flash_label = flash_labels.get(news_id)
        plus_label = plus_labels.get(news_id)
        if flash_label is None or plus_label is None:
            continue
        pairs.append(
            {
                "news_id": news_id,
                "category": str(row.get("category", "")),
                "subcategory": str(row.get("subcategory", "")),
                "title": str(row.get("title", "")),
                "flash": flash_label,
                "plus": plus_label,
            }
        )
    return pairs


def _metric_rows(pairs: Sequence[dict[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    metric_targets = [(dim, VALUE_DIMENSION_CN[dim], 0, 3) for dim in VALUE_DIMENSIONS]
    metric_targets.append(("total", "总分", 0, 15))
    for key, label, min_score, max_score in metric_targets:
        if key == "total":
            flash_values = [int(pair["flash"].total) for pair in pairs]
            plus_values = [int(pair["plus"].total) for pair in pairs]
        else:
            flash_values = [int(pair["flash"].scores[key]) for pair in pairs]
            plus_values = [int(pair["plus"].scores[key]) for pair in pairs]
        deltas = [plus - flash for flash, plus in zip(flash_values, plus_values, strict=True)]
        exact = _mean([1.0 if delta == 0 else 0.0 for delta in deltas])
        within_one = _mean([1.0 if abs(delta) <= 1 else 0.0 for delta in deltas])
        mae = _mean([abs(delta) for delta in deltas])
        rows.append(
            {
                "target": key,
                "target_cn": label,
                "n": str(len(pairs)),
                "flash_mean": _format_float(_mean(flash_values)),
                "plus_mean": _format_float(_mean(plus_values)),
                "mean_delta": _format_float(_mean(deltas)),
                "exact_agreement": _format_float(exact),
                "within_one_agreement": _format_float(within_one),
                "mae": _format_float(mae),
                "pearson": _format_float(_pearson(flash_values, plus_values)),
                "spearman": _format_float(_spearman(flash_values, plus_values)),
                "quadratic_weighted_kappa": _format_float(
                    _quadratic_weighted_kappa(flash_values, plus_values, min_score, max_score)
                ),
            }
        )
    return rows


def _example_rows(pairs: Sequence[dict[str, Any]]) -> list[dict[str, str]]:
    scored = []
    for pair in pairs:
        flash_label: ValueLabel = pair["flash"]
        plus_label: ValueLabel = pair["plus"]
        dim_abs = [abs(int(plus_label.scores[dim]) - int(flash_label.scores[dim])) for dim in VALUE_DIMENSIONS]
        scored.append(
            {
                "pair": pair,
                "sum_abs_delta": sum(dim_abs),
                "max_dim_delta": max(dim_abs),
                "total_abs_delta": abs(int(plus_label.total) - int(flash_label.total)),
            }
        )

    agreement = sorted(scored, key=lambda item: (item["sum_abs_delta"], item["total_abs_delta"], item["pair"]["news_id"]))
    disagreement = sorted(
        (item for item in scored if item["total_abs_delta"] >= 3 or item["max_dim_delta"] >= 2),
        key=lambda item: (-item["total_abs_delta"], -item["max_dim_delta"], -item["sum_abs_delta"], item["pair"]["news_id"]),
    )
    chosen = [("高度一致", item) for item in agreement[:3]]
    chosen.extend(("明显分歧", item) for item in disagreement[:3])

    rows = []
    seen: set[tuple[str, str]] = set()
    for example_type, item in chosen:
        pair = item["pair"]
        key = (example_type, pair["news_id"])
        if key in seen:
            continue
        seen.add(key)
        flash_label = pair["flash"]
        plus_label = pair["plus"]
        rows.append(
            {
                "example_type": example_type,
                "news_id": pair["news_id"],
                "category": pair["category"],
                "subcategory": pair["subcategory"],
                "title": pair["title"],
                "flash_scores": _score_text(flash_label.scores),
                "plus_scores": _score_text(plus_label.scores),
                "flash_total": str(flash_label.total),
                "plus_total": str(plus_label.total),
                "total_delta": str(int(plus_label.total) - int(flash_label.total)),
                "flash_reason": flash_label.reason,
                "plus_reason": plus_label.reason,
            }
        )
    return rows


def _save_figure(fig, output_dir: Path, stem: str) -> Path:
    ensure_dir(output_dir)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.08, facecolor=PALETTE["paper"])
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.08, facecolor=PALETTE["paper"])
    plt.close(fig)
    return png_path


def _style_axis(ax, xgrid: bool = False, ygrid: bool = False) -> None:
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


def _draw_total_scatter(pairs: Sequence[dict[str, Any]], metric_rows: Sequence[dict[str, str]], figure_dir: Path) -> Path:
    counts = Counter((int(pair["flash"].total), int(pair["plus"].total)) for pair in pairs)
    xs = [flash for flash, _ in counts]
    ys = [plus for _, plus in counts]
    sizes = [32 + count * 9 for count in counts.values()]
    total_metrics = next(row for row in metric_rows if row["target"] == "total")

    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    ax.scatter(xs, ys, s=sizes, color=PALETTE["blue"], alpha=0.72, edgecolor=PALETTE["ink"], linewidth=0.35)
    domain = np.arange(0, 16)
    ax.plot(domain, domain, color=PALETTE["ink"], linewidth=1.3, label="y=x")
    for delta, color, linestyle in [(1, PALETTE["green"], "--"), (2, PALETTE["amber"], ":")]:
        ax.plot(domain, domain + delta, color=color, linewidth=1.0, linestyle=linestyle, label=f"±{delta} 分")
        ax.plot(domain, domain - delta, color=color, linewidth=1.0, linestyle=linestyle)
    ax.set_xlim(-0.4, 15.4)
    ax.set_ylim(-0.4, 15.4)
    ax.set_xticks(range(0, 16, 3))
    ax.set_yticks(range(0, 16, 3))
    ax.set_xlabel("qwen3.5-flash 总分")
    ax.set_ylabel("qwen3.5-plus 总分")
    ax.set_title("双模型新闻价值总分一致性", fontsize=15, weight="bold", pad=12)
    ax.text(
        0.03,
        0.97,
        f"n={len(pairs)}\nPearson={float(total_metrics['pearson']):.3f}\nMAE={float(total_metrics['mae']):.3f}",
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": PALETTE["soft"], "edgecolor": PALETTE["line"]},
    )
    ax.legend(frameon=False, loc="lower right")
    _style_axis(ax, xgrid=True, ygrid=True)
    fig.tight_layout()
    return _save_figure(fig, figure_dir, "cross_model_total_scatter")


def _confusion_matrix(pairs: Sequence[dict[str, Any]], dim: str) -> np.ndarray:
    matrix = np.zeros((4, 4), dtype=int)
    for pair in pairs:
        flash = int(pair["flash"].scores[dim])
        plus = int(pair["plus"].scores[dim])
        matrix[plus, flash] += 1
    return matrix


def _draw_confusion_matrices(pairs: Sequence[dict[str, Any]], figure_dir: Path) -> Path:
    matrices = [_confusion_matrix(pairs, dim) for dim in VALUE_DIMENSIONS]
    vmax = max(int(matrix.max()) for matrix in matrices) if matrices else 1
    fig, axes = plt.subplots(1, len(VALUE_DIMENSIONS), figsize=(14.2, 3.5), constrained_layout=True)
    for ax, dim, matrix in zip(axes, VALUE_DIMENSIONS, matrices, strict=True):
        image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=vmax)
        ax.set_title(VALUE_DIMENSION_CN[dim], fontsize=12, weight="bold")
        ax.set_xticks(range(4), range(4))
        ax.set_yticks(range(4), range(4))
        ax.set_xlabel("flash")
        if dim == VALUE_DIMENSIONS[0]:
            ax.set_ylabel("plus")
        for row in range(4):
            for col in range(4):
                value = int(matrix[row, col])
                color = "white" if value > vmax * 0.55 else PALETTE["ink"]
                ax.text(col, row, str(value), ha="center", va="center", fontsize=8.5, color=color)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02, label="样本数")
    fig.suptitle("五维新闻价值评分混淆矩阵", fontsize=15, weight="bold")
    return _save_figure(fig, figure_dir, "cross_model_confusion_matrices")


def _draw_agreement_rates(metric_rows: Sequence[dict[str, str]], figure_dir: Path) -> Path:
    rows = [row for row in metric_rows if row["target"] in VALUE_DIMENSIONS]
    labels = [row["target_cn"] for row in rows]
    exact = [float(row["exact_agreement"]) * 100 for row in rows]
    within_one = [float(row["within_one_agreement"]) * 100 for row in rows]
    x = np.arange(len(rows))
    width = 0.34

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    bars_a = ax.bar(x - width / 2, exact, width=width, label="完全一致", color=PALETTE["blue"], edgecolor=PALETTE["ink"], linewidth=0.35)
    bars_b = ax.bar(x + width / 2, within_one, width=width, label="|差值|≤1", color=PALETTE["green"], edgecolor=PALETTE["ink"], linewidth=0.35)
    for bars in [bars_a, bars_b]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2, f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x, labels)
    ax.set_ylabel("一致率")
    ax.set_ylim(0, 108)
    ax.set_title("五维新闻价值双模型一致率", fontsize=15, weight="bold", pad=12)
    ax.legend(frameon=False, loc="upper right")
    _style_axis(ax, ygrid=True)
    fig.tight_layout()
    return _save_figure(fig, figure_dir, "cross_model_agreement_rates")


def _draw_delta_distribution(pairs: Sequence[dict[str, Any]], figure_dir: Path) -> Path:
    deltas = list(range(-3, 4))
    matrix = np.zeros((len(VALUE_DIMENSIONS), len(deltas)), dtype=int)
    for row_idx, dim in enumerate(VALUE_DIMENSIONS):
        counter = Counter(int(pair["plus"].scores[dim]) - int(pair["flash"].scores[dim]) for pair in pairs)
        for col_idx, delta in enumerate(deltas):
            matrix[row_idx, col_idx] = counter.get(delta, 0)

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    image = ax.imshow(matrix, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(np.arange(len(deltas)), [str(delta) for delta in deltas])
    ax.set_yticks(np.arange(len(VALUE_DIMENSIONS)), [VALUE_DIMENSION_CN[dim] for dim in VALUE_DIMENSIONS])
    ax.set_xlabel("plus - flash")
    ax.set_title("双模型五维评分差异分布", fontsize=15, weight="bold", pad=12)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = int(matrix[row, col])
            color = "white" if value > matrix.max() * 0.55 else PALETTE["ink"]
            ax.text(col, row, str(value), ha="center", va="center", fontsize=9, color=color)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.04)
    cbar.set_label("样本数", rotation=270, labelpad=14)
    fig.tight_layout()
    return _save_figure(fig, figure_dir, "cross_model_score_delta_distribution")


def analyze_cross_model_validation(
    sample_path: str | Path = "artifacts/labels/cross_model_sample.jsonl",
    flash_label_path: str | Path = "artifacts/labels/news_value_labels.jsonl",
    plus_label_path: str | Path = "artifacts/labels/news_value_labels_qwen35_plus_sample.jsonl",
    output_dir: str | Path = "artifacts/thesis",
    write_figures: bool = True,
    strict: bool = True,
) -> dict[str, Any]:
    sample_path = Path(sample_path)
    flash_label_path = Path(flash_label_path)
    plus_label_path = Path(plus_label_path)
    output_dir = Path(output_dir)
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"

    if not sample_path.exists():
        if strict:
            raise FileNotFoundError(f"Cross-model sample file does not exist: {sample_path}")
        return {"status": "missing_sample", "sample_path": str(sample_path)}
    if not plus_label_path.exists():
        if strict:
            raise FileNotFoundError(f"Plus label file does not exist: {plus_label_path}")
        return {"status": "missing_plus_labels", "plus_label_path": str(plus_label_path)}

    pairs = _load_pairs(sample_path, flash_label_path, plus_label_path)
    if not pairs:
        if strict:
            raise ValueError("No overlapping flash/plus labels found for the cross-model sample.")
        return {"status": "no_overlap", "plus_label_path": str(plus_label_path)}

    metric_rows = _metric_rows(pairs)
    example_rows = _example_rows(pairs)
    metric_path = table_dir / "cross_model_agreement.csv"
    example_path = table_dir / "cross_model_examples.csv"
    _write_csv(
        metric_path,
        [
            "target",
            "target_cn",
            "n",
            "flash_mean",
            "plus_mean",
            "mean_delta",
            "exact_agreement",
            "within_one_agreement",
            "mae",
            "pearson",
            "spearman",
            "quadratic_weighted_kappa",
        ],
        metric_rows,
    )
    _write_csv(
        example_path,
        [
            "example_type",
            "news_id",
            "category",
            "subcategory",
            "title",
            "flash_scores",
            "plus_scores",
            "flash_total",
            "plus_total",
            "total_delta",
            "flash_reason",
            "plus_reason",
        ],
        example_rows,
    )

    figure_paths: dict[str, str] = {}
    if write_figures:
        figure_paths = {
            "total_scatter": str(_draw_total_scatter(pairs, metric_rows, figure_dir)),
            "confusion_matrices": str(_draw_confusion_matrices(pairs, figure_dir)),
            "agreement_rates": str(_draw_agreement_rates(metric_rows, figure_dir)),
            "delta_distribution": str(_draw_delta_distribution(pairs, figure_dir)),
        }

    return {
        "status": "created",
        "pairs": len(pairs),
        "expected_sample_size": CROSS_MODEL_SAMPLE_SIZE,
        "metric_path": str(metric_path),
        "example_path": str(example_path),
        "figure_paths": figure_paths,
        "total_metrics": next(row for row in metric_rows if row["target"] == "total"),
    }
