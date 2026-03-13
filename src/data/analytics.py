from __future__ import annotations

import csv
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


_TIME_FORMATS = (
    "%m/%d/%Y %I:%M:%S %p",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
)


def build_dataset_statistics(
    news_dict: dict[str, dict[str, Any]],
    train_behaviors: list[dict[str, Any]],
    dev_behaviors: list[dict[str, Any]],
    top_n_subcategories: int = 20,
    sample_size: int = 3,
) -> dict[str, Any]:
    news_category_counter = Counter(_normalized(item.get("category")) for item in news_dict.values())
    news_subcategory_counter = Counter(_normalized(item.get("subcategory")) for item in news_dict.values())
    news_to_category = {news_id: _normalized(item.get("category")) for news_id, item in news_dict.items()}

    train_stats = _analyze_behaviors(train_behaviors, news_to_category)
    dev_stats = _analyze_behaviors(dev_behaviors, news_to_category)
    combined_stats = _analyze_behaviors(train_behaviors + dev_behaviors, news_to_category)

    sample_news = _collect_news_examples(news_dict, sample_size)
    sample_train_behaviors = _collect_behavior_examples(train_behaviors, sample_size)
    sample_dev_behaviors = _collect_behavior_examples(dev_behaviors, sample_size)

    return {
        "summary": {
            "unique_news": len(news_dict),
            "categories": len(news_category_counter),
            "subcategories": len(news_subcategory_counter),
            "train_behaviors": len(train_behaviors),
            "dev_behaviors": len(dev_behaviors),
            "train_unique_users": train_stats["unique_users"],
            "dev_unique_users": dev_stats["unique_users"],
            "train_impressions": train_stats["total_impressions"],
            "dev_impressions": dev_stats["total_impressions"],
            "train_clicks": train_stats["total_clicks"],
            "dev_clicks": dev_stats["total_clicks"],
            "train_ctr": train_stats["ctr"],
            "dev_ctr": dev_stats["ctr"],
        },
        "distributions": {
            "news_category": _counter_to_rows(news_category_counter, "category"),
            "news_subcategory": _counter_to_rows(news_subcategory_counter, "subcategory"),
            "news_subcategory_top": _counter_to_rows(news_subcategory_counter, "subcategory", limit=top_n_subcategories),
            "history_length_train": _counter_to_numeric_rows(train_stats["history_length_counter"], "length"),
            "history_length_dev": _counter_to_numeric_rows(dev_stats["history_length_counter"], "length"),
            "impression_length_train": _counter_to_numeric_rows(train_stats["impression_length_counter"], "length"),
            "impression_length_dev": _counter_to_numeric_rows(dev_stats["impression_length_counter"], "length"),
            "hourly_behavior_train": _hourly_rows(train_stats["hourly_counter"]),
            "hourly_behavior_dev": _hourly_rows(dev_stats["hourly_counter"]),
        },
        "engagement": {
            "train": _behavior_summary(train_stats),
            "dev": _behavior_summary(dev_stats),
            "combined": _behavior_summary(combined_stats),
            "category_ctr_train": _category_ctr_rows(train_stats),
            "category_ctr_dev": _category_ctr_rows(dev_stats),
        },
        "examples": {
            "news": sample_news,
            "train_behaviors": sample_train_behaviors,
            "dev_behaviors": sample_dev_behaviors,
        },
    }


def export_statistics_csv(statistics: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    exported: dict[str, Path] = {}

    exported["news_category"] = output_dir / "news_category_distribution.csv"
    _write_csv(exported["news_category"], ["category", "count"], statistics["distributions"]["news_category"])

    exported["news_subcategory_top"] = output_dir / "news_subcategory_top.csv"
    _write_csv(
        exported["news_subcategory_top"],
        ["subcategory", "count"],
        statistics["distributions"]["news_subcategory_top"],
    )

    exported["news_subcategory"] = output_dir / "news_subcategory_distribution.csv"
    _write_csv(
        exported["news_subcategory"],
        ["subcategory", "count"],
        statistics["distributions"]["news_subcategory"],
    )

    exported["history_length_train"] = output_dir / "history_length_train.csv"
    _write_csv(
        exported["history_length_train"],
        ["length", "count"],
        statistics["distributions"]["history_length_train"],
    )

    exported["history_length_dev"] = output_dir / "history_length_dev.csv"
    _write_csv(
        exported["history_length_dev"],
        ["length", "count"],
        statistics["distributions"]["history_length_dev"],
    )

    exported["impression_length_train"] = output_dir / "impression_length_train.csv"
    _write_csv(
        exported["impression_length_train"],
        ["length", "count"],
        statistics["distributions"]["impression_length_train"],
    )

    exported["impression_length_dev"] = output_dir / "impression_length_dev.csv"
    _write_csv(
        exported["impression_length_dev"],
        ["length", "count"],
        statistics["distributions"]["impression_length_dev"],
    )

    exported["hourly_behavior_train"] = output_dir / "hourly_behavior_train.csv"
    _write_csv(
        exported["hourly_behavior_train"],
        ["hour", "count"],
        statistics["distributions"]["hourly_behavior_train"],
    )

    exported["hourly_behavior_dev"] = output_dir / "hourly_behavior_dev.csv"
    _write_csv(
        exported["hourly_behavior_dev"],
        ["hour", "count"],
        statistics["distributions"]["hourly_behavior_dev"],
    )

    exported["category_ctr_train"] = output_dir / "category_ctr_train.csv"
    _write_csv(
        exported["category_ctr_train"],
        ["category", "impressions", "clicks", "ctr"],
        statistics["engagement"]["category_ctr_train"],
    )

    exported["category_ctr_dev"] = output_dir / "category_ctr_dev.csv"
    _write_csv(
        exported["category_ctr_dev"],
        ["category", "impressions", "clicks", "ctr"],
        statistics["engagement"]["category_ctr_dev"],
    )

    return exported


def export_statistics_plots(statistics: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    plots: dict[str, Path] = {}

    plots["news_category"] = output_dir / "news_category_distribution.png"
    _plot_bar(
        rows=statistics["distributions"]["news_category"],
        x_key="category",
        y_key="count",
        title="News Category Distribution",
        x_label="Category",
        y_label="News Count",
        output_path=plots["news_category"],
        rotation=35,
    )

    plots["news_subcategory_top"] = output_dir / "news_subcategory_top.png"
    _plot_horizontal_bar(
        rows=statistics["distributions"]["news_subcategory_top"],
        label_key="subcategory",
        value_key="count",
        title="Top Subcategory Distribution",
        x_label="News Count",
        output_path=plots["news_subcategory_top"],
    )

    plots["history_length_train"] = output_dir / "history_length_train.png"
    _plot_bar(
        rows=statistics["distributions"]["history_length_train"],
        x_key="length",
        y_key="count",
        title="Train History Length Distribution",
        x_label="History Length",
        y_label="Behavior Count",
        output_path=plots["history_length_train"],
    )

    plots["history_length_dev"] = output_dir / "history_length_dev.png"
    _plot_bar(
        rows=statistics["distributions"]["history_length_dev"],
        x_key="length",
        y_key="count",
        title="Dev History Length Distribution",
        x_label="History Length",
        y_label="Behavior Count",
        output_path=plots["history_length_dev"],
    )

    plots["impression_length_train"] = output_dir / "impression_length_train.png"
    _plot_bar(
        rows=statistics["distributions"]["impression_length_train"],
        x_key="length",
        y_key="count",
        title="Train Impression Length Distribution",
        x_label="Impression Length",
        y_label="Behavior Count",
        output_path=plots["impression_length_train"],
    )

    plots["impression_length_dev"] = output_dir / "impression_length_dev.png"
    _plot_bar(
        rows=statistics["distributions"]["impression_length_dev"],
        x_key="length",
        y_key="count",
        title="Dev Impression Length Distribution",
        x_label="Impression Length",
        y_label="Behavior Count",
        output_path=plots["impression_length_dev"],
    )

    plots["hourly_behavior_train"] = output_dir / "hourly_behavior_train.png"
    _plot_bar(
        rows=statistics["distributions"]["hourly_behavior_train"],
        x_key="hour",
        y_key="count",
        title="Train Behavior by Hour",
        x_label="Hour",
        y_label="Behavior Count",
        output_path=plots["hourly_behavior_train"],
    )

    plots["hourly_behavior_dev"] = output_dir / "hourly_behavior_dev.png"
    _plot_bar(
        rows=statistics["distributions"]["hourly_behavior_dev"],
        x_key="hour",
        y_key="count",
        title="Dev Behavior by Hour",
        x_label="Hour",
        y_label="Behavior Count",
        output_path=plots["hourly_behavior_dev"],
    )

    plots["category_ctr_train"] = output_dir / "category_ctr_train.png"
    _plot_bar(
        rows=statistics["engagement"]["category_ctr_train"],
        x_key="category",
        y_key="ctr",
        title="Train Category CTR",
        x_label="Category",
        y_label="CTR",
        output_path=plots["category_ctr_train"],
        rotation=35,
    )

    plots["category_ctr_dev"] = output_dir / "category_ctr_dev.png"
    _plot_bar(
        rows=statistics["engagement"]["category_ctr_dev"],
        x_key="category",
        y_key="ctr",
        title="Dev Category CTR",
        x_label="Category",
        y_label="CTR",
        output_path=plots["category_ctr_dev"],
        rotation=35,
    )

    return plots


def render_markdown_report(
    statistics: dict[str, Any],
    report_path: Path,
    figure_paths: dict[str, Path],
    table_paths: dict[str, Path],
) -> None:
    summary = statistics["summary"]
    train_engagement = statistics["engagement"]["train"]
    dev_engagement = statistics["engagement"]["dev"]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "# 数据集统计报告",
        "",
        "本报告由 `uv run python main.py dataset-report` 自动生成。",
        "",
        "## 数据规模",
        "",
        "| 指标 | 数值 |",
        "| --- | ---: |",
        f"| unique_news | {summary['unique_news']} |",
        f"| categories | {summary['categories']} |",
        f"| subcategories | {summary['subcategories']} |",
        f"| train_behaviors | {summary['train_behaviors']} |",
        f"| dev_behaviors | {summary['dev_behaviors']} |",
        f"| train_unique_users | {summary['train_unique_users']} |",
        f"| dev_unique_users | {summary['dev_unique_users']} |",
        f"| train_impressions | {summary['train_impressions']} |",
        f"| dev_impressions | {summary['dev_impressions']} |",
        f"| train_clicks | {summary['train_clicks']} |",
        f"| dev_clicks | {summary['dev_clicks']} |",
        f"| train_ctr | {summary['train_ctr']:.4f} |",
        f"| dev_ctr | {summary['dev_ctr']:.4f} |",
        "",
        "## 行为分布摘要",
        "",
        "| Split | avg_history_len | p50_history_len | p90_history_len | avg_impression_len | p50_impression_len | p90_impression_len |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| train | {train_engagement['history_length']['avg']:.2f} | {train_engagement['history_length']['p50']:.2f} | {train_engagement['history_length']['p90']:.2f} | {train_engagement['impression_length']['avg']:.2f} | {train_engagement['impression_length']['p50']:.2f} | {train_engagement['impression_length']['p90']:.2f} |",
        f"| dev | {dev_engagement['history_length']['avg']:.2f} | {dev_engagement['history_length']['p50']:.2f} | {dev_engagement['history_length']['p90']:.2f} | {dev_engagement['impression_length']['avg']:.2f} | {dev_engagement['impression_length']['p50']:.2f} | {dev_engagement['impression_length']['p90']:.2f} |",
        "",
        "## 图表清单",
        "",
    ]

    for key, path in figure_paths.items():
        relative = path.relative_to(report_path.parent).as_posix()
        lines.append(f"- {key}: [{path.name}]({relative})")

    lines.extend(["", "## 统计表清单", ""])
    for key, path in table_paths.items():
        relative = path.relative_to(report_path.parent).as_posix()
        lines.append(f"- {key}: [{path.name}]({relative})")

    lines.extend(["", "## 新闻样例", ""])
    lines.extend(_render_news_examples(statistics["examples"]["news"]))

    lines.extend(["", "## 行为样例（Train）", ""])
    lines.extend(_render_behavior_examples(statistics["examples"]["train_behaviors"]))

    lines.extend(["", "## 行为样例（Dev）", ""])
    lines.extend(_render_behavior_examples(statistics["examples"]["dev_behaviors"]))

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _collect_news_examples(news_dict: dict[str, dict[str, Any]], sample_size: int) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for news_id in sorted(news_dict)[:sample_size]:
        item = news_dict[news_id]
        samples.append(
            {
                "news_id": news_id,
                "category": _normalized(item.get("category")),
                "subcategory": _normalized(item.get("subcategory")),
                "title": _truncate_text(_normalized(item.get("title")), 140),
                "abstract": _truncate_text(_normalized(item.get("abstract")), 180),
            }
        )
    return samples


def _collect_behavior_examples(behaviors: list[dict[str, Any]], sample_size: int) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for behavior in behaviors[:sample_size]:
        impressions = behavior.get("impressions", [])
        positive = [news_id for news_id, label in impressions if int(label) == 1]
        negative = [news_id for news_id, label in impressions if int(label) == 0]

        samples.append(
            {
                "impression_id": _normalized(behavior.get("impression_id")),
                "user_id": _normalized(behavior.get("user_id")),
                "time": _normalized(behavior.get("time")),
                "history_len": len(behavior.get("history", [])),
                "history_preview": " ".join((behavior.get("history", [])[:8])),
                "impression_len": len(impressions),
                "clicked_preview": " ".join(positive[:8]),
                "unclicked_preview": " ".join(negative[:8]),
            }
        )
    return samples


def _analyze_behaviors(behaviors: list[dict[str, Any]], news_to_category: dict[str, str]) -> dict[str, Any]:
    history_counter: Counter[int] = Counter()
    impression_counter: Counter[int] = Counter()
    hourly_counter: Counter[int] = Counter()
    category_impression_counter: Counter[str] = Counter()
    category_click_counter: Counter[str] = Counter()

    unique_users: set[str] = set()
    total_impressions = 0
    total_clicks = 0

    for behavior in behaviors:
        user_id = _normalized(behavior.get("user_id"))
        if user_id:
            unique_users.add(user_id)

        history = behavior.get("history", [])
        impressions = behavior.get("impressions", [])

        history_counter[len(history)] += 1
        impression_counter[len(impressions)] += 1

        parsed_hour = _parse_hour(_normalized(behavior.get("time")))
        if parsed_hour is not None:
            hourly_counter[parsed_hour] += 1

        total_impressions += len(impressions)
        for news_id, label in impressions:
            category = news_to_category.get(news_id, "unknown")
            category_impression_counter[category] += 1
            if int(label) == 1:
                total_clicks += 1
                category_click_counter[category] += 1

    ctr = total_clicks / total_impressions if total_impressions else 0.0

    return {
        "behavior_count": len(behaviors),
        "unique_users": len(unique_users),
        "total_impressions": total_impressions,
        "total_clicks": total_clicks,
        "ctr": ctr,
        "history_length_counter": history_counter,
        "impression_length_counter": impression_counter,
        "hourly_counter": hourly_counter,
        "category_impression_counter": category_impression_counter,
        "category_click_counter": category_click_counter,
    }


def _behavior_summary(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "behavior_count": stats["behavior_count"],
        "unique_users": stats["unique_users"],
        "total_impressions": stats["total_impressions"],
        "total_clicks": stats["total_clicks"],
        "ctr": stats["ctr"],
        "history_length": _summarize_numeric_counter(stats["history_length_counter"]),
        "impression_length": _summarize_numeric_counter(stats["impression_length_counter"]),
    }


def _category_ctr_rows(stats: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for category, impressions in sorted(
        stats["category_impression_counter"].items(),
        key=lambda item: (-item[1], item[0]),
    ):
        clicks = stats["category_click_counter"].get(category, 0)
        ctr = clicks / impressions if impressions else 0.0
        rows.append(
            {
                "category": category,
                "impressions": impressions,
                "clicks": clicks,
                "ctr": ctr,
            }
        )
    return rows


def _summarize_numeric_counter(counter: Counter[int]) -> dict[str, float]:
    values: list[int] = []
    for value, count in counter.items():
        values.extend([value] * count)

    if not values:
        return {
            "count": 0,
            "min": 0.0,
            "max": 0.0,
            "avg": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
        }

    sorted_values = sorted(values)
    return {
        "count": float(len(values)),
        "min": float(sorted_values[0]),
        "max": float(sorted_values[-1]),
        "avg": float(mean(values)),
        "p50": _percentile(sorted_values, 0.5),
        "p90": _percentile(sorted_values, 0.9),
        "p95": _percentile(sorted_values, 0.95),
    }


def _counter_to_rows(counter: Counter[str], key: str, limit: int | None = None) -> list[dict[str, Any]]:
    rows = [{key: name, "count": count} for name, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))]
    if limit is not None:
        return rows[:limit]
    return rows


def _counter_to_numeric_rows(counter: Counter[int], key: str) -> list[dict[str, int]]:
    return [{key: value, "count": count} for value, count in sorted(counter.items(), key=lambda item: item[0])]


def _hourly_rows(counter: Counter[int]) -> list[dict[str, int]]:
    return [{"hour": hour, "count": counter.get(hour, 0)} for hour in range(24)]


def _parse_hour(raw_time: str) -> int | None:
    if not raw_time:
        return None

    for fmt in _TIME_FORMATS:
        try:
            return datetime.strptime(raw_time, fmt).hour
        except ValueError:
            continue

    try:
        normalized = raw_time.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized).hour
    except ValueError:
        return None


def _percentile(sorted_values: list[int], quantile: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])

    index = (len(sorted_values) - 1) * quantile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return float(sorted_values[lower])

    lower_value = sorted_values[lower]
    upper_value = sorted_values[upper]
    return float(lower_value + (upper_value - lower_value) * (index - lower))


def _write_csv(filepath: Path, headers: list[str], rows: list[dict[str, Any]]) -> None:
    with filepath.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_bar(
    rows: list[dict[str, Any]],
    x_key: str,
    y_key: str,
    title: str,
    x_label: str,
    y_label: str,
    output_path: Path,
    rotation: int = 0,
) -> None:
    plt.figure(figsize=(12, 5))
    x_values = [row[x_key] for row in rows]
    y_values = [row[y_key] for row in rows]

    positions = list(range(len(x_values)))
    plt.bar(positions, y_values, color="#3B82F6")
    plt.xticks(positions, x_values, rotation=rotation, ha="right" if rotation else "center")
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_horizontal_bar(
    rows: list[dict[str, Any]],
    label_key: str,
    value_key: str,
    title: str,
    x_label: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(12, 8))
    labels = [row[label_key] for row in rows][::-1]
    values = [row[value_key] for row in rows][::-1]

    positions = list(range(len(labels)))
    plt.barh(positions, values, color="#10B981")
    plt.yticks(positions, labels)
    plt.title(title)
    plt.xlabel(x_label)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def _render_news_examples(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["无可用新闻样例。"]

    lines = [
        "| news_id | category | subcategory | title | abstract |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['news_id']} | {row['category']} | {row['subcategory']} | "
            f"{_escape_table(row['title'])} | {_escape_table(row['abstract'])} |"
        )
    return lines


def _render_behavior_examples(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["无可用行为样例。"]

    lines = [
        "| impression_id | user_id | time | history_len | impression_len | clicked_preview | history_preview |",
        "| --- | --- | --- | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['impression_id']} | {row['user_id']} | {row['time']} | {row['history_len']} | "
            f"{row['impression_len']} | {_escape_table(row['clicked_preview'])} | {_escape_table(row['history_preview'])} |"
        )
    return lines


def _escape_table(text: str) -> str:
    return text.replace("|", "\\|")


def _truncate_text(text: str, max_length: int) -> str:
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def _normalized(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return text if text else "unknown"
