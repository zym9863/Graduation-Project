from pathlib import Path

from src.data.analytics import build_dataset_statistics, export_statistics_csv, export_statistics_plots, render_markdown_report


def _mock_news() -> dict[str, dict[str, str]]:
    return {
        "N1": {
            "category": "news",
            "subcategory": "newsworld",
            "title": "Title One",
            "abstract": "Abstract One",
        },
        "N2": {
            "category": "sports",
            "subcategory": "baseball",
            "title": "Title Two",
            "abstract": "Abstract Two",
        },
    }


def _mock_train_behaviors() -> list[dict[str, object]]:
    return [
        {
            "impression_id": "1",
            "user_id": "U1",
            "time": "11/11/2019 9:00:00 AM",
            "history": ["N1"],
            "impressions": [("N1", 1), ("N2", 0), ("N3", 0)],
        },
        {
            "impression_id": "2",
            "user_id": "U2",
            "time": "11/11/2019 10:00:00 AM",
            "history": [],
            "impressions": [("N2", 1)],
        },
    ]


def _mock_dev_behaviors() -> list[dict[str, object]]:
    return [
        {
            "impression_id": "3",
            "user_id": "U3",
            "time": "2019-11-12 21:00:00",
            "history": ["N1", "N2"],
            "impressions": [("N2", 0)],
        }
    ]


def test_build_dataset_statistics_contains_expected_summary() -> None:
    stats = build_dataset_statistics(
        news_dict=_mock_news(),
        train_behaviors=_mock_train_behaviors(),
        dev_behaviors=_mock_dev_behaviors(),
        top_n_subcategories=1,
        sample_size=1,
    )

    assert stats["summary"]["unique_news"] == 2
    assert stats["summary"]["categories"] == 2
    assert stats["summary"]["subcategories"] == 2
    assert stats["summary"]["train_impressions"] == 4
    assert stats["summary"]["train_clicks"] == 2
    assert stats["summary"]["train_ctr"] == 0.5

    assert len(stats["distributions"]["news_subcategory_top"]) == 1
    assert len(stats["examples"]["news"]) == 1
    assert len(stats["examples"]["train_behaviors"]) == 1


def test_export_csv_plots_and_report(tmp_path: Path) -> None:
    stats = build_dataset_statistics(
        news_dict=_mock_news(),
        train_behaviors=_mock_train_behaviors(),
        dev_behaviors=_mock_dev_behaviors(),
        top_n_subcategories=2,
        sample_size=2,
    )

    figures_dir = tmp_path / "figures"
    tables_dir = tmp_path / "tables"
    report_path = tmp_path / "data_report.md"

    table_paths = export_statistics_csv(stats, tables_dir)
    figure_paths = export_statistics_plots(stats, figures_dir)
    render_markdown_report(stats, report_path, figure_paths, table_paths)

    assert table_paths["news_category"].exists()
    assert table_paths["news_subcategory"].exists()
    assert table_paths["category_ctr_train"].exists()

    assert figure_paths["news_category"].exists()
    assert figure_paths["history_length_train"].exists()
    assert figure_paths["category_ctr_dev"].exists()

    assert report_path.exists()
    report_content = report_path.read_text(encoding="utf-8")
    assert "# 数据集统计报告" in report_content
    assert "| unique_news |" in report_content
    assert "figures/news_category_distribution.png" in report_content
