from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gp_newsrec.cross_validation import analyze_cross_model_validation, prepare_cross_model_sample
from gp_newsrec.io import read_jsonl, write_jsonl
from gp_newsrec.labels import validate_value_payload


def scores_from_total(total: int) -> dict[str, int]:
    scores = {}
    for dim in ["importance", "prominence", "conflict", "novelty", "human_interest"]:
        value = min(3, total)
        scores[dim] = value
        total -= value
    return scores


def label_record(news_id: str, total: int, reason: str = "测试理由") -> dict:
    return asdict(validate_value_payload(news_id, {"scores": scores_from_total(total), "reason": reason}))


class CrossModelValidationTest(unittest.TestCase):
    def test_prepare_cross_model_sample_uses_score_bands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            flash_path = root / "flash.jsonl"
            sample_path = root / "sample.jsonl"
            news_rows = [
                {"news_id": "L1", "category": "news", "subcategory": "x", "title": "low 1", "abstract": ""},
                {"news_id": "L2", "category": "sports", "subcategory": "x", "title": "low 2", "abstract": ""},
                {"news_id": "L3", "category": "sports", "subcategory": "x", "title": "low 3", "abstract": ""},
                {"news_id": "M1", "category": "news", "subcategory": "x", "title": "mid 1", "abstract": ""},
                {"news_id": "M2", "category": "finance", "subcategory": "x", "title": "mid 2", "abstract": ""},
                {"news_id": "H1", "category": "news", "subcategory": "x", "title": "high 1", "abstract": ""},
                {"news_id": "H2", "category": "sports", "subcategory": "x", "title": "high 2", "abstract": ""},
            ]
            write_jsonl(data_dir / "news.jsonl", news_rows)
            write_jsonl(
                flash_path,
                [
                    label_record("L1", 5),
                    label_record("L2", 6),
                    label_record("L3", 4),
                    label_record("M1", 8),
                    label_record("M2", 9),
                    label_record("H1", 10),
                    label_record("H2", 12),
                ],
            )

            stats = prepare_cross_model_sample(
                data_dir=data_dir,
                flash_label_path=flash_path,
                output_path=sample_path,
                bands=(("low", 0, 6, 2), ("medium", 7, 9, 1), ("high", 10, 15, 1)),
                anchor_news_ids=(),
            )
            rows = list(read_jsonl(sample_path))

        self.assertEqual(stats["sample_size"], 4)
        self.assertEqual(len({row["news_id"] for row in rows}), 4)
        self.assertEqual({"low": 2, "medium": 1, "high": 1}, {band: sum(row["score_band"] == band for row in rows) for band in ["low", "medium", "high"]})

    def test_analyze_cross_model_validation_writes_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample_path = root / "sample.jsonl"
            flash_path = root / "flash.jsonl"
            plus_path = root / "plus.jsonl"
            output_dir = root / "thesis"
            write_jsonl(
                sample_path,
                [
                    {"news_id": "N1", "category": "news", "subcategory": "x", "title": "Story 1"},
                    {"news_id": "N2", "category": "sports", "subcategory": "x", "title": "Story 2"},
                    {"news_id": "N3", "category": "finance", "subcategory": "x", "title": "Story 3"},
                ],
            )
            write_jsonl(
                flash_path,
                [
                    label_record("N1", 5, "flash1"),
                    label_record("N2", 8, "flash2"),
                    label_record("N3", 10, "flash3"),
                ],
            )
            write_jsonl(
                plus_path,
                [
                    label_record("N1", 5, "plus1"),
                    label_record("N2", 9, "plus2"),
                    label_record("N3", 8, "plus3"),
                ],
            )

            stats = analyze_cross_model_validation(
                sample_path=sample_path,
                flash_label_path=flash_path,
                plus_label_path=plus_path,
                output_dir=output_dir,
                write_figures=False,
            )
            with (output_dir / "tables" / "cross_model_agreement.csv").open("r", encoding="utf-8-sig", newline="") as file:
                metric_rows = list(csv.DictReader(file))
            example_exists = (output_dir / "tables" / "cross_model_examples.csv").exists()

        self.assertEqual(stats["status"], "created")
        self.assertEqual(stats["pairs"], 3)
        self.assertTrue(example_exists)
        self.assertEqual("total", metric_rows[-1]["target"])
        self.assertEqual("3", metric_rows[-1]["n"])


if __name__ == "__main__":
    unittest.main()
