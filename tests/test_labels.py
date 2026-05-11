from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gp_newsrec.constants import VALUE_DIMENSIONS
from gp_newsrec.io import read_jsonl, write_jsonl
from gp_newsrec.labels import (
    build_aliyun_batch_request,
    build_value_prompt,
    merge_aliyun_batch_results,
    prepare_aliyun_batch_input,
    validate_value_payload,
)


class ValueLabelTest(unittest.TestCase):
    def test_value_dimensions_exclude_data_deficient_fields(self) -> None:
        self.assertNotIn("timeliness", VALUE_DIMENSIONS)
        self.assertNotIn("proximity", VALUE_DIMENSIONS)

    def test_validate_value_payload(self) -> None:
        payload = {
            "scores": {
                "importance": 2,
                "prominence": 1,
                "conflict": 3,
                "novelty": 1,
                "human_interest": 0,
            },
            "reason": "测试理由",
        }
        label = validate_value_payload("N1", payload)
        self.assertEqual(label.total, 7)
        self.assertEqual(label.scores["conflict"], 3)

    def test_prompt_mentions_excluded_dimensions(self) -> None:
        prompt = build_value_prompt({"title": "A title", "abstract": "Body"})
        self.assertIn("不要评估时效性和接近性", prompt)

    def test_build_aliyun_batch_request_uses_chat_completion_shape(self) -> None:
        request = build_aliyun_batch_request(
            {
                "news_id": "N1",
                "category": "news",
                "subcategory": "world",
                "title": "A title",
                "abstract": "An abstract",
            }
        )

        self.assertEqual(request["custom_id"], "N1")
        self.assertEqual(request["method"], "POST")
        self.assertEqual(request["url"], "/v1/chat/completions")
        self.assertEqual(request["body"]["model"], "qwen3.5-flash")
        self.assertFalse(request["body"]["enable_thinking"])
        self.assertEqual(request["body"]["response_format"], {"type": "json_object"})

    def test_prepare_aliyun_batch_input_skips_cached_news(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            output_path = root / "labels.jsonl"
            input_path = root / "input.jsonl"
            write_jsonl(
                data_dir / "news.jsonl",
                [
                    {"news_id": "N1", "title": "Cached", "abstract": "Already labeled"},
                    {"news_id": "N2", "title": "Fresh", "abstract": "Needs labeling"},
                ],
            )
            with (data_dir / "news_frequency.json").open("w", encoding="utf-8") as file:
                json.dump({"N1": 10, "N2": 5}, file)
            cached = validate_value_payload(
                "N1",
                {
                    "scores": {
                        "importance": 1,
                        "prominence": 1,
                        "conflict": 1,
                        "novelty": 1,
                        "human_interest": 1,
                    },
                    "reason": "已缓存",
                },
            )
            write_jsonl(output_path, [asdict(cached)])

            stats = prepare_aliyun_batch_input(
                data_dir=data_dir,
                output_path=output_path,
                input_path=input_path,
                max_news=2,
            )
            rows = list(read_jsonl(input_path))

        self.assertEqual(stats["request_count"], 1)
        self.assertEqual(stats["cached_in_target"], 1)
        self.assertEqual(rows[0]["custom_id"], "N2")

    def test_prepare_aliyun_batch_input_can_use_sample_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            output_path = root / "plus_labels.jsonl"
            input_path = root / "input.jsonl"
            sample_path = root / "sample.jsonl"
            write_jsonl(
                data_dir / "news.jsonl",
                [
                    {"news_id": "N1", "title": "Outside sample", "abstract": ""},
                    {"news_id": "N2", "title": "Needs plus check", "abstract": ""},
                    {"news_id": "N3", "title": "Cached plus check", "abstract": ""},
                ],
            )
            write_jsonl(sample_path, [{"news_id": "N2"}, {"news_id": "N3"}])
            cached = validate_value_payload(
                "N3",
                {
                    "scores": {
                        "importance": 1,
                        "prominence": 1,
                        "conflict": 1,
                        "novelty": 1,
                        "human_interest": 1,
                    },
                    "reason": "plus 已缓存",
                },
            )
            write_jsonl(output_path, [asdict(cached)])

            stats = prepare_aliyun_batch_input(
                data_dir=data_dir,
                output_path=output_path,
                input_path=input_path,
                max_news=3000,
                model="qwen3.5-plus",
                sample_path=sample_path,
            )
            rows = list(read_jsonl(input_path))

        self.assertEqual(stats["target"], 2)
        self.assertEqual(stats["request_count"], 1)
        self.assertEqual(stats["cached_in_target"], 1)
        self.assertEqual(rows[0]["custom_id"], "N2")
        self.assertEqual(rows[0]["body"]["model"], "qwen3.5-plus")

    def test_merge_aliyun_batch_results_appends_valid_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_path = root / "result.jsonl"
            output_path = root / "labels.jsonl"
            invalid_path = root / "invalid.jsonl"
            payload = {
                "scores": {
                    "importance": 2,
                    "prominence": 1,
                    "conflict": 3,
                    "novelty": 1,
                    "human_interest": 0,
                },
                "reason": "有效结果",
            }
            write_jsonl(
                result_path,
                [
                    {
                        "custom_id": "N1",
                        "response": {
                            "status_code": 200,
                            "body": {"choices": [{"message": {"content": json.dumps(payload)}}]},
                        },
                    }
                ],
            )

            stats = merge_aliyun_batch_results(result_path, output_path, invalid_path)
            labels = list(read_jsonl(output_path))
            invalid = list(read_jsonl(invalid_path))

        self.assertEqual(stats["created"], 1)
        self.assertEqual(stats["invalid"], 0)
        self.assertEqual(labels[0]["news_id"], "N1")
        self.assertEqual(labels[0]["total"], 7)
        self.assertEqual(invalid, [])

    def test_merge_aliyun_batch_results_records_invalid_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_path = root / "result.jsonl"
            output_path = root / "labels.jsonl"
            invalid_path = root / "invalid.jsonl"
            payload = {
                "scores": {
                    "importance": 4,
                    "prominence": 1,
                    "conflict": 1,
                    "novelty": 1,
                    "human_interest": 1,
                },
                "reason": "非法分数",
            }
            write_jsonl(
                result_path,
                [
                    {
                        "custom_id": "N1",
                        "response": {
                            "status_code": 200,
                            "body": {"choices": [{"message": {"content": json.dumps(payload)}}]},
                        },
                    }
                ],
            )

            stats = merge_aliyun_batch_results(result_path, output_path, invalid_path)
            invalid = list(read_jsonl(invalid_path))

        self.assertEqual(stats["created"], 0)
        self.assertEqual(stats["invalid"], 1)
        self.assertFalse(output_path.exists())
        self.assertEqual(invalid[0]["custom_id"], "N1")


if __name__ == "__main__":
    unittest.main()
