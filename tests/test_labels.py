from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gp_newsrec.constants import VALUE_DIMENSIONS
from gp_newsrec.labels import build_value_prompt, validate_value_payload


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


if __name__ == "__main__":
    unittest.main()
