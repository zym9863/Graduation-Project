from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gp_newsrec.mind import parse_behavior_line, parse_news_line


class MindParsingTest(unittest.TestCase):
    def test_parse_news_line(self) -> None:
        line = (
            "N1\tnews\tworld\tTitle\tAbstract text\t"
            "https://example.com\t[]\t[]\n"
        )
        record = parse_news_line(line)
        self.assertEqual(record.news_id, "N1")
        self.assertEqual(record.category, "news")
        self.assertIn("Title", record.text_for_encoder)

    def test_parse_behavior_line(self) -> None:
        line = "1\tU1\t11/11/2019 9:05:58 AM\tN1 N2\tN3-1 N4-0\n"
        record = parse_behavior_line(line)
        self.assertEqual(record.user_id, "U1")
        self.assertEqual(record.history, ["N1", "N2"])
        self.assertEqual(record.impressions, [("N3", 1), ("N4", 0)])


if __name__ == "__main__":
    unittest.main()
