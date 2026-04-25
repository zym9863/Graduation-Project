from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gp_newsrec.metrics import aggregate_ranking_metrics, group_auc, mrr_score, ndcg_score


class MetricsTest(unittest.TestCase):
    def test_group_auc_pairwise(self) -> None:
        self.assertEqual(group_auc([1, 0, 0], [0.9, 0.2, 0.1]), 1.0)
        self.assertEqual(group_auc([1, 0], [0.5, 0.5]), 0.5)

    def test_ranking_metrics(self) -> None:
        labels = [0, 1, 0]
        scores = [0.2, 0.9, 0.1]
        self.assertEqual(mrr_score(labels, scores), 1.0)
        self.assertEqual(ndcg_score(labels, scores, 2), 1.0)
        metrics = aggregate_ranking_metrics([(labels, scores)])
        self.assertEqual(metrics["auc"], 1.0)
        self.assertEqual(metrics["mrr"], 1.0)


if __name__ == "__main__":
    unittest.main()
