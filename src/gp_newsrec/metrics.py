from __future__ import annotations

import math
from typing import Iterable, Sequence


def group_auc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    positives = [score for label, score in zip(labels, scores, strict=True) if label == 1]
    negatives = [score for label, score in zip(labels, scores, strict=True) if label == 0]
    if not positives or not negatives:
        return None
    wins = 0.0
    total = 0
    for pos in positives:
        for neg in negatives:
            total += 1
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / total


def mrr_score(labels: Sequence[int], scores: Sequence[float]) -> float:
    ranked = sorted(zip(labels, scores, strict=True), key=lambda item: item[1], reverse=True)
    for rank, (label, _) in enumerate(ranked, start=1):
        if label == 1:
            return 1.0 / rank
    return 0.0


def ndcg_score(labels: Sequence[int], scores: Sequence[float], k: int) -> float:
    ranked = sorted(zip(labels, scores, strict=True), key=lambda item: item[1], reverse=True)[:k]
    dcg = sum((2**label - 1) / math.log2(idx + 2) for idx, (label, _) in enumerate(ranked))
    ideal = sorted(labels, reverse=True)[:k]
    idcg = sum((2**label - 1) / math.log2(idx + 2) for idx, label in enumerate(ideal))
    return 0.0 if idcg == 0 else dcg / idcg


def aggregate_ranking_metrics(groups: Iterable[tuple[Sequence[int], Sequence[float]]]) -> dict[str, float]:
    values = {"auc": [], "mrr": [], "ndcg5": [], "ndcg10": []}
    for labels, scores in groups:
        auc = group_auc(labels, scores)
        if auc is not None:
            values["auc"].append(auc)
        values["mrr"].append(mrr_score(labels, scores))
        values["ndcg5"].append(ndcg_score(labels, scores, 5))
        values["ndcg10"].append(ndcg_score(labels, scores, 10))
    return {
        name: (sum(metric_values) / len(metric_values) if metric_values else 0.0)
        for name, metric_values in values.items()
    }
