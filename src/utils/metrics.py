from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import roc_auc_score


def _dcg(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    if labels.size == 0:
        return 0.0
    order = np.argsort(scores)[::-1][:k]
    ranked = labels[order]
    discounts = 1.0 / np.log2(np.arange(2, ranked.size + 2))
    return float(np.sum(ranked * discounts))


def mean_reciprocal_rank(labels: Sequence[int], scores: Sequence[float]) -> float:
    label_array = np.asarray(labels)
    score_array = np.asarray(scores)
    order = np.argsort(score_array)[::-1]
    ranked = label_array[order]
    positives = np.where(ranked == 1)[0]
    if positives.size == 0:
        return 0.0
    return float(1.0 / (positives[0] + 1))


def ndcg_at_k(labels: Sequence[int], scores: Sequence[float], k: int) -> float:
    label_array = np.asarray(labels)
    score_array = np.asarray(scores)
    ideal = _dcg(label_array, label_array, k)
    if ideal == 0.0:
        return 0.0
    return _dcg(label_array, score_array, k) / ideal


def group_auc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    label_array = np.asarray(labels)
    if np.unique(label_array).size < 2:
        return None
    return float(roc_auc_score(label_array, np.asarray(scores)))


def compute_ranking_metrics(
    label_groups: Sequence[Sequence[int]],
    score_groups: Sequence[Sequence[float]],
) -> dict[str, float]:
    auc_values: list[float] = []
    mrr_values: list[float] = []
    ndcg5_values: list[float] = []
    ndcg10_values: list[float] = []

    for labels, scores in zip(label_groups, score_groups, strict=True):
        auc = group_auc(labels, scores)
        if auc is not None:
            auc_values.append(auc)
        mrr_values.append(mean_reciprocal_rank(labels, scores))
        ndcg5_values.append(ndcg_at_k(labels, scores, 5))
        ndcg10_values.append(ndcg_at_k(labels, scores, 10))

    def _safe_mean(values: Sequence[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    return {
        "auc": _safe_mean(auc_values),
        "mrr": _safe_mean(mrr_values),
        "ndcg@5": _safe_mean(ndcg5_values),
        "ndcg@10": _safe_mean(ndcg10_values),
    }