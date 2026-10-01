"""Refusal gate: refuse without calling the LLM when retrieval support is too weak.

The gate score of a question is the reranker score of its best context
article. Below a threshold tuned on held-out questions, the system answers
with the fixed refusal sentence instead of generating. This moves the
"is there a legal basis at all?" decision from the 3B generator, which tends
to answer anyway, to the cross-encoder, which was trained for relevance.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def sweep_thresholds(scores: Sequence[float], answerable: Sequence[bool]) -> list[dict]:
    """Gate quality at every candidate threshold.

    keep_answerable: share of answerable questions that pass the gate (go to the LLM).
    block_negative:  share of unanswerable questions the gate refuses.
    """
    scores = np.asarray(scores, dtype=float)
    answerable = np.asarray(answerable, dtype=bool)
    n_pos, n_neg = answerable.sum(), (~answerable).sum()
    if not n_pos or not n_neg:
        raise ValueError("need both answerable and unanswerable questions to tune the gate")

    candidates = np.unique(np.concatenate([[0.0], scores, [np.inf]]))
    points = []
    for t in candidates:
        passed = scores >= t
        keep = (passed & answerable).sum() / n_pos
        block = (~passed & ~answerable).sum() / n_neg
        points.append({"threshold": float(t), "keep_answerable": float(keep), "block_negative": float(block),
                       "balanced_accuracy": float((keep + block) / 2)})
    return points


def choose_threshold(points: Sequence[dict], min_keep: float | None = None) -> dict:
    """Best balanced accuracy, or (with ``min_keep``) the strictest gate that still
    lets at least ``min_keep`` of answerable questions through."""
    if min_keep is None:
        return max(points, key=lambda p: (p["balanced_accuracy"], -p["threshold"]))
    feasible = [p for p in points if p["keep_answerable"] >= min_keep]
    return max(feasible, key=lambda p: (p["block_negative"], -p["threshold"]))
