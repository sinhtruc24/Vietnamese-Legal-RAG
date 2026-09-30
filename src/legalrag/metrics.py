"""Ranking metrics for retrieval evaluation."""
from __future__ import annotations

import math
from typing import Mapping, Sequence


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def mrr_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    for rank, doc in enumerate(ranked[:k], start=1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(rank + 1) for rank, doc in enumerate(ranked[:k], start=1) if doc in relevant)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def evaluate_run(
    run: Mapping[str, Sequence[str]],
    qrels: Mapping[str, set[str]],
    ks: Sequence[int] = (1, 5, 10),
) -> dict[str, float]:
    """Average metrics over all queries in ``qrels`` (a missing run counts as empty)."""
    totals: dict[str, float] = {}
    for qid, relevant in qrels.items():
        ranked = list(run.get(qid, []))
        for k in ks:
            totals[f"recall@{k}"] = totals.get(f"recall@{k}", 0.0) + recall_at_k(ranked, relevant, k)
        kmax = max(ks)
        totals[f"mrr@{kmax}"] = totals.get(f"mrr@{kmax}", 0.0) + mrr_at_k(ranked, relevant, kmax)
        totals[f"ndcg@{kmax}"] = totals.get(f"ndcg@{kmax}", 0.0) + ndcg_at_k(ranked, relevant, kmax)
    n = max(len(qrels), 1)
    return {name: value / n for name, value in totals.items()}
