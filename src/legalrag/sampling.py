"""Controlled contexts ("oracle" gold + hard negatives) for SFT data and generation eval."""
from __future__ import annotations

import random
from typing import Mapping

from .data import Article
from .pipeline import build_context_docs
from .prompts import build_messages
from .retrieval import Hit, Retriever


def sample_context(
    qid: str,
    question: str,
    gold: set[str],
    articles: Mapping[str, Article],
    retriever: Retriever,
    n_contexts: int = 3,
    negative_pool: int = 10,
    neg_ratio: float = 0.15,
    max_chars: int = 1500,
    seed: int = 13,
) -> dict:
    """Return a prompt whose context is gold article(s) + hard negatives (shuffled),
    or, with probability ``neg_ratio``, hard negatives only ("negative" sample).

    Seeded per query, so the same query gets the same context for every model
    being compared.
    """
    rng = random.Random(f"{seed}-{qid}")
    hits = retriever.search(question, negative_pool)
    by_id = {h.article_id: h for h in hits}
    negatives = [h for h in hits if h.article_id not in gold]

    if rng.random() < neg_ratio:
        kind, chosen = "negative", negatives[:n_contexts]
    else:
        kind = "answerable"
        gold_hits = []
        for aid in sorted(gold)[:n_contexts]:
            if aid in by_id:
                gold_hits.append(by_id[aid])
            else:  # retriever missed it: use the start of the article
                article = articles[aid]
                gold_hits.append(Hit(aid, 0.0, f"{aid}#0", f"{article.title}\n{article.text}"))
        chosen = gold_hits + negatives[: n_contexts - len(gold_hits)]
        rng.shuffle(chosen)

    docs = build_context_docs(chosen, articles, max_chars)
    return {
        "id": qid,
        "type": kind,
        "gold": [i for i, d in enumerate(docs, start=1) if d.article_id in gold],
        "n_docs": len(docs),
        "prompt": build_messages(question, docs),
    }
