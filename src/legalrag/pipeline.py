"""End-to-end RAG: retrieve -> (rerank) -> build cited context -> generate."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from .data import Article, format_citation
from .generator import Generator
from .prompts import ContextDoc, build_messages, is_refusal, parse_citations
from .retrieval import CrossEncoderReranker, Hit, Retriever


def build_context_docs(
    hits: Sequence[Hit], articles: Mapping[str, Article], max_chars: int = 1500
) -> list[ContextDoc]:
    """Give the LLM the whole article when it is short, otherwise the matched chunk."""
    docs = []
    for hit in hits:
        article = articles.get(hit.article_id)
        if article is not None:
            full = f"{article.title}\n{article.text}".strip()
            text = full if len(full) <= max_chars else hit.text[:max_chars]
            citation = article.citation
        else:
            text, citation = hit.text[:max_chars], format_citation(hit.article_id)
        docs.append(ContextDoc(hit.article_id, citation, text))
    return docs


@dataclass
class RAGAnswer:
    question: str
    answer: str
    refused: bool
    citations: list[dict]
    contexts: list[ContextDoc]
    timings_ms: dict[str, float] = field(default_factory=dict)


class RAGPipeline:
    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
        articles: Mapping[str, Article],
        reranker: CrossEncoderReranker | None = None,
        candidates: int = 30,
        context_k: int = 3,
        max_context_chars: int = 1500,
    ):
        self.retriever = retriever
        self.generator = generator
        self.articles = articles
        self.reranker = reranker
        self.candidates = candidates
        self.context_k = context_k
        self.max_context_chars = max_context_chars

    def retrieve(self, question: str, k: int | None = None, timings: dict | None = None) -> list[Hit]:
        k = k or self.context_k
        t0 = time.perf_counter()
        hits = self.retriever.search(question, max(k, self.candidates) if self.reranker else k)
        t1 = time.perf_counter()
        if self.reranker:
            hits = self.reranker.rerank(question, hits, k)
        t2 = time.perf_counter()
        if timings is not None:
            timings["retrieve"] = (t1 - t0) * 1000
            timings["rerank"] = (t2 - t1) * 1000
        return hits[:k]

    def answer(self, question: str) -> RAGAnswer:
        timings: dict[str, float] = {}
        hits = self.retrieve(question, timings=timings)
        docs = build_context_docs(hits, self.articles, self.max_context_chars)

        t0 = time.perf_counter()
        text = self.generator.generate(build_messages(question, docs))
        timings["generate"] = (time.perf_counter() - t0) * 1000
        timings["total"] = sum(timings.values())

        valid, _ = parse_citations(text, len(docs))
        citations = [
            {"index": i, "article_id": docs[i - 1].article_id, "citation": docs[i - 1].citation}
            for i in valid
        ]
        return RAGAnswer(question, text, is_refusal(text), citations, docs, timings)
