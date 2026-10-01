import pytest

from legalrag.gate import choose_threshold, sweep_thresholds
from legalrag.pipeline import RAGPipeline
from legalrag.prompts import REFUSAL, ContextDoc, build_messages, parse_user_message
from legalrag.retrieval import BM25Retriever


def test_parse_user_message_inverts_build_messages():
    docs = [
        ContextDoc("a+1", "Điều 1, A", "Điều 1. Tiêu đề\nKhoản 1.\n\nKhoản 2 có dòng trống ở trên."),
        ContextDoc("b+2", "Điều 2, B", "Điều 2. Khác\nNội dung [1] có ngoặc vuông."),
    ]
    user = build_messages("Hỏi gì vậy?", docs)[1]["content"]
    question, texts = parse_user_message(user, len(docs))
    assert question == "Hỏi gì vậy?"
    assert texts == [d.text for d in docs]


def test_sweep_and_choose_threshold():
    scores = [0.9, 0.8, 0.7, 0.2, 0.1, 0.6]
    answerable = [True, True, True, False, False, False]
    points = sweep_thresholds(scores, answerable)
    best = choose_threshold(points)
    assert best["balanced_accuracy"] == pytest.approx(1.0)
    assert 0.6 < best["threshold"] <= 0.7
    # demanding that every answerable question passes forces a looser gate
    loose = choose_threshold(points, min_keep=1.0)
    assert loose["keep_answerable"] == 1.0 and loose["threshold"] <= 0.7


class _Reranker:
    def __init__(self, score):
        self._score = score

    def rerank(self, query, hits, k):
        from dataclasses import replace
        return [replace(h, score=self._score) for h in hits[:k]]


class _LLM:
    calls = 0

    def generate(self, messages):
        _LLM.calls += 1
        return "Trả lời [1]."


def test_pipeline_gate_skips_llm_below_threshold(articles, chunks):
    bm25 = BM25Retriever().fit(chunks)
    _LLM.calls = 0
    low = RAGPipeline(bm25, _LLM(), articles, reranker=_Reranker(0.05), refusal_threshold=0.3)
    result = low.answer("không đội mũ bảo hiểm")
    assert result.gated and result.refused and result.answer == REFUSAL and _LLM.calls == 0
    assert result.top_score == pytest.approx(0.05)

    high = RAGPipeline(bm25, _LLM(), articles, reranker=_Reranker(0.9), refusal_threshold=0.3)
    result = high.answer("không đội mũ bảo hiểm")
    assert not result.gated and result.answer == "Trả lời [1]." and _LLM.calls == 1


def test_gate_requires_reranker(articles, chunks):
    with pytest.raises(ValueError):
        RAGPipeline(BM25Retriever().fit(chunks), _LLM(), articles, refusal_threshold=0.3)
