import math

import pytest

from legalrag.metrics import evaluate_run, mrr_at_k, ndcg_at_k, recall_at_k
from legalrag.pipeline import RAGPipeline
from legalrag.prompts import REFUSAL, build_messages, clean_answer, is_refusal, parse_citations, ContextDoc
from legalrag.retrieval import BM25Retriever
from legalrag.sampling import sample_context


def test_ranking_metrics():
    ranked = ["x", "a", "y", "b"]
    assert recall_at_k(ranked, {"a", "b"}, 2) == 0.5
    assert mrr_at_k(ranked, {"a", "b"}, 10) == 0.5
    expected = (1 / math.log2(3) + 1 / math.log2(5)) / (1 + 1 / math.log2(3))
    assert ndcg_at_k(ranked, {"a", "b"}, 10) == pytest.approx(expected)


def test_evaluate_run_counts_missing_queries_as_zero():
    metrics = evaluate_run({"q1": ["a"]}, {"q1": {"a"}, "q2": {"b"}}, ks=(1,))
    assert metrics["recall@1"] == 0.5 and metrics["mrr@1"] == 0.5


def test_parse_citations_splits_valid_and_invalid():
    valid, invalid = parse_citations("Theo [2] và [1], cũng như [2] và [7].", n_docs=3)
    assert valid == [2, 1] and invalid == [7]


def test_prompt_and_refusal_helpers():
    docs = [ContextDoc("a+1", "Điều 1, A", "nội dung")]
    messages = build_messages("Hỏi gì?", docs)
    assert messages[0]["role"] == "system" and REFUSAL in messages[0]["content"]
    assert "[1] (Điều 1, A)\nnội dung" in messages[1]["content"]
    assert is_refusal(REFUSAL) and not is_refusal("Có, theo [1].")
    assert clean_answer("<think>\nsuy nghĩ\n</think>\nĐáp án [1]") == "Đáp án [1]"


class _FakeLLM:
    def __init__(self, answer):
        self.answer = answer
        self.messages = None

    def generate(self, messages):
        self.messages = messages
        return self.answer


def test_pipeline_end_to_end_with_fake_llm(articles, chunks):
    llm = _FakeLLM("Bị phạt tiền [1]. Xem thêm [9].")
    pipeline = RAGPipeline(BM25Retriever().fit(chunks), llm, articles, context_k=2)
    result = pipeline.answer("Không đội mũ bảo hiểm bị phạt thế nào?")

    assert result.contexts[0].article_id == "100/2019/nđ-cp+6"
    assert result.citations == [{"index": 1, "article_id": "100/2019/nđ-cp+6",
                                  "citation": "Điều 6, 100/2019/NĐ-CP"}]
    assert not result.refused
    assert "mũ bảo hiểm" in llm.messages[1]["content"]
    assert set(result.timings_ms) >= {"retrieve", "generate", "total"}


def test_sample_context_is_deterministic_and_contains_gold(articles, chunks):
    bm25 = BM25Retriever().fit(chunks)
    gold = {"45/2019/qh14+35"}
    args = dict(qid="q1", question="nghỉ việc báo trước", gold=gold, articles=articles,
                retriever=bm25, n_contexts=3, neg_ratio=0.0)
    first, second = sample_context(**args), sample_context(**args)
    assert first == second
    assert first["type"] == "answerable" and len(first["gold"]) == 1

    negative = sample_context(**{**args, "neg_ratio": 1.0})
    assert negative["type"] == "negative" and negative["gold"] == []


def test_answer_to_dict_is_json_ready(articles, chunks):
    import json

    pipeline = RAGPipeline(BM25Retriever().fit(chunks), _FakeLLM("Bị phạt [1]."), articles, context_k=2)
    data = pipeline.answer("không đội mũ bảo hiểm").to_dict()
    assert set(data) >= {"answer", "refused", "gated", "top_score", "citations", "contexts", "timings_ms"}
    assert data["contexts"][0]["article_id"] == "100/2019/nđ-cp+6"
    json.dumps(data, ensure_ascii=False)  # must be serialisable for the API
