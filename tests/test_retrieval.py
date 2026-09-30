from legalrag.data import Article, Chunk, chunk_article
from legalrag.retrieval import BM25Retriever, Hit, HybridRetriever, reciprocal_rank_fusion
from legalrag.text import tokenize


def test_tokenize_normalizes_unicode_and_case():
    precomposed = "Luật"
    decomposed = "Luật"  # same word written with combining marks
    assert tokenize(precomposed) == tokenize(decomposed) == ["luật"]
    assert tokenize("xử phạt vi phạm", bigrams=True) == ["xử", "phạt", "vi", "phạm", "xử_phạt", "phạt_vi", "vi_phạm"]


def test_bm25_ranks_relevant_article_first(chunks):
    bm25 = BM25Retriever().fit(chunks)
    hits = bm25.search("không đội mũ bảo hiểm bị phạt", k=2)
    assert hits[0].article_id == "100/2019/nđ-cp+6"

    hits = bm25.search("báo trước bao nhiêu ngày khi nghỉ việc chấm dứt hợp đồng lao động", k=1)
    assert hits[0].article_id == "45/2019/qh14+35"


def test_bm25_unknown_terms_return_nothing(chunks):
    assert BM25Retriever().fit(chunks).search("zzzz qqqq", k=3) == []


def test_bm25_save_load_roundtrip(tmp_path, chunks):
    bm25 = BM25Retriever(bigrams=True).fit(chunks)
    bm25.save(tmp_path)
    loaded = BM25Retriever.load(tmp_path, chunks)
    query = "mũ bảo hiểm"
    assert [h.article_id for h in loaded.search(query, 3)] == [h.article_id for h in bm25.search(query, 3)]


def test_long_article_returns_once_with_matching_chunk():
    filler = "\n".join(f"Khoản {i}. quy định chung về thủ tục hành chính" for i in range(60))
    article = Article("x+1", "Điều 1. Dài", filler + "\nKhoản cuối. mức phạt đối với hành vi đua xe trái phép")
    other = Article("y+1", "Điều 1. Khác", "quy định về đê điều")
    chunks = chunk_article(article, max_chars=400, overlap_chars=50) + chunk_article(other)
    hits = BM25Retriever().fit(chunks).search("đua xe trái phép", k=5)
    assert [h.article_id for h in hits].count("x+1") == 1
    assert "đua xe" in hits[0].text


def _hit(aid):
    return Hit(aid, 0.0, f"{aid}#0", aid)


def test_rrf_rewards_agreement():
    fused = reciprocal_rank_fusion([[_hit("a"), _hit("b"), _hit("c")], [_hit("b"), _hit("d"), _hit("a")]], k=4)
    assert [h.article_id for h in fused][:2] in (["a", "b"], ["b", "a"])
    assert {h.article_id for h in fused} == {"a", "b", "c", "d"}


class _Static:
    def __init__(self, ids):
        self.ids = ids

    def batch_search(self, queries, k):
        return [[_hit(a) for a in self.ids[:k]] for _ in queries]


def test_hybrid_fuses_retrievers():
    hybrid = HybridRetriever([_Static(["a", "b"]), _Static(["b", "c"])], depth=10)
    assert hybrid.search("q", 3)[0].article_id == "b"
