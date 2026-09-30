import json

from legalrag.data import Article, LegalDataset, chunk_article, format_citation, subsample_corpus


def test_format_citation_uses_title_and_uppercases_law_id():
    assert format_citation("100/2019/nđ-cp+5", "Điều 5. Xử phạt") == "Điều 5, 100/2019/NĐ-CP"
    assert format_citation("47/2011/tt-bca+7") == "Điều 7, 47/2011/TT-BCA"


def test_short_article_is_a_single_chunk():
    article = Article("a+1", "Điều 1. Tiêu đề", "Nội dung ngắn.")
    chunks = chunk_article(article)
    assert [c.id for c in chunks] == ["a+1#0"]
    assert chunks[0].text == "Điều 1. Tiêu đề\nNội dung ngắn."


def test_long_article_is_split_with_title_and_overlap():
    lines = [f"Khoản {i}. " + "nội dung " * 20 for i in range(1, 21)]
    article = Article("a+2", "Điều 2. Dài", "\n".join(lines))
    chunks = chunk_article(article, max_chars=500, overlap_chars=200)

    assert len(chunks) > 1
    assert all(c.article_id == "a+2" and c.text.startswith("Điều 2. Dài\n") for c in chunks)
    assert all(len(c.text) <= 500 + 200 + len("Điều 2. Dài\n") for c in chunks)
    # consecutive chunks share at least one line
    for prev, nxt in zip(chunks, chunks[1:]):
        assert set(prev.text.split("\n")[1:]) & set(nxt.text.split("\n")[1:])
    # nothing is lost
    covered = {line for c in chunks for line in c.text.split("\n")[1:]}
    assert {l.strip() for l in lines} <= covered


def test_single_huge_line_is_hard_split():
    article = Article("a+3", "Điều 3.", "x" * 5000)
    chunks = chunk_article(article, max_chars=1000, overlap_chars=100)
    assert len(chunks) >= 5
    assert all(len(c.text) <= 1000 + 100 + len("Điều 3.\n") for c in chunks)


def test_subsample_keeps_gold(articles):
    subset = subsample_corpus(articles, ["45/2019/qh14+35", "missing"], size=2, seed=1)
    assert "45/2019/qh14+35" in subset and len(subset) == 2


def test_dataset_loading(tmp_path):
    def dump(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")

    dump(tmp_path / "corpus.jsonl", [{"_id": "l+1", "title": "Điều 1.", "text": "abc"}])
    dump(tmp_path / "queries.jsonl", [{"_id": "q1", "text": "câu hỏi?"}, {"_id": "q2", "text": "khác?"}])
    dump(tmp_path / "qrels/test.jsonl", [
        {"query-id": "q1", "corpus-id": "l+1", "score": 1},
        {"query-id": "q2", "corpus-id": "not-in-corpus", "score": 1},
    ])
    ds = LegalDataset.load(tmp_path)
    assert ds.qrels("test") == {"q1": {"l+1"}}
    assert ds.split_queries("test") == {"q1": "câu hỏi?"}
