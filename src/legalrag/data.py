"""Loading the Zalo AI 2021 Legal Text Retrieval dataset (MTEB/BEIR layout).

Expected files inside ``data_dir``::

    corpus.jsonl        {"_id": "47/2011/tt-bca+7", "title": "Điều 7. ...", "text": "..."}
    queries.jsonl       {"_id": "<hash>", "text": "Công an xã xử phạt ...?"}
    qrels/train.jsonl   {"query-id": "<hash>", "corpus-id": "47/2011/tt-bca+7", "score": 1}
    qrels/test.jsonl
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator


@dataclass(frozen=True)
class Article:
    id: str
    title: str
    text: str

    @property
    def law_id(self) -> str:
        return self.id.rpartition("+")[0]

    @property
    def citation(self) -> str:
        return format_citation(self.id, self.title)


@dataclass(frozen=True)
class Chunk:
    id: str
    article_id: str
    text: str


def format_citation(article_id: str, title: str = "") -> str:
    """'47/2011/tt-bca+7', 'Điều 7. ...' -> 'Điều 7, 47/2011/TT-BCA'."""
    law, _, number = article_id.rpartition("+")
    head = title.split(".", 1)[0].strip()
    label = head if head.lower().startswith("điều") else f"Điều {number}"
    return f"{label}, {law.upper()}"


# ---------------------------------------------------------------- io helpers
def read_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_articles(path: str | Path) -> dict[str, Article]:
    articles = {}
    for row in read_jsonl(path):
        article_id = row.get("_id", row.get("id"))
        articles[article_id] = Article(article_id, row.get("title", ""), row.get("text", ""))
    return articles


def save_articles(path: str | Path, articles: Iterable[Article]) -> None:
    write_jsonl(path, ({"_id": a.id, "title": a.title, "text": a.text} for a in articles))


def load_queries(path: str | Path) -> dict[str, str]:
    return {row.get("_id", row.get("id")): row["text"] for row in read_jsonl(path)}


def load_qrels(path: str | Path) -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = defaultdict(set)
    for row in read_jsonl(path):
        if float(row.get("score", 1)) > 0:
            qrels[row["query-id"]].add(row["corpus-id"])
    return dict(qrels)


def load_chunks(path: str | Path) -> list[Chunk]:
    return [Chunk(**row) for row in read_jsonl(path)]


def save_chunks(path: str | Path, chunks: Iterable[Chunk]) -> None:
    write_jsonl(path, (asdict(c) for c in chunks))


# ---------------------------------------------------------------- dataset
@dataclass
class LegalDataset:
    corpus: dict[str, Article]
    queries: dict[str, str]
    qrels_by_split: dict[str, dict[str, set[str]]]

    @classmethod
    def load(cls, data_dir: str | Path) -> "LegalDataset":
        data_dir = Path(data_dir)
        qrels = {
            p.stem: load_qrels(p) for p in sorted((data_dir / "qrels").glob("*.jsonl"))
        }
        return cls(
            corpus=load_articles(data_dir / "corpus.jsonl"),
            queries=load_queries(data_dir / "queries.jsonl"),
            qrels_by_split=qrels,
        )

    def qrels(self, split: str) -> dict[str, set[str]]:
        """Qrels of a split, restricted to queries whose gold articles exist in the corpus."""
        return {
            qid: gold
            for qid, gold in self.qrels_by_split[split].items()
            if qid in self.queries and gold & self.corpus.keys()
        }

    def split_queries(self, split: str) -> dict[str, str]:
        return {qid: self.queries[qid] for qid in self.qrels(split)}


def subsample_corpus(
    corpus: dict[str, Article], keep: Iterable[str], size: int, seed: int = 0
) -> dict[str, Article]:
    """Small corpus for fast iteration on CPU: all ``keep`` ids + random fillers.

    Scores on a sub-sampled corpus are optimistic; report numbers on the full one.
    """
    keep = [a for a in dict.fromkeys(keep) if a in corpus]
    keep_set = set(keep)
    rest = [a for a in corpus if a not in keep_set]
    rng = random.Random(seed)
    fill = rng.sample(rest, max(0, min(len(rest), size - len(keep))))
    return {a: corpus[a] for a in keep + fill}


# ---------------------------------------------------------------- chunking
def chunk_article(article: Article, max_chars: int = 1200, overlap_chars: int = 200) -> list[Chunk]:
    """Split a long article into overlapping windows along line (clause) boundaries.

    Most articles fit in one chunk, but some exceed 100k characters; embedding
    models would silently truncate them. Every chunk keeps the article title so
    it stays self-describing.
    """
    if overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be smaller than max_chars")
    header = article.title.strip()
    body = article.text.strip()
    if len(body) <= max_chars:
        return [Chunk(f"{article.id}#0", article.id, f"{header}\n{body}".strip())]

    pieces: list[str] = []
    for line in body.split("\n"):
        line = line.strip()
        while len(line) > max_chars:  # a single huge clause: hard split
            pieces.append(line[:max_chars])
            line = line[max_chars - overlap_chars:]
        if line:
            pieces.append(line)

    windows: list[list[str]] = []
    current: list[str] = []
    size = 0
    for piece in pieces:
        if current and size + len(piece) > max_chars:
            windows.append(current)
            tail: list[str] = []  # carry the last lines over as overlap
            tail_size = 0
            for prev in reversed(current):
                if tail_size + len(prev) > overlap_chars:
                    break
                tail.insert(0, prev)
                tail_size += len(prev) + 1
            current, size = tail, tail_size
        current.append(piece)
        size += len(piece) + 1
    if current:
        windows.append(current)

    return [
        Chunk(f"{article.id}#{i}", article.id, f"{header}\n" + "\n".join(w))
        for i, w in enumerate(windows)
    ]


def chunk_corpus(articles: Iterable[Article], max_chars: int = 1200, overlap_chars: int = 200) -> list[Chunk]:
    return [c for a in articles for c in chunk_article(a, max_chars, overlap_chars)]
