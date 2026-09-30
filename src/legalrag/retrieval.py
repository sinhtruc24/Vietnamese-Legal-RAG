"""Retrievers: BM25 (sparse), bi-encoder (dense), hybrid (RRF) and a cross-encoder reranker.

All retrievers score *chunks* and return *articles*: an article's score is the
score of its best chunk (max-pooling), and the matching chunk is kept so the
reranker and the LLM see the relevant part of long articles.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Protocol, Sequence

import numpy as np
from scipy import sparse

from .data import Chunk
from .text import tokenize

# How many chunks to pull per requested article before max-pooling, so that
# several chunks of one long article cannot crowd out other articles.
CHUNK_FANOUT = 4


@dataclass(frozen=True)
class Hit:
    article_id: str
    score: float
    chunk_id: str
    text: str


class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[Hit]: ...

    def batch_search(self, queries: Sequence[str], k: int) -> list[list[Hit]]: ...


def _collect_hits(chunks: Sequence[Chunk], ranked: Iterable[tuple[int, float]], k: int) -> list[Hit]:
    """Max-pool ranked (chunk_index, score) pairs into at most k article hits."""
    hits: list[Hit] = []
    seen: set[str] = set()
    for idx, score in ranked:
        chunk = chunks[idx]
        if chunk.article_id in seen:
            continue
        seen.add(chunk.article_id)
        hits.append(Hit(chunk.article_id, float(score), chunk.id, chunk.text))
        if len(hits) == k:
            break
    return hits


# ====================================================================== BM25
class BM25Retriever:
    """Okapi BM25 over a precomputed sparse (chunk x term) weight matrix.

    Scoring a query is a column slice + row sum, so the whole 60k-article
    corpus is searched in milliseconds without an external search engine.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75, bigrams: bool = False):
        self.k1, self.b, self.bigrams = k1, b, bigrams
        self.vocab: dict[str, int] = {}
        self.matrix: sparse.csc_matrix | None = None
        self.chunks: list[Chunk] = []

    def fit(self, chunks: Sequence[Chunk]) -> "BM25Retriever":
        self.chunks = list(chunks)
        # Build CSR arrays directly; per-entry Python lists would need GBs of RAM on the full corpus.
        indptr, indices, counts = [0], [], []
        for chunk in self.chunks:
            tf_chunk = Counter(tokenize(chunk.text, self.bigrams))
            indices.append(np.fromiter(
                (self.vocab.setdefault(t, len(self.vocab)) for t in tf_chunk),
                dtype=np.int32, count=len(tf_chunk),
            ))
            counts.append(np.fromiter(tf_chunk.values(), dtype=np.float32, count=len(tf_chunk)))
            indptr.append(indptr[-1] + len(tf_chunk))
        n_docs, n_terms = len(self.chunks), len(self.vocab)
        tf = sparse.csr_matrix(
            (
                np.concatenate(counts) if counts else np.zeros(0, np.float32),
                np.concatenate(indices) if indices else np.zeros(0, np.int32),
                np.asarray(indptr, dtype=np.int64),
            ),
            shape=(n_docs, n_terms),
        )

        doc_len = np.asarray(tf.sum(axis=1)).ravel()
        avg_len = doc_len.mean() if n_docs else 0.0
        df = np.bincount(tf.indices, minlength=n_terms)
        idf = np.log1p((n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)

        row_of_entry = np.repeat(np.arange(n_docs), np.diff(tf.indptr))
        norm = self.k1 * (1 - self.b + self.b * doc_len[row_of_entry] / max(avg_len, 1e-9))
        tf.data = idf[tf.indices] * tf.data * (self.k1 + 1) / (tf.data + norm)
        self.matrix = tf.tocsc()
        return self

    def scores(self, query: str) -> np.ndarray:
        term_ids = sorted({self.vocab[t] for t in tokenize(query, self.bigrams) if t in self.vocab})
        if not term_ids:
            return np.zeros(len(self.chunks), dtype=np.float32)
        return np.asarray(self.matrix[:, term_ids].sum(axis=1)).ravel()

    def search(self, query: str, k: int) -> list[Hit]:
        scores = self.scores(query)
        n = min(len(scores), k * CHUNK_FANOUT)
        if n == 0:
            return []
        top = np.argpartition(-scores, n - 1)[:n]
        top = top[np.argsort(-scores[top])]
        return _collect_hits(self.chunks, ((i, scores[i]) for i in top if scores[i] > 0), k)

    def batch_search(self, queries: Sequence[str], k: int) -> list[list[Hit]]:
        return [self.search(q, k) for q in queries]

    # ---------------------------------------------------------------- io
    def save(self, index_dir: str | Path) -> None:
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(index_dir / "bm25_matrix.npz", self.matrix)
        with open(index_dir / "bm25_meta.json", "w", encoding="utf-8") as f:
            json.dump({"k1": self.k1, "b": self.b, "bigrams": self.bigrams, "vocab": self.vocab}, f, ensure_ascii=False)

    @classmethod
    def load(cls, index_dir: str | Path, chunks: Sequence[Chunk]) -> "BM25Retriever":
        index_dir = Path(index_dir)
        with open(index_dir / "bm25_meta.json", encoding="utf-8") as f:
            meta = json.load(f)
        retriever = cls(meta["k1"], meta["b"], meta["bigrams"])
        retriever.vocab = meta["vocab"]
        retriever.matrix = sparse.load_npz(index_dir / "bm25_matrix.npz").tocsc()
        retriever.chunks = list(chunks)
        if retriever.matrix.shape[0] != len(retriever.chunks):
            raise ValueError("BM25 index does not match chunks.jsonl; rebuild the index")
        return retriever


# ===================================================================== dense
def _resolve_device(device: str | None) -> str:
    if device:
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class DenseRetriever:
    """Bi-encoder (default BAAI/bge-m3) + exact inner-product search in FAISS."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str | None = None,
        batch_size: int = 32,
        max_seq_length: int = 512,
        query_prefix: str = "",
        passage_prefix: str = "",
    ):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.device = _resolve_device(device)
        self.model = SentenceTransformer(model_name, device=self.device)
        if self.device.startswith("cuda"):
            self.model.half()
        self.model.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.query_prefix, self.passage_prefix = query_prefix, passage_prefix
        self.index = None
        self.chunks: list[Chunk] = []

    def _encode(self, texts: Sequence[str], prefix: str, progress: bool = False) -> np.ndarray:
        emb = self.model.encode(
            [prefix + t for t in texts],
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=progress,
        )
        return np.ascontiguousarray(emb, dtype=np.float32)

    def fit(self, chunks: Sequence[Chunk]) -> "DenseRetriever":
        import faiss

        self.chunks = list(chunks)
        # Encode longest-first: batches of similar length waste less padding.
        order = sorted(range(len(self.chunks)), key=lambda i: -len(self.chunks[i].text))
        emb = self._encode([self.chunks[i].text for i in order], self.passage_prefix, progress=True)
        vectors = np.empty_like(emb)
        vectors[order] = emb
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        return self

    def batch_search(self, queries: Sequence[str], k: int) -> list[list[Hit]]:
        emb = self._encode(queries, self.query_prefix)
        scores, ids = self.index.search(emb, min(k * CHUNK_FANOUT, len(self.chunks)))
        return [
            _collect_hits(self.chunks, ((int(i), s) for i, s in zip(row_ids, row_scores) if i >= 0), k)
            for row_ids, row_scores in zip(ids, scores)
        ]

    def search(self, query: str, k: int) -> list[Hit]:
        return self.batch_search([query], k)[0]

    def save(self, index_dir: str | Path) -> None:
        import faiss

        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_dir / "dense.faiss"))
        with open(index_dir / "dense_meta.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_name": self.model_name,
                    "max_seq_length": self.model.max_seq_length,
                    "query_prefix": self.query_prefix,
                    "passage_prefix": self.passage_prefix,
                },
                f,
                ensure_ascii=False,
            )

    @classmethod
    def load(cls, index_dir: str | Path, chunks: Sequence[Chunk], device: str | None = None) -> "DenseRetriever":
        import faiss

        index_dir = Path(index_dir)
        with open(index_dir / "dense_meta.json", encoding="utf-8") as f:
            meta = json.load(f)
        retriever = cls(device=device, **meta)
        retriever.index = faiss.read_index(str(index_dir / "dense.faiss"))
        retriever.chunks = list(chunks)
        if retriever.index.ntotal != len(retriever.chunks):
            raise ValueError("Dense index does not match chunks.jsonl; rebuild the index")
        return retriever


# ==================================================================== hybrid
def reciprocal_rank_fusion(runs: Sequence[Sequence[Hit]], k: int, rrf_k: int = 60) -> list[Hit]:
    """Fuse ranked lists by sum of 1 / (rrf_k + rank).

    RRF only uses ranks, so BM25 scores (unbounded) and cosine similarities
    (in [-1, 1]) can be combined without any score calibration.
    """
    fused: dict[str, float] = {}
    best: dict[str, Hit] = {}
    for run in runs:
        for rank, hit in enumerate(run, start=1):
            fused[hit.article_id] = fused.get(hit.article_id, 0.0) + 1.0 / (rrf_k + rank)
            best.setdefault(hit.article_id, hit)  # keep the chunk from the first run
    ranked = sorted(fused, key=fused.get, reverse=True)[:k]
    return [replace(best[a], score=fused[a]) for a in ranked]


class HybridRetriever:
    def __init__(self, retrievers: Sequence[Retriever], depth: int = 100, rrf_k: int = 60):
        self.retrievers = list(retrievers)
        self.depth, self.rrf_k = depth, rrf_k

    def batch_search(self, queries: Sequence[str], k: int) -> list[list[Hit]]:
        depth = max(k, self.depth)
        per_retriever = [r.batch_search(queries, depth) for r in self.retrievers]
        return [
            reciprocal_rank_fusion([runs[i] for runs in per_retriever], k, self.rrf_k)
            for i in range(len(queries))
        ]

    def search(self, query: str, k: int) -> list[Hit]:
        return self.batch_search([query], k)[0]


# ================================================================== reranker
class CrossEncoderReranker:
    """Re-scores (query, chunk) pairs jointly; slower but far more precise than a bi-encoder."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str | None = None,
                 max_length: int = 512, batch_size: int = 16):
        from sentence_transformers import CrossEncoder

        device = _resolve_device(device)
        self.model = CrossEncoder(model_name, max_length=max_length, device=device)
        if device.startswith("cuda"):
            self.model.model.half()
        self.batch_size = batch_size

    def rerank(self, query: str, hits: Sequence[Hit], k: int) -> list[Hit]:
        if not hits:
            return []
        scores = self.model.predict([(query, h.text) for h in hits], batch_size=self.batch_size)
        order = np.argsort(-np.asarray(scores))[:k]
        return [replace(hits[i], score=float(scores[i])) for i in order]
