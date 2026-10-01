"""Build retrievers / pipeline from an index directory produced by scripts/build_index.py."""
from __future__ import annotations

from pathlib import Path

from .config import Settings
from .data import Article, Chunk, load_articles, load_chunks


class IndexBundle:
    """Lazily loads the pieces of an index directory and caches them."""

    def __init__(self, index_dir: str | Path, device: str | None = None):
        self.index_dir = Path(index_dir)
        self.device = device
        self._cache: dict[str, object] = {}

    def _get(self, key, build):
        if key not in self._cache:
            self._cache[key] = build()
        return self._cache[key]

    @property
    def chunks(self) -> list[Chunk]:
        return self._get("chunks", lambda: load_chunks(self.index_dir / "chunks.jsonl"))

    @property
    def articles(self) -> dict[str, Article]:
        return self._get("articles", lambda: load_articles(self.index_dir / "articles.jsonl"))

    def retriever(self, method: str):
        from .retrieval import BM25Retriever, DenseRetriever, HybridRetriever

        if method == "bm25":
            return self._get("bm25", lambda: BM25Retriever.load(self.index_dir, self.chunks))
        if method == "dense":
            return self._get("dense", lambda: DenseRetriever.load(self.index_dir, self.chunks, self.device))
        if method == "hybrid":
            return self._get("hybrid", lambda: HybridRetriever([self.retriever("bm25"), self.retriever("dense")]))
        raise ValueError(f"Unknown retriever '{method}' (expected bm25, dense or hybrid)")

    def reranker(self, model_name: str):
        from .retrieval import CrossEncoderReranker

        return self._get(f"reranker:{model_name}", lambda: CrossEncoderReranker(model_name, self.device))


def build_pipeline(settings: Settings, bundle: IndexBundle | None = None):
    from .pipeline import RAGPipeline

    bundle = bundle or IndexBundle(settings.index_dir, settings.device)
    if settings.llm_backend == "hf":
        from .generator import HFGenerator

        generator = HFGenerator(settings.llm_model, settings.llm_adapter)
    elif settings.llm_backend == "openai":
        from .generator import OpenAICompatibleGenerator

        generator = OpenAICompatibleGenerator(settings.llm_base_url, settings.llm_model, settings.llm_api_key)
    else:
        raise ValueError(f"Unknown LLM_BACKEND '{settings.llm_backend}' (expected openai or hf)")
    return RAGPipeline(
        retriever=bundle.retriever(settings.retriever),
        generator=generator,
        articles=bundle.articles,
        reranker=bundle.reranker(settings.reranker_model) if settings.use_reranker else None,
        candidates=settings.candidates,
        context_k=settings.context_k,
        max_context_chars=settings.max_context_chars,
        refusal_threshold=settings.refusal_threshold if settings.use_reranker else None,
    )
