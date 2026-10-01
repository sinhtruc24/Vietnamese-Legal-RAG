"""Runtime settings, overridable through environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    data_dir: Path = Path("data/zalo_legal")
    index_dir: Path = Path("indexes")

    retriever: str = "hybrid"  # bm25 | dense | hybrid
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    use_reranker: bool = True
    device: str | None = None  # None = auto (cuda if available)

    candidates: int = 20  # hits passed to the reranker (20 keeps ~all of the gain of 30, 30% faster)
    context_k: int = 3  # articles given to the LLM
    max_context_chars: int = 1500
    refusal_threshold: float | None = None  # reranker-score gate, see scripts/tune_refusal_gate.py

    llm_base_url: str = "http://localhost:8000/v1"
    llm_model: str = "Qwen/Qwen2.5-3B-Instruct"
    llm_api_key: str = "EMPTY"

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.environ.get
        return cls(
            data_dir=Path(env("DATA_DIR", str(cls.data_dir))),
            index_dir=Path(env("INDEX_DIR", str(cls.index_dir))),
            retriever=env("RETRIEVER", cls.retriever),
            embedding_model=env("EMBEDDING_MODEL", cls.embedding_model),
            reranker_model=env("RERANKER_MODEL", cls.reranker_model),
            use_reranker=_bool(env("USE_RERANKER", str(cls.use_reranker))),
            device=env("DEVICE") or None,
            candidates=int(env("CANDIDATES", cls.candidates)),
            context_k=int(env("CONTEXT_K", cls.context_k)),
            max_context_chars=int(env("MAX_CONTEXT_CHARS", cls.max_context_chars)),
            refusal_threshold=float(env("REFUSAL_THRESHOLD")) if env("REFUSAL_THRESHOLD") else None,
            llm_base_url=env("LLM_BASE_URL", cls.llm_base_url),
            llm_model=env("LLM_MODEL", cls.llm_model),
            llm_api_key=env("LLM_API_KEY", cls.llm_api_key),
        )
