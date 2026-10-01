"""REST API for the legal assistant.

    uvicorn app.api:app --host 0.0.0.0 --port 8080

Configuration comes from environment variables (see legalrag/config.py and .env.example).
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from legalrag.config import Settings
from legalrag.factory import IndexBundle, build_pipeline

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings.from_env()
    state["settings"] = settings
    state["pipeline"] = build_pipeline(settings, IndexBundle(settings.index_dir, settings.device))
    yield
    state.clear()


app = FastAPI(title="Vietnamese Legal RAG", version="0.1.0", lifespan=lifespan)


class SearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=1000)
    k: int = Field(5, ge=1, le=50)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


@app.get("/health")
def health() -> dict:
    settings: Settings = state["settings"]
    return {
        "status": "ok",
        "retriever": settings.retriever,
        "reranker": settings.reranker_model if settings.use_reranker else None,
        "refusal_threshold": settings.refusal_threshold if settings.use_reranker else None,
        "llm": settings.llm_model,
    }


# Plain `def` endpoints: FastAPI runs them in a threadpool, so model calls do not block the event loop.
@app.post("/search")
def search(req: SearchRequest) -> dict:
    pipeline = state["pipeline"]
    hits = pipeline.retrieve(req.query, k=req.k)
    return {
        "query": req.query,
        "hits": [
            {"article_id": h.article_id, "citation": pipeline.articles[h.article_id].citation
             if h.article_id in pipeline.articles else h.article_id, "score": h.score, "text": h.text}
            for h in hits
        ],
    }


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    try:
        result = state["pipeline"].answer(req.question)
    except Exception as exc:  # most often: the LLM server is unreachable
        raise HTTPException(status_code=502, detail=f"generation failed: {exc}") from exc
    return {
        "question": result.question,
        "answer": result.answer,
        "refused": result.refused,
        "gated": result.gated,
        "top_score": result.top_score,
        "citations": result.citations,
        "contexts": [asdict(d) for d in result.contexts],
        "timings_ms": {k: round(v, 1) for k, v in result.timings_ms.items()},
    }
