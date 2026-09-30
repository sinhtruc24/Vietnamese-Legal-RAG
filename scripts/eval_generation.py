"""Evaluate answer generation (base model vs. LoRA) on test queries.

Two modes:
* ``oracle`` (default): same controlled contexts as the SFT data (gold + hard
  negatives, some negative-only). Isolates the generator from retrieval, so
  base vs. LoRA is a fair comparison.
* ``pipeline``: real end-to-end RAG (retriever [+ reranker] -> LLM).

Reference-free metrics (the dataset has no gold answers):
    answer_rate        answerable context -> model answers (does not refuse)
    gold_cite_rate     answerable context -> cites at least one gold article
    cite_precision     share of cited articles that are gold
    invalid_cite_rate  cites an index that is not in the context (hallucinated source)
    uncited_rate       answers without any citation
    refusal_acc        negative context -> model correctly refuses
    faithfulness       optional LLM-as-judge: is every claim supported by the context?

    python scripts/eval_generation.py --index-dir indexes/full --limit 300 \
        --llm-base-url http://localhost:8000/v1 --llm-model legal-lora --name lora
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

from legalrag.config import Settings
from legalrag.data import LegalDataset, write_jsonl
from legalrag.factory import IndexBundle, build_pipeline
from legalrag.generator import OpenAICompatibleGenerator
from legalrag.prompts import build_messages, is_refusal, parse_citations
from legalrag.sampling import sample_context

JUDGE_PROMPT = """Bạn là giám khảo. Cho NGỮ CẢNH và CÂU TRẢ LỜI, hãy kiểm tra mọi thông tin trong CÂU TRẢ LỜI có được NGỮ CẢNH hỗ trợ hay không.
Chỉ trả về JSON: {{"faithful": true}} hoặc {{"faithful": false, "reason": "<ngắn gọn>"}}

NGỮ CẢNH:
{context}

CÂU TRẢ LỜI:
{answer}"""


def judge_faithfulness(judge, context: str, answer: str) -> bool | None:
    raw = judge.generate([{"role": "user", "content": JUDGE_PROMPT.format(context=context, answer=answer)}])
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        return bool(json.loads(match.group(0))["faithful"]) if match else None
    except (json.JSONDecodeError, KeyError):
        return None


def score(row: dict) -> dict:
    valid, invalid = parse_citations(row["answer"], row["n_docs"])
    gold = set(row["gold"])
    refused = is_refusal(row["answer"])
    return {
        **row,
        "answerable": bool(gold),
        "refused": refused,
        "cited": valid,
        "invalid_cite": bool(invalid),
        "uncited": not refused and not valid,
        "cites_gold": bool(set(valid) & gold),
        "cite_precision": len(set(valid) & gold) / len(valid) if valid else None,
    }


def summarize(rows: list[dict]) -> dict:
    def mean(values):
        values = [v for v in values if v is not None]
        return round(sum(values) / len(values), 4) if values else None

    answerable = [r for r in rows if r["answerable"]]
    negative = [r for r in rows if not r["answerable"]]
    answered = [r for r in rows if not r["refused"]]
    return {
        "n": len(rows),
        "n_answerable": len(answerable),
        "n_negative": len(negative),
        "answer_rate": mean([not r["refused"] for r in answerable]),
        "gold_cite_rate": mean([r["cites_gold"] for r in answerable]),
        "cite_precision": mean([r["cite_precision"] for r in answered]),
        "invalid_cite_rate": mean([r["invalid_cite"] for r in answered]),
        "uncited_rate": mean([r["uncited"] for r in answered]),
        "refusal_acc": mean([r["refused"] for r in negative]),
        "faithfulness": mean([r.get("faithful") for r in answered]),
        "latency_ms": mean([r["latency_ms"] for r in rows]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data/zalo_legal"))
    parser.add_argument("--index-dir", type=Path, default=Path("indexes/full"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--mode", choices=["oracle", "pipeline"], default="oracle")
    parser.add_argument("--retriever", default="bm25", help="oracle: negatives source; pipeline: retriever")
    parser.add_argument("--use-reranker", action="store_true", help="pipeline mode only")
    parser.add_argument("--backend", choices=["openai", "hf"], default="openai",
                        help="openai: any OpenAI-compatible server; hf: load the model locally (oracle mode)")
    parser.add_argument("--llm-base-url", default="http://localhost:8000/v1")
    parser.add_argument("--llm-model", required=True, help="served model name, or HF model id with --backend hf")
    parser.add_argument("--adapter", default=None, help="LoRA adapter dir (--backend hf)")
    parser.add_argument("--judge-base-url", default=None)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--n-contexts", type=int, default=3)
    parser.add_argument("--neg-ratio", type=float, default=0.2)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--name", default=None, help="run name used for output files")
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    if args.backend == "hf" and args.mode == "pipeline":
        parser.error("--backend hf is only supported in oracle mode; serve the model for pipeline mode")

    ds = LegalDataset.load(args.data_dir)
    bundle = IndexBundle(args.index_dir)
    indexed = bundle.articles.keys()
    qrels = {q: g & indexed for q, g in ds.qrels(args.split).items() if g & indexed}
    qids = list(qrels)[: args.limit or None]

    if args.mode == "oracle":
        if args.backend == "hf":
            from legalrag.generator import HFGenerator

            llm = HFGenerator(args.llm_model, args.adapter)
        else:
            llm = OpenAICompatibleGenerator(args.llm_base_url, args.llm_model, os.environ.get("LLM_API_KEY", "EMPTY"),
                                            max_tokens=2048)
        retriever = bundle.retriever(args.retriever)
        samples = [
            sample_context(q, ds.queries[q], qrels[q], bundle.articles, retriever,
                           args.n_contexts, 10, args.neg_ratio, seed=args.seed)
            for q in tqdm(qids, desc="contexts")
        ]

        def run(sample: dict) -> dict:
            t0 = time.perf_counter()
            answer = llm.generate(sample["prompt"])
            return {"id": sample["id"], "gold": sample["gold"], "n_docs": sample["n_docs"],
                    "context": sample["prompt"][1]["content"], "answer": answer,
                    "latency_ms": (time.perf_counter() - t0) * 1000}
        inputs = samples
    else:
        settings = Settings(
            index_dir=args.index_dir, retriever=args.retriever, use_reranker=args.use_reranker,
            context_k=args.n_contexts, llm_base_url=args.llm_base_url, llm_model=args.llm_model,
            llm_api_key=os.environ.get("LLM_API_KEY", "EMPTY"),
        )
        pipeline = build_pipeline(settings, bundle)

        def run(qid: str) -> dict:
            result = pipeline.answer(ds.queries[qid])
            docs = result.contexts
            return {"id": qid, "gold": [i for i, d in enumerate(docs, 1) if d.article_id in qrels[qid]],
                    "n_docs": len(docs), "context": build_messages(result.question, docs)[1]["content"],
                    "answer": result.answer, "latency_ms": result.timings_ms["total"]}
        inputs = qids

    # Local models (HF backend, pipeline reranker) are not thread-safe: keep them sequential.
    workers = args.workers if args.mode == "oracle" and args.backend == "openai" else 1
    with ThreadPoolExecutor(workers) as pool:
        rows = [score(r) for r in tqdm(pool.map(run, inputs), total=len(inputs), desc="generate")]

    if args.judge_model:
        judge = OpenAICompatibleGenerator(args.judge_base_url or args.llm_base_url, args.judge_model,
                                          os.environ.get("JUDGE_API_KEY", "EMPTY"), temperature=0.0, max_tokens=1024)
        todo = [r for r in rows if not r["refused"]]
        with ThreadPoolExecutor(args.workers) as pool:
            verdicts = list(tqdm(pool.map(lambda r: judge_faithfulness(judge, r["context"], r["answer"]), todo),
                                 total=len(todo), desc="judge"))
        for row, verdict in zip(todo, verdicts):
            row["faithful"] = verdict

    name = args.name or f"{args.mode}_{args.llm_model.replace('/', '_')}"
    summary = {"name": name, "mode": args.mode, "model": args.llm_model, "adapter": args.adapter,
               **summarize(rows)}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / f"gen_{name}.jsonl", rows)
    (args.out_dir / f"gen_{name}_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
