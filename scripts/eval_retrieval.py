"""Compare retrieval configurations on the labelled queries.

    python scripts/eval_retrieval.py --index-dir indexes/full \
        --methods bm25 dense hybrid hybrid+rerank --output results/retrieval_test.json

Methods: bm25, dense, hybrid, and any of them with "+rerank".
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from tqdm import tqdm

from legalrag.data import LegalDataset
from legalrag.factory import IndexBundle
from legalrag.metrics import evaluate_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data/zalo_legal"))
    parser.add_argument("--index-dir", type=Path, default=Path("indexes/full"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--methods", nargs="+", default=["bm25", "dense", "hybrid", "hybrid+rerank"])
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--rerank-candidates", type=int, default=30)
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--limit", type=int, default=0, help="evaluate on the first N queries only")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    ds = LegalDataset.load(args.data_dir)
    bundle = IndexBundle(args.index_dir, args.device)
    indexed = {c.article_id for c in bundle.chunks}
    qrels = {q: g & indexed for q, g in ds.qrels(args.split).items() if g & indexed}
    qids = list(qrels)[: args.limit or None]
    qrels = {q: qrels[q] for q in qids}
    questions = [ds.queries[q] for q in qids]
    print(f"{args.split}: {len(qids)} queries, {len(indexed):,} indexed articles")

    kmax = max(args.ks)
    results = {}
    for method in args.methods:
        base, _, rerank = method.partition("+")
        retriever = bundle.retriever(base)
        reranker = bundle.reranker(args.reranker_model) if rerank else None
        depth = max(kmax, args.rerank_candidates) if reranker else kmax

        t0 = time.perf_counter()
        runs = []
        for start in tqdm(range(0, len(questions), args.batch_size), desc=method):
            batch = questions[start:start + args.batch_size]
            hits = retriever.batch_search(batch, depth)
            if reranker:
                hits = [reranker.rerank(q, h, kmax) for q, h in zip(batch, hits)]
            runs.extend(hits)
        elapsed = time.perf_counter() - t0

        run = {qid: [h.article_id for h in hits] for qid, hits in zip(qids, runs)}
        metrics = evaluate_run(run, qrels, args.ks)
        metrics["ms_per_query"] = 1000 * elapsed / max(len(qids), 1)
        results[method] = metrics

    names = list(next(iter(results.values())))
    print("\n| method | " + " | ".join(names) + " |")
    print("|---" * (len(names) + 1) + "|")
    for method, metrics in results.items():
        cells = [f"{metrics[n]:.1f}" if n == "ms_per_query" else f"{metrics[n]:.4f}" for n in names]
        print(f"| {method} | " + " | ".join(cells) + " |")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {"split": args.split, "n_queries": len(qids), "index_dir": str(args.index_dir), "results": results}
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nsaved {args.output}")


if __name__ == "__main__":
    main()
