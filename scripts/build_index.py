"""Chunk the corpus and build the BM25 and dense indexes.

Full corpus on a GPU (Kaggle T4 / Colab):
    python scripts/build_index.py --data-dir data/zalo_legal --index-dir indexes/full

Quick CPU iteration on a sub-sampled corpus (all gold articles + random fillers):
    python scripts/build_index.py --corpus-limit 5000 --skip-dense --index-dir indexes/dev
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from legalrag.data import LegalDataset, chunk_corpus, save_articles, save_chunks, subsample_corpus
from legalrag.retrieval import BM25Retriever


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data/zalo_legal"))
    parser.add_argument("--index-dir", type=Path, default=Path("indexes/full"))
    parser.add_argument("--corpus-limit", type=int, default=0, help="0 = full corpus")
    parser.add_argument("--max-chars", type=int, default=1200, help="max characters per chunk")
    parser.add_argument("--overlap", type=int, default=200, help="overlap between chunks (characters)")
    parser.add_argument("--bigrams", action="store_true", help="add syllable bigrams to BM25")
    parser.add_argument("--skip-dense", action="store_true")
    parser.add_argument("--embedding-model", default="BAAI/bge-m3")
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    ds = LegalDataset.load(args.data_dir)
    articles = ds.corpus
    if args.corpus_limit:
        gold = [a for split in ds.qrels_by_split for q in ds.qrels(split).values() for a in q]
        articles = subsample_corpus(articles, gold, args.corpus_limit)

    chunks = chunk_corpus(articles.values(), args.max_chars, args.overlap)
    args.index_dir.mkdir(parents=True, exist_ok=True)
    save_articles(args.index_dir / "articles.jsonl", articles.values())
    save_chunks(args.index_dir / "chunks.jsonl", chunks)
    print(f"{len(articles):,} articles -> {len(chunks):,} chunks")

    meta = {"articles": len(articles), "chunks": len(chunks), "corpus_limit": args.corpus_limit,
            "max_chars": args.max_chars, "overlap": args.overlap}

    t0 = time.perf_counter()
    BM25Retriever(bigrams=args.bigrams).fit(chunks).save(args.index_dir)
    meta["bm25_build_s"] = round(time.perf_counter() - t0, 1)
    print(f"BM25 built in {meta['bm25_build_s']} s")

    if not args.skip_dense:
        from legalrag.retrieval import DenseRetriever

        t0 = time.perf_counter()
        dense = DenseRetriever(args.embedding_model, args.device, args.batch_size, args.max_seq_length)
        dense.fit(chunks).save(args.index_dir)
        meta["dense_build_s"] = round(time.perf_counter() - t0, 1)
        meta["embedding_model"] = args.embedding_model
        print(f"Dense index built in {meta['dense_build_s']} s")

    (args.index_dir / "index_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
