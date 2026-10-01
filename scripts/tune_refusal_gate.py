"""Tune the reranker refusal gate on held-out questions, then apply it to saved test results.

1. Validation: train questions NOT used for SFT, with oracle contexts (gold + hard
   negatives, or hard negatives only). Gate score = best reranker score among the
   context articles. Pick the threshold with the best balanced accuracy
   (or the strictest one keeping >= --min-keep of answerable questions).
2. Test: for every results/gen_*.jsonl, re-score exactly the context each model saw
   and replace answers whose gate score is below the threshold by the refusal
   sentence. No generation is re-run, so base / LoRA / teacher are compared on
   identical decisions.

    python scripts/tune_refusal_gate.py results/gen_base.jsonl results/gen_lora.jsonl \
        results/gen_lora_v2.jsonl results/gen_teacher.jsonl --device cpu
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

from tqdm import tqdm

from legalrag.data import LegalDataset, read_jsonl
from legalrag.evaluation import score, summarize
from legalrag.factory import IndexBundle
from legalrag.gate import choose_threshold, sweep_thresholds
from legalrag.prompts import REFUSAL, parse_user_message
from legalrag.retrieval import CrossEncoderReranker
from legalrag.sampling import sample_docs

METRICS = ["answer_rate", "gold_cite_rate", "cite_precision", "refusal_acc", "faithfulness", "n_judged"]


def load_cache(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", nargs="+", type=Path, help="results/gen_*.jsonl files to gate")
    parser.add_argument("--data-dir", type=Path, default=Path("data/zalo_legal"))
    parser.add_argument("--index-dir", type=Path, default=Path("indexes/full"))
    parser.add_argument("--sft-raw", type=Path, default=Path("data/sft/raw.jsonl"),
                        help="SFT questions are excluded from validation")
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--device", default=None)
    parser.add_argument("--val-offset", type=int, default=0, help="applied after removing SFT questions")
    parser.add_argument("--val-limit", type=int, default=300)
    parser.add_argument("--val-neg-ratio", type=float, default=0.5)
    parser.add_argument("--min-keep", type=float, default=None,
                        help="instead of best balanced accuracy, keep at least this share of answerable questions")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--cache", type=Path, default=Path("results/refusal_gate_scores.json"))
    parser.add_argument("--output", type=Path, default=Path("results/refusal_gate.json"))
    args = parser.parse_args()

    cache = load_cache(args.cache)
    reranker = None

    def gate_score(key: str, question: str, texts: list[str]) -> float:
        nonlocal reranker
        if key not in cache:
            if reranker is None:
                reranker = CrossEncoderReranker(args.reranker_model, args.device)
            cache[key] = float(max(reranker.score(question, texts))) if texts else 0.0
        return cache[key]

    # ---------------------------------------------------------------- validation
    # Build every validation context first, then free the corpus + BM25 index before the
    # reranker (~2.3 GB) is loaded: keeps peak RAM low enough for an 8 GB laptop.
    ds = LegalDataset.load(args.data_dir)
    bundle = IndexBundle(args.index_dir)
    retriever = bundle.retriever("bm25")
    sft_ids = {row["id"] for row in read_jsonl(args.sft_raw)} if args.sft_raw.exists() else set()
    indexed = bundle.articles.keys()
    train = {q: g & indexed for q, g in ds.qrels("train").items() if g & indexed and q not in sft_ids}
    val_ids = list(train)[args.val_offset: args.val_offset + args.val_limit]
    if not val_ids:
        raise SystemExit(f"no validation questions: {len(train)} train questions left after removing SFT ones")
    val_items = []
    for qid in tqdm(val_ids, desc="validation contexts"):
        kind, docs = sample_docs(qid, ds.queries[qid], train[qid], bundle.articles, retriever,
                                 neg_ratio=args.val_neg_ratio, seed=args.seed)
        val_items.append((qid, ds.queries[qid], [d.text for d in docs], kind == "answerable"))
    del ds, bundle, retriever, train, indexed
    gc.collect()

    val_scores, val_labels = [], []
    for qid, question, texts, answerable in tqdm(val_items, desc="validation scores"):
        val_scores.append(gate_score(f"val:{qid}:{args.val_neg_ratio}", question, texts))
        val_labels.append(answerable)
        if len(val_scores) % 25 == 0:  # checkpoint: a crash does not lose reranker work
            args.cache.parent.mkdir(parents=True, exist_ok=True)
            args.cache.write_text(json.dumps(cache), encoding="utf-8")
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    args.cache.write_text(json.dumps(cache), encoding="utf-8")

    points = sweep_thresholds(val_scores, val_labels)
    chosen = choose_threshold(points, args.min_keep)
    threshold = chosen["threshold"]
    print(f"\nvalidation: {len(val_ids)} questions ({sum(val_labels)} answerable, "
          f"{len(val_labels) - sum(val_labels)} negative), none used for SFT")
    print(f"threshold = {threshold:.4f} | keeps {chosen['keep_answerable']:.1%} of answerable, "
          f"blocks {chosen['block_negative']:.1%} of negative (balanced acc {chosen['balanced_accuracy']:.1%})")

    # ---------------------------------------------------------------- test
    report = {"threshold": threshold, "selection": "min_keep" if args.min_keep else "balanced_accuracy",
              "min_keep": args.min_keep, "validation": {"n": len(val_ids), **chosen}, "models": {}}
    test_gate = None
    for path in args.results:
        rows = list(read_jsonl(path))
        gated_rows = []
        for row in tqdm(rows, desc=path.stem):
            question, texts = parse_user_message(row["context"], row["n_docs"])
            digest = hashlib.md5(row["context"].encode("utf-8")).hexdigest()[:12]
            s = gate_score(f"test:{row['id']}:{digest}", question, texts)
            if s < threshold:
                new = score({k: v for k, v in row.items() if k != "faithful"} | {"answer": REFUSAL})
                gated_rows.append({**new, "gated": True, "gate_score": s})
            else:
                gated_rows.append({**row, "gated": False, "gate_score": s})
        args.cache.write_text(json.dumps(cache), encoding="utf-8")

        name = path.stem.removeprefix("gen_")
        before, after = summarize(rows), summarize(gated_rows)
        report["models"][name] = {"before": before, "after": after}
        if test_gate is None:  # identical contexts for every model -> same gate decisions
            pos = [r for r in gated_rows if r["answerable"]]
            neg = [r for r in gated_rows if not r["answerable"]]
            test_gate = {"keep_answerable": sum(not r["gated"] for r in pos) / len(pos),
                         "block_negative": sum(r["gated"] for r in neg) / len(neg)}
    report["test_gate"] = test_gate

    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\ntest gate: keeps {test_gate['keep_answerable']:.1%} of answerable, "
          f"blocks {test_gate['block_negative']:.1%} of negative\n")
    print("| model | " + " | ".join(f"{m} (before -> after)" for m in METRICS) + " |")
    print("|---" * (len(METRICS) + 1) + "|")
    for name, r in report["models"].items():
        cells = []
        for m in METRICS:
            b, a = r["before"][m], r["after"][m]
            fmt = (lambda v: "-" if v is None else str(v)) if m == "n_judged" else (lambda v: "-" if v is None else f"{v:.1%}")
            cells.append(f"{fmt(b)} -> {fmt(a)}")
        print(f"| {name} | " + " | ".join(cells) + " |")
    print(f"\nsaved {args.output}")


if __name__ == "__main__":
    main()
