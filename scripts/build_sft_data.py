"""Build the SFT dataset for QLoRA by distilling a stronger "teacher" LLM.

The Zalo dataset has questions + gold articles but no written answers, so:

* answerable samples: context = gold article(s) + hard negatives from the
  retriever (shuffled). The teacher writes a cited answer; we keep it only if
  it cites a gold article and no out-of-range index.
* negative samples (``--neg-ratio``): context = hard negatives only; the
  target is the fixed refusal sentence. This teaches the student to say
  "not found" instead of hallucinating when retrieval misses.

Any OpenAI-compatible endpoint can be the teacher (a hosted API or a large
open model served with vLLM). Progress is appended to raw.jsonl, so an
interrupted run resumes where it stopped (rejected samples are retried).

    export TEACHER_API_KEY=...
    python scripts/build_sft_data.py --index-dir indexes/full \
        --teacher-base-url https://api.openai.com/v1 --teacher-model gpt-4o-mini
"""
from __future__ import annotations

import argparse
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from legalrag.data import LegalDataset, read_jsonl, write_jsonl
from legalrag.factory import IndexBundle
from legalrag.prompts import REFUSAL, is_refusal, parse_citations
from legalrag.sampling import sample_context


FATAL_STATUS = {
    400: "Kiểm tra API key (Gemini báo key sai bằng lỗi 400), --teacher-model và --teacher-base-url.",
    401: "API key sai hoặc không khớp nhà cung cấp: key Gemini phải đi với endpoint Gemini, key OpenAI với endpoint OpenAI.",
    403: "Key không có quyền dùng model/endpoint này.",
    404: "Sai --teacher-base-url hoặc tên model không tồn tại.",
}


def label(sample: dict, teacher) -> dict | None:
    """Attach the target answer, or return None if the teacher output is unusable."""
    if sample["type"] == "negative":
        answer = REFUSAL
    else:
        answer = teacher.generate(sample["prompt"])
        valid, invalid = parse_citations(answer, sample["n_docs"])
        if is_refusal(answer) or invalid or not set(valid) & set(sample["gold"]):
            return None
    return {**sample, "completion": [{"role": "assistant", "content": answer}]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data/zalo_legal"))
    parser.add_argument("--index-dir", type=Path, default=Path("indexes/full"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/sft"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--retriever", default="bm25", help="source of hard negatives")
    parser.add_argument("--teacher-base-url", required=True)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--n-contexts", type=int, default=3)
    parser.add_argument("--negative-pool", type=int, default=10)
    parser.add_argument("--neg-ratio", type=float, default=0.15)
    parser.add_argument("--max-context-chars", type=int, default=1500)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--rpm", type=float, default=None,
                        help="max teacher requests per minute (free tiers: e.g. 4); default unlimited")
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="teacher output budget; reasoning models spend part of it thinking")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    from legalrag.generator import OpenAICompatibleGenerator, QuotaExhausted

    teacher = OpenAICompatibleGenerator(
        args.teacher_base_url, args.teacher_model,
        api_key=os.environ.get("TEACHER_API_KEY", "EMPTY"), temperature=0.2, max_tokens=args.max_tokens,
        requests_per_minute=args.rpm,
    )
    ds = LegalDataset.load(args.data_dir)
    bundle = IndexBundle(args.index_dir)
    retriever = bundle.retriever(args.retriever)

    indexed = bundle.articles.keys()
    qrels = {q: g & indexed for q, g in ds.qrels(args.split).items() if g & indexed}
    qids = list(qrels)[: args.limit or None]

    raw_path = args.out_dir / "raw.jsonl"
    done = {row["id"] for row in read_jsonl(raw_path)} if raw_path.exists() else set()
    todo = [q for q in qids if q not in done]
    print(f"{len(qids)} queries, {len(done)} already done, {len(todo)} to go")

    samples = [
        sample_context(q, ds.queries[q], qrels[q], bundle.articles, retriever, args.n_contexts,
                       args.negative_pool, args.neg_ratio, args.max_context_chars, args.seed)
        for q in tqdm(todo, desc="contexts")
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rejected = errors = 0
    quota_exhausted = False
    with open(raw_path, "a", encoding="utf-8") as out, ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(label, s, teacher) for s in samples]
        for future in tqdm(as_completed(futures), total=len(futures), desc="teacher"):
            # After the quota runs out keep draining: answers that already finished must still be saved.
            if future.cancelled():
                continue
            try:
                row = future.result()
            except QuotaExhausted:
                if not quota_exhausted:
                    quota_exhausted = True
                    for pending in futures:
                        pending.cancel()
                continue
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                if status in FATAL_STATUS:  # wrong key / URL / model: every call would fail the same way
                    pool.shutdown(cancel_futures=True)
                    raise SystemExit(f"\nTeacher API error {status}: {exc}\n{FATAL_STATUS[status]}") from None
                errors += 1  # transient (rate limit, network): skipped now, retried on the next run
                if errors <= 3:
                    print(f"error: {exc}")
                continue
            if row is None:
                rejected += 1
                continue
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()

    if quota_exhausted:
        print("\nHết quota trong ngày của teacher. Dữ liệu đã làm được vẫn được giữ; "
              "chạy lại lệnh này sau khi quota reset, script sẽ làm tiếp phần còn lại.")
    rows = list(read_jsonl(raw_path))
    random.Random(args.seed).shuffle(rows)
    n_val = max(1, int(len(rows) * args.val_ratio))
    write_jsonl(args.out_dir / "val.jsonl", rows[:n_val])
    write_jsonl(args.out_dir / "train.jsonl", rows[n_val:])
    kinds = {k: sum(r["type"] == k for r in rows) for k in ("answerable", "negative")}
    print(f"rejected this run: {rejected} | API errors: {errors} (rerun to retry) | "
          f"kept: {len(rows)} {kinds} | train {len(rows) - n_val} / val {n_val}")


if __name__ == "__main__":
    main()
