"""Add LLM-as-judge faithfulness to existing generation results, after the fact.

Lets the GPU part (generation on Kaggle) and the API part (judging) run on
different days / machines: download results/gen_*.jsonl from Kaggle, then

    export JUDGE_API_KEY=...
    python scripts/judge_results.py results/gen_base.jsonl results/gen_lora.jsonl \
        --judge-base-url https://generativelanguage.googleapis.com/v1beta/openai/ \
        --judge-model gemini-3.1-flash-lite --rpm 14

Only answered rows without a verdict are judged, and the file is saved after
every verdict, so a run stopped by the daily quota continues where it left off.
The matching *_summary.json is recomputed.
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from legalrag.data import read_jsonl, write_jsonl
from legalrag.evaluation import judge_faithfulness, summarize
from legalrag.generator import OpenAICompatibleGenerator, QuotaExhausted


def update_summary(path: Path, rows: list[dict], judge_model: str) -> dict:
    summary_path = path.with_name(path.stem + "_summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {"name": path.stem}
    summary.update(summarize(rows), judge_model=judge_model)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def judge_file(path: Path, judge, args) -> bool:
    """Judge one results file. Returns False if the daily quota ran out."""
    rows = list(read_jsonl(path))
    todo = [i for i, r in enumerate(rows) if not r["refused"] and r.get("faithful") is None]
    print(f"{path.name}: {len(todo)} answers to judge")
    quota_left = True
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(judge_faithfulness, judge, rows[i]["context"], rows[i]["answer"]): i for i in todo}
        for future in tqdm(as_completed(futures), total=len(futures), desc=path.stem):
            # Keep draining after the quota runs out: verdicts that already finished must not be lost.
            if future.cancelled():
                continue
            try:
                rows[futures[future]]["faithful"] = future.result()
            except QuotaExhausted:
                if quota_left:
                    quota_left = False
                    for pending in futures:
                        pending.cancel()
                continue
            write_jsonl(path, rows)  # checkpoint after every verdict

    write_jsonl(path, rows)
    summary = update_summary(path, rows, args.judge_model)
    print(f"  faithfulness={summary['faithfulness']} ({summary['n_judged']} judged)")
    return quota_left


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", type=Path, help="results/gen_*.jsonl files")
    parser.add_argument("--judge-base-url", required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--rpm", type=float, default=None)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    judge = OpenAICompatibleGenerator(
        args.judge_base_url, args.judge_model, os.environ.get("JUDGE_API_KEY", "EMPTY"),
        temperature=0.0, max_tokens=1024, requests_per_minute=args.rpm,
    )
    for path in args.files:
        if not judge_file(path, judge, args):
            print("\nHết quota trong ngày của judge. Kết quả đã chấm được lưu lại; "
                  "chạy lại lệnh này sau khi quota reset để chấm tiếp.")
            break


if __name__ == "__main__":
    main()
