"""Reference-free metrics for generated answers, shared by eval_generation and judge_results."""
from __future__ import annotations

import json
import re

from .generator import QuotaExhausted
from .prompts import is_refusal, parse_citations

JUDGE_PROMPT = """Bạn là giám khảo. Cho NGỮ CẢNH và CÂU TRẢ LỜI, hãy kiểm tra mọi thông tin trong CÂU TRẢ LỜI có được NGỮ CẢNH hỗ trợ hay không.
Chỉ trả về JSON: {{"faithful": true}} hoặc {{"faithful": false, "reason": "<ngắn gọn>"}}

NGỮ CẢNH:
{context}

CÂU TRẢ LỜI:
{answer}"""


def judge_faithfulness(judge, context: str, answer: str) -> bool | None:
    """True/False verdict, or None if the judge failed or answered unparseably.

    ``QuotaExhausted`` is re-raised so callers can stop instead of burning
    through every remaining sample with failing requests.
    """
    try:
        raw = judge.generate([{"role": "user", "content": JUDGE_PROMPT.format(context=context, answer=answer)}])
    except QuotaExhausted:
        raise
    except Exception as exc:  # network / transient API error: leave this sample unjudged
        print(f"judge error: {str(exc)[:200]}")
        return None
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
        "n_judged": sum(r.get("faithful") is not None for r in answered),
        "latency_ms": mean([r["latency_ms"] for r in rows]),
    }
