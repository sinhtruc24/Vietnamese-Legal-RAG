"""Prompt format shared by inference, SFT data generation and evaluation.

Keeping one format everywhere matters: the LoRA adapter learns exactly this
layout (numbered context, [n] citations, fixed refusal sentence).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

REFUSAL = "Tôi không tìm thấy căn cứ pháp lý phù hợp trong các văn bản được cung cấp."

SYSTEM_PROMPT = f"""Bạn là trợ lý tư vấn pháp luật Việt Nam. Chỉ được trả lời dựa trên các điều luật trong phần NGỮ CẢNH.

Quy tắc:
1. Trả lời ngắn gọn, chính xác, bằng tiếng Việt; đi thẳng vào câu hỏi.
2. Mỗi ý đều phải có trích dẫn nguồn dạng [1], [2] theo số thứ tự điều luật trong NGỮ CẢNH.
3. Không suy diễn hay bổ sung thông tin không có trong NGỮ CẢNH.
4. Nếu NGỮ CẢNH không có căn cứ để trả lời, chỉ trả lời đúng một câu: "{REFUSAL}\""""

_CITATION_RE = re.compile(r"\[(\d+)\]")
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


@dataclass(frozen=True)
class ContextDoc:
    article_id: str
    citation: str
    text: str


def format_context(docs: list[ContextDoc]) -> str:
    return "\n\n".join(f"[{i}] ({d.citation})\n{d.text.strip()}" for i, d in enumerate(docs, start=1))


def build_messages(question: str, docs: list[ContextDoc]) -> list[dict[str, str]]:
    user = f"NGỮ CẢNH:\n{format_context(docs)}\n\nCÂU HỎI: {question.strip()}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def clean_answer(text: str) -> str:
    """Drop reasoning blocks some models (e.g. Qwen3) emit before the answer."""
    return _THINK_RE.sub("", text).strip()


def parse_citations(answer: str, n_docs: int) -> tuple[list[int], list[int]]:
    """Return (valid, invalid) 1-based citation indices, in order of first use."""
    valid: list[int] = []
    invalid: list[int] = []
    for match in _CITATION_RE.finditer(answer):
        idx = int(match.group(1))
        bucket = valid if 1 <= idx <= n_docs else invalid
        if idx not in bucket:
            bucket.append(idx)
    return valid, invalid


def is_refusal(answer: str) -> bool:
    return "không tìm thấy căn cứ" in answer.lower()
