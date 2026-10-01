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


def parse_user_message(user: str, n_docs: int) -> tuple[str, list[str]]:
    """Inverse of :func:`build_messages` for the user turn: (question, [doc text, ...]).

    Lets evaluation re-score exactly the context a model saw, from saved results.
    """
    head, sep, question = user.rpartition("\n\nCÂU HỎI: ")
    if not sep or not head.startswith("NGỮ CẢNH:\n"):
        raise ValueError("not a message produced by build_messages")
    body = head[len("NGỮ CẢNH:\n"):]
    starts = []
    pos = 0
    for i in range(1, n_docs + 1):
        marker = f"[{i}] ("
        found = body.find(marker if i == 1 else f"\n\n{marker}", pos)
        if found < 0:
            raise ValueError(f"context block [{i}] not found")
        starts.append(found if i == 1 else found + 2)
        pos = found + 1
    texts = []
    for i, start in enumerate(starts):
        end = starts[i + 1] - 2 if i + 1 < len(starts) else len(body)
        block = body[start:end]
        texts.append(block.split("\n", 1)[1] if "\n" in block else "")  # drop the "[i] (citation)" line
    return question.strip(), texts


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
