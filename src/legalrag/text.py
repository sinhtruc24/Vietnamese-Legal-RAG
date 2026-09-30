"""Text normalisation and tokenisation for Vietnamese."""
from __future__ import annotations

import re
import unicodedata

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def normalize(text: str) -> str:
    # NFC matters for Vietnamese: the same syllable can be encoded with
    # precomposed or combining diacritics depending on the source document.
    return unicodedata.normalize("NFC", text).lower()


def tokenize(text: str, bigrams: bool = False) -> list[str]:
    """Split into syllables; optionally add syllable bigrams.

    Vietnamese words are often two syllables ("pháp luật", "xử phạt"), so
    bigrams give BM25 a cheap approximation of word segmentation.
    """
    tokens = _TOKEN_RE.findall(normalize(text))
    if bigrams:
        tokens += [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
    return tokens
