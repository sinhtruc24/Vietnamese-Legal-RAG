"""LLM client for any OpenAI-compatible server (vLLM, llama.cpp, Ollama, hosted APIs)."""
from __future__ import annotations

import re
import threading
import time
from typing import Protocol

from .prompts import clean_answer

_RETRY_IN_RE = re.compile(r"retry in ([\d.]+)\s*s", re.IGNORECASE)


class Generator(Protocol):
    def generate(self, messages: list[dict[str, str]]) -> str: ...


class QuotaExhausted(RuntimeError):
    """The provider's daily quota is used up: retrying today is pointless."""


class RateLimiter:
    """Spaces calls evenly so that at most ``requests_per_minute`` start per minute (thread-safe)."""

    def __init__(self, requests_per_minute: float | None):
        self.interval = 60.0 / requests_per_minute if requests_per_minute else 0.0
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        if not self.interval:
            return
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next)
            self._next = start + self.interval
        time.sleep(max(0.0, start - now))


def _retry_delay(message: str, default: float = 30.0) -> float:
    match = _RETRY_IN_RE.search(message)
    return float(match.group(1)) if match else default


class OpenAICompatibleGenerator:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        temperature: float = 0.1,
        max_tokens: int = 512,
        timeout: float = 120.0,
        requests_per_minute: float | None = None,
        rate_limit_retries: int = 5,
    ):
        from openai import InternalServerError, OpenAI, RateLimitError

        # 429 and 5xx are retried below with provider-aware waits; the SDK only retries connection errors quickly.
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=2)
        self._rate_limit_error = RateLimitError
        self._server_error = InternalServerError
        self.limiter = RateLimiter(requests_per_minute)
        self.rate_limit_retries = rate_limit_retries
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, messages: list[dict[str, str]]) -> str:
        for attempt in range(self.rate_limit_retries + 1):
            self.limiter.wait()
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return clean_answer(response.choices[0].message.content or "")
            except self._rate_limit_error as exc:
                text = str(exc)
                if "perday" in text.lower().replace(" ", ""):
                    raise QuotaExhausted(text) from exc
                if attempt == self.rate_limit_retries:
                    raise
                time.sleep(_retry_delay(text) + 1.0)
            except self._server_error:  # 5xx, e.g. 503 "model overloaded": usually clears within a minute
                if attempt == self.rate_limit_retries:
                    raise
                time.sleep(min(10.0 * 2**attempt, 120.0))
        raise AssertionError("unreachable")


class HFGenerator:
    """Local transformers model (+ optional LoRA adapter), greedy decoding.

    Slower than vLLM but has no server and no extra dependencies, which makes
    it the reliable choice on Kaggle/Colab. Not thread-safe: call sequentially.
    """

    def __init__(self, model: str, adapter: str | None = None, max_new_tokens: int = 512):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        # One GPU is enough for a 3B model in fp16 (~6 GB) and avoids slow cross-GPU layer splitting.
        self.model = AutoModelForCausalLM.from_pretrained(
            model, dtype=torch.bfloat16 if bf16 else torch.float16,
            device_map={"": 0} if torch.cuda.is_available() else None,
        )
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    def generate(self, messages: list[dict[str, str]]) -> str:
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.model.device)
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        new_tokens = output[0, inputs["input_ids"].shape[1]:]
        return clean_answer(self.tokenizer.decode(new_tokens, skip_special_tokens=True))
