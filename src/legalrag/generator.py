"""LLM client for any OpenAI-compatible server (vLLM, llama.cpp, Ollama, hosted APIs)."""
from __future__ import annotations

from typing import Protocol

from .prompts import clean_answer


class Generator(Protocol):
    def generate(self, messages: list[dict[str, str]]) -> str: ...


class OpenAICompatibleGenerator:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        temperature: float = 0.1,
        max_tokens: int = 512,
        timeout: float = 120.0,
    ):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=3)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, messages: list[dict[str, str]]) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return clean_answer(response.choices[0].message.content or "")


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
        self.model = AutoModelForCausalLM.from_pretrained(
            model, torch_dtype=torch.bfloat16 if bf16 else torch.float16, device_map="auto"
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
