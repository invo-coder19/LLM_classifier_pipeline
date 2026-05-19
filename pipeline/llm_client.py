"""
Unified LLM client supporting OpenAI, Anthropic, Ollama, and Mock backends.
"""
from __future__ import annotations

import time
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from pipeline.config import PipelineConfig


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    model: str
    provider: str


class BaseLLMClient(ABC):
    def __init__(self, config: PipelineConfig):
        self.config = config

    @abstractmethod
    def complete(self, system_prompt: str, user_message: str) -> LLMResponse:
        ...


class OpenAIClient(BaseLLMClient):
    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=config.openai_api_key)
        except ImportError:
            raise ImportError("openai package not installed.")

    def complete(self, system_prompt: str, user_message: str) -> LLMResponse:
        start = time.perf_counter()
        resp = self._client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            timeout=self.config.timeout_seconds,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        choice = resp.choices[0]
        return LLMResponse(
            text=choice.message.content or "",
            prompt_tokens=resp.usage.prompt_tokens,
            completion_tokens=resp.usage.completion_tokens,
            latency_ms=latency_ms,
            model=self.config.model,
            provider="openai",
        )


class AnthropicClient(BaseLLMClient):
    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=config.anthropic_api_key)
        except ImportError:
            raise ImportError("anthropic package not installed.")

    def complete(self, system_prompt: str, user_message: str) -> LLMResponse:
        start = time.perf_counter()
        resp = self._client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        latency_ms = (time.perf_counter() - start) * 1000
        text = resp.content[0].text if resp.content else ""
        return LLMResponse(
            text=text,
            prompt_tokens=resp.usage.input_tokens,
            completion_tokens=resp.usage.output_tokens,
            latency_ms=latency_ms,
            model=self.config.model,
            provider="anthropic",
        )


class OllamaClient(BaseLLMClient):
    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        import httpx
        self._http = httpx.Client(base_url=config.ollama_base_url, timeout=config.timeout_seconds)

    def complete(self, system_prompt: str, user_message: str) -> LLMResponse:
        start = time.perf_counter()
        payload = {
            "model": self.config.model,
            "prompt": f"{system_prompt}\n\nUser: {user_message}\nAssistant:",
            "stream": False,
        }
        resp = self._http.post("/api/generate", json=payload)
        resp.raise_for_status()
        data = resp.json()
        latency_ms = (time.perf_counter() - start) * 1000
        return LLMResponse(
            text=data.get("response", ""),
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            latency_ms=latency_ms,
            model=self.config.model,
            provider="ollama",
        )


class MockLLMClient(BaseLLMClient):
    """Deterministic mock — used in CI without real API keys."""

    CANNED = [
        "The answer is based on the provided context.",
        "According to the source documents, this is the expected response.",
        "Based on the retrieved information, the answer follows logically.",
        "The context supports this conclusion clearly.",
        "This is a hallucinated answer not from any source.",  # deliberate for testing
    ]

    def complete(self, system_prompt: str, user_message: str) -> LLMResponse:
        # Stable seed per question for reproducibility
        idx = hash(user_message) % (len(self.CANNED) - 1)  # avoids hallucination entry normally
        text = self.CANNED[idx]
        latency_ms = random.uniform(80, 400)
        time.sleep(latency_ms / 1000)
        tokens = len(text.split())
        return LLMResponse(
            text=text,
            prompt_tokens=len(user_message.split()),
            completion_tokens=tokens,
            latency_ms=latency_ms,
            model="mock-model",
            provider="mock",
        )


def get_llm_client(config: Optional[PipelineConfig] = None) -> BaseLLMClient:
    """Factory — returns the right client based on config."""
    cfg = config or PipelineConfig()
    provider = cfg.provider.lower()
    if provider == "openai":
        return OpenAIClient(cfg)
    if provider == "anthropic":
        return AnthropicClient(cfg)
    if provider == "ollama":
        return OllamaClient(cfg)
    return MockLLMClient(cfg)
