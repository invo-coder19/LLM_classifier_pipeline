"""
Cost estimation per query based on token usage and model pricing.

Pricing table (USD per 1M tokens) — update as needed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

# Pricing: (input_per_1M, output_per_1M) in USD
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o":            (5.00,   15.00),
    "gpt-4o-mini":       (0.15,    0.60),
    "gpt-4-turbo":       (10.00,  30.00),
    "gpt-3.5-turbo":     (0.50,   1.50),
    "claude-3-5-sonnet": (3.00,   15.00),
    "claude-3-haiku":    (0.25,   1.25),
    "claude-3-opus":     (15.00,  75.00),
    "mock-model":        (0.00,    0.00),
}

DEFAULT_PRICING = (1.00, 3.00)  # fallback for unknown models


@dataclass
class CostResult:
    cost_usd: float
    prompt_tokens: int
    completion_tokens: int
    model: str


def estimate_cost(prompt_tokens: int, completion_tokens: int, model: str) -> CostResult:
    pricing = MODEL_PRICING.get(model, DEFAULT_PRICING)
    input_cost = (prompt_tokens / 1_000_000) * pricing[0]
    output_cost = (completion_tokens / 1_000_000) * pricing[1]
    return CostResult(
        cost_usd=input_cost + output_cost,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        model=model,
    )


@dataclass
class CostSummary:
    total_cost_usd: float
    mean_cost_per_query: float
    total_prompt_tokens: int
    total_completion_tokens: int
    total_queries: int


def compute_cost_summary(results: List[CostResult]) -> CostSummary:
    if not results:
        return CostSummary(0, 0, 0, 0, 0)
    total = sum(r.cost_usd for r in results)
    return CostSummary(
        total_cost_usd=total,
        mean_cost_per_query=total / len(results),
        total_prompt_tokens=sum(r.prompt_tokens for r in results),
        total_completion_tokens=sum(r.completion_tokens for r in results),
        total_queries=len(results),
    )
