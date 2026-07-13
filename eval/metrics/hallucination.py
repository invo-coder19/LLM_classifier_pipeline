"""
Hallucination detection using NLI (Natural Language Inference).

Strategy: For each answer, check whether any claim in the answer is
CONTRADICTED by the source documents. Answers with no entailment
from any source are flagged as hallucinated.

Model: cross-encoder/nli-deberta-v3-small  (fast, ~180MB, runs on CPU)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

# Module-level singleton — lazy-loaded on first call.
# To mock in tests, patch the exact path: 'eval.metrics.hallucination._model'
# (each metric module has its own independent singleton — patch them separately).
_model = None  # lazy-loaded singleton


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder
        _model = CrossEncoder("cross-encoder/nli-deberta-v3-small")
    return _model


@dataclass
class HallucinationResult:
    is_hallucinated: bool
    max_entailment_score: float   # 0..1, higher = more grounded
    contradiction_score: float    # 0..1, higher = more hallucinated
    verdict: str                  # "grounded" | "uncertain" | "hallucinated"


def check_hallucination(answer: str, source_docs: List[str]) -> HallucinationResult:
    """
    Returns hallucination verdict for a single answer vs. its source docs.

    NLI label mapping (DeBERTa-v3):
        0 = contradiction, 1 = entailment, 2 = neutral
    """
    if not answer.strip():
        return HallucinationResult(True, 0.0, 1.0, "hallucinated")

    if not source_docs:
        # No sources at all — cannot verify
        return HallucinationResult(False, 0.5, 0.5, "uncertain")

    model = _get_model()

    # Pair each source doc with the answer
    pairs = [(doc, answer) for doc in source_docs]
    scores = model.predict(pairs, apply_softmax=True)  # shape: (N, 3)

    entailment_scores = [s[1] for s in scores]
    contradiction_scores = [s[0] for s in scores]

    max_entailment = max(entailment_scores)
    max_contradiction = max(contradiction_scores)

    # Decision rules
    if max_entailment >= 0.60:
        verdict = "grounded"
        is_hallucinated = False
    elif max_contradiction >= 0.50:
        verdict = "hallucinated"
        is_hallucinated = True
    else:
        verdict = "uncertain"
        is_hallucinated = False

    return HallucinationResult(
        is_hallucinated=is_hallucinated,
        max_entailment_score=max_entailment,
        contradiction_score=max_contradiction,
        verdict=verdict,
    )


def hallucination_rate(results: List[HallucinationResult]) -> float:
    """Fraction of answers flagged as hallucinated."""
    if not results:
        return 0.0
    return sum(1 for r in results if r.is_hallucinated) / len(results)
