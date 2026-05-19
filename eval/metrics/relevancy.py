"""
Answer Relevancy — measures how well the generated answer addresses the question.

Uses cosine similarity between question embedding and answer embedding.
Model: all-MiniLM-L6-v2 (fast, 80MB, runs on CPU)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List
import numpy as np

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


@dataclass
class RelevancyResult:
    score: float        # 0..1, higher is more relevant
    question: str
    answer: str


def compute_relevancy(question: str, answer: str) -> RelevancyResult:
    """Cosine similarity between question and answer embeddings."""
    if not answer.strip():
        return RelevancyResult(0.0, question, answer)

    model = _get_model()
    embeddings = model.encode([question, answer], normalize_embeddings=True)
    score = float(np.dot(embeddings[0], embeddings[1]))
    # Clamp to [0, 1]
    score = max(0.0, min(1.0, score))
    return RelevancyResult(score=score, question=question, answer=answer)


def mean_relevancy(results: List[RelevancyResult]) -> float:
    if not results:
        return 0.0
    return float(np.mean([r.score for r in results]))
