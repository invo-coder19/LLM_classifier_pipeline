"""
Faithfulness — measures whether factual claims in the answer
are supported by the retrieved source documents.

Approach: split answer into sentences, check each sentence for
entailment against the concatenated source context.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List
import re

# Module-level singleton — lazy-loaded on first call.
# To mock in tests, patch the exact path: 'eval.metrics.faithfulness._model'
# (each metric module has its own independent singleton — patch them separately).
_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder
        _model = CrossEncoder("cross-encoder/nli-deberta-v3-small")
    return _model


def _split_sentences(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if len(s.split()) >= 3]


@dataclass
class FaithfulnessResult:
    score: float               # fraction of sentences that are entailed
    supported_count: int
    total_sentences: int
    unsupported_sentences: List[str]


def compute_faithfulness(answer: str, source_docs: List[str]) -> FaithfulnessResult:
    """
    For each sentence in the answer, check if it is entailed by any source doc.
    Faithfulness = supported_sentences / total_sentences
    """
    sentences = _split_sentences(answer)
    if not sentences:
        return FaithfulnessResult(1.0, 0, 0, [])

    if not source_docs:
        return FaithfulnessResult(0.5, 0, len(sentences), sentences)

    model = _get_model()
    context = " ".join(source_docs)[:2000]  # truncate to avoid OOM

    pairs = [(context, sent) for sent in sentences]
    scores = model.predict(pairs, apply_softmax=True)  # (N, 3)

    supported = []
    unsupported = []
    for sent, s in zip(sentences, scores):
        entailment = s[1]
        if entailment >= 0.50:
            supported.append(sent)
        else:
            unsupported.append(sent)

    score = len(supported) / len(sentences)
    return FaithfulnessResult(
        score=score,
        supported_count=len(supported),
        total_sentences=len(sentences),
        unsupported_sentences=unsupported,
    )


def mean_faithfulness(results: List[FaithfulnessResult]) -> float:
    if not results:
        return 0.0
    import numpy as np
    return float(np.mean([r.score for r in results]))
