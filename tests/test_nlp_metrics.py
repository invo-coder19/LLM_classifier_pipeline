"""
Unit tests for hallucination, faithfulness, and relevancy metrics.
All ML models are replaced with deterministic mocks via conftest.py fixtures.
"""
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from eval.metrics.hallucination import (
    HallucinationResult,
    check_hallucination,
    hallucination_rate,
)
from eval.metrics.faithfulness import (
    FaithfulnessResult,
    compute_faithfulness,
    mean_faithfulness,
)
from eval.metrics.relevancy import (
    RelevancyResult,
    compute_relevancy,
    mean_relevancy,
)


# ── Hallucination ─────────────────────────────────────────────────────────────

class TestHallucination:
    def test_empty_answer_is_hallucinated(self):
        result = check_hallucination("", ["some source doc"])
        assert result.is_hallucinated is True
        assert result.verdict == "hallucinated"

    def test_no_source_docs_is_uncertain(self):
        result = check_hallucination("Some answer", [])
        assert result.is_hallucinated is False
        assert result.verdict == "uncertain"
        assert result.max_entailment_score == 0.5

    def test_grounded_answer(self, mock_nli_model):
        """High entailment score → grounded."""
        mock_nli_model.predict.return_value = [[0.02, 0.92, 0.06]]
        result = check_hallucination("Paris is in France.", ["Paris is the capital of France."])
        assert result.verdict == "grounded"
        assert result.is_hallucinated is False
        assert result.max_entailment_score > 0.60

    def test_hallucinated_answer(self, mock_nli_model):
        """High contradiction score → hallucinated."""
        mock_nli_model.predict.return_value = [[0.80, 0.10, 0.10]]
        result = check_hallucination("Paris is in Germany.", ["Paris is in France."])
        assert result.verdict == "hallucinated"
        assert result.is_hallucinated is True

    def test_uncertain_answer(self, mock_nli_model):
        """Low entailment, low contradiction → uncertain."""
        mock_nli_model.predict.return_value = [[0.30, 0.35, 0.35]]
        result = check_hallucination("Something vague.", ["Unrelated source doc."])
        assert result.verdict == "uncertain"
        assert result.is_hallucinated is False

    def test_hallucination_rate_empty(self):
        assert hallucination_rate([]) == 0.0

    def test_hallucination_rate_none(self):
        results = [
            HallucinationResult(False, 0.8, 0.1, "grounded"),
            HallucinationResult(False, 0.9, 0.05, "grounded"),
        ]
        assert hallucination_rate(results) == 0.0

    def test_hallucination_rate_all(self):
        results = [
            HallucinationResult(True, 0.1, 0.8, "hallucinated"),
            HallucinationResult(True, 0.1, 0.9, "hallucinated"),
        ]
        assert hallucination_rate(results) == 1.0

    def test_hallucination_rate_partial(self):
        results = [
            HallucinationResult(True, 0.1, 0.8, "hallucinated"),
            HallucinationResult(False, 0.9, 0.05, "grounded"),
            HallucinationResult(False, 0.8, 0.1, "grounded"),
            HallucinationResult(True, 0.1, 0.7, "hallucinated"),
        ]
        assert hallucination_rate(results) == pytest.approx(0.5)


# ── Faithfulness ──────────────────────────────────────────────────────────────

class TestFaithfulness:
    def test_empty_answer_returns_perfect(self):
        result = compute_faithfulness("", ["source doc"])
        assert result.score == 1.0
        assert result.total_sentences == 0

    def test_no_sources_returns_half(self):
        result = compute_faithfulness("This is a factual claim.", [])
        assert result.score == 0.5

    def test_fully_supported(self, mock_nli_model):
        """All sentences entailed → faithfulness = 1.0."""
        mock_nli_model.predict.return_value = [[0.05, 0.90, 0.05], [0.05, 0.85, 0.10]]
        result = compute_faithfulness(
            "The Eiffel Tower is in Paris. It was built in 1889.",
            ["The Eiffel Tower is located in Paris, France, built in 1889."],
        )
        assert result.score == 1.0
        assert result.supported_count == result.total_sentences
        assert result.unsupported_sentences == []

    def test_partially_supported(self, mock_nli_model):
        """One supported, one not → faithfulness = 0.5."""
        mock_nli_model.predict.return_value = [
            [0.05, 0.90, 0.05],  # supported
            [0.70, 0.15, 0.15],  # not supported
        ]
        result = compute_faithfulness(
            "Paris is in France. The moon is made of cheese.",
            ["Paris is the capital of France."],
        )
        assert result.score == pytest.approx(0.5)
        assert len(result.unsupported_sentences) == 1

    def test_mean_faithfulness_empty(self):
        assert mean_faithfulness([]) == 0.0

    def test_mean_faithfulness_single(self):
        r = FaithfulnessResult(0.8, 4, 5, [])
        assert mean_faithfulness([r]) == pytest.approx(0.8)

    def test_mean_faithfulness_multiple(self):
        results = [
            FaithfulnessResult(1.0, 3, 3, []),
            FaithfulnessResult(0.5, 1, 2, ["unsup"]),
            FaithfulnessResult(0.0, 0, 2, ["a", "b"]),
        ]
        assert mean_faithfulness(results) == pytest.approx(0.5)


# ── Relevancy ─────────────────────────────────────────────────────────────────

class TestRelevancy:
    def test_empty_answer_returns_zero(self):
        result = compute_relevancy("What is Python?", "")
        assert result.score == 0.0

    def test_whitespace_only_answer_returns_zero(self):
        result = compute_relevancy("What is Python?", "   ")
        assert result.score == 0.0

    def test_identical_embeddings_gives_score_one(self, mock_sentence_transformer):
        """Mock returns identical vectors → dot product = 1.0."""
        result = compute_relevancy("What is Python?", "Python is a programming language.")
        assert result.score == pytest.approx(1.0)
        assert result.question == "What is Python?"

    def test_score_clamped_to_zero_one(self, mock_sentence_transformer):
        """Negative cosine similarity gets clamped to 0."""
        mock_sentence_transformer.encode.return_value = np.array([
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
        ])
        result = compute_relevancy("What?", "Opposite answer.")
        assert result.score == 0.0

    def test_mean_relevancy_empty(self):
        assert mean_relevancy([]) == 0.0

    def test_mean_relevancy_single(self):
        r = RelevancyResult(0.75, "q", "a")
        assert mean_relevancy([r]) == pytest.approx(0.75)

    def test_mean_relevancy_multiple(self):
        results = [
            RelevancyResult(0.8, "q1", "a1"),
            RelevancyResult(0.6, "q2", "a2"),
            RelevancyResult(1.0, "q3", "a3"),
        ]
        assert mean_relevancy(results) == pytest.approx(0.8)
