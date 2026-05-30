"""
Global pytest configuration and fixtures.

Sets environment variables before any imports so that PipelineConfig
always uses the mock backend in unit tests — no API keys or real models needed.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── Environment setup (must happen before pipeline imports) ───────────────────
os.environ.setdefault("LLM_PROVIDER", "mock")
os.environ.setdefault("LLM_MODEL", "mock-model")
os.environ.setdefault("RAG_BACKEND", "mock")
os.environ.setdefault("OPENAI_API_KEY", "")
os.environ.setdefault("ANTHROPIC_API_KEY", "")

# Ensure project root is on sys.path for CI environments
_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ── Mock heavy ML model fixtures ──────────────────────────────────────────────

@pytest.fixture(autouse=False)
def mock_nli_model():
    """
    Replace CrossEncoder with a deterministic mock so NLI-based metrics
    (hallucination, faithfulness) run without downloading any models.

    Scores returned: [contradiction=0.05, entailment=0.90, neutral=0.05]
    → answer is always considered 'grounded'.
    """
    mock_model = MagicMock()
    mock_model.predict.return_value = [[0.05, 0.90, 0.05]]

    with patch("eval.metrics.hallucination._model", mock_model), \
         patch("eval.metrics.faithfulness._model", mock_model):
        yield mock_model


@pytest.fixture(autouse=False)
def mock_sentence_transformer():
    """
    Replace SentenceTransformer with a mock that returns fixed unit vectors
    so relevancy scores are deterministic (cosine similarity = 1.0).
    """
    import numpy as np

    mock_model = MagicMock()
    # Identical embeddings → cosine similarity = 1.0 → relevancy = 1.0
    mock_model.encode.return_value = np.array([
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])

    with patch("eval.metrics.relevancy._model", mock_model):
        yield mock_model


@pytest.fixture
def mock_config():
    """Returns a PipelineConfig with all-mock settings and known thresholds."""
    from pipeline.config import PipelineConfig

    cfg = PipelineConfig()
    cfg.provider = "mock"
    cfg.model = "mock-model"
    cfg.rag_backend = "mock"
    cfg.hallucination_threshold = 0.05
    cfg.p95_latency_ms = 5000.0
    cfg.min_answer_relevancy = 0.70
    cfg.min_faithfulness = 0.75
    cfg.max_cost_per_query = 0.01
    return cfg
