"""
Unit tests for LLM clients, RAG retriever, and config loading.
No real API calls are made — all external clients are mocked.
"""
from __future__ import annotations

import os
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from pipeline.config import PipelineConfig
from pipeline.llm_client import (
    LLMResponse,
    MockLLMClient,
    get_llm_client,
)
from pipeline.rag import MockRetriever, RetrievedDoc, get_retriever


# ── PipelineConfig ────────────────────────────────────────────────────────────

class TestPipelineConfig:
    def test_defaults(self):
        cfg = PipelineConfig()
        # In CI env vars are set by conftest.py
        assert cfg.provider in ("mock", "openai", "anthropic", "ollama")
        assert cfg.max_tokens > 0
        assert 0.0 <= cfg.temperature <= 2.0

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        monkeypatch.setenv("LLM_MODEL", "llama3")
        monkeypatch.setenv("MAX_TOKENS", "1024")
        cfg = PipelineConfig()
        assert cfg.provider == "ollama"
        assert cfg.model == "llama3"
        assert cfg.max_tokens == 1024

    def test_threshold_fields(self):
        cfg = PipelineConfig()
        assert 0.0 < cfg.hallucination_threshold <= 1.0
        assert cfg.p95_latency_ms > 0
        assert 0.0 < cfg.min_answer_relevancy <= 1.0
        assert 0.0 < cfg.min_faithfulness <= 1.0
        assert cfg.max_cost_per_query > 0


# ── MockLLMClient ─────────────────────────────────────────────────────────────

class TestMockLLMClient:
    def test_returns_llm_response(self):
        client = MockLLMClient(PipelineConfig())
        resp = client.complete("You are an assistant.", "What is Python?")
        assert isinstance(resp, LLMResponse)
        assert isinstance(resp.text, str)
        assert len(resp.text) > 0
        assert resp.provider == "mock"
        assert resp.model == "mock-model"

    def test_latency_is_positive(self):
        client = MockLLMClient(PipelineConfig())
        resp = client.complete("System.", "Question?")
        assert resp.latency_ms > 0

    def test_tokens_are_non_negative(self):
        client = MockLLMClient(PipelineConfig())
        resp = client.complete("System.", "Question?")
        assert resp.prompt_tokens >= 0
        assert resp.completion_tokens >= 0

    def test_deterministic_for_same_input(self):
        """Same question → same canned answer (hash-based selection)."""
        client = MockLLMClient(PipelineConfig())
        r1 = client.complete("System.", "What is Python?")
        r2 = client.complete("System.", "What is Python?")
        assert r1.text == r2.text

    def test_canned_responses_cover_all_questions(self):
        """Different questions should map to valid canned answers."""
        client = MockLLMClient(PipelineConfig())
        questions = [
            "What is the capital of France?",
            "How does machine learning work?",
            "Explain the transformer architecture.",
            "What year was the Eiffel Tower built?",
            "How fast is light?",
        ]
        for q in questions:
            resp = client.complete("System.", q)
            assert resp.text in MockLLMClient.CANNED


# ── get_llm_client factory ────────────────────────────────────────────────────

class TestGetLLMClientFactory:
    def test_mock_provider_returns_mock_client(self, mock_config):
        client = get_llm_client(mock_config)
        assert isinstance(client, MockLLMClient)

    def test_default_config_uses_mock(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "mock")
        client = get_llm_client()
        assert isinstance(client, MockLLMClient)

    def test_openai_provider_raises_without_package(self, mock_config):
        """OpenAI client raises ImportError if openai package missing."""
        mock_config.provider = "openai"
        with patch.dict("sys.modules", {"openai": None}):
            with pytest.raises((ImportError, Exception)):
                from pipeline.llm_client import OpenAIClient
                OpenAIClient(mock_config)

    def test_anthropic_provider_raises_without_package(self, mock_config):
        """Anthropic client raises ImportError if anthropic package missing."""
        mock_config.provider = "anthropic"
        with patch.dict("sys.modules", {"anthropic": None}):
            with pytest.raises((ImportError, Exception)):
                from pipeline.llm_client import AnthropicClient
                AnthropicClient(mock_config)


# ── MockRetriever ─────────────────────────────────────────────────────────────

class TestMockRetriever:
    def test_returns_list_of_retrieved_docs(self):
        retriever = MockRetriever()
        docs = retriever.retrieve("What is Python?")
        assert isinstance(docs, list)
        assert len(docs) > 0
        assert all(isinstance(d, RetrievedDoc) for d in docs)

    def test_top_k_limits_results(self):
        retriever = MockRetriever()
        docs = retriever.retrieve("Machine learning AI transformer", top_k=2)
        assert len(docs) <= 2

    def test_top_k_default_is_three(self):
        retriever = MockRetriever()
        docs = retriever.retrieve("some query")
        assert len(docs) <= 3

    def test_docs_have_required_fields(self):
        retriever = MockRetriever()
        docs = retriever.retrieve("What is the Eiffel Tower?")
        for doc in docs:
            assert isinstance(doc.doc_id, str)
            assert isinstance(doc.content, str)
            assert isinstance(doc.score, float)
            assert isinstance(doc.source, str)
            assert 0.0 <= doc.score <= 1.0

    def test_keyword_overlap_scores_relevant_docs_higher(self):
        retriever = MockRetriever()
        # "Python programming language" should surface Python doc
        docs = retriever.retrieve("Python programming language", top_k=1)
        assert "Python" in docs[0].content or "programming" in docs[0].content

    def test_empty_query_returns_docs(self):
        retriever = MockRetriever()
        docs = retriever.retrieve("")
        assert isinstance(docs, list)


# ── get_retriever factory ─────────────────────────────────────────────────────

class TestGetRetrieverFactory:
    def test_mock_backend_returns_mock_retriever(self, mock_config):
        retriever = get_retriever(mock_config)
        assert isinstance(retriever, MockRetriever)

    def test_default_config_returns_mock(self, monkeypatch):
        monkeypatch.setenv("RAG_BACKEND", "mock")
        retriever = get_retriever()
        assert isinstance(retriever, MockRetriever)

    def test_chromadb_backend_raises_without_package(self, mock_config):
        mock_config.rag_backend = "chromadb"
        with patch.dict("sys.modules", {"chromadb": None}):
            with pytest.raises((ImportError, Exception)):
                from pipeline.rag import ChromaRetriever
                ChromaRetriever(mock_config)
