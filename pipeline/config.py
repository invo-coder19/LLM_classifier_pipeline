"""
Pipeline configuration — loads from environment / .env file.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class PipelineConfig:
    # LLM
    provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "mock"))
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini"))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    ollama_base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    max_tokens: int = field(default_factory=lambda: int(os.getenv("MAX_TOKENS", "512")))
    temperature: float = field(default_factory=lambda: float(os.getenv("TEMPERATURE", "0.0")))
    timeout_seconds: int = field(default_factory=lambda: int(os.getenv("TIMEOUT_SECONDS", "30")))

    # RAG
    rag_backend: str = field(default_factory=lambda: os.getenv("RAG_BACKEND", "mock"))
    chroma_persist_dir: str = field(default_factory=lambda: os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db"))
    top_k_docs: int = field(default_factory=lambda: int(os.getenv("TOP_K_DOCS", "3")))

    # Thresholds
    hallucination_threshold: float = field(default_factory=lambda: float(os.getenv("HALLUCINATION_THRESHOLD", "0.05")))
    p95_latency_ms: float = field(default_factory=lambda: float(os.getenv("P95_LATENCY_MS", "5000")))
    min_answer_relevancy: float = field(default_factory=lambda: float(os.getenv("MIN_ANSWER_RELEVANCY", "0.70")))
    min_faithfulness: float = field(default_factory=lambda: float(os.getenv("MIN_FAITHFULNESS", "0.75")))
    max_cost_per_query: float = field(default_factory=lambda: float(os.getenv("MAX_COST_PER_QUERY", "0.01")))

    # Results
    results_dir: str = field(default_factory=lambda: os.getenv("RESULTS_DIR", "./results"))
