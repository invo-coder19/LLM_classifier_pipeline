"""
Integration-level tests for the eval runner and merge_and_gate script.
Uses mock backends — no real API calls, no model downloads.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pipeline.config import PipelineConfig
from eval.runner import build_prompt, run_evaluation
from scripts.merge_and_gate import merge_and_gate


# ── build_prompt ──────────────────────────────────────────────────────────────

class TestBuildPrompt:
    def test_includes_question(self):
        from pipeline.rag import RetrievedDoc
        docs = [RetrievedDoc("d1", "Paris is in France.", 0.9, "kb/geo.txt")]
        prompt = build_prompt("Where is Paris?", docs)
        assert "Where is Paris?" in prompt

    def test_includes_source_content(self):
        from pipeline.rag import RetrievedDoc
        docs = [RetrievedDoc("d1", "Paris is in France.", 0.9, "kb/geo.txt")]
        prompt = build_prompt("Where is Paris?", docs)
        assert "Paris is in France." in prompt

    def test_multiple_sources_numbered(self):
        from pipeline.rag import RetrievedDoc
        docs = [
            RetrievedDoc("d1", "Content A.", 0.9, "src1"),
            RetrievedDoc("d2", "Content B.", 0.8, "src2"),
        ]
        prompt = build_prompt("Query?", docs)
        assert "[Source 1]" in prompt
        assert "[Source 2]" in prompt


# ── run_evaluation ────────────────────────────────────────────────────────────

class TestRunEvaluationMock:
    """
    Tests run_evaluation end-to-end with fully mocked ML models
    so no sentence-transformer or NLI model is loaded.
    """

    @pytest.fixture
    def tiny_dataset(self, tmp_path):
        """Write a minimal valid golden dataset to a temp file."""
        data = {
            "version": "1.0",
            "items": [
                {
                    "id": "q001",
                    "question": "Where is the Eiffel Tower?",
                    "expected_answer": "Paris, France",
                    "category": "factual",
                    "difficulty": "easy",
                },
                {
                    "id": "q002",
                    "question": "What is machine learning?",
                    "expected_answer": "A subset of AI that learns from data.",
                    "category": "factual",
                    "difficulty": "medium",
                },
            ],
        }
        dataset_file = tmp_path / "test_dataset.json"
        dataset_file.write_text(json.dumps(data))
        return str(dataset_file)

    def test_run_evaluation_completes(self, tiny_dataset, mock_nli_model, mock_sentence_transformer):
        """Evaluation should run without errors with mock models."""
        cfg = PipelineConfig()
        result = run_evaluation(
            dataset_path=tiny_dataset,
            shard=0,
            total_shards=1,
            config=cfg,
        )

        assert "run_id" in result
        assert "metrics" in result
        assert "gate" in result
        assert "per_query" in result
        assert result["total_questions"] == 2

    def test_run_evaluation_metrics_present(self, tiny_dataset, mock_nli_model, mock_sentence_transformer):
        cfg = PipelineConfig()
        result = run_evaluation(
            dataset_path=tiny_dataset,
            shard=0,
            total_shards=1,
            config=cfg,
        )
        m = result["metrics"]
        assert "hallucination_rate" in m
        assert "mean_answer_relevancy" in m
        assert "mean_faithfulness" in m
        assert "latency" in m
        assert "cost" in m

    def test_sharding_splits_dataset(self, tiny_dataset, mock_nli_model, mock_sentence_transformer):
        """With 2 shards and 2 questions, each shard should get 1 question."""
        cfg = PipelineConfig()
        result_0 = run_evaluation(tiny_dataset, shard=0, total_shards=2, config=cfg)
        result_1 = run_evaluation(tiny_dataset, shard=1, total_shards=2, config=cfg)
        assert result_0["total_questions"] == 1
        assert result_1["total_questions"] == 1

    def test_gate_is_dict_with_passed_field(self, tiny_dataset, mock_nli_model, mock_sentence_transformer):
        cfg = PipelineConfig()
        result = run_evaluation(tiny_dataset, shard=0, total_shards=1, config=cfg)
        gate = result["gate"]
        assert isinstance(gate, dict)
        assert "passed" in gate
        assert isinstance(gate["passed"], bool)


# ── merge_and_gate ────────────────────────────────────────────────────────────

class TestMergeAndGate:
    def _make_shard_result(self, tmp_path: Path, shard: int, questions: list) -> None:
        """Write a fake shard result JSON."""
        data = {
            "run_id": f"shard{shard}_test",
            "model": "mock-model",
            "provider": "mock",
            "total_questions": len(questions),
            "per_query": questions,
        }
        (tmp_path / f"shard{shard}_result.json").write_text(json.dumps(data))

    def _make_query(
        self,
        qid: str,
        verdict: str = "grounded",
        rel: float = 0.85,
        faith: float = 0.90,
        latency: float = 300.0,
        cost: float = 0.0,
    ) -> dict:
        return {
            "id": qid,
            "question": f"Q{qid}",
            "expected_answer": "Expected",
            "generated_answer": "Generated",
            "hallucination_verdict": verdict,
            "hallucination_score": 0.9 if verdict == "grounded" else 0.1,
            "relevancy_score": rel,
            "faithfulness_score": faith,
            "latency_ms": latency,
            "cost_usd": cost,
        }

    def test_gate_passes_with_good_metrics(self, tmp_path):
        queries = [self._make_query(f"q{i:03d}") for i in range(10)]
        self._make_shard_result(tmp_path, 0, queries)
        output = tmp_path / "gate_result.json"
        result = merge_and_gate(str(tmp_path), str(output))
        assert result is True
        gate_data = json.loads(output.read_text())
        assert gate_data["passed"] is True

    def test_gate_fails_with_high_hallucination(self, tmp_path):
        # 50% hallucinated (> 5% threshold)
        queries = (
            [self._make_query(f"q{i:03d}", verdict="hallucinated") for i in range(5)]
            + [self._make_query(f"q{i+5:03d}") for i in range(5)]
        )
        self._make_shard_result(tmp_path, 0, queries)
        output = tmp_path / "gate_result.json"
        result = merge_and_gate(str(tmp_path), str(output))
        assert result is False
        gate_data = json.loads(output.read_text())
        assert gate_data["passed"] is False

    def test_gate_fails_with_high_latency(self, tmp_path):
        # p95 latency >> 5000ms
        queries = [self._make_query(f"q{i:03d}", latency=9000.0) for i in range(20)]
        self._make_shard_result(tmp_path, 0, queries)
        output = tmp_path / "gate_result.json"
        result = merge_and_gate(str(tmp_path), str(output))
        assert result is False

    def test_merges_multiple_shards(self, tmp_path):
        for shard in range(3):
            queries = [self._make_query(f"q{shard}{i:02d}") for i in range(5)]
            self._make_shard_result(tmp_path, shard, queries)
        output = tmp_path / "gate_result.json"
        result = merge_and_gate(str(tmp_path), str(output))
        gate_data = json.loads(output.read_text())
        assert gate_data["total_questions"] == 15
        assert gate_data["shards"] == 3

    def test_output_has_all_metric_fields(self, tmp_path):
        queries = [self._make_query(f"q{i:03d}") for i in range(5)]
        self._make_shard_result(tmp_path, 0, queries)
        output = tmp_path / "gate_result.json"
        merge_and_gate(str(tmp_path), str(output))
        data = json.loads(output.read_text())
        assert "metrics" in data
        m = data["metrics"]
        assert "hallucination_rate" in m
        assert "mean_relevancy" in m
        assert "mean_faithfulness" in m
        assert "p95_latency_ms" in m
        assert "mean_cost_per_query" in m
