"""Unit tests for gate logic."""
import pytest
from eval.gates import run_gates
from pipeline.config import PipelineConfig


@pytest.fixture
def cfg():
    c = PipelineConfig()
    c.hallucination_threshold = 0.05
    c.p95_latency_ms = 5000.0
    c.min_answer_relevancy = 0.70
    c.min_faithfulness = 0.75
    c.max_cost_per_query = 0.01
    return c


def test_all_pass(cfg):
    result = run_gates(0.02, 3000, 0.85, 0.90, 0.005, cfg)
    assert result.passed is True
    assert len(result.violations) == 0


def test_hallucination_blocks(cfg):
    result = run_gates(0.10, 3000, 0.85, 0.90, 0.005, cfg)
    assert result.passed is False
    assert any(v.metric == "hallucination_rate" for v in result.violations)


def test_latency_blocks(cfg):
    result = run_gates(0.02, 6000, 0.85, 0.90, 0.005, cfg)
    assert result.passed is False
    assert any(v.metric == "p95_latency_ms" for v in result.violations)


def test_relevancy_warning_not_blocking(cfg):
    result = run_gates(0.02, 3000, 0.50, 0.90, 0.005, cfg)
    assert result.passed is True
    assert any(w.metric == "answer_relevancy" for w in result.warnings)


def test_faithfulness_warning_not_blocking(cfg):
    result = run_gates(0.02, 3000, 0.85, 0.50, 0.005, cfg)
    assert result.passed is True
    assert any(w.metric == "faithfulness" for w in result.warnings)


def test_cost_warning_not_blocking(cfg):
    result = run_gates(0.02, 3000, 0.85, 0.90, 0.05, cfg)
    assert result.passed is True
    assert any(w.metric == "cost_per_query" for w in result.warnings)


def test_multiple_violations(cfg):
    result = run_gates(0.10, 7000, 0.40, 0.30, 0.10, cfg)
    assert result.passed is False
    assert len(result.violations) == 2


def test_gate_to_dict(cfg):
    result = run_gates(0.02, 3000, 0.85, 0.90, 0.005, cfg)
    d = result.to_dict()
    assert "passed" in d
    assert "violations" in d
    assert "summary" in d
