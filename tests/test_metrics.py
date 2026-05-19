"""Unit tests for metric functions (no real model — uses mock data)."""
import pytest
from eval.metrics.latency import compute_latency_stats
from eval.metrics.cost import estimate_cost, compute_cost_summary, CostResult


# ── Latency ──────────────────────────────────────────────────────────────────

def test_latency_empty():
    stats = compute_latency_stats([])
    assert stats.p50_ms == 0
    assert stats.p95_ms == 0


def test_latency_single():
    stats = compute_latency_stats([500.0])
    assert stats.p50_ms == 500.0
    assert stats.p95_ms == 500.0
    assert stats.mean_ms == 500.0


def test_latency_p95():
    latencies = list(range(1, 101))  # 1..100 ms
    stats = compute_latency_stats(latencies)
    assert stats.p95_ms == pytest.approx(95.05, abs=1.0)
    assert stats.p50_ms == pytest.approx(50.5, abs=1.0)


# ── Cost ──────────────────────────────────────────────────────────────────────

def test_cost_mock_model():
    r = estimate_cost(1000, 500, "mock-model")
    assert r.cost_usd == 0.0


def test_cost_gpt4o_mini():
    r = estimate_cost(10_000, 2_000, "gpt-4o-mini")
    # input: 10000/1M * 0.15 = 0.0015; output: 2000/1M * 0.60 = 0.0012
    assert r.cost_usd == pytest.approx(0.0027, rel=0.01)


def test_cost_summary():
    results = [
        CostResult(0.001, 100, 50, "gpt-4o-mini"),
        CostResult(0.002, 200, 80, "gpt-4o-mini"),
        CostResult(0.003, 300, 100, "gpt-4o-mini"),
    ]
    s = compute_cost_summary(results)
    assert s.total_cost_usd == pytest.approx(0.006)
    assert s.mean_cost_per_query == pytest.approx(0.002)
    assert s.total_queries == 3


def test_cost_unknown_model():
    r = estimate_cost(1_000_000, 1_000_000, "unknown-model-xyz")
    assert r.cost_usd > 0  # uses default pricing
