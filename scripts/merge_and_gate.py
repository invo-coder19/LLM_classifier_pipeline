"""
Merge shard result JSONs and run the final gate check.
Called by the gate-check CI job.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pipeline.config import PipelineConfig
from eval.gates import run_gates
from eval.metrics.latency import compute_latency_stats
from eval.metrics.cost import compute_cost_summary, CostResult


def merge_and_gate(results_dir: str, output_path: str) -> bool:
    cfg = PipelineConfig()
    result_files = list(Path(results_dir).glob("*.json"))

    if not result_files:
        print("❌ No result files found in", results_dir)
        sys.exit(1)

    all_queries, all_lat, all_hall, all_rel, all_faith, all_costs = [], [], [], [], [], []

    for f in result_files:
        data = json.loads(f.read_text())
        queries = data.get("per_query", [])
        all_queries.extend(queries)
        for q in queries:
            all_lat.append(q.get("latency_ms", 0))
            all_hall.append(q.get("hallucination_verdict") == "hallucinated")
            all_rel.append(q.get("relevancy_score", 0))
            all_faith.append(q.get("faithfulness_score", 0))
            all_costs.append(CostResult(
                cost_usd=q.get("cost_usd", 0.0),
                prompt_tokens=0,
                completion_tokens=0,
                model=data.get("model", "mock-model"),
            ))

    total = len(all_queries)
    if total == 0:
        print("❌ No queries found")
        sys.exit(1)

    lat_stats = compute_latency_stats(all_lat)
    cost_summary = compute_cost_summary(all_costs)
    h_rate = sum(all_hall) / total
    m_rel = sum(all_rel) / total
    m_faith = sum(all_faith) / total

    gate = run_gates(h_rate, lat_stats.p95_ms, m_rel, m_faith,
                     cost_summary.mean_cost_per_query, cfg)

    merged = {
        "passed": gate.passed,
        "total_questions": total,
        "shards": len(result_files),
        "metrics": {
            "hallucination_rate": h_rate,
            "mean_relevancy": m_rel,
            "mean_faithfulness": m_faith,
            "p95_latency_ms": lat_stats.p95_ms,
            "p50_latency_ms": lat_stats.p50_ms,
            "mean_latency_ms": lat_stats.mean_ms,
            "mean_cost_per_query": cost_summary.mean_cost_per_query,
            "total_cost_usd": cost_summary.total_cost_usd,
        },
        "violations": [v.__dict__ for v in gate.violations],
        "warnings": [w.__dict__ for w in gate.warnings],
    }

    Path(output_path).write_text(json.dumps(merged, indent=2))
    print(f"{'✅ GATE PASSED' if gate.passed else '🚨 GATE FAILED'} — {total} questions across {len(result_files)} shards")
    print(f"  hallucination={h_rate:.1%}  p95={lat_stats.p95_ms:.0f}ms  relevancy={m_rel:.3f}")
    return gate.passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge shard results and run gate check")
    parser.add_argument("--results-dir", default="results",
                        help="Directory containing shard result JSON files")
    parser.add_argument("--output", default="gate_result.json",
                        help="Output path for the merged gate result JSON")
    parser.add_argument("--fail-on-gate", action="store_true",
                        help="Exit 1 if gate fails (useful for local runs; in CI the workflow handles this)")
    args = parser.parse_args()
    passed = merge_and_gate(args.results_dir, args.output)
    # In CI: always exit 0 here — the workflow reads gate_passed from the JSON output
    # and handles the blocking failure in a separate step.
    # With --fail-on-gate: exit 1 for local developer convenience.
    sys.exit(0 if (passed or not args.fail_on_gate) else 1)
