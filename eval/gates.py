"""
Gate Check — enforces quality thresholds.
Returns structured pass/fail results that GitHub Actions reads.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any

from pipeline.config import PipelineConfig


@dataclass
class GateViolation:
    metric: str
    threshold: float
    actual: float
    message: str
    severity: str  # "blocking" | "warning"


@dataclass
class GateResult:
    passed: bool
    violations: List[GateViolation] = field(default_factory=list)
    warnings: List[GateViolation] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "violations": [
                {
                    "metric": v.metric,
                    "threshold": v.threshold,
                    "actual": v.actual,
                    "message": v.message,
                    "severity": v.severity,
                }
                for v in self.violations
            ],
            "warnings": [
                {
                    "metric": v.metric,
                    "threshold": v.threshold,
                    "actual": v.actual,
                    "message": v.message,
                    "severity": v.severity,
                }
                for v in self.warnings
            ],
            "summary": self.summary,
        }


def run_gates(
    hallucination_rate: float,
    p95_latency_ms: float,
    mean_relevancy: float,
    mean_faithfulness: float,
    mean_cost_per_query: float,
    config: PipelineConfig | None = None,
) -> GateResult:
    """
    Evaluate all metrics against configured thresholds.
    Blocking violations cause CI to fail (exit 1).
    Warnings are informational only.
    """
    cfg = config or PipelineConfig()
    violations: List[GateViolation] = []
    warnings: List[GateViolation] = []

    # ── Blocking Gates ─────────────────────────────────────────────────────────
    if hallucination_rate > cfg.hallucination_threshold:
        violations.append(GateViolation(
            metric="hallucination_rate",
            threshold=cfg.hallucination_threshold,
            actual=hallucination_rate,
            message=(
                f"Hallucination rate {hallucination_rate:.1%} exceeds "
                f"threshold {cfg.hallucination_threshold:.1%}"
            ),
            severity="blocking",
        ))

    if p95_latency_ms > cfg.p95_latency_ms:
        violations.append(GateViolation(
            metric="p95_latency_ms",
            threshold=cfg.p95_latency_ms,
            actual=p95_latency_ms,
            message=(
                f"p95 latency {p95_latency_ms:.0f}ms exceeds SLA "
                f"{cfg.p95_latency_ms:.0f}ms"
            ),
            severity="blocking",
        ))

    # ── Warning Gates ──────────────────────────────────────────────────────────
    if mean_relevancy < cfg.min_answer_relevancy:
        warnings.append(GateViolation(
            metric="answer_relevancy",
            threshold=cfg.min_answer_relevancy,
            actual=mean_relevancy,
            message=(
                f"Mean relevancy {mean_relevancy:.2f} below "
                f"target {cfg.min_answer_relevancy:.2f}"
            ),
            severity="warning",
        ))

    if mean_faithfulness < cfg.min_faithfulness:
        warnings.append(GateViolation(
            metric="faithfulness",
            threshold=cfg.min_faithfulness,
            actual=mean_faithfulness,
            message=(
                f"Mean faithfulness {mean_faithfulness:.2f} below "
                f"target {cfg.min_faithfulness:.2f}"
            ),
            severity="warning",
        ))

    if mean_cost_per_query > cfg.max_cost_per_query:
        warnings.append(GateViolation(
            metric="cost_per_query",
            threshold=cfg.max_cost_per_query,
            actual=mean_cost_per_query,
            message=(
                f"Mean cost ${mean_cost_per_query:.4f} exceeds "
                f"budget ${cfg.max_cost_per_query:.4f}"
            ),
            severity="warning",
        ))

    passed = len(violations) == 0
    summary = {
        "hallucination_rate": hallucination_rate,
        "p95_latency_ms": p95_latency_ms,
        "mean_relevancy": mean_relevancy,
        "mean_faithfulness": mean_faithfulness,
        "mean_cost_per_query": mean_cost_per_query,
        "blocking_violations": len(violations),
        "warnings": len(warnings),
    }

    return GateResult(passed=passed, violations=violations, warnings=warnings, summary=summary)
