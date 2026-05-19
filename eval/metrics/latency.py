"""
Latency metrics — p50 and p95 across evaluation runs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class LatencyStats:
    p50_ms: float
    p95_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float
    total_queries: int


def compute_latency_stats(latencies_ms: List[float]) -> LatencyStats:
    if not latencies_ms:
        return LatencyStats(0, 0, 0, 0, 0, 0)
    arr = np.array(latencies_ms)
    return LatencyStats(
        p50_ms=float(np.percentile(arr, 50)),
        p95_ms=float(np.percentile(arr, 95)),
        mean_ms=float(np.mean(arr)),
        min_ms=float(np.min(arr)),
        max_ms=float(np.max(arr)),
        total_queries=len(latencies_ms),
    )
