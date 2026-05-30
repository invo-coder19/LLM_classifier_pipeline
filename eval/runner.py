"""
Evaluation Runner — orchestrates the full eval suite.

Usage:
    python -m eval.runner --dataset data/golden_dataset.json --shard 0 --total-shards 1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from pipeline.config import PipelineConfig
from pipeline.llm_client import get_llm_client
from pipeline.rag import get_retriever

from eval.metrics.hallucination import check_hallucination, hallucination_rate
from eval.metrics.relevancy import compute_relevancy, mean_relevancy
from eval.metrics.faithfulness import compute_faithfulness, mean_faithfulness
from eval.metrics.latency import compute_latency_stats
from eval.metrics.cost import estimate_cost, compute_cost_summary
from eval.gates import run_gates

# force_terminal=None lets Rich auto-detect TTY (correct for both CI and local)
console = Console(highlight=False)

SYSTEM_PROMPT = """You are a helpful assistant. Answer the user's question based ONLY on the 
provided context documents. If the answer is not in the context, say 'I don't know based on 
the provided information.' Be concise and factual."""


def build_prompt(question: str, docs: list) -> str:
    context = "\n\n".join([f"[Source {i+1}]: {d.content}" for i, d in enumerate(docs)])
    return f"Context:\n{context}\n\nQuestion: {question}"


def run_evaluation(
    dataset_path: str = "data/golden_dataset.json",
    shard: int = 0,
    total_shards: int = 1,
    config: PipelineConfig | None = None,
) -> Dict[str, Any]:
    cfg = config or PipelineConfig()
    llm = get_llm_client(cfg)
    retriever = get_retriever(cfg)

    with open(dataset_path) as f:
        dataset = json.load(f)["items"]

    # Shard the dataset for parallelism in CI
    dataset = [item for i, item in enumerate(dataset) if i % total_shards == shard]
    console.print(f"[bold cyan]Running shard {shard+1}/{total_shards} -- {len(dataset)} questions[/]")

    per_query_results: List[Dict[str, Any]] = []
    hall_results, rel_results, faith_results, lat_ms, cost_results = [], [], [], [], []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Evaluating...", total=len(dataset))

        for item in dataset:
            qid = item["id"]
            question = item["question"]
            expected = item.get("expected_answer", "")

            # Retrieve sources
            docs = retriever.retrieve(question, top_k=cfg.top_k_docs)
            source_texts = [d.content for d in docs]
            source_refs = [d.source for d in docs]

            # Generate answer
            prompt = build_prompt(question, docs)
            try:
                response = llm.complete(SYSTEM_PROMPT, prompt)
                answer = response.text
                latency = response.latency_ms
                p_tokens = response.prompt_tokens
                c_tokens = response.completion_tokens
                model = response.model
            except Exception as e:  # noqa: BLE001
                console.print(f"[red]Error on {qid}: {e}[/]")
                answer = ""
                latency = float(cfg.timeout_seconds * 1000)
                p_tokens = c_tokens = 0
                model = cfg.model

            # Metrics
            hall = check_hallucination(answer, source_texts)
            rel = compute_relevancy(question, answer)
            faith = compute_faithfulness(answer, source_texts)
            cost = estimate_cost(p_tokens, c_tokens, model)

            hall_results.append(hall)
            rel_results.append(rel)
            faith_results.append(faith)
            lat_ms.append(latency)
            cost_results.append(cost)

            per_query_results.append({
                "id": qid,
                "question": question,
                "expected_answer": expected,
                "generated_answer": answer,
                "source_docs": source_refs,
                "hallucination_verdict": hall.verdict,
                "hallucination_score": hall.max_entailment_score,
                "relevancy_score": rel.score,
                "faithfulness_score": faith.score,
                "latency_ms": latency,
                "cost_usd": cost.cost_usd,
                "category": item.get("category", "general"),
                "difficulty": item.get("difficulty", "medium"),
            })

            progress.advance(task)

    # Aggregate
    lat_stats = compute_latency_stats(lat_ms)
    cost_summary = compute_cost_summary(cost_results)
    h_rate = hallucination_rate(hall_results)
    m_rel = mean_relevancy(rel_results)
    m_faith = mean_faithfulness(faith_results)

    # Gate check
    gate = run_gates(
        hallucination_rate=h_rate,
        p95_latency_ms=lat_stats.p95_ms,
        mean_relevancy=m_rel,
        mean_faithfulness=m_faith,
        mean_cost_per_query=cost_summary.mean_cost_per_query,
        config=cfg,
    )

    # Build result payload
    run_result = {
        "run_id": f"shard{shard}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "shard": shard,
        "total_shards": total_shards,
        "model": cfg.model,
        "provider": cfg.provider,
        "total_questions": len(dataset),
        "metrics": {
            "hallucination_rate": h_rate,
            "mean_answer_relevancy": m_rel,
            "mean_faithfulness": m_faith,
            "latency": {
                "p50_ms": lat_stats.p50_ms,
                "p95_ms": lat_stats.p95_ms,
                "mean_ms": lat_stats.mean_ms,
            },
            "cost": {
                "total_usd": cost_summary.total_cost_usd,
                "mean_per_query_usd": cost_summary.mean_cost_per_query,
            },
        },
        "gate": gate.to_dict(),
        "per_query": per_query_results,
    }

    _print_summary(run_result, gate)
    return run_result


def _print_summary(result: Dict[str, Any], gate) -> None:
    m = result["metrics"]
    table = Table(title="[[ Evaluation Summary ]]", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    table.add_column("Status", style="white")

    def status(ok: bool) -> str:
        return "[green]PASS[/]" if ok else "[red]FAIL[/]"

    hall_ok = m["hallucination_rate"] <= 0.05
    lat_ok = m["latency"]["p95_ms"] <= 5000

    table.add_row("Hallucination Rate", f"{m['hallucination_rate']:.1%}", status(hall_ok))
    table.add_row("Answer Relevancy (mean)", f"{m['mean_answer_relevancy']:.3f}", "")
    table.add_row("Faithfulness (mean)", f"{m['mean_faithfulness']:.3f}", "")
    table.add_row("Latency p50", f"{m['latency']['p50_ms']:.0f} ms", "")
    table.add_row("Latency p95", f"{m['latency']['p95_ms']:.0f} ms", status(lat_ok))
    table.add_row("Cost / query", f"${m['cost']['mean_per_query_usd']:.5f}", "")
    table.add_row("Total Cost", f"${m['cost']['total_usd']:.4f}", "")

    console.print(table)

    if gate.passed:
        console.print("\n[bold green]ALL GATES PASSED -- merge allowed[/]")
    else:
        console.print("\n[bold red]GATE FAILURES -- merge BLOCKED[/]")
        for v in gate.violations:
            console.print(f"  [red]- {v.message}[/]")

    if gate.warnings:
        console.print("\n[yellow]Warnings:[/]")
        for w in gate.warnings:
            console.print(f"  [yellow]- {w.message}[/]")


def main():
    parser = argparse.ArgumentParser(description="LLM Evaluation Runner")
    parser.add_argument("--dataset", default="data/golden_dataset.json")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--total-shards", type=int, default=1)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--fail-on-gate", action="store_true",
                        help="Exit 1 if gate fails (used in CI)")
    args = parser.parse_args()

    cfg = PipelineConfig()
    result = run_evaluation(
        dataset_path=args.dataset,
        shard=args.shard,
        total_shards=args.total_shards,
        config=cfg,
    )

    # Save result
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{result['run_id']}.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    console.print(f"\n[dim]Results saved -> {out_file}[/]")

    if args.fail_on_gate and not result["gate"]["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
