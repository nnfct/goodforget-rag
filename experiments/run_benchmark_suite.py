"""Run the main synthetic validation benchmark suite."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from goodforget_rag.benchmark import evaluate_benchmark, failure_cases, generate_benchmark, summarize, write_jsonl
from goodforget_rag.configs import load_config
from goodforget_rag.spans import documents_to_spans, span_queries
from goodforget_rag.statistics import bootstrap_confidence_intervals


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/benchmark_default.yaml")
    args = parser.parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    documents, queries = generate_benchmark(config, config.seeds[0])
    write_jsonl(results_dir / "benchmark_corpus.jsonl", documents)
    write_jsonl(results_dir / "benchmark_queries.jsonl", queries)

    rows = evaluate_benchmark(documents, queries, config)
    span_documents = documents_to_spans(documents)
    span_q = span_queries(queries, span_documents)
    span_rows = evaluate_benchmark(
        span_documents,
        span_q,
        config,
        representations=["tfidf_word"],
        methods=["GoodForget-RAG"],
    )
    span_rows["method"] = "GoodForget-RAG span-level"
    rows = pd.concat([rows, span_rows], ignore_index=True)

    rows.to_csv(results_dir / "benchmark_query_results.csv", index=False)
    summarize(rows, ["method", "representation"]).to_csv(results_dir / "benchmark_summary.csv", index=False)
    summarize(rows, ["split", "method", "representation"]).to_csv(results_dir / "benchmark_by_split.csv", index=False)
    summarize(rows, ["domain", "method", "representation"]).to_csv(results_dir / "benchmark_by_domain.csv", index=False)
    summarize(rows, ["attack_type", "method", "representation"]).to_csv(results_dir / "benchmark_by_attack.csv", index=False)
    bootstrap_confidence_intervals(
        rows,
        group_cols=["method", "representation"],
        samples=config.bootstrap_samples,
    ).to_csv(results_dir / "bootstrap_confidence_intervals.csv", index=False)
    failure_cases(rows).to_csv(results_dir / "failure_cases.csv", index=False)
    print("\nBenchmark summary")
    print(pd.read_csv(results_dir / "benchmark_summary.csv").to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
