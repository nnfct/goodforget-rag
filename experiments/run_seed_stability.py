"""Run synthetic benchmark across seeds."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from goodforget_rag.benchmark import evaluate_benchmark, generate_benchmark
from goodforget_rag.configs import load_config
from goodforget_rag.statistics import seed_stability


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/benchmark_fast.yaml")
    args = parser.parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    all_rows = []
    for seed in config.seeds:
        documents, queries = generate_benchmark(config, seed)
        rows = evaluate_benchmark(
            documents,
            queries,
            config,
            representations=["tfidf_word"],
            methods=["Vanilla RAG", "Metadata Filter", "GoodForget-RAG", "GoodForget-RAG hard threshold"],
        )
        rows["seed"] = seed
        all_rows.append(rows)
    frame = pd.concat(all_rows, ignore_index=True)
    summary = seed_stability(frame)
    summary.to_csv(PROJECT_ROOT / "results" / "seed_stability_summary.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
