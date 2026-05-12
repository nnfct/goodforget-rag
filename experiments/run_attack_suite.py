"""Summarize benchmark results by attack type."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from goodforget_rag.benchmark import evaluate_benchmark, generate_benchmark, summarize
from goodforget_rag.configs import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/benchmark_default.yaml")
    args = parser.parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    documents, queries = generate_benchmark(config, config.seeds[0])
    rows = evaluate_benchmark(documents, queries, config)
    output = summarize(rows, ["attack_type", "method", "representation"])
    output.to_csv(PROJECT_ROOT / "results" / "benchmark_by_attack.csv", index=False)
    print(output.head(40).to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
