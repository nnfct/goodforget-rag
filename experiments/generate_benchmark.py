"""Generate synthetic benchmark corpus and queries."""

from __future__ import annotations

import argparse
from pathlib import Path

from goodforget_rag.benchmark import generate_benchmark, write_jsonl
from goodforget_rag.configs import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/benchmark_default.yaml")
    args = parser.parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    seed = config.seeds[0]
    documents, queries = generate_benchmark(config, seed)
    write_jsonl(PROJECT_ROOT / "results" / "benchmark_corpus.jsonl", documents)
    write_jsonl(PROJECT_ROOT / "results" / "benchmark_queries.jsonl", queries)
    print(f"Generated {len(documents)} documents and {len(queries)} queries with seed={seed}")


if __name__ == "__main__":
    main()
