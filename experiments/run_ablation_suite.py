"""Run one-factor GoodForget-RAG ablations."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from goodforget_rag.benchmark import evaluate_benchmark, generate_benchmark, summarize
from goodforget_rag.configs import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/benchmark_fast.yaml")
    args = parser.parse_args()
    base = load_config(PROJECT_ROOT / args.config)
    documents, queries = generate_benchmark(base, base.seeds[0])
    rows = []
    grids = {
        "alpha": [0.0, 0.5, 1.0],
        "beta": [0.0, 0.5, 1.0],
        "gamma": [0.0, 0.5, 1.0, 1.5, 2.0],
        "top_k": [3, 5, 8],
        "candidate_k": [10, 20, 40],
    }
    for parameter, values in grids.items():
        for value in values:
            config = replace(base, **{parameter: value})
            result = evaluate_benchmark(
                documents,
                queries,
                config,
                representations=["tfidf_word"],
                methods=["GoodForget-RAG"],
            )
            summary = summarize(result, ["method", "representation"])
            summary["parameter"] = parameter
            summary["value"] = value
            rows.append(summary)
    for value in [None, 0.10, 0.15, 0.20, 0.25, 0.30]:
        config = replace(base, forget_threshold=0.0 if value is None else float(value))
        method = "GoodForget-RAG" if value is None else "GoodForget-RAG hard threshold"
        result = evaluate_benchmark(
            documents,
            queries,
            config,
            representations=["tfidf_word"],
            methods=[method],
        )
        summary = summarize(result, ["method", "representation"])
        summary["parameter"] = "forget_threshold"
        summary["value"] = "None" if value is None else value
        rows.append(summary)
    output = pd.concat(rows, ignore_index=True)
    output.to_csv(PROJECT_ROOT / "results" / "ablation_summary.csv", index=False)
    print(output.to_string(index=False, float_format=lambda metric: f"{metric:.3f}"))


if __name__ == "__main__":
    main()
