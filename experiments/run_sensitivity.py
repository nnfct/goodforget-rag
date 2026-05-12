"""Sensitivity analysis for GoodForget-RAG forget penalty and threshold."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

from goodforget_rag.data import load_jsonl, vectorizer_training_texts
from goodforget_rag.eval import (
    build_query_result_row,
    empty_context_rate,
    leakage_rate,
    over_filter_rate,
    safe_context_recall,
    utility_recall,
)
from goodforget_rag.retrieval import goodforget_retrieve
from goodforget_rag.vectorizer import TfidfEncoder


SEED = 7
TOP_K = 2
CANDIDATE_K = 5
GAMMAS = [0.5, 1.0, 1.5, 2.0]
FORGET_THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30]
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    documents = load_jsonl(PROJECT_ROOT / "experiments" / "toy_corpus.jsonl")
    queries = load_jsonl(PROJECT_ROOT / "experiments" / "toy_queries.jsonl")
    encoder = TfidfEncoder().fit(vectorizer_training_texts(documents, queries))

    sensitivity_rows: list[dict[str, object]] = []
    for gamma in GAMMAS:
        for threshold in FORGET_THRESHOLDS:
            query_rows = []
            for query in queries:
                result = goodforget_retrieve(
                    encoder,
                    documents,
                    str(query["query"]),
                    str(query["positive_intent"]),
                    [str(item) for item in query["forget_set"]],
                    alpha=0.4,
                    beta=0.8,
                    gamma=gamma,
                    top_k=TOP_K,
                    candidate_k=CANDIDATE_K,
                    forget_threshold=threshold,
                    use_threshold=True,
                )
                query_rows.append(
                    build_query_result_row(
                        "GoodForget-RAG",
                        query,
                        result,
                        documents,
                    )
                )
            sensitivity_rows.append(
                {
                    "gamma": gamma,
                    "forget_threshold": threshold,
                    "leakage_rate": leakage_rate(query_rows),
                    "utility_recall": utility_recall(query_rows),
                    "safe_context_recall": safe_context_recall(query_rows),
                    "over_filter_rate": over_filter_rate(query_rows),
                    "empty_context_rate": empty_context_rate(query_rows),
                }
            )

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    sensitivity_df = pd.DataFrame(sensitivity_rows)
    sensitivity_df.to_csv(results_dir / "sensitivity.csv", index=False)

    print("\nGoodForget-RAG sensitivity analysis")
    print(
        sensitivity_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )
    print(f"\nWrote: {results_dir / 'sensitivity.csv'}")


if __name__ == "__main__":
    main()
