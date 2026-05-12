"""Run the deterministic GoodForget-RAG toy experiment."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np

from goodforget_rag.eval import build_query_result_row, results_dataframe, summarize_results
from goodforget_rag.retrieval import (
    goodforget_retrieve,
    positive_only_retrieve,
    vanilla_retrieve,
)
from goodforget_rag.vectorizer import TfidfEncoder, document_to_text


SEED = 7
TOP_K = 2
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def vectorizer_training_texts(
    documents: Iterable[dict[str, object]],
    queries: Iterable[dict[str, object]],
) -> list[str]:
    texts = [document_to_text(doc) for doc in documents]
    for query in queries:
        texts.append(str(query["query"]))
        texts.append(str(query["positive_intent"]))
        texts.extend(str(item) for item in query["forget_set"])
    return texts


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    corpus_path = PROJECT_ROOT / "experiments" / "toy_corpus.jsonl"
    queries_path = PROJECT_ROOT / "experiments" / "toy_queries.jsonl"
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    documents = load_jsonl(corpus_path)
    queries = load_jsonl(queries_path)
    encoder = TfidfEncoder().fit(vectorizer_training_texts(documents, queries))

    rows: list[dict[str, object]] = []
    for query in queries:
        rows.append(
            build_query_result_row(
                "Vanilla RAG",
                query,
                vanilla_retrieve(encoder, documents, str(query["query"]), top_k=TOP_K),
            )
        )
        rows.append(
            build_query_result_row(
                "Positive-only RAG",
                query,
                positive_only_retrieve(
                    encoder,
                    documents,
                    str(query["positive_intent"]),
                    top_k=TOP_K,
                ),
            )
        )
        rows.append(
            build_query_result_row(
                "GoodForget-RAG",
                query,
                goodforget_retrieve(
                    encoder,
                    documents,
                    str(query["query"]),
                    str(query["positive_intent"]),
                    [str(item) for item in query["forget_set"]],
                    alpha=0.4,
                    beta=0.8,
                    gamma=1.2,
                    top_k=TOP_K,
                ),
            )
        )

    results_df = results_dataframe(rows)
    summary_df = summarize_results(rows)
    results_df.to_csv(results_dir / "toy_results.csv", index=False)
    summary_df.to_csv(results_dir / "summary.csv", index=False)

    print("\nGoodForget-RAG toy experiment")
    print(f"Top-k: {TOP_K}")
    print(summary_df.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nWrote: {results_dir / 'summary.csv'}")
    print(f"Wrote: {results_dir / 'toy_results.csv'}")


if __name__ == "__main__":
    main()
