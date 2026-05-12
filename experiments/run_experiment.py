"""Run the deterministic GoodForget-RAG toy experiment."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable

import numpy as np

from goodforget_rag.data import load_jsonl, vectorizer_training_texts
from goodforget_rag.eval import (
    build_query_result_row,
    results_dataframe,
    summarize_by_split,
    summarize_results,
)
from goodforget_rag.retrieval import (
    RetrievalResult,
    goodforget_retrieve,
    keyword_blocklist_retrieve,
    metadata_filter_retrieve,
    positive_only_retrieve,
    query_rewrite_retrieve,
    vanilla_retrieve,
)
from goodforget_rag.vectorizer import TfidfEncoder


SEED = 7
TOP_K = 2
CANDIDATE_K = 5
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_methods(
    encoder: TfidfEncoder,
    documents: list[dict[str, object]],
    query: dict[str, object],
) -> list[tuple[str, RetrievalResult]]:
    """Run all retrieval methods for one query."""

    query_text = str(query["query"])
    positive_intent = str(query["positive_intent"])
    forget_set = [str(item) for item in query["forget_set"]]
    methods: list[tuple[str, Callable[[], RetrievalResult]]] = [
        (
            "Vanilla RAG",
            lambda: vanilla_retrieve(
                encoder,
                documents,
                query_text,
                top_k=TOP_K,
                candidate_k=CANDIDATE_K,
            ),
        ),
        (
            "Positive-only RAG",
            lambda: positive_only_retrieve(
                encoder,
                documents,
                positive_intent,
                top_k=TOP_K,
                candidate_k=CANDIDATE_K,
            ),
        ),
        (
            "Query Rewrite RAG",
            lambda: query_rewrite_retrieve(
                encoder,
                documents,
                positive_intent,
                top_k=TOP_K,
                candidate_k=CANDIDATE_K,
            ),
        ),
        (
            "Keyword Blocklist",
            lambda: keyword_blocklist_retrieve(
                encoder,
                documents,
                query_text,
                top_k=TOP_K,
                candidate_k=CANDIDATE_K,
            ),
        ),
        (
            "Metadata Filter",
            lambda: metadata_filter_retrieve(
                encoder,
                documents,
                query_text,
                top_k=TOP_K,
                candidate_k=CANDIDATE_K,
            ),
        ),
        (
            "GoodForget-RAG",
            lambda: goodforget_retrieve(
                encoder,
                documents,
                query_text,
                positive_intent,
                forget_set,
                alpha=0.4,
                beta=0.8,
                gamma=0.2,
                top_k=TOP_K,
                candidate_k=CANDIDATE_K,
                forget_threshold=None,
                use_threshold=False,
            ),
        ),
    ]
    return [(name, runner()) for name, runner in methods]


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
        for method, result in run_methods(encoder, documents, query):
            rows.append(build_query_result_row(method, query, result, documents))

    results_df = results_dataframe(rows)
    summary_df = summarize_results(rows)
    split_summary_df = summarize_by_split(rows)

    results_df.to_csv(results_dir / "toy_results.csv", index=False)
    summary_df.to_csv(results_dir / "summary.csv", index=False)
    split_summary_df.to_csv(results_dir / "split_summary.csv", index=False)

    print("\nGoodForget-RAG red-team toy experiment")
    print(f"Top-k: {TOP_K} | Candidate-k: {CANDIDATE_K}")
    print(summary_df.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("\nBy split")
    print(split_summary_df.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nWrote: {results_dir / 'summary.csv'}")
    print(f"Wrote: {results_dir / 'split_summary.csv'}")
    print(f"Wrote: {results_dir / 'toy_results.csv'}")


if __name__ == "__main__":
    main()
