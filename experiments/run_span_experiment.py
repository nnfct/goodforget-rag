"""Span-level variant for mixed-evidence limitation checks."""

from __future__ import annotations

from pathlib import Path

from goodforget_rag.data import load_jsonl, vectorizer_training_texts
from goodforget_rag.eval import build_query_result_row, results_dataframe, summarize_results
from goodforget_rag.retrieval import goodforget_retrieve, metadata_filter_retrieve, vanilla_retrieve
from goodforget_rag.spans import documents_to_spans, span_queries
from goodforget_rag.vectorizer import TfidfEncoder


TOP_K = 3
CANDIDATE_K = 8
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    documents = load_jsonl(PROJECT_ROOT / "experiments" / "toy_corpus.jsonl")
    queries = load_jsonl(PROJECT_ROOT / "experiments" / "toy_queries.jsonl")
    spans = documents_to_spans(documents)
    mapped_queries = span_queries(queries, spans)
    encoder = TfidfEncoder().fit(vectorizer_training_texts(spans, mapped_queries))

    rows: list[dict[str, object]] = []
    for query in mapped_queries:
        query_text = str(query["query"])
        rows.append(
            build_query_result_row(
                "Span Vanilla RAG",
                query,
                vanilla_retrieve(encoder, spans, query_text, top_k=TOP_K, candidate_k=CANDIDATE_K),
                spans,
            )
        )
        rows.append(
            build_query_result_row(
                "Span Metadata Filter",
                query,
                metadata_filter_retrieve(encoder, spans, query_text, top_k=TOP_K, candidate_k=CANDIDATE_K),
                spans,
            )
        )
        rows.append(
            build_query_result_row(
                "Span GoodForget-RAG",
                query,
                goodforget_retrieve(
                    encoder,
                    spans,
                    query_text,
                    str(query["positive_intent"]),
                    [str(item) for item in query["forget_set"]],
                    alpha=0.4,
                    beta=0.8,
                    gamma=0.2,
                    top_k=TOP_K,
                    candidate_k=CANDIDATE_K,
                ),
                spans,
            )
        )

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    results = results_dataframe(rows)
    summary = summarize_results(rows)
    results.to_csv(results_dir / "span_results.csv", index=False)
    summary.to_csv(results_dir / "span_summary.csv", index=False)
    print("\nSpan-level limitation check")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nWrote: {results_dir / 'span_summary.csv'}")
    print(f"Wrote: {results_dir / 'span_results.csv'}")


if __name__ == "__main__":
    main()
