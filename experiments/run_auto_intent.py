"""Compare oracle intents with simple heuristic intents."""

from __future__ import annotations

from pathlib import Path

from goodforget_rag.data import load_jsonl, vectorizer_training_texts
from goodforget_rag.eval import build_query_result_row, results_dataframe, summarize_results
from goodforget_rag.intents import heuristic_forget_set, heuristic_positive_intent
from goodforget_rag.retrieval import goodforget_retrieve, vanilla_retrieve
from goodforget_rag.vectorizer import TfidfEncoder


TOP_K = 2
CANDIDATE_K = 5
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    documents = load_jsonl(PROJECT_ROOT / "experiments" / "toy_corpus.jsonl")
    queries = load_jsonl(PROJECT_ROOT / "experiments" / "toy_queries.jsonl")
    encoder = TfidfEncoder().fit(vectorizer_training_texts(documents, queries))

    rows: list[dict[str, object]] = []
    for query in queries:
        query_text = str(query["query"])
        rows.append(
            build_query_result_row(
                "Vanilla RAG",
                query,
                vanilla_retrieve(encoder, documents, query_text, top_k=TOP_K, candidate_k=CANDIDATE_K),
                documents,
            )
        )
        rows.append(
            build_query_result_row(
                "Oracle GoodForget-RAG",
                query,
                goodforget_retrieve(
                    encoder,
                    documents,
                    query_text,
                    str(query["positive_intent"]),
                    [str(item) for item in query["forget_set"]],
                    alpha=0.4,
                    beta=0.8,
                    gamma=0.2,
                    top_k=TOP_K,
                    candidate_k=CANDIDATE_K,
                ),
                documents,
            )
        )
        rows.append(
            build_query_result_row(
                "Heuristic GoodForget-RAG",
                query,
                goodforget_retrieve(
                    encoder,
                    documents,
                    query_text,
                    heuristic_positive_intent(query_text),
                    heuristic_forget_set(query_text),
                    alpha=0.4,
                    beta=0.8,
                    gamma=0.2,
                    top_k=TOP_K,
                    candidate_k=CANDIDATE_K,
                ),
                documents,
            )
        )

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    results = results_dataframe(rows)
    summary = summarize_results(rows)
    results.to_csv(results_dir / "auto_intent_results.csv", index=False)
    summary.to_csv(results_dir / "auto_intent_summary.csv", index=False)
    print("\nOracle vs heuristic intent comparison")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nWrote: {results_dir / 'auto_intent_summary.csv'}")
    print(f"Wrote: {results_dir / 'auto_intent_results.csv'}")


if __name__ == "__main__":
    main()
