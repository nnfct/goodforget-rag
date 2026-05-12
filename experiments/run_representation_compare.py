"""Compare TF-IDF with a local LSA representation."""

from __future__ import annotations

from pathlib import Path

from goodforget_rag.data import load_jsonl, vectorizer_training_texts
from goodforget_rag.eval import build_query_result_row, results_dataframe, summarize_results
from goodforget_rag.retrieval import goodforget_retrieve, vanilla_retrieve
from goodforget_rag.vectorizer import LsaEncoder, TfidfEncoder


TOP_K = 2
CANDIDATE_K = 5
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    documents = load_jsonl(PROJECT_ROOT / "experiments" / "toy_corpus.jsonl")
    queries = load_jsonl(PROJECT_ROOT / "experiments" / "toy_queries.jsonl")
    training_texts = vectorizer_training_texts(documents, queries)
    encoders = {
        "TF-IDF": TfidfEncoder().fit(training_texts),
        "Local LSA": LsaEncoder(max_components=8).fit(training_texts),
    }

    rows: list[dict[str, object]] = []
    for representation, encoder in encoders.items():
        for query in queries:
            query_text = str(query["query"])
            rows.append(
                {
                    **build_query_result_row(
                        "Vanilla RAG",
                        query,
                        vanilla_retrieve(encoder, documents, query_text, top_k=TOP_K, candidate_k=CANDIDATE_K),
                        documents,
                    ),
                    "representation": representation,
                }
            )
            rows.append(
                {
                    **build_query_result_row(
                        "GoodForget-RAG",
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
                    ),
                    "representation": representation,
                }
            )

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    results = results_dataframe(rows)
    summary = (
        results.groupby(["representation", "method"], as_index=False)
        .agg(
            leakage_rate=("leaked", "mean"),
            utility_recall=("utility_recall", "mean"),
            safe_context_recall=("safe_context_recall", "mean"),
            over_filter_rate=("over_filter_rate", "mean"),
            answer_leakage_rate=("answer_leaked", "mean"),
            num_queries=("query_id", "count"),
        )
        .sort_values(["representation", "method"])
    )
    results.to_csv(results_dir / "representation_results.csv", index=False)
    summary.to_csv(results_dir / "representation_summary.csv", index=False)
    print("\nRepresentation sensitivity comparison")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nWrote: {results_dir / 'representation_summary.csv'}")
    print(f"Wrote: {results_dir / 'representation_results.csv'}")


if __name__ == "__main__":
    main()
