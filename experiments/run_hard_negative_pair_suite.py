"""Run hard negative pair stress tests for GoodForget-RAG."""

from __future__ import annotations

from pathlib import Path

from goodforget_rag.benchmark import write_jsonl
from goodforget_rag.hard_negatives import (
    HardNegativeConfig,
    evaluate_hard_negative_pairs,
    generate_hard_negative_pairs,
    label_consistency_check,
    mine_hard_negative_cases,
    pairwise_deltas,
    summarize_hard_negative,
    write_hard_negative_report,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    results_dir = PROJECT_ROOT / "results"
    docs_dir = PROJECT_ROOT / "docs"
    results_dir.mkdir(exist_ok=True)
    docs_dir.mkdir(exist_ok=True)

    config = HardNegativeConfig()
    documents, queries = generate_hard_negative_pairs(config)
    rows = evaluate_hard_negative_pairs(documents, queries, config)
    summary = summarize_hard_negative(rows)
    cases = mine_hard_negative_cases(rows)
    deltas = pairwise_deltas(summary)
    label_check = label_consistency_check(documents)

    write_jsonl(results_dir / "hard_negative_pair_corpus.jsonl", documents)
    write_jsonl(results_dir / "hard_negative_pair_queries.jsonl", queries)
    rows.to_csv(results_dir / "hard_negative_pair_query_results.csv", index=False)
    summary.to_csv(results_dir / "hard_negative_pair_summary.csv", index=False)
    cases.to_csv(results_dir / "hard_negative_pair_cases.csv", index=False)
    deltas.to_csv(results_dir / "hard_negative_pair_pairwise_deltas.csv", index=False)
    label_check.to_csv(results_dir / "label_consistency_check.csv", index=False)
    write_hard_negative_report(summary, cases, docs_dir / "hard_negative_pair_report.html")

    print("\nHard negative pair suite")
    print(f"Documents: {len(documents)} | Queries: {len(queries)}")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nWrote: {results_dir / 'hard_negative_pair_summary.csv'}")
    print(f"Wrote: {results_dir / 'hard_negative_pair_cases.csv'}")
    print(f"Wrote: {results_dir / 'hard_negative_pair_pairwise_deltas.csv'}")
    print(f"Wrote: {results_dir / 'label_consistency_check.csv'}")
    print(f"Wrote: {docs_dir / 'hard_negative_pair_report.html'}")


if __name__ == "__main__":
    main()
