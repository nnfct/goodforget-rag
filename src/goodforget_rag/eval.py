"""Evaluation helpers for the toy retrieval comparison."""

from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd

from goodforget_rag.retrieval import RetrievalHit


def leakage_rate(rows: Sequence[dict[str, object]]) -> float:
    """Fraction of queries where at least one forbidden document is retrieved."""

    if not rows:
        return 0.0
    leaked = sum(bool(row["leaked"]) for row in rows)
    return leaked / len(rows)


def utility_recall(rows: Sequence[dict[str, object]]) -> float:
    """Mean fraction of expected useful documents retrieved."""

    if not rows:
        return 0.0
    return sum(float(row["utility_recall"]) for row in rows) / len(rows)


def empty_context_rate(rows: Sequence[dict[str, object]]) -> float:
    """Fraction of queries where retrieval returns no selected context."""

    if not rows:
        return 0.0
    empty = sum(int(row["num_retrieved"]) == 0 for row in rows)
    return empty / len(rows)


def build_query_result_row(
    method: str,
    query: dict[str, object],
    hits: Sequence[RetrievalHit],
) -> dict[str, object]:
    """Build one query-level evaluation row."""

    retrieved_ids = [hit.doc_id for hit in hits]
    forbidden_ids = set(_as_str_list(query["forbidden_doc_ids"]))
    expected_ids = set(_as_str_list(query["expected_relevant_doc_ids"]))
    leaked_ids = sorted(forbidden_ids.intersection(retrieved_ids))
    useful_ids = sorted(expected_ids.intersection(retrieved_ids))
    utility = len(useful_ids) / len(expected_ids) if expected_ids else 0.0
    return {
        "method": method,
        "query_id": str(query["query_id"]),
        "retrieved_doc_ids": "|".join(retrieved_ids),
        "leaked_doc_ids": "|".join(leaked_ids),
        "useful_doc_ids": "|".join(useful_ids),
        "leaked": bool(leaked_ids),
        "num_retrieved": len(retrieved_ids),
        "utility_recall": utility,
        "top_score": hits[0].score if hits else float("nan"),
    }


def summarize_results(rows: Sequence[dict[str, object]]) -> pd.DataFrame:
    """Aggregate query-level rows into the required metrics by method."""

    summary_rows = []
    for method in sorted({str(row["method"]) for row in rows}):
        method_rows = [row for row in rows if row["method"] == method]
        summary_rows.append(
            {
                "method": method,
                "leakage_rate": leakage_rate(method_rows),
                "utility_recall": utility_recall(method_rows),
                "empty_context_rate": empty_context_rate(method_rows),
                "num_queries": len(method_rows),
            }
        )
    order = {"Vanilla RAG": 0, "Positive-only RAG": 1, "GoodForget-RAG": 2}
    return pd.DataFrame(summary_rows).sort_values(
        by="method",
        key=lambda series: series.map(lambda value: order.get(value, 99)),
    )


def results_dataframe(rows: Iterable[dict[str, object]]) -> pd.DataFrame:
    """Return a stable DataFrame for writing per-query CSV output."""

    return pd.DataFrame(list(rows))


def _as_str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]
