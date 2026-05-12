"""Evaluation helpers for the toy retrieval comparison."""

from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd

from goodforget_rag.data import document_lookup, short_snippet
from goodforget_rag.retrieval import RetrievalResult


def leakage_rate(rows: Sequence[dict[str, object]]) -> float:
    """Fraction of queries where a forbidden document appears in top-k context."""

    if not rows:
        return 0.0
    return sum(bool(row["leaked"]) for row in rows) / len(rows)


def utility_recall(rows: Sequence[dict[str, object]]) -> float:
    """Mean fraction of expected useful documents retrieved."""

    if not rows:
        return 0.0
    return sum(float(row["utility_recall"]) for row in rows) / len(rows)


def safe_context_recall(rows: Sequence[dict[str, object]]) -> float:
    """Mean recall over expected relevant documents that are not forbidden."""

    if not rows:
        return 0.0
    return sum(float(row["safe_context_recall"]) for row in rows) / len(rows)


def over_filter_rate(rows: Sequence[dict[str, object]]) -> float:
    """Mean fraction of expected safe documents not preserved in context."""

    if not rows:
        return 0.0
    return sum(float(row["over_filter_rate"]) for row in rows) / len(rows)


def forbidden_removal_precision(rows: Sequence[dict[str, object]]) -> float:
    """Among removed documents, fraction that were actually forbidden."""

    removed = sum(int(row["num_removed"]) for row in rows)
    if removed == 0:
        return 0.0
    forbidden_removed = sum(int(row["num_forbidden_removed"]) for row in rows)
    return forbidden_removed / removed


def empty_context_rate(rows: Sequence[dict[str, object]]) -> float:
    """Fraction of queries where retrieval returns no selected context."""

    if not rows:
        return 0.0
    return sum(int(row["num_retrieved"]) == 0 for row in rows) / len(rows)


def answer_leakage_rate(rows: Sequence[dict[str, object]]) -> float:
    """Fraction of deterministic simulated answers with forbidden markers."""

    if not rows:
        return 0.0
    return sum(bool(row["answer_leaked"]) for row in rows) / len(rows)


def build_query_result_row(
    method: str,
    query: dict[str, object],
    result: RetrievalResult,
    documents: Sequence[dict[str, object]],
) -> dict[str, object]:
    """Build one query-level evaluation row."""

    docs_by_id = document_lookup(documents)
    retrieved_ids = [hit.doc_id for hit in result.selected]
    removed_ids = result.removed_doc_ids
    forbidden_ids = set(_as_str_list(query["forbidden_doc_ids"]))
    expected_ids = set(_as_str_list(query["expected_relevant_doc_ids"]))
    expected_safe_ids = expected_ids - forbidden_ids
    leaked_ids = sorted(forbidden_ids.intersection(retrieved_ids))
    useful_ids = sorted(expected_ids.intersection(retrieved_ids))
    safe_useful_ids = sorted(expected_safe_ids.intersection(retrieved_ids))
    removed_forbidden_ids = sorted(
        item.doc_id
        for item in result.diagnostics
        if (item.removed_by_filter or item.removed_by_threshold) and item.is_forbidden
    )
    utility = len(useful_ids) / len(expected_ids) if expected_ids else 0.0
    safe_recall = len(safe_useful_ids) / len(expected_safe_ids) if expected_safe_ids else 0.0
    over_filter = 1.0 - safe_recall if expected_safe_ids else 0.0
    simulated_answer = simulate_answer(result, documents)
    answer_leaked = detect_answer_leakage(simulated_answer, query, documents)

    return {
        "method": method,
        "split": str(query["split"]),
        "query_id": str(query["query_id"]),
        "retrieved_doc_ids": "|".join(retrieved_ids),
        "removed_doc_ids": "|".join(removed_ids),
        "leaked_doc_ids": "|".join(leaked_ids),
        "useful_doc_ids": "|".join(useful_ids),
        "safe_useful_doc_ids": "|".join(safe_useful_ids),
        "removed_forbidden_doc_ids": "|".join(removed_forbidden_ids),
        "leaked": bool(leaked_ids),
        "answer_leaked": bool(answer_leaked),
        "num_retrieved": len(retrieved_ids),
        "num_removed": len(removed_ids),
        "num_forbidden_removed": len(removed_forbidden_ids),
        "utility_recall": utility,
        "safe_context_recall": safe_recall,
        "over_filter_rate": over_filter,
        "top_score": result.selected[0].final_score if result.selected else float("nan"),
        "simulated_answer": simulated_answer,
    }


def summarize_results(rows: Sequence[dict[str, object]]) -> pd.DataFrame:
    """Aggregate query-level rows into required metrics by method."""

    return _summarize(rows, group_cols=["method"])


def summarize_by_split(rows: Sequence[dict[str, object]]) -> pd.DataFrame:
    """Aggregate query-level rows into required metrics by method and split."""

    return _summarize(rows, group_cols=["split", "method"])


def results_dataframe(rows: Iterable[dict[str, object]]) -> pd.DataFrame:
    """Return a stable DataFrame for writing per-query CSV output."""

    return pd.DataFrame(list(rows))


def simulate_answer(result: RetrievalResult, documents: Sequence[dict[str, object]]) -> str:
    """Create a deterministic evidence-only answer proxy.

    This is not an LLM answer. It concatenates selected evidence titles and short
    snippets so leakage can be measured without external services.
    """

    docs_by_id = document_lookup(documents)
    parts = []
    for hit in result.selected:
        document = docs_by_id[hit.doc_id]
        parts.append(f"{hit.doc_id}: {document['title']} - {short_snippet(document)}")
    return " || ".join(parts)


def detect_answer_leakage(
    simulated_answer: str,
    query: dict[str, object],
    documents: Sequence[dict[str, object]],
) -> bool:
    """Detect forbidden titles, IDs, or key phrases in the answer proxy."""

    lowered = simulated_answer.lower()
    docs_by_id = document_lookup(documents)
    for doc_id in _as_str_list(query["forbidden_doc_ids"]):
        document = docs_by_id[doc_id]
        title = str(document["title"]).lower()
        if doc_id.lower() in lowered or title in lowered:
            return True
        for phrase in _forbidden_phrases(document):
            if phrase and phrase in lowered:
                return True
    return False


def _summarize(rows: Sequence[dict[str, object]], group_cols: list[str]) -> pd.DataFrame:
    summary_rows = []
    sorted_groups = sorted({tuple(str(row[col]) for col in group_cols) for row in rows})
    for group_key in sorted_groups:
        group = dict(zip(group_cols, group_key))
        group_rows = [
            row
            for row in rows
            if tuple(str(row[col]) for col in group_cols) == group_key
        ]
        summary_rows.append(
            {
                **group,
                "leakage_rate": leakage_rate(group_rows),
                "utility_recall": utility_recall(group_rows),
                "safe_context_recall": safe_context_recall(group_rows),
                "over_filter_rate": over_filter_rate(group_rows),
                "forbidden_removal_precision": forbidden_removal_precision(group_rows),
                "empty_context_rate": empty_context_rate(group_rows),
                "answer_leakage_rate": answer_leakage_rate(group_rows),
                "num_queries": len(group_rows),
            }
        )
    frame = pd.DataFrame(summary_rows)
    method_order = {
        "Vanilla RAG": 0,
        "Positive-only RAG": 1,
        "Query Rewrite RAG": 2,
        "Keyword Blocklist": 3,
        "Metadata Filter": 4,
        "GoodForget-RAG": 5,
    }
    sort_cols = [col for col in ["split", "method"] if col in frame.columns]
    if "method" in sort_cols:
        return frame.sort_values(
            by=sort_cols,
            key=lambda series: series.map(lambda value: method_order.get(value, value)),
        )
    return frame


def _forbidden_phrases(document: dict[str, object]) -> list[str]:
    title = str(document.get("title", "")).lower()
    tags = " ".join(str(tag) for tag in document.get("tags", [])).lower()
    notes = str(document.get("notes", "")).lower()
    return [title, tags, notes]


def _as_str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]
