"""Sentence-span helpers for document-level limitation checks."""

from __future__ import annotations

import re
from typing import Sequence


FORBIDDEN_SPAN_HINTS = (
    "project maple",
    "obsolete flat credit",
    "confidential",
    "secret",
    "leaked",
    "private customer",
    "temporary passwords",
    "orchard route",
    "board-level path",
    "quiet entry",
)


def documents_to_spans(documents: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    """Split documents into sentence-like spans with parent metadata."""

    spans: list[dict[str, object]] = []
    for document in documents:
        text = str(document.get("text", ""))
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", text)
            if sentence.strip()
        ]
        if not sentences:
            sentences = [text]
        for idx, sentence in enumerate(sentences):
            parent_id = str(document["doc_id"])
            is_forbidden = _is_forbidden_span(sentence, document)
            spans.append(
                {
                    "doc_id": f"{parent_id}::span_{idx}",
                    "parent_doc_id": parent_id,
                    "title": f"{document['title']} [span {idx}]",
                    "text": sentence,
                    "tags": document.get("tags", []),
                    "split": document.get("split", ""),
                    "is_forbidden": is_forbidden,
                    "notes": f"parent={parent_id}; span_forbidden={is_forbidden}",
                }
            )
    return spans


def span_queries(queries: Sequence[dict[str, object]], spans: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    """Map document-level query labels to span-level labels."""

    spans_by_parent: dict[str, list[dict[str, object]]] = {}
    for span in spans:
        spans_by_parent.setdefault(str(span["parent_doc_id"]), []).append(span)

    mapped_queries = []
    for query in queries:
        expected_span_ids = []
        for doc_id in query["expected_relevant_doc_ids"]:
            expected_span_ids.extend(
                str(span["doc_id"])
                for span in spans_by_parent.get(str(doc_id), [])
                if not bool(span.get("is_forbidden", False))
            )
        forbidden_span_ids = []
        for doc_id in query["forbidden_doc_ids"]:
            expected_span_ids.extend(
                str(span["doc_id"])
                for span in spans_by_parent.get(str(doc_id), [])
                if not bool(span.get("is_forbidden", False))
            )
            forbidden_span_ids.extend(
                str(span["doc_id"])
                for span in spans_by_parent.get(str(doc_id), [])
                if bool(span.get("is_forbidden", False))
            )
        mapped = dict(query)
        mapped["expected_relevant_doc_ids"] = expected_span_ids
        mapped["forbidden_doc_ids"] = forbidden_span_ids
        mapped["notes"] = f"{query.get('notes', '')} Span-level labels are derived heuristically."
        mapped_queries.append(mapped)
    return mapped_queries


def _is_forbidden_span(sentence: str, document: dict[str, object]) -> bool:
    if not bool(document.get("is_forbidden", False)):
        return False
    lowered = sentence.lower()
    return any(hint in lowered for hint in FORBIDDEN_SPAN_HINTS)
