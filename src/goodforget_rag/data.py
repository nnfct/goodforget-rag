"""Data loading helpers for the toy experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

from goodforget_rag.vectorizer import document_to_text


def load_jsonl(path: Path) -> list[dict[str, object]]:
    """Load UTF-8 JSONL records."""

    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def document_lookup(documents: Sequence[dict[str, object]]) -> dict[str, dict[str, object]]:
    """Map document IDs to document dictionaries."""

    return {str(document["doc_id"]): document for document in documents}


def vectorizer_training_texts(
    documents: Iterable[dict[str, object]],
    queries: Iterable[dict[str, object]],
) -> list[str]:
    """Collect all text seen by the deterministic TF-IDF encoder."""

    texts = [document_to_text(document) for document in documents]
    for query in queries:
        texts.append(str(query["query"]))
        texts.append(str(query["positive_intent"]))
        texts.extend(str(item) for item in query["forget_set"])
    return texts


def short_snippet(document: dict[str, object], *, max_chars: int = 140) -> str:
    """Return a compact evidence snippet for deterministic answer simulation."""

    text = " ".join(str(document.get("text", "")).split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."
