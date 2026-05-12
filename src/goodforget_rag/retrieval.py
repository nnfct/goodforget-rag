"""Retrieval functions and diagnostics for the GoodForget-RAG toy experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from goodforget_rag.vectorizer import TfidfEncoder, cosine_scores, document_to_text


DEFAULT_BLOCKED_KEYWORDS = (
    "leaked",
    "confidential",
    "private",
    "internal memo",
    "secret",
    "unreleased",
)


@dataclass(frozen=True)
class RetrievalHit:
    """A selected document and its scoring diagnostics."""

    doc_id: str
    score: float
    query_similarity: float
    positive_similarity: float
    max_forget_similarity: float
    final_score: float
    removed_by_threshold: bool
    removed_by_filter: bool
    is_forbidden: bool


@dataclass(frozen=True)
class DocumentDiagnostic:
    """Per-document retrieval diagnostic row."""

    doc_id: str
    query_similarity: float
    positive_similarity: float
    max_forget_similarity: float
    final_score: float
    removed_by_threshold: bool
    removed_by_filter: bool
    is_forbidden: bool


@dataclass(frozen=True)
class RetrievalResult:
    """Selected hits plus full candidate diagnostics."""

    selected: list[RetrievalHit]
    diagnostics: list[DocumentDiagnostic]

    @property
    def removed_doc_ids(self) -> list[str]:
        return [
            item.doc_id
            for item in self.diagnostics
            if item.removed_by_filter or item.removed_by_threshold
        ]


def vanilla_retrieve(
    encoder: TfidfEncoder,
    documents: Sequence[dict[str, object]],
    query: str,
    *,
    top_k: int = 3,
    candidate_k: int | None = None,
) -> RetrievalResult:
    """Retrieve documents by similarity to the original user query only."""

    vectors = _score_vectors(encoder, documents, query, "", [])
    return _select(
        documents,
        final_scores=vectors["query"],
        query_scores=vectors["query"],
        positive_scores=np.zeros_like(vectors["query"]),
        forget_scores=np.zeros_like(vectors["query"]),
        top_k=top_k,
        candidate_k=candidate_k,
    )


def positive_only_retrieve(
    encoder: TfidfEncoder,
    documents: Sequence[dict[str, object]],
    positive_intent: str,
    *,
    top_k: int = 3,
    candidate_k: int | None = None,
) -> RetrievalResult:
    """Retrieve documents using only the positive/useful information intent."""

    vectors = _score_vectors(encoder, documents, "", positive_intent, [])
    return _select(
        documents,
        final_scores=vectors["positive"],
        query_scores=np.zeros_like(vectors["positive"]),
        positive_scores=vectors["positive"],
        forget_scores=np.zeros_like(vectors["positive"]),
        top_k=top_k,
        candidate_k=candidate_k,
    )


def query_rewrite_retrieve(
    encoder: TfidfEncoder,
    documents: Sequence[dict[str, object]],
    positive_intent: str,
    *,
    top_k: int = 3,
    candidate_k: int | None = None,
) -> RetrievalResult:
    """Baseline using the positive intent as a simplified rewritten query."""

    return positive_only_retrieve(
        encoder,
        documents,
        positive_intent,
        top_k=top_k,
        candidate_k=candidate_k,
    )


def keyword_blocklist_retrieve(
    encoder: TfidfEncoder,
    documents: Sequence[dict[str, object]],
    query: str,
    *,
    blocked_keywords: Sequence[str] = DEFAULT_BLOCKED_KEYWORDS,
    top_k: int = 3,
    candidate_k: int | None = None,
) -> RetrievalResult:
    """Retrieve by query, then remove documents containing blocked keywords."""

    vectors = _score_vectors(encoder, documents, query, "", [])
    removed = [
        _contains_blocked_keyword(document, blocked_keywords)
        for document in documents
    ]
    return _select(
        documents,
        final_scores=vectors["query"],
        query_scores=vectors["query"],
        positive_scores=np.zeros_like(vectors["query"]),
        forget_scores=np.zeros_like(vectors["query"]),
        top_k=top_k,
        candidate_k=candidate_k,
        removed_by_filter=np.asarray(removed, dtype=bool),
    )


def metadata_filter_retrieve(
    encoder: TfidfEncoder,
    documents: Sequence[dict[str, object]],
    query: str,
    *,
    top_k: int = 3,
    candidate_k: int | None = None,
) -> RetrievalResult:
    """Oracle baseline that removes documents labeled as forbidden."""

    vectors = _score_vectors(encoder, documents, query, "", [])
    removed = np.asarray([bool(document.get("is_forbidden", False)) for document in documents])
    return _select(
        documents,
        final_scores=vectors["query"],
        query_scores=vectors["query"],
        positive_scores=np.zeros_like(vectors["query"]),
        forget_scores=np.zeros_like(vectors["query"]),
        top_k=top_k,
        candidate_k=candidate_k,
        removed_by_filter=removed,
    )


def goodforget_retrieve(
    encoder: TfidfEncoder,
    documents: Sequence[dict[str, object]],
    query: str,
    positive_intent: str,
    forget_set: Sequence[str],
    *,
    alpha: float = 0.4,
    beta: float = 0.8,
    gamma: float = 1.2,
    top_k: int = 3,
    candidate_k: int | None = None,
    forget_threshold: float | None = None,
    use_threshold: bool = False,
) -> RetrievalResult:
    """Retrieve with positive relevance and a forget-intent penalty.

    The score is:
        S(d) = alpha * sim(q, d) + beta * sim(p, d) - gamma * max_i sim(f_i, d)

    The forget input is a separate representation of what should not be used,
    not the arithmetic negative vector ``-v``.
    """

    vectors = _score_vectors(encoder, documents, query, positive_intent, forget_set)
    final_scores = alpha * vectors["query"] + beta * vectors["positive"] - gamma * vectors["forget"]
    removed_by_threshold = np.zeros(len(documents), dtype=bool)
    if use_threshold and forget_threshold is not None:
        removed_by_threshold = vectors["forget"] >= forget_threshold
    return _select(
        documents,
        final_scores=final_scores,
        query_scores=vectors["query"],
        positive_scores=vectors["positive"],
        forget_scores=vectors["forget"],
        top_k=top_k,
        candidate_k=candidate_k,
        removed_by_threshold=removed_by_threshold,
    )


def _score_vectors(
    encoder: TfidfEncoder,
    documents: Sequence[dict[str, object]],
    query: str,
    positive_intent: str,
    forget_set: Sequence[str],
) -> dict[str, np.ndarray]:
    doc_vectors = encoder.encode_documents(documents)
    query_scores = (
        cosine_scores(encoder.encode_texts([query]), doc_vectors)
        if query
        else np.zeros(len(documents), dtype=float)
    )
    positive_scores = (
        cosine_scores(encoder.encode_texts([positive_intent]), doc_vectors)
        if positive_intent
        else np.zeros(len(documents), dtype=float)
    )
    forget_scores = _max_forget_scores(encoder, doc_vectors, forget_set)
    return {"query": query_scores, "positive": positive_scores, "forget": forget_scores}


def _max_forget_scores(
    encoder: TfidfEncoder,
    doc_vectors: object,
    forget_set: Sequence[str],
) -> np.ndarray:
    if not forget_set:
        return np.zeros(doc_vectors.shape[0], dtype=float)

    forget_vectors = encoder.encode_texts(list(forget_set))
    scores_by_forget_text = [
        cosine_scores(forget_vectors[row_idx : row_idx + 1], doc_vectors)
        for row_idx in range(forget_vectors.shape[0])
    ]
    return np.max(np.vstack(scores_by_forget_text), axis=0)


def _select(
    documents: Sequence[dict[str, object]],
    *,
    final_scores: np.ndarray,
    query_scores: np.ndarray,
    positive_scores: np.ndarray,
    forget_scores: np.ndarray,
    top_k: int,
    candidate_k: int | None,
    removed_by_filter: np.ndarray | None = None,
    removed_by_threshold: np.ndarray | None = None,
) -> RetrievalResult:
    doc_ids = [str(document["doc_id"]) for document in documents]
    removed_by_filter = (
        removed_by_filter
        if removed_by_filter is not None
        else np.zeros(len(documents), dtype=bool)
    )
    removed_by_threshold = (
        removed_by_threshold
        if removed_by_threshold is not None
        else np.zeros(len(documents), dtype=bool)
    )
    ranked_indices = sorted(range(len(documents)), key=lambda idx: (-float(final_scores[idx]), doc_ids[idx]))
    candidate_limit = candidate_k if candidate_k is not None else len(documents)
    candidate_indices = set(ranked_indices[:candidate_limit])

    diagnostics = [
        DocumentDiagnostic(
            doc_id=doc_ids[idx],
            query_similarity=float(query_scores[idx]),
            positive_similarity=float(positive_scores[idx]),
            max_forget_similarity=float(forget_scores[idx]),
            final_score=float(final_scores[idx]),
            removed_by_threshold=bool(idx in candidate_indices and removed_by_threshold[idx]),
            removed_by_filter=bool(idx in candidate_indices and removed_by_filter[idx]),
            is_forbidden=bool(documents[idx].get("is_forbidden", False)),
        )
        for idx in range(len(documents))
    ]

    selected: list[RetrievalHit] = []
    for idx in ranked_indices:
        if len(selected) >= top_k:
            break
        if idx not in candidate_indices:
            continue
        if removed_by_filter[idx] or removed_by_threshold[idx]:
            continue
        selected.append(
            RetrievalHit(
                doc_id=doc_ids[idx],
                score=float(final_scores[idx]),
                query_similarity=float(query_scores[idx]),
                positive_similarity=float(positive_scores[idx]),
                max_forget_similarity=float(forget_scores[idx]),
                final_score=float(final_scores[idx]),
                removed_by_threshold=bool(removed_by_threshold[idx]),
                removed_by_filter=bool(removed_by_filter[idx]),
                is_forbidden=bool(documents[idx].get("is_forbidden", False)),
            )
        )
    return RetrievalResult(selected=selected, diagnostics=diagnostics)


def _contains_blocked_keyword(
    document: dict[str, object],
    blocked_keywords: Sequence[str],
) -> bool:
    text = document_to_text(document).lower()
    return any(keyword.lower() in text for keyword in blocked_keywords)
