"""Retrieval functions for the GoodForget-RAG toy experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from goodforget_rag.vectorizer import TfidfEncoder, cosine_scores


@dataclass(frozen=True)
class RetrievalHit:
    """A selected document and its scoring diagnostics."""

    doc_id: str
    score: float
    query_similarity: float
    positive_similarity: float
    max_forget_similarity: float


def vanilla_retrieve(
    encoder: TfidfEncoder,
    documents: Sequence[dict[str, object]],
    query: str,
    *,
    top_k: int = 3,
    score_threshold: float = -float("inf"),
) -> list[RetrievalHit]:
    """Retrieve documents by similarity to the original user query only."""

    doc_vectors = encoder.encode_documents(documents)
    query_scores = cosine_scores(encoder.encode_texts([query]), doc_vectors)
    zeros = np.zeros_like(query_scores)
    return _rank(documents, query_scores, query_scores, zeros, zeros, top_k, score_threshold)


def positive_only_retrieve(
    encoder: TfidfEncoder,
    documents: Sequence[dict[str, object]],
    positive_intent: str,
    *,
    top_k: int = 3,
    score_threshold: float = -float("inf"),
) -> list[RetrievalHit]:
    """Retrieve documents using only the positive/useful information intent."""

    doc_vectors = encoder.encode_documents(documents)
    positive_scores = cosine_scores(encoder.encode_texts([positive_intent]), doc_vectors)
    zeros = np.zeros_like(positive_scores)
    return _rank(documents, positive_scores, zeros, positive_scores, zeros, top_k, score_threshold)


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
    score_threshold: float = -float("inf"),
) -> list[RetrievalHit]:
    """Retrieve with positive relevance and a forget-intent penalty.

    The score is:
        S(d) = alpha * sim(q, d) + beta * sim(p, d) - gamma * max_i sim(f_i, d)

    The forget inputs are encoded as explicit forget-intent text, not by taking
    an arithmetic negative vector.
    """

    doc_vectors = encoder.encode_documents(documents)
    query_scores = cosine_scores(encoder.encode_texts([query]), doc_vectors)
    positive_scores = cosine_scores(encoder.encode_texts([positive_intent]), doc_vectors)
    forget_scores = _max_forget_scores(encoder, doc_vectors, forget_set)
    combined_scores = alpha * query_scores + beta * positive_scores - gamma * forget_scores
    return _rank(
        documents,
        combined_scores,
        query_scores,
        positive_scores,
        forget_scores,
        top_k,
        score_threshold,
    )


def _max_forget_scores(
    encoder: TfidfEncoder,
    doc_vectors: object,
    forget_set: Sequence[str],
) -> np.ndarray:
    if not forget_set:
        return np.zeros(doc_vectors.shape[0], dtype=float)

    forget_vectors = encoder.encode_texts(list(forget_set))
    scores_by_forget_text = []
    for row_idx in range(forget_vectors.shape[0]):
        scores_by_forget_text.append(cosine_scores(forget_vectors[row_idx], doc_vectors))
    return np.max(np.vstack(scores_by_forget_text), axis=0)


def _rank(
    documents: Sequence[dict[str, object]],
    scores: np.ndarray,
    query_scores: np.ndarray,
    positive_scores: np.ndarray,
    forget_scores: np.ndarray,
    top_k: int,
    score_threshold: float,
) -> list[RetrievalHit]:
    doc_ids = [str(doc["doc_id"]) for doc in documents]
    ranked_indices = sorted(
        range(len(documents)),
        key=lambda idx: (-float(scores[idx]), doc_ids[idx]),
    )
    hits: list[RetrievalHit] = []
    for idx in ranked_indices:
        if len(hits) >= top_k:
            break
        if float(scores[idx]) < score_threshold:
            continue
        hits.append(
            RetrievalHit(
                doc_id=doc_ids[idx],
                score=float(scores[idx]),
                query_similarity=float(query_scores[idx]),
                positive_similarity=float(positive_scores[idx]),
                max_forget_similarity=float(forget_scores[idx]),
            )
        )
    return hits
