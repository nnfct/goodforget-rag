"""Ablation retrieval variants."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from goodforget_rag.retrieval import RetrievalResult, _select
from goodforget_rag.vectorizer import TfidfEncoder, cosine_scores


def negative_vector_retrieve(
    encoder: TfidfEncoder,
    documents: Sequence[dict[str, object]],
    query: str,
    positive_intent: str,
    forget_set: Sequence[str],
    *,
    alpha: float,
    beta: float,
    gamma: float,
    top_k: int,
    candidate_k: int,
) -> RetrievalResult:
    """Baseline for the flawed arithmetic ``-v`` idea.

    This computes a mean forget vector, multiplies it by -1, and rewards cosine
    similarity to that arithmetic opposite. It is included as a diagnostic
    baseline, not as a recommended forgetting mechanism.
    """

    doc_vectors = encoder.encode_documents(documents)
    query_scores = cosine_scores(encoder.encode_texts([query]), doc_vectors)
    positive_scores = cosine_scores(encoder.encode_texts([positive_intent]), doc_vectors)
    if forget_set:
        forget_vectors = encoder.encode_texts(list(forget_set))
        negative_vector = -forget_vectors.mean(axis=0)
        negative_array = np.asarray(negative_vector)
        if negative_array.ndim == 1:
            negative_array = negative_array.reshape(1, -1)
        negative_scores = cosine_scores(negative_array, doc_vectors)
    else:
        negative_scores = np.zeros(len(documents), dtype=float)
    final_scores = alpha * query_scores + beta * positive_scores + gamma * negative_scores
    return _select(
        documents,
        final_scores=final_scores,
        query_scores=query_scores,
        positive_scores=positive_scores,
        forget_scores=-negative_scores,
        top_k=top_k,
        candidate_k=candidate_k,
    )
