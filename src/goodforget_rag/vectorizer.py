"""Small deterministic TF-IDF encoding utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


SparseMatrix = sparse.spmatrix


@dataclass
class TfidfEncoder:
    """Wrapper around scikit-learn TF-IDF vectors.

    The encoder intentionally uses a local bag-of-ngrams representation so the
    experiment is reproducible on a laptop and requires no API keys.
    """

    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 1
    max_df: float = 1.0
    vectorizer: TfidfVectorizer = field(init=False)

    def __post_init__(self) -> None:
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            norm="l2",
        )

    def fit(self, texts: Iterable[str]) -> "TfidfEncoder":
        """Fit the vectorizer on all texts used in the toy experiment."""

        self.vectorizer.fit(list(texts))
        return self

    def encode_documents(self, documents: Sequence[dict[str, object]]) -> SparseMatrix:
        """Encode document dictionaries using their title, body, and tags."""

        return self.encode_texts([document_to_text(doc) for doc in documents])

    def encode_texts(self, texts: Sequence[str]) -> SparseMatrix:
        """Encode arbitrary text strings in the fitted TF-IDF space."""

        return self.vectorizer.transform(list(texts))


def document_to_text(document: dict[str, object]) -> str:
    """Create a stable searchable text field from a toy document."""

    title = str(document.get("title", ""))
    text = str(document.get("text", ""))
    notes = str(document.get("notes", ""))
    tags = document.get("tags", [])
    tag_text = " ".join(str(tag) for tag in tags) if isinstance(tags, list) else str(tags)
    return f"{title}\n{text}\n{tag_text}\n{notes}".strip()


def cosine_scores(query_vector: SparseMatrix, document_vectors: SparseMatrix) -> np.ndarray:
    """Return cosine similarities between one query vector and many documents."""

    scores = cosine_similarity(query_vector, document_vectors).ravel()
    return np.asarray(scores, dtype=float)
