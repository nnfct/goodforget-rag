"""Small deterministic TF-IDF encoding utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer


Matrix = sparse.spmatrix | np.ndarray


@dataclass
class TfidfEncoder:
    """Wrapper around scikit-learn TF-IDF vectors.

    The encoder intentionally uses a local bag-of-ngrams representation so the
    experiment is reproducible on a laptop and requires no API keys.
    """

    ngram_range: tuple[int, int] = (1, 2)
    analyzer: str = "word"
    min_df: int = 1
    max_df: float = 1.0
    vectorizer: TfidfVectorizer = field(init=False)

    def __post_init__(self) -> None:
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            analyzer=self.analyzer,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            norm="l2",
        )

    def fit(self, texts: Iterable[str]) -> "TfidfEncoder":
        """Fit the vectorizer on all texts used in the toy experiment."""

        self.vectorizer.fit(list(texts))
        return self

    def encode_documents(self, documents: Sequence[dict[str, object]]) -> Matrix:
        """Encode document dictionaries using their title, body, and tags."""

        return self.encode_texts([document_to_text(doc) for doc in documents])

    def encode_texts(self, texts: Sequence[str]) -> Matrix:
        """Encode arbitrary text strings in the fitted TF-IDF space."""

        return self.vectorizer.transform(list(texts))


@dataclass
class LsaEncoder:
    """Local TF-IDF + truncated SVD encoder for representation sensitivity.

    This is not a downloaded dense embedding model. It is a deterministic local
    latent semantic analysis proxy used to test whether conclusions depend on
    raw lexical TF-IDF features.
    """

    ngram_range: tuple[int, int] = (1, 2)
    analyzer: str = "word"
    max_components: int = 8
    min_df: int = 1
    max_df: float = 1.0
    vectorizer: TfidfVectorizer = field(init=False)
    svd: TruncatedSVD | None = field(default=None, init=False)
    normalizer: Normalizer = field(init=False)

    def __post_init__(self) -> None:
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            analyzer=self.analyzer,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            norm="l2",
        )
        self.normalizer = Normalizer(copy=False)

    def fit(self, texts: Iterable[str]) -> "LsaEncoder":
        training_texts = list(texts)
        tfidf = self.vectorizer.fit_transform(training_texts)
        max_valid_components = max(1, min(tfidf.shape[0], tfidf.shape[1]) - 1)
        n_components = min(self.max_components, max_valid_components)
        if n_components <= 1:
            self.svd = None
        else:
            self.svd = TruncatedSVD(n_components=n_components, random_state=7)
            self.normalizer.fit(self.svd.fit_transform(tfidf))
        return self

    def encode_documents(self, documents: Sequence[dict[str, object]]) -> Matrix:
        return self.encode_texts([document_to_text(doc) for doc in documents])

    def encode_texts(self, texts: Sequence[str]) -> Matrix:
        tfidf = self.vectorizer.transform(list(texts))
        if self.svd is None:
            return tfidf
        return self.normalizer.transform(self.svd.transform(tfidf))


@dataclass
class HybridEncoder:
    """Hybrid local lexical encoder combining word and character TF-IDF."""

    word_encoder: TfidfEncoder = field(default_factory=TfidfEncoder)
    char_encoder: TfidfEncoder = field(
        default_factory=lambda: TfidfEncoder(analyzer="char_wb", ngram_range=(3, 5))
    )

    def fit(self, texts: Iterable[str]) -> "HybridEncoder":
        training_texts = list(texts)
        self.word_encoder.fit(training_texts)
        self.char_encoder.fit(training_texts)
        return self

    def encode_documents(self, documents: Sequence[dict[str, object]]) -> Matrix:
        return self.encode_texts([document_to_text(doc) for doc in documents])

    def encode_texts(self, texts: Sequence[str]) -> Matrix:
        word = self.word_encoder.encode_texts(texts)
        char = self.char_encoder.encode_texts(texts)
        return sparse.hstack([word, char], format="csr")


def document_to_text(document: dict[str, object]) -> str:
    """Create a stable searchable text field from a toy document."""

    title = str(document.get("title", ""))
    text = str(document.get("text", ""))
    notes = str(document.get("notes", ""))
    tags = document.get("tags", [])
    tag_text = " ".join(str(tag) for tag in tags) if isinstance(tags, list) else str(tags)
    return f"{title}\n{text}\n{tag_text}\n{notes}".strip()


def cosine_scores(query_vector: Matrix, document_vectors: Matrix) -> np.ndarray:
    """Return cosine similarities between one query vector and many documents."""

    scores = cosine_similarity(query_vector, document_vectors).ravel()
    return np.asarray(scores, dtype=float)
