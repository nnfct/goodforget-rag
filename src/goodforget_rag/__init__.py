"""GoodForget-RAG toy retrieval package."""

from goodforget_rag.retrieval import (
    goodforget_retrieve,
    keyword_blocklist_retrieve,
    metadata_filter_retrieve,
    query_rewrite_retrieve,
    positive_only_retrieve,
    vanilla_retrieve,
)
from goodforget_rag.vectorizer import HybridEncoder, LsaEncoder, TfidfEncoder
from goodforget_rag.benchmark import generate_benchmark, evaluate_benchmark

__all__ = [
    "TfidfEncoder",
    "LsaEncoder",
    "HybridEncoder",
    "generate_benchmark",
    "evaluate_benchmark",
    "vanilla_retrieve",
    "positive_only_retrieve",
    "query_rewrite_retrieve",
    "keyword_blocklist_retrieve",
    "metadata_filter_retrieve",
    "goodforget_retrieve",
]
