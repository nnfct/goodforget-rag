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
from goodforget_rag.hard_negatives import (
    evaluate_hard_negative_pairs,
    generate_hard_negative_pairs,
)

__all__ = [
    "TfidfEncoder",
    "LsaEncoder",
    "HybridEncoder",
    "generate_benchmark",
    "evaluate_benchmark",
    "generate_hard_negative_pairs",
    "evaluate_hard_negative_pairs",
    "vanilla_retrieve",
    "positive_only_retrieve",
    "query_rewrite_retrieve",
    "keyword_blocklist_retrieve",
    "metadata_filter_retrieve",
    "goodforget_retrieve",
]
