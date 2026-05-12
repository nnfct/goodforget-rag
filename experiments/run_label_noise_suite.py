"""Stress-test label-aware metadata filtering with noisy labels."""

from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path

import pandas as pd

from goodforget_rag.benchmark import build_benchmark_row, generate_benchmark, make_encoder, run_method, summarize
from goodforget_rag.configs import load_config
from goodforget_rag.data import vectorizer_training_texts
from goodforget_rag.retrieval import goodforget_retrieve


PROJECT_ROOT = Path(__file__).resolve().parents[1]


SETTINGS = [
    ("0% missing forbidden labels", "missing", 0.0),
    ("10% missing forbidden labels", "missing", 0.10),
    ("25% missing forbidden labels", "missing", 0.25),
    ("50% missing forbidden labels", "missing", 0.50),
    ("10% false positive forbidden labels", "false_positive", 0.10),
    ("25% false positive forbidden labels", "false_positive", 0.25),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/benchmark_default.yaml")
    args = parser.parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    documents, queries = generate_benchmark(config, config.seeds[0])
    rows = []
    for label, mode, rate in SETTINGS:
        noisy_docs = _apply_noise(documents, mode, rate, seed=config.seeds[0])
        encoder = make_encoder("tfidf_word", vectorizer_training_texts(noisy_docs, queries))
        for query in queries:
            for method in ["Metadata Filter", "GoodForget-RAG"]:
                result = run_method(method, encoder, noisy_docs, query, config)
                row = build_benchmark_row(method, "tfidf_word", query, result, noisy_docs)
                row["noise_setting"] = label
                rows.append(row)
            filtered_docs = [doc for doc in noisy_docs if not bool(doc["is_forbidden"])]
            hybrid_result = goodforget_retrieve(
                encoder,
                filtered_docs,
                str(query["query"]),
                str(query["positive_intent"]),
                [str(item) for item in query["forget_set"]],
                alpha=config.alpha,
                beta=config.beta,
                gamma=config.gamma,
                top_k=config.top_k,
                candidate_k=min(config.candidate_k, len(filtered_docs)),
            )
            row = build_benchmark_row("Hybrid Metadata + GoodForget-RAG", "tfidf_word", query, hybrid_result, noisy_docs)
            row["noise_setting"] = label
            rows.append(row)
    frame = pd.DataFrame(rows)
    summary = summarize(frame, ["noise_setting", "method", "representation"])
    summary.to_csv(PROJECT_ROOT / "results" / "label_noise_summary.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


def _apply_noise(documents: list[dict[str, object]], mode: str, rate: float, *, seed: int) -> list[dict[str, object]]:
    rng = random.Random(seed + int(rate * 1000))
    docs = copy.deepcopy(documents)
    candidates = [doc for doc in docs if bool(doc["is_forbidden"])] if mode == "missing" else [doc for doc in docs if not bool(doc["is_forbidden"])]
    rng.shuffle(candidates)
    for doc in candidates[: int(round(len(candidates) * rate))]:
        doc["is_forbidden"] = False if mode == "missing" else True
        doc["forbidden_reason"] = f"label_noise_{mode}"
    return docs


if __name__ == "__main__":
    main()
