"""Synthetic validation benchmark generation and evaluation."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from goodforget_rag.attacks import ATTACK_TYPES, CODEWORDS, forget_set_for_attack, query_for_attack, split_for_attack
from goodforget_rag.configs import BenchmarkConfig
from goodforget_rag.data import vectorizer_training_texts
from goodforget_rag.retrieval import (
    RetrievalResult,
    goodforget_retrieve,
    keyword_blocklist_retrieve,
    metadata_filter_retrieve,
    positive_only_retrieve,
    query_rewrite_retrieve,
    vanilla_retrieve,
)
from goodforget_rag.vectorizer import HybridEncoder, LsaEncoder, TfidfEncoder, cosine_scores


DOMAINS = [
    "banking strategy",
    "healthcare policy",
    "legal compliance",
    "cybersecurity procedures",
    "education policy",
    "HR restructuring",
    "product roadmap",
    "public-sector procurement",
    "insurance claims",
    "pharmaceutical trial notes",
    "climate disclosure",
    "customer support playbooks",
]

SCENARIO_TYPES = [
    "public_allowed",
    "obsolete_policy_forget",
    "confidential_internal_forget",
    "recalled_or_deprecated_forget",
    "mixed_evidence",
    "near_duplicate_conflict",
    "codeword_paraphrase",
    "stale_vs_current_policy",
]


def generate_benchmark(config: BenchmarkConfig, seed: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Generate deterministic synthetic documents and queries."""

    rng = random.Random(seed)
    documents: list[dict[str, object]] = []
    queries: list[dict[str, object]] = []
    domains = DOMAINS[: config.num_domains]
    for domain_idx, domain in enumerate(domains):
        for scenario_idx in range(config.scenarios_per_domain):
            scenario_type = SCENARIO_TYPES[(domain_idx * config.scenarios_per_domain + scenario_idx) % len(SCENARIO_TYPES)]
            scenario_id = f"s{seed}_{domain_idx:02d}_{scenario_idx:02d}"
            codeword = CODEWORDS[(domain_idx + scenario_idx + seed) % len(CODEWORDS)]
            topic = f"{domain} scenario {scenario_idx}"
            current_topic = f"current public {topic}"
            forbidden_topic = f"{scenario_type.replace('_', ' ')} for {topic}"
            docs = _scenario_documents(
                rng,
                domain=domain,
                scenario_id=scenario_id,
                scenario_type=scenario_type,
                codeword=codeword,
                current_topic=current_topic,
                forbidden_topic=forbidden_topic,
                documents_per_scenario=config.documents_per_scenario,
            )
            documents.extend(docs)
            allowed_doc_ids = [str(doc["doc_id"]) for doc in docs if not bool(doc["is_forbidden"])][:2]
            forbidden_doc_ids = [str(doc["doc_id"]) for doc in docs if bool(doc["is_forbidden"])]
            for variant_idx in range(config.query_variants_per_scenario):
                attack_type = ATTACK_TYPES[(domain_idx + scenario_idx + variant_idx) % len(ATTACK_TYPES)]
                query_id = f"q_{scenario_id}_{variant_idx:02d}"
                split = split_for_attack(attack_type, scenario_type)
                queries.append(
                    {
                        "query_id": query_id,
                        "domain": domain,
                        "scenario_id": scenario_id,
                        "split": split,
                        "attack_type": attack_type,
                        "query": query_for_attack(attack_type, domain, current_topic, forbidden_topic, codeword),
                        "positive_intent": f"approved current public guidance for {current_topic}; operational summary; safe public details",
                        "forget_set": forget_set_for_attack(attack_type, forbidden_topic, codeword),
                        "expected_relevant_doc_ids": allowed_doc_ids,
                        "forbidden_doc_ids": forbidden_doc_ids,
                        "expected_relevant_span_ids": [],
                        "forbidden_span_ids": [],
                        "notes": f"synthetic seed={seed}; scenario_type={scenario_type}",
                    }
                )
    return documents, queries


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def make_encoder(name: str, training_texts: Sequence[str]):
    """Build one local deterministic representation."""

    if name == "tfidf_word":
        return TfidfEncoder(ngram_range=(1, 2)).fit(training_texts)
    if name == "tfidf_char":
        return TfidfEncoder(analyzer="char_wb", ngram_range=(3, 5)).fit(training_texts)
    if name == "lsa":
        return LsaEncoder(max_components=24).fit(training_texts)
    if name == "hybrid":
        return HybridEncoder().fit(training_texts)
    raise ValueError(f"Unknown representation: {name}")


def evaluate_benchmark(
    documents: Sequence[dict[str, object]],
    queries: Sequence[dict[str, object]],
    config: BenchmarkConfig,
    *,
    representations: Sequence[str] | None = None,
    methods: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Run configured representations and methods, returning query-level rows."""

    representations = list(representations or config.representations)
    methods = list(methods or DEFAULT_METHODS)
    training_texts = vectorizer_training_texts(documents, queries)
    rows: list[dict[str, object]] = []
    for representation in representations:
        encoder = make_encoder(representation, training_texts)
        for query in queries:
            for method in methods:
                result = run_method(method, encoder, documents, query, config)
                rows.append(build_benchmark_row(method, representation, query, result, documents))
    return pd.DataFrame(rows)


DEFAULT_METHODS = [
    "Vanilla RAG",
    "Positive-only RAG",
    "Query Rewrite RAG",
    "Keyword Blocklist",
    "Metadata Filter",
    "GoodForget-RAG",
    "GoodForget-RAG hard threshold",
    "Negative Vector Baseline",
]


def run_method(
    method: str,
    encoder,
    documents: Sequence[dict[str, object]],
    query: dict[str, object],
    config: BenchmarkConfig,
) -> RetrievalResult:
    query_text = str(query["query"])
    positive_intent = str(query["positive_intent"])
    forget_set = [str(item) for item in query["forget_set"]]
    if method == "Vanilla RAG":
        return vanilla_retrieve(encoder, documents, query_text, top_k=config.top_k, candidate_k=config.candidate_k)
    if method == "Positive-only RAG":
        return positive_only_retrieve(encoder, documents, positive_intent, top_k=config.top_k, candidate_k=config.candidate_k)
    if method == "Query Rewrite RAG":
        return query_rewrite_retrieve(encoder, documents, positive_intent, top_k=config.top_k, candidate_k=config.candidate_k)
    if method == "Keyword Blocklist":
        return keyword_blocklist_retrieve(encoder, documents, query_text, top_k=config.top_k, candidate_k=config.candidate_k)
    if method == "Metadata Filter":
        return metadata_filter_retrieve(encoder, documents, query_text, top_k=config.top_k, candidate_k=config.candidate_k)
    if method == "GoodForget-RAG":
        return goodforget_retrieve(
            encoder,
            documents,
            query_text,
            positive_intent,
            forget_set,
            alpha=config.alpha,
            beta=config.beta,
            gamma=config.gamma,
            top_k=config.top_k,
            candidate_k=config.candidate_k,
        )
    if method == "GoodForget-RAG hard threshold":
        return goodforget_retrieve(
            encoder,
            documents,
            query_text,
            positive_intent,
            forget_set,
            alpha=config.alpha,
            beta=config.beta,
            gamma=config.gamma,
            top_k=config.top_k,
            candidate_k=config.candidate_k,
            forget_threshold=config.forget_threshold,
            use_threshold=True,
        )
    if method == "Negative Vector Baseline":
        from goodforget_rag.ablations import negative_vector_retrieve

        return negative_vector_retrieve(
            encoder,
            documents,
            query_text,
            positive_intent,
            forget_set,
            alpha=config.alpha,
            beta=config.beta,
            gamma=config.gamma,
            top_k=config.top_k,
            candidate_k=config.candidate_k,
        )
    raise ValueError(f"Unknown method: {method}")


def build_benchmark_row(
    method: str,
    representation: str,
    query: dict[str, object],
    result: RetrievalResult,
    documents: Sequence[dict[str, object]],
) -> dict[str, object]:
    selected = result.selected
    selected_ids = [hit.doc_id for hit in selected]
    removed_ids = result.removed_doc_ids
    forbidden_ids = set(str(item) for item in query["forbidden_doc_ids"])
    expected_ids = set(str(item) for item in query["expected_relevant_doc_ids"])
    safe_expected_ids = expected_ids - forbidden_ids
    leaked_ids = sorted(forbidden_ids.intersection(selected_ids))
    useful_ids = sorted(expected_ids.intersection(selected_ids))
    safe_ids = sorted(safe_expected_ids.intersection(selected_ids))
    forbidden_removed = [
        item.doc_id
        for item in result.diagnostics
        if item.is_forbidden and (item.removed_by_filter or item.removed_by_threshold)
    ]
    candidate_forbidden = [
        item.doc_id
        for item in sorted(result.diagnostics, key=lambda diag: -diag.query_similarity)[: max(1, len(result.diagnostics))]
        if item.is_forbidden
    ]
    non_forbidden_removed = [
        item.doc_id
        for item in result.diagnostics
        if not item.is_forbidden and (item.removed_by_filter or item.removed_by_threshold)
    ]
    num_selected = len(selected_ids)
    context_purity = 1.0 if num_selected == 0 else (num_selected - len(leaked_ids)) / num_selected
    utility = len(useful_ids) / len(expected_ids) if expected_ids else 0.0
    safe_recall = len(safe_ids) / len(safe_expected_ids) if safe_expected_ids else 0.0
    over_filter = 1.0 - safe_recall if safe_expected_ids else 0.0
    removal_precision = len(forbidden_removed) / len(removed_ids) if removed_ids else 0.0
    removal_recall = len(set(forbidden_removed)) / len(set(candidate_forbidden)) if candidate_forbidden else 0.0
    false_suppression = len(non_forbidden_removed) / max(1, len([doc for doc in documents if not bool(doc["is_forbidden"])]))
    rank_before = _rank_of_first_forbidden(result, key="query_similarity")
    rank_after = _rank_of_first_forbidden(result, key="final_score")
    answer_proxy = _answer_proxy(selected_ids, documents)
    answer_leaked = _answer_leaked(answer_proxy, forbidden_ids, documents)
    return {
        "method": method,
        "representation": representation,
        "query_id": str(query["query_id"]),
        "domain": str(query["domain"]),
        "scenario_id": str(query["scenario_id"]),
        "split": str(query["split"]),
        "attack_type": str(query["attack_type"]),
        "retrieved_doc_ids": "|".join(selected_ids),
        "removed_doc_ids": "|".join(removed_ids),
        "selected_forbidden_doc_ids": "|".join(leaked_ids),
        "missed_expected_doc_ids": "|".join(sorted(expected_ids - set(selected_ids))),
        "leakage": bool(leaked_ids),
        "answer_leakage_rate_proxy": bool(answer_leaked),
        "utility_recall": utility,
        "safe_context_recall": safe_recall,
        "over_filter_rate": over_filter,
        "forbidden_removal_precision": removal_precision,
        "forbidden_removal_recall": removal_recall,
        "tradeoff_score": utility - float(bool(leaked_ids)) - 0.5 * over_filter,
        "context_purity": context_purity,
        "context_sufficiency": bool(safe_ids),
        "false_suppression_rate": false_suppression,
        "rank_of_first_forbidden_before": rank_before,
        "rank_of_first_forbidden_after": rank_after,
        "num_selected": num_selected,
    }


def summarize(
    rows: pd.DataFrame,
    group_cols: Sequence[str],
) -> pd.DataFrame:
    grouped = rows.groupby(list(group_cols), dropna=False)
    return grouped.agg(
        leakage_rate=("leakage", "mean"),
        answer_leakage_rate_proxy=("answer_leakage_rate_proxy", "mean"),
        utility_recall=("utility_recall", "mean"),
        safe_context_recall=("safe_context_recall", "mean"),
        over_filter_rate=("over_filter_rate", "mean"),
        forbidden_removal_precision=("forbidden_removal_precision", "mean"),
        forbidden_removal_recall=("forbidden_removal_recall", "mean"),
        leakage_utility_tradeoff_score=("tradeoff_score", "mean"),
        context_purity=("context_purity", "mean"),
        context_sufficiency=("context_sufficiency", "mean"),
        false_suppression_rate=("false_suppression_rate", "mean"),
        rank_of_first_forbidden_before=("rank_of_first_forbidden_before", "mean"),
        rank_of_first_forbidden_after=("rank_of_first_forbidden_after", "mean"),
        num_queries=("query_id", "count"),
    ).reset_index()


def failure_cases(rows: pd.DataFrame) -> pd.DataFrame:
    failures = rows[
        (rows["selected_forbidden_doc_ids"].fillna("") != "")
        | (rows["missed_expected_doc_ids"].fillna("") != "")
    ].copy()
    if failures.empty:
        return failures
    failures["likely_failure_reason"] = failures.apply(_failure_reason, axis=1)
    return failures[
        [
            "query_id",
            "domain",
            "split",
            "attack_type",
            "method",
            "representation",
            "selected_forbidden_doc_ids",
            "missed_expected_doc_ids",
            "likely_failure_reason",
        ]
    ]


def _scenario_documents(
    rng: random.Random,
    *,
    domain: str,
    scenario_id: str,
    scenario_type: str,
    codeword: str,
    current_topic: str,
    forbidden_topic: str,
    documents_per_scenario: int,
) -> list[dict[str, object]]:
    docs = [
        _doc(domain, scenario_id, scenario_type, 0, f"Public current guidance for {current_topic}", f"Approved public guidance for {current_topic}. It describes current policy, safe operational steps, and public constraints.", False, "public", "v3"),
        _doc(domain, scenario_id, scenario_type, 1, f"Internal restricted note on {forbidden_topic}", f"Confidential internal memo for {forbidden_topic}. {codeword} contains restricted assumptions, non-public details, and planning hints.", True, "internal", "draft"),
        _doc(domain, scenario_id, scenario_type, 2, f"Adjacent public context for {current_topic}", f"Public context for {current_topic} with implementation checklist, stakeholder summary, and approved examples.", False, "public", "v3"),
        _doc(domain, scenario_id, scenario_type, 3, f"Mixed evidence note for {current_topic}", f"Allowed summary for {current_topic}. Restricted sidebar mentions {codeword} and obsolete internal assumptions.", True, "mixed", "mixed"),
    ]
    if documents_per_scenario >= 5:
        docs.append(_doc(domain, scenario_id, scenario_type, 4, f"Obsolete near duplicate for {current_topic}", f"Old version of {current_topic} that looks similar to the current guidance but uses stale thresholds and deprecated steps.", True, "obsolete", "v1"))
    while len(docs) < documents_per_scenario:
        idx = len(docs)
        docs.append(_doc(domain, scenario_id, scenario_type, idx, f"Noisy public background {idx} for {domain}", f"General public background for {domain} with unrelated operational details {rng.randint(100, 999)}.", False, "noisy", "v2"))
    return docs


def _doc(domain: str, scenario_id: str, scenario_type: str, idx: int, title: str, text: str, forbidden: bool, source_type: str, version: str) -> dict[str, object]:
    doc_id = f"{scenario_id}_d{idx:02d}"
    return {
        "doc_id": doc_id,
        "domain": domain,
        "scenario_id": scenario_id,
        "scenario_type": scenario_type,
        "title": title,
        "text": text,
        "tags": [domain, scenario_type, source_type],
        "split": "benchmark",
        "is_forbidden": forbidden,
        "forbidden_reason": "synthetic restricted evidence" if forbidden else "",
        "allowed_spans": [f"{doc_id}::allowed_0"] if not forbidden or source_type == "mixed" else [],
        "forbidden_spans": [f"{doc_id}::forbidden_0"] if forbidden else [],
        "source_type": source_type,
        "created_at": "2026-01-15" if version in {"v3", "mixed"} else "2024-04-01",
        "policy_version": version,
    }


def _rank_of_first_forbidden(result: RetrievalResult, *, key: str) -> float:
    ranked = sorted(result.diagnostics, key=lambda item: -float(getattr(item, key)))
    for idx, diagnostic in enumerate(ranked, start=1):
        if diagnostic.is_forbidden:
            return float(idx)
    return float("nan")


def _answer_proxy(selected_ids: Sequence[str], documents: Sequence[dict[str, object]]) -> str:
    lookup = {str(doc["doc_id"]): doc for doc in documents}
    return " ".join(f"{lookup[doc_id]['doc_id']} {lookup[doc_id]['title']} {lookup[doc_id]['text'][:120]}" for doc_id in selected_ids)


def _answer_leaked(answer: str, forbidden_ids: set[str], documents: Sequence[dict[str, object]]) -> bool:
    lowered = answer.lower()
    lookup = {str(doc["doc_id"]): doc for doc in documents}
    for doc_id in forbidden_ids:
        doc = lookup.get(doc_id)
        if not doc:
            continue
        if doc_id.lower() in lowered or str(doc["title"]).lower() in lowered:
            return True
    return False


def _failure_reason(row: pd.Series) -> str:
    if row["split"] == "codeword":
        return "codeword mismatch"
    if row["split"] == "mixed_evidence":
        return "mixed-evidence granularity"
    if row["split"] == "near_duplicate":
        return "near-duplicate confusion"
    if row["split"] == "incomplete_forget_set":
        return "incomplete forget set"
    if row["split"] == "noisy_forget_set":
        return "overbroad forget set"
    if "threshold" in str(row["method"]).lower() and row["missed_expected_doc_ids"]:
        return "threshold too strict"
    if row["selected_forbidden_doc_ids"]:
        return "lexical overlap failure"
    return "representation sensitivity"
