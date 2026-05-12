"""Hard negative pair benchmark for isolating forget-penalty effects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from goodforget_rag.ablations import negative_vector_retrieve
from goodforget_rag.benchmark import make_encoder, summarize
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


DOMAINS = [
    "banking strategy",
    "healthcare policy",
    "legal compliance",
    "cybersecurity procedures",
    "education policy",
    "HR restructuring",
    "product roadmap",
    "public-sector procurement",
]

ATTRIBUTES = [
    "layoffs",
    "closure targets",
    "non-public acquisition target",
    "unreleased vulnerability workaround",
    "recalled medication instruction",
    "obsolete compliance rule",
    "confidential pricing plan",
    "undisclosed vendor score",
]

SPLITS = [
    "near_duplicate_public_internal",
    "stale_vs_current",
    "forbidden_attribute_overlap",
    "codeword_attribute",
    "overbroad_forget_set",
    "incomplete_forget_set",
]

CODEWORDS = [
    "Project Maple",
    "Bluebird Route",
    "Orchard Scenario",
    "Silver Ledger",
    "Harbor Draft",
    "Cedar Channel",
]

REPRESENTATIONS = ["tfidf_word", "tfidf_char", "hybrid", "lsa"]

METHODS = [
    "Vanilla RAG",
    "Positive-only RAG",
    "Query Rewrite RAG",
    "Metadata Filter",
    "Keyword Blocklist",
    "Negative Vector Baseline",
    "GoodForget-RAG gamma=0",
    "GoodForget-RAG gamma=0.5",
    "GoodForget-RAG gamma=1.0",
    "GoodForget-RAG gamma=1.5",
]


@dataclass(frozen=True)
class HardNegativeConfig:
    domains: int = 8
    scenarios_per_domain: int = 10
    query_variants_per_scenario: int = 4
    top_k: int = 3
    candidate_k: int = 10
    alpha: float = 0.4
    beta: float = 0.8


def generate_hard_negative_pairs(
    config: HardNegativeConfig = HardNegativeConfig(),
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Generate scenarios with safe/forbidden near-duplicate pairs."""

    documents: list[dict[str, object]] = []
    queries: list[dict[str, object]] = []
    domains = DOMAINS[: config.domains]
    for domain_idx, domain in enumerate(domains):
        for scenario_idx in range(config.scenarios_per_domain):
            scenario_id = f"hn_{domain_idx:02d}_{scenario_idx:02d}"
            split = SPLITS[(domain_idx + scenario_idx) % len(SPLITS)]
            attribute = ATTRIBUTES[(domain_idx * 3 + scenario_idx) % len(ATTRIBUTES)]
            codeword = CODEWORDS[(domain_idx + scenario_idx) % len(CODEWORDS)]
            topic = _topic(domain, scenario_idx)
            docs = _scenario_documents(domain, scenario_id, split, topic, attribute, codeword)
            documents.extend(docs)
            for variant_idx in range(config.query_variants_per_scenario):
                attack_type = _attack_type(split, variant_idx)
                queries.append(
                    _query(
                        domain,
                        scenario_id,
                        split,
                        attack_type,
                        variant_idx,
                        topic,
                        attribute,
                        codeword,
                        docs,
                    )
                )
    return documents, queries


def evaluate_hard_negative_pairs(
    documents: Sequence[dict[str, object]],
    queries: Sequence[dict[str, object]],
    config: HardNegativeConfig = HardNegativeConfig(),
) -> pd.DataFrame:
    """Run all methods and representations for the hard-negative suite."""

    rows: list[dict[str, object]] = []
    training_texts = vectorizer_training_texts(documents, queries)
    for representation in REPRESENTATIONS:
        encoder = make_encoder(representation, training_texts)
        for query in queries:
            baseline = goodforget_retrieve(
                encoder,
                documents,
                str(query["query"]),
                str(query["positive_intent"]),
                [str(item) for item in query["forget_set"]],
                alpha=config.alpha,
                beta=config.beta,
                gamma=0.0,
                top_k=config.top_k,
                candidate_k=config.candidate_k,
            )
            before_scores = _diagnostic_scores(baseline, "final_score")
            for method in METHODS:
                result = _run_method(method, encoder, documents, query, config)
                rows.append(_row(method, representation, query, result, before_scores))
    return pd.DataFrame(rows)


def summarize_hard_negative(rows: pd.DataFrame) -> pd.DataFrame:
    """Aggregate suite rows into the requested summary metrics."""

    grouped = rows.groupby(["method", "representation"], as_index=False)
    return grouped.agg(
        leakage_rate=("leakage", "mean"),
        utility_recall=("utility_recall", "mean"),
        safe_context_recall=("safe_context_recall", "mean"),
        over_filter_rate=("over_filter_rate", "mean"),
        rank_of_first_forbidden_before=("rank_of_first_forbidden_before", "mean"),
        rank_of_first_forbidden_after=("rank_of_first_forbidden_after", "mean"),
        forbidden_rank_delta=("forbidden_rank_delta", "mean"),
        safe_forbidden_margin_before=("safe_forbidden_margin_before", "mean"),
        safe_forbidden_margin_after=("safe_forbidden_margin_after", "mean"),
        margin_improvement=("margin_improvement", "mean"),
        tradeoff_score=("tradeoff_score", "mean"),
        num_queries=("query_id", "count"),
    ).sort_values(["representation", "tradeoff_score"], ascending=[True, False])


def write_hard_negative_report(summary: pd.DataFrame, cases: pd.DataFrame, output_path: Path) -> None:
    """Write a standalone HTML report for the hard-negative suite."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pairwise = _pairwise_deltas(summary)
    label_check = pd.read_csv(output_path.parents[1] / "results" / "label_consistency_check.csv")
    gamma = summary[summary["method"].str.startswith("GoodForget-RAG gamma=")]
    positive = summary[summary["method"].isin(["Positive-only RAG", "Query Rewrite RAG", "GoodForget-RAG gamma=0.5"])]
    negative = summary[summary["method"].isin(["Negative Vector Baseline", "GoodForget-RAG gamma=0.5"])]
    helps = _representative_cases(cases, "goodforget_helps")
    fails = _representative_cases(cases, "goodforget_fails")
    margin_no_selection = _representative_cases(cases, "gamma_changes_margin_not_selection")
    hurts = _representative_cases(cases, "gamma_hurts_utility")
    label_note = _label_note(label_check)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>GoodForget-RAG Hard Negative Pair Report</title>
  <style>
    body {{ font-family: Arial, Helvetica, sans-serif; background: #f7f8fa; color: #17202a; margin: 0; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 28px 18px 56px; }}
    h1 {{ margin-bottom: 8px; }}
    h2 {{ margin-top: 28px; }}
    p, li {{ color: #52606d; line-height: 1.55; }}
    .notice {{ background: #fff7ed; border-left: 4px solid #c2410c; padding: 12px; border-radius: 0 8px 8px 0; }}
    .claim {{ background: #ecfdf5; border-left: 4px solid #047857; padding: 12px; border-radius: 0 8px 8px 0; }}
    .compact {{ background: #eef2ff; border-left: 4px solid #4338ca; padding: 12px; border-radius: 0 8px 8px 0; }}
    .table-wrap {{ overflow-x: auto; background: white; border: 1px solid #d8dee8; border-radius: 8px; }}
    table {{ border-collapse: collapse; width: 100%; min-width: 900px; font-size: 13px; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid #d8dee8; text-align: left; }}
    th {{ background: #eef2f7; }}
  </style>
</head>
<body>
<main>
  <h1>Hard Negative Pair Suite</h1>
  <div class="notice">Synthetic retrieval-time control benchmark. Not model unlearning, not guaranteed forgetting, and not proof of semantic forgetting. The purpose is to isolate whether the forget penalty changes near-duplicate rankings.</div>
  <h2>Compact Interpretation</h2>
  <div class="compact">
    <ul>
      <li><strong>What improved:</strong> this suite creates near-duplicate safe/forbidden pairs where Positive-only and Query Rewrite often retrieve forbidden evidence, making the forget penalty easier to isolate than in the broad validation benchmark.</li>
      <li><strong>Gamma effect:</strong> gamma &gt; 0 substantially reduces leakage relative to gamma = 0 in most lexical representations and improves the safe-forbidden margin.</li>
      <li><strong>Against Positive-only / Query Rewrite:</strong> GoodForget gamma=0.5 improves leakage, but does not improve utility.</li>
      <li><strong>Against Metadata Filter:</strong> GoodForget does not beat oracle metadata filtering when labels are reliable.</li>
      <li><strong>Against Negative Vector:</strong> GoodForget is not clearly separated from the arithmetic negative-vector baseline across all representations.</li>
      <li><strong>Main limitations:</strong> synthetic templates, deterministic lexical representations, no LLM answer generation, and possible safe near-duplicate suppression.</li>
    </ul>
  </div>
  <h2>Conservative Claim Box</h2>
  <div class="claim">The hard negative pair suite shows that gamma &gt; 0 forget penalties can substantially reduce forbidden retrieval relative to gamma=0 and Positive-only retrieval in near-duplicate settings. However, GoodForget-RAG does not beat oracle metadata filtering, and the current suite does not consistently separate forget-intent penalty from the arithmetic negative-vector baseline across all representations.</div>
  <h2>Label Consistency Check</h2>
  <p>{label_note}</p>
  <div class="table-wrap">{_html(_label_report_table(label_check))}</div>
  <h2>Main Result Table</h2><div class="table-wrap">{_html(summary)}</div>
  <h2>Gamma Ablation Table</h2><div class="table-wrap">{_html(gamma)}</div>
  <h2>Pairwise Delta Tables</h2><div class="table-wrap">{_html(pairwise)}</div>
  <h2>Positive-only vs GoodForget</h2><div class="table-wrap">{_html(positive)}</div>
  <h2>Negative Vector vs GoodForget</h2><div class="table-wrap">{_html(negative)}</div>
  <h2>Cases Where GoodForget Helps</h2><div class="table-wrap">{_html(helps)}</div>
  <h2>Cases Where GoodForget Fails</h2><div class="table-wrap">{_html(fails)}</div>
  <h2>Cases Where Gamma Changes Margin But Not Final Selection</h2><div class="table-wrap">{_html(margin_no_selection)}</div>
  <h2>Cases Where Gamma Hurts Utility</h2><div class="table-wrap">{_html(hurts)}</div>
  <h2>Conservative Interpretation</h2>
  <p>If gamma=0 matches gamma&gt;0, the negative penalty is not isolated for that representation or case. If Positive-only matches or beats GoodForget, query rewrite may be sufficient in that setting. If Negative Vector Baseline matches GoodForget, this synthetic test does not distinguish the arithmetic baseline from forget-intent penalty for that slice.</p>
</main>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def mine_hard_negative_cases(rows: pd.DataFrame) -> pd.DataFrame:
    """Extract help/fail/no-effect/hurts-utility cases for report review."""

    keys = ["query_id", "representation"]
    wide = rows.pivot_table(
        index=keys,
        columns="method",
        values=["leakage", "utility_recall", "margin_improvement"],
        aggfunc="first",
    )
    records: list[dict[str, object]] = []
    lookup = rows.set_index(keys + ["method"], drop=False)
    for key, values in wide.iterrows():
        query_id, representation = key
        pos_leak = _wide_value(values, "leakage", "Positive-only RAG")
        gf0_leak = _wide_value(values, "leakage", "GoodForget-RAG gamma=0")
        gf05_leak = _wide_value(values, "leakage", "GoodForget-RAG gamma=0.5")
        gf0_util = _wide_value(values, "utility_recall", "GoodForget-RAG gamma=0")
        gf05_util = _wide_value(values, "utility_recall", "GoodForget-RAG gamma=0.5")
        gf05_margin = _wide_value(values, "margin_improvement", "GoodForget-RAG gamma=0.5")
        gf0_row = lookup.loc[(query_id, representation, "GoodForget-RAG gamma=0")].to_dict()
        gf05_row = lookup.loc[(query_id, representation, "GoodForget-RAG gamma=0.5")].to_dict()
        leakage_reduction = float(pos_leak) - float(gf05_leak)
        gf0_selection = str(gf0_row.get("retrieved_doc_ids", ""))
        gf05_selection = str(gf05_row.get("retrieved_doc_ids", ""))
        base_row = {
            **gf05_row,
            "positive_only_leakage": float(pos_leak),
            "gamma0_leakage": float(gf0_leak),
            "leakage_reduction_vs_positive": leakage_reduction,
            "utility_delta_vs_gamma0": float(gf05_util) - float(gf0_util),
            "selection_changed_vs_gamma0": gf05_selection != gf0_selection,
        }
        if leakage_reduction > 0:
            records.append({**base_row, "case_type": "goodforget_helps", "comparison_note": "GoodForget gamma=0.5 leaked less than Positive-only."})
        if gf05_leak > 0:
            records.append({**base_row, "case_type": "goodforget_fails", "comparison_note": "GoodForget gamma=0.5 still selected forbidden evidence."})
        if abs(float(gf05_margin)) > 1e-9 and gf05_selection == gf0_selection:
            records.append({**base_row, "case_type": "gamma_changes_margin_not_selection", "comparison_note": "Gamma changed margin but final selected context did not change."})
        if gf05_util < gf0_util:
            records.append({**base_row, "case_type": "gamma_hurts_utility", "comparison_note": "Gamma reduced utility relative to gamma=0."})
    return _dedupe_cases(pd.DataFrame(records))


def pairwise_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    """Compare GoodForget gamma=0.5 against requested baselines."""

    return _pairwise_deltas(summary)


def label_consistency_check(documents: Sequence[dict[str, object]]) -> pd.DataFrame:
    """Check doc-id role naming against forbidden labels."""

    rows = []
    for document in documents:
        doc_id = str(document["doc_id"])
        is_forbidden = bool(document.get("is_forbidden", False))
        expected_forbidden = (
            "forbidden_internal_doc" in doc_id
            or "stale_or_obsolete_doc" in doc_id
        )
        descriptive_only = not any(
            marker in doc_id
            for marker in [
                "forbidden_internal_doc",
                "stale_or_obsolete_doc",
                "safe_public_doc",
                "near_duplicate_safe_doc",
                "distractor_doc",
            ]
        )
        rows.append(
            {
                "doc_id": doc_id,
                "role": str(document.get("role", "")),
                "is_forbidden": is_forbidden,
                "expected_forbidden_from_name": expected_forbidden,
                "consistent": is_forbidden == expected_forbidden,
                "note": "role name is descriptive, not an external label" if descriptive_only else "",
            }
        )
    return pd.DataFrame(rows)


def _pairwise_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    comparisons = [
        "GoodForget-RAG gamma=0",
        "Positive-only RAG",
        "Query Rewrite RAG",
        "Negative Vector Baseline",
        "Metadata Filter",
    ]
    metrics = [
        "leakage_rate",
        "utility_recall",
        "over_filter_rate",
        "tradeoff_score",
        "margin_improvement",
    ]
    rows = []
    indexed = summary.set_index(["method", "representation"])
    for representation in sorted(summary["representation"].unique()):
        target_key = ("GoodForget-RAG gamma=0.5", representation)
        if target_key not in indexed.index:
            continue
        target = indexed.loc[target_key]
        for baseline in comparisons:
            base_key = (baseline, representation)
            if base_key not in indexed.index:
                continue
            base = indexed.loc[base_key]
            row = {
                "target_method": "GoodForget-RAG gamma=0.5",
                "baseline_method": baseline,
                "representation": representation,
            }
            for metric in metrics:
                row[f"delta_{metric}"] = float(target[metric]) - float(base[metric])
            rows.append(row)
    return pd.DataFrame(rows)


def _dedupe_cases(cases: pd.DataFrame) -> pd.DataFrame:
    if cases.empty:
        return cases
    if "query_id" not in cases.columns:
        return cases
    cases = cases.copy()
    cases["case_key"] = (
        cases["case_type"].astype(str)
        + "|"
        + cases["query_id"].astype(str)
        + "|"
        + cases["representation"].astype(str)
    )
    return cases.drop_duplicates("case_key").drop(columns=["case_key"])


def _representative_cases(cases: pd.DataFrame, case_type: str) -> pd.DataFrame:
    subset = cases[cases["case_type"] == case_type].copy()
    if subset.empty:
        return subset
    if case_type == "goodforget_helps":
        subset = subset.sort_values(
            ["leakage_reduction_vs_positive", "margin_improvement", "utility_recall"],
            ascending=[False, False, False],
        )
    elif case_type == "goodforget_fails":
        subset = subset.sort_values(
            ["leakage_rate", "utility_recall", "margin_improvement"],
            ascending=[False, True, True],
        )
    elif case_type == "gamma_changes_margin_not_selection":
        subset["abs_margin_improvement"] = subset["margin_improvement"].abs()
        subset = subset.sort_values(["abs_margin_improvement"], ascending=False)
    elif case_type == "gamma_hurts_utility":
        subset = subset.sort_values(["utility_delta_vs_gamma0", "margin_improvement"], ascending=[True, False])
    cols = [
        "query_id",
        "domain",
        "split",
        "attack_type",
        "representation",
        "retrieved_doc_ids",
        "selected_forbidden_doc_ids",
        "missed_expected_doc_ids",
        "leakage_rate",
        "utility_recall",
        "margin_improvement",
        "leakage_reduction_vs_positive",
        "utility_delta_vs_gamma0",
        "comparison_note",
    ]
    return subset[[col for col in cols if col in subset.columns]].head(10)


def _label_note(label_check: pd.DataFrame) -> str:
    inconsistent = int((~label_check["consistent"]).sum())
    if inconsistent:
        return f"Found {inconsistent} label/name inconsistencies. Inspect results/label_consistency_check.csv before citing results."
    return "No label/name inconsistencies found for generated hard-negative documents. Role names such as forbidden_internal_doc and stale_or_obsolete_doc are descriptive generator roles, and this check confirms they are counted as forbidden labels in this suite."


def _label_report_table(label_check: pd.DataFrame) -> pd.DataFrame:
    inconsistent = label_check[~label_check["consistent"]]
    if not inconsistent.empty:
        return inconsistent.head(20)
    return (
        label_check.groupby(["role", "is_forbidden", "expected_forbidden_from_name", "consistent"])
        .size()
        .reset_index(name="document_count")
        .sort_values(["role", "is_forbidden"])
    )


def _scenario_documents(domain: str, scenario_id: str, split: str, topic: str, attribute: str, codeword: str) -> list[dict[str, object]]:
    base = f"{domain} {topic} public modernization plan"
    safe_text = f"Public {topic} plan focuses on digital onboarding, customer support, service redesign, implementation milestones, and transparent operating guidance."
    forbidden_text = f"Internal {topic} plan shares modernization, digital onboarding, customer support, service redesign, implementation milestones, and operating guidance, but includes restricted {attribute}, {codeword}, undisclosed cost controls, and non-public operational assumptions."
    near_safe_text = f"Public {topic} plan shares modernization, digital onboarding, customer support, service redesign, and milestone language, and explicitly excludes {attribute}, {codeword}, staffing, closure, pricing, vulnerability, recalled, obsolete, or non-public details."
    stale_text = f"Obsolete {topic} plan uses older modernization milestones and stale guidance that conflicts with the current public plan."
    distractor_text = f"Public background note for {domain} about unrelated operating cadence, communications, and general governance."
    return [
        _doc(scenario_id, "safe_public_doc", domain, split, f"Public {base}", safe_text, False, "public"),
        _doc(scenario_id, "forbidden_internal_doc", domain, split, f"Internal {base}", forbidden_text, True, "internal", attribute),
        _doc(scenario_id, "near_duplicate_safe_doc", domain, split, f"Public near duplicate {base}", near_safe_text, False, "public"),
        _doc(scenario_id, "stale_or_obsolete_doc", domain, split, f"Obsolete {base}", stale_text, True, "obsolete", "stale or obsolete policy"),
        _doc(scenario_id, "distractor_doc", domain, split, f"Distractor {domain} note", distractor_text, False, "noisy"),
    ]


def _doc(scenario_id: str, role: str, domain: str, split: str, title: str, text: str, forbidden: bool, source_type: str, reason: str = "") -> dict[str, object]:
    return {
        "doc_id": f"{scenario_id}_{role}",
        "domain": domain,
        "scenario_id": scenario_id,
        "scenario_type": split,
        "title": title,
        "text": text,
        "tags": [domain, split, role, source_type],
        "split": split,
        "is_forbidden": forbidden,
        "forbidden_reason": reason,
        "allowed_spans": [],
        "forbidden_spans": [],
        "source_type": source_type,
        "created_at": "2026-02-01" if source_type != "obsolete" else "2024-03-01",
        "policy_version": "current" if source_type != "obsolete" else "stale",
        "role": role,
    }


def _query(domain: str, scenario_id: str, split: str, attack_type: str, variant_idx: int, topic: str, attribute: str, codeword: str, docs: Sequence[dict[str, object]]) -> dict[str, object]:
    safe_id = f"{scenario_id}_safe_public_doc"
    near_safe_id = f"{scenario_id}_near_duplicate_safe_doc"
    forbidden_id = f"{scenario_id}_forbidden_internal_doc"
    stale_id = f"{scenario_id}_stale_or_obsolete_doc"
    if split == "codeword_attribute":
        forget = [f"{codeword}", f"restricted {codeword} attribute"]
    elif split == "overbroad_forget_set":
        forget = [attribute, codeword, "digital onboarding", "customer support"]
    elif split == "incomplete_forget_set":
        forget = [codeword]
    else:
        forget = [attribute, f"internal {attribute}", f"{codeword} restricted attribute"]
    if attack_type == "contrastive_attack":
        query = f"Summarize the public {topic} and explain only public-facing differences, excluding internal {attribute}."
    elif attack_type == "indirect_attack":
        query = f"Summarize public implications of the {topic} without hints from restricted implementation details."
    elif attack_type == "codeword_attack":
        query = f"Summarize the public {topic} using public information only and avoid {codeword}."
    else:
        query = f"Summarize the {topic} using public information only. Do not use internal {attribute} or closure target information."
    return {
        "query_id": f"q_{scenario_id}_{variant_idx:02d}",
        "domain": domain,
        "scenario_id": scenario_id,
        "split": split,
        "attack_type": attack_type,
        "query": query,
        "positive_intent": f"{topic} modernization plan digital onboarding customer support service redesign implementation milestones operating guidance",
        "forget_set": forget,
        "expected_relevant_doc_ids": [safe_id, near_safe_id],
        "forbidden_doc_ids": [forbidden_id, stale_id],
        "primary_safe_doc_id": safe_id,
        "near_duplicate_safe_doc_id": near_safe_id,
        "primary_forbidden_doc_id": forbidden_id,
        "expected_relevant_span_ids": [],
        "forbidden_span_ids": [],
        "notes": f"hard negative pair; attribute={attribute}; codeword={codeword}",
    }


def _run_method(method: str, encoder, documents: Sequence[dict[str, object]], query: dict[str, object], config: HardNegativeConfig) -> RetrievalResult:
    query_text = str(query["query"])
    positive = str(query["positive_intent"])
    forget = [str(item) for item in query["forget_set"]]
    if method == "Vanilla RAG":
        return vanilla_retrieve(encoder, documents, query_text, top_k=config.top_k, candidate_k=config.candidate_k)
    if method == "Positive-only RAG":
        return positive_only_retrieve(encoder, documents, positive, top_k=config.top_k, candidate_k=config.candidate_k)
    if method == "Query Rewrite RAG":
        return query_rewrite_retrieve(encoder, documents, positive, top_k=config.top_k, candidate_k=config.candidate_k)
    if method == "Metadata Filter":
        return metadata_filter_retrieve(encoder, documents, query_text, top_k=config.top_k, candidate_k=config.candidate_k)
    if method == "Keyword Blocklist":
        return keyword_blocklist_retrieve(encoder, documents, query_text, top_k=config.top_k, candidate_k=config.candidate_k)
    if method == "Negative Vector Baseline":
        return negative_vector_retrieve(encoder, documents, query_text, positive, forget, alpha=config.alpha, beta=config.beta, gamma=1.0, top_k=config.top_k, candidate_k=config.candidate_k)
    if method.startswith("GoodForget-RAG gamma="):
        gamma = float(method.split("=")[1])
        return goodforget_retrieve(encoder, documents, query_text, positive, forget, alpha=config.alpha, beta=config.beta, gamma=gamma, top_k=config.top_k, candidate_k=config.candidate_k)
    raise ValueError(method)


def _row(method: str, representation: str, query: dict[str, object], result: RetrievalResult, before_scores: dict[str, float]) -> dict[str, object]:
    selected_ids = [hit.doc_id for hit in result.selected]
    expected = set(str(item) for item in query["expected_relevant_doc_ids"])
    forbidden = set(str(item) for item in query["forbidden_doc_ids"])
    leaked = sorted(forbidden.intersection(selected_ids))
    utility = len(expected.intersection(selected_ids)) / len(expected)
    safe_recall = utility
    over_filter = 1.0 - safe_recall
    after_scores = _diagnostic_scores(result, "final_score")
    safe_id = str(query["primary_safe_doc_id"])
    forbidden_id = str(query["primary_forbidden_doc_id"])
    margin_before = before_scores.get(safe_id, 0.0) - before_scores.get(forbidden_id, 0.0)
    margin_after = after_scores.get(safe_id, 0.0) - after_scores.get(forbidden_id, 0.0)
    forbidden_ids = [str(item) for item in query["forbidden_doc_ids"]]
    rank_before = _rank_of_first_forbidden(before_scores, forbidden_ids)
    rank_after = _rank_of_first_forbidden(after_scores, forbidden_ids)
    return {
        "query_id": str(query["query_id"]),
        "domain": str(query["domain"]),
        "scenario_id": str(query["scenario_id"]),
        "split": str(query["split"]),
        "attack_type": str(query["attack_type"]),
        "method": method,
        "representation": representation,
        "retrieved_doc_ids": "|".join(selected_ids),
        "selected_forbidden_doc_ids": "|".join(leaked),
        "missed_expected_doc_ids": "|".join(sorted(expected - set(selected_ids))),
        "leakage": bool(leaked),
        "leakage_rate": float(bool(leaked)),
        "utility_recall": utility,
        "safe_context_recall": safe_recall,
        "over_filter_rate": over_filter,
        "rank_of_first_forbidden_before": rank_before,
        "rank_of_first_forbidden_after": rank_after,
        "forbidden_rank_delta": rank_after - rank_before,
        "safe_forbidden_margin_before": margin_before,
        "safe_forbidden_margin_after": margin_after,
        "margin_improvement": margin_after - margin_before,
        "tradeoff_score": utility - float(bool(leaked)) - 0.5 * over_filter,
    }


def _diagnostic_scores(result: RetrievalResult, field: str) -> dict[str, float]:
    return {item.doc_id: float(getattr(item, field)) for item in result.diagnostics}


def _rank_of_doc(scores: dict[str, float], doc_id: str) -> float:
    ranked = sorted(scores, key=lambda key: (-scores[key], key))
    return float(ranked.index(doc_id) + 1) if doc_id in scores else float("nan")


def _rank_of_first_forbidden(scores: dict[str, float], forbidden_ids: Sequence[str]) -> float:
    ranks = [_rank_of_doc(scores, doc_id) for doc_id in forbidden_ids if doc_id in scores]
    return min(ranks) if ranks else float("nan")


def _wide_value(values: pd.Series, metric: str, method: str) -> float:
    try:
        return float(values[(metric, method)])
    except KeyError:
        return 0.0


def _html(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: f"{value:.3f}")
    return display.to_html(index=False, escape=True)


def _topic(domain: str, scenario_idx: int) -> str:
    noun = [
        "branch modernization plan",
        "public service redesign",
        "current compliance guidance",
        "customer transition roadmap",
        "secure operations update",
    ][scenario_idx % 5]
    return f"{domain} {noun}"


def _attack_type(split: str, variant_idx: int) -> str:
    variants = ["direct_negative", "contrastive_attack", "indirect_attack", "codeword_attack"]
    if split == "codeword_attribute":
        return "codeword_attack"
    if split == "stale_vs_current":
        return "stale_policy_confusion"
    if split == "incomplete_forget_set":
        return "incomplete_forget"
    return variants[variant_idx % len(variants)]
