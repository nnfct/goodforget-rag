# GoodForget-RAG: Negative-Aware Retrieval for Selective Non-Use of Knowledge

## 1. Abstract

Retrieval-augmented generation (RAG) systems usually optimize for selecting evidence that is relevant to a user query. In some settings, a system must also avoid using particular evidence, such as outdated instructions, confidential memos, non-public plans, or mixed documents containing restricted concepts. This note presents **GoodForget-RAG**, a toy retrieval-control mechanism that penalizes candidate documents close to an explicit forget set. The current implementation uses deterministic TF-IDF vectors and cosine similarity. It does not modify model parameters, does not solve model unlearning, and does not provide a safety guarantee. The goal is narrower: evaluate retrieval-time suppression of forbidden evidence under a chosen lexical vector representation.

**Good forgetting is the ability to suppress forbidden evidence while preserving utility on adjacent, non-forbidden knowledge.**

## 2. Motivation

RAG systems influence generation by deciding which external evidence enters the context window. A standard retriever asks which documents are most relevant to the query. A retrieval-control system also asks which documents should not be used, even when they are close to the topic.

This distinction matters because many governance problems appear at retrieval time. A corpus may contain obsolete rules, private memos, unreleased plans, or mixed documents with both allowed and forbidden evidence. Removing all related content can damage utility, while retrieving the nearest evidence can leak restricted information. GoodForget-RAG explores a simple middle ground: reward positive relevance and penalize closeness to forget intents.

## 3. Problem Framing

Given a corpus, a user query, a positive intent, and a forget set, the retriever must select a top-k context. The desired behavior is selective non-use of forbidden evidence while preserving utility on adjacent non-forbidden knowledge.

This project uses the following framing:

- **Retrieval-time forgetting**: changing retrieval behavior at inference time.
- **Negative-aware retrieval**: including an explicit forget penalty in document scoring.
- **Selective non-use of evidence**: avoiding evidence that should not be used.
- **Behavioral forgetting in RAG**: observable retrieval behavior, not parameter-level erasure.

The forget-intent vector is not the arithmetic negative vector `-v`. It is a separate representation of what should not be used.

## 4. Method: GoodForget-RAG

For each candidate document `d`, GoodForget-RAG computes:

```text
S(d) = alpha * sim(q, d) + beta * sim(p, d) - gamma * max_i sim(f_i, d)
```

Where:

- `q` is the original user query.
- `p` is the positive intent / useful information intent.
- `f_i` is a forget-intent vector.
- `d` is a candidate document.
- `sim` is cosine similarity.
- `alpha`, `beta`, and `gamma` are configurable weights.

The implementation uses scikit-learn TF-IDF vectors. TF-IDF is transparent and deterministic, but it is a lexical proxy. It does not validate dense semantic retrieval performance.

The retrieval function returns selected documents and per-document diagnostics:

- query similarity;
- positive similarity;
- maximum forget similarity;
- final score;
- threshold removal flag;
- filter removal flag;
- forbidden label.

## 5. Experimental Setup

The dataset is synthetic and intentionally small. It contains three splits.

The `literal` split uses obvious forbidden terms such as leaked, confidential, private, internal memo, and secret. This split is expected to be relatively easy for keyword filtering.

The `paraphrase` split uses code names and indirect descriptions, including Project Maple, branch optimization plan, non-public market-entry plan, and unreleased board scenario. This split exposes the limitation of lexical matching.

The `mixed_evidence` split includes documents that combine allowed public information with forbidden concepts. This tests document-level limitations: a whole document may be penalized or filtered even when only one span is problematic.

All experiments are local and deterministic. The vectorizer is fit on document titles, document text, query texts, positive intents, and forget-set strings. No API keys, paid services, or runtime downloads are used.

## 6. Baselines

The experiment compares six methods.

**Vanilla RAG** retrieves by original query similarity only. It is a weak baseline when the query includes negative instructions because those negative terms can increase similarity to forbidden documents.

**Positive-only RAG** retrieves by positive intent only. This baseline tests whether simply removing negative phrasing from the query is enough.

**Query Rewrite RAG** uses the positive intent as a simplified rewritten query. It is intentionally similar to Positive-only RAG, included because raw vanilla retrieval is disadvantaged by negative terms.

**Keyword Blocklist** retrieves candidates and removes documents containing obvious blocked keywords. It is expected to work better on literal cases than paraphrases.

**Metadata Filter** removes documents labeled `is_forbidden=true` before retrieval. This is an oracle / label-aware baseline. It can outperform GoodForget-RAG when reliable labels are available, but it is not a general solution when labels are absent, incomplete, or too coarse.

**GoodForget-RAG** combines query relevance, positive relevance, and a forget-intent penalty.

## 7. Metrics

The evaluation reports:

- `leakage_rate`: fraction of queries where at least one forbidden document appears in selected top-k context.
- `utility_recall`: retrieved expected relevant documents divided by total expected relevant documents, averaged across queries.
- `safe_context_recall`: recall over expected relevant non-forbidden documents.
- `over_filter_rate`: fraction of expected safe documents not preserved in context.
- `forbidden_removal_precision`: among removed documents, fraction that were actually forbidden.
- `empty_context_rate`: fraction of queries where no context remains.
- `answer_leakage_rate`: deterministic proxy based on selected evidence titles and snippets.

The current experiment evaluates retrieval-context leakage, not full LLM answer leakage. The `answer_leakage_rate` is only a deterministic proxy; it does not simulate free-form generation, model priors, prompt injection, or adversarial behavior.

## 8. Results

The current toy run produces:

```text
           method  leakage_rate  utility_recall  safe_context_recall  over_filter_rate  forbidden_removal_precision  empty_context_rate  answer_leakage_rate  num_queries
      Vanilla RAG         0.833           0.833                0.833             0.167                        0.000               0.000                0.833            6
Positive-only RAG         0.333           0.833                0.833             0.167                        0.000               0.000                0.333            6
Query Rewrite RAG         0.333           0.833                0.833             0.167                        0.000               0.000                0.333            6
Keyword Blocklist         0.333           0.833                0.833             0.167                        1.000               0.000                0.333            6
  Metadata Filter         0.000           0.917                0.917             0.083                        1.000               0.000                0.000            6
   GoodForget-RAG         0.333           0.833                0.833             0.167                        0.000               0.000                0.333            6
```

The result should be read narrowly. GoodForget-RAG reduces leakage relative to vanilla retrieval, but it is not perfect. Positive-only and query rewrite retrieval are strong in some cases because the positive intent is provided as an oracle input. Metadata filtering performs best because it has direct access to forbidden labels.

Split-level results are especially important. Keyword filtering works in literal cases but fails when forbidden evidence is paraphrased. GoodForget-RAG can also fail when the TF-IDF representation does not capture the relationship between a forget intent and an indirectly worded document.

## 9. Sensitivity Analysis

The sensitivity script varies:

- `gamma`: 0.5, 1.0, 1.5, 2.0;
- `forget_threshold`: 0.10, 0.15, 0.20, 0.25, 0.30.

This shows a tradeoff. Lower thresholds can remove more forbidden material but may reduce safe context recall. Higher thresholds preserve more context but may depend more heavily on ranking rather than explicit removal. The sensitivity results are written to `results/sensitivity.csv`.

## 10. Red-Team Limitations

This project has several limitations by design.

Oracle positive and forget intents are assumed. A real system would need to create or validate those intents, and errors there could dominate the retrieval behavior.

The dataset is synthetic and may favor the proposed method. It is useful for inspection and reproducibility, not broad empirical claims.

The current experiment evaluates retrieval-context leakage, not full LLM answer leakage. The answer proxy concatenates selected evidence titles and snippets and checks for forbidden markers. It is not a generated-answer evaluation.

TF-IDF is a lexical proxy, not dense semantic embedding. Dense embedding performance is not validated.

The method operates at document level, not span level. Mixed-evidence documents can force a tradeoff between utility and suppression.

Metadata filtering can outperform GoodForget-RAG when reliable labels are available. In practice, labels may be unavailable, stale, or too coarse, but this repository does not solve that data-governance problem.

GoodForget-RAG is not a substitute for privacy review, policy enforcement, or model unlearning. It is a retrieval-time control experiment.

## 11. Conclusion

GoodForget-RAG is a small, reproducible experiment in negative-aware retrieval for selective non-use of evidence. The contribution is not a claim that forgetting is guaranteed, and it is not a claim of parameter-level deletion. It is a concrete toy framework for asking a practical RAG question: how can a retriever preserve useful adjacent evidence while reducing the chance that forbidden evidence enters the context?
