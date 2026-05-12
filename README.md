# GoodForget-RAG

**GoodForget-RAG: Negative-Aware Retrieval for Selective Non-Use of Knowledge** is a lightweight technical-note project that demonstrates a toy retrieval-control mechanism for RAG systems. It studies whether a retriever can suppress forbidden evidence while preserving adjacent useful evidence under a chosen vector representation.

**Good forgetting is the ability to suppress forbidden evidence while preserving utility on adjacent, non-forbidden knowledge.**

## Strong Disclaimer

This repository is intentionally conservative in scope:

- This is a toy retrieval-control demo.
- This is not model unlearning.
- This is not a safety guarantee.
- This does not prove semantic forgetting.
- This does not modify model parameters or remove knowledge from an LLM.
- This uses TF-IDF as a lexical proxy for vector similarity.
- Dense semantic retrieval is discussed as a possible extension, but it is not empirically validated here.

## Method

GoodForget-RAG reranks candidate documents with positive relevance and a forget-intent penalty:

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

The forget-intent vector is **not** the arithmetic negative vector `-v`. It is a separate representation of what should not be used.

## Baselines

The experiment compares:

- **Vanilla RAG**: retrieves by original query only.
- **Positive-only RAG**: retrieves by positive intent only.
- **Query Rewrite RAG**: uses the positive intent as a simplified rewritten query.
- **Keyword Blocklist**: retrieves candidates, then removes documents containing obvious blocked keywords.
- **Metadata Filter**: removes `is_forbidden=true` documents before retrieval. This is an oracle / label-aware baseline.
- **GoodForget-RAG**: uses positive relevance and the negative-aware forget penalty.

## Repository Structure

```text
goodforget-rag/
├── README.md
├── LICENSE
├── CITATION.cff
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── src/goodforget_rag/
│   ├── __init__.py
│   ├── vectorizer.py
│   ├── retrieval.py
│   ├── eval.py
│   └── data.py
├── experiments/
│   ├── run_experiment.py
│   ├── run_sensitivity.py
│   ├── toy_corpus.jsonl
│   └── toy_queries.jsonl
├── results/
│   ├── summary.csv
│   ├── split_summary.csv
│   ├── toy_results.csv
│   └── sensitivity.csv
├── paper/
│   └── technical_note.md
├── docs/
│   ├── architecture.html
│   ├── linkedin_post_kr.md
│   ├── linkedin_post_en.md
│   └── red_team_notes.md
└── notebooks/
    └── goodforget_demo.ipynb
```

## Quickstart

```bash
python -m pip install -e .
python experiments/run_experiment.py
python experiments/run_sensitivity.py
```

The scripts use only local files and do not download models or call external APIs at runtime.

## Expected Outputs

The main experiment writes:

- `results/summary.csv`
- `results/split_summary.csv`
- `results/toy_results.csv`

The sensitivity run writes:

- `results/sensitivity.csv`

Example summary from the current toy run:

```text
           method  leakage_rate  utility_recall  safe_context_recall  over_filter_rate  forbidden_removal_precision  empty_context_rate  answer_leakage_rate  num_queries
      Vanilla RAG         0.833           0.833                0.833             0.167                        0.000               0.000                0.833            6
Positive-only RAG         0.333           0.833                0.833             0.167                        0.000               0.000                0.333            6
Query Rewrite RAG         0.333           0.833                0.833             0.167                        0.000               0.000                0.333            6
Keyword Blocklist         0.333           0.833                0.833             0.167                        1.000               0.000                0.333            6
  Metadata Filter         0.000           0.917                0.917             0.083                        1.000               0.000                0.000            6
   GoodForget-RAG         0.333           0.833                0.833             0.167                        0.000               0.000                0.333            6
```

## Metrics

- `leakage_rate`: fraction of queries where at least one forbidden document appears in selected top-k context.
- `utility_recall`: average retrieved expected relevant documents divided by total expected relevant documents.
- `safe_context_recall`: recall over expected relevant documents that are not forbidden.
- `over_filter_rate`: fraction of expected safe documents not preserved in context.
- `forbidden_removal_precision`: among removed documents, fraction that were actually forbidden.
- `empty_context_rate`: fraction of queries where no context remains.
- `answer_leakage_rate`: deterministic proxy based on concatenated selected evidence titles and snippets. This is not an LLM answer evaluation.

## Result Interpretation

The toy dataset has three splits:

- `literal`: obvious words such as leaked, confidential, private, internal memo.
- `paraphrase`: indirect code names and non-obvious phrasing such as Project Maple and orchard route scenario.
- `mixed_evidence`: documents that combine allowed public information with forbidden concepts.

The results are intentionally not perfect. Vanilla RAG leaks frequently. Positive-only and query rewrite baselines reduce some leakage but still select forbidden evidence in close-topic cases. Keyword blocklists handle literal cases but fail on paraphrases. Metadata filtering is strong when reliable labels exist, but that assumption is often unrealistic. GoodForget-RAG reduces leakage relative to vanilla retrieval in this toy setup, but it still leaks in some literal and mixed-evidence cases with the default weights.

## Red-Team Limitations

- Oracle positive and forget intents are assumed.
- The dataset is synthetic and may favor the proposed method.
- TF-IDF is a lexical proxy, not a dense semantic embedding.
- Dense embedding performance is not validated here.
- The experiment evaluates retrieval-context leakage, not full LLM answer leakage.
- `answer_leakage_rate` is only a deterministic proxy.
- The method operates at document level, not span level.
- Metadata filtering can outperform GoodForget-RAG when reliable labels are available.
- GoodForget-RAG is not a substitute for privacy review, policy enforcement, or model unlearning.

## LinkedIn-Friendly Positioning

GoodForget-RAG frames forgetting in RAG as **selective non-use of evidence**. The practical question is not only "what should we retrieve?" but also "what should we avoid using?" This repository is a reproducible technical note for exploring that question with local TF-IDF retrieval, conservative baselines, and explicit red-team limitations.

## Suggested Next Steps

- Replace TF-IDF with local dense embeddings and rerun the same evaluation.
- Add span-level filtering for mixed-evidence documents.
- Add adversarial paraphrase generation and human-reviewed labels.
- Evaluate with real corpora where policy labels are incomplete or noisy.
- Add tests for metric definitions and ranking stability.
