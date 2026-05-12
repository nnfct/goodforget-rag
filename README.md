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
│   ├── data.py
│   ├── intents.py
│   └── spans.py
├── experiments/
│   ├── run_experiment.py
│   ├── run_sensitivity.py
│   ├── run_auto_intent.py
│   ├── run_representation_compare.py
│   ├── run_span_experiment.py
│   ├── write_html_brief.py
│   ├── toy_corpus.jsonl
│   └── toy_queries.jsonl
├── results/
│   ├── summary.csv
│   ├── split_summary.csv
│   ├── toy_results.csv
│   ├── sensitivity.csv
│   ├── auto_intent_summary.csv
│   ├── representation_summary.csv
│   └── span_summary.csv
├── paper/
│   └── technical_note.md
├── docs/
│   ├── architecture.html
│   ├── experiment_brief.html
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
python experiments/run_auto_intent.py
python experiments/run_representation_compare.py
python experiments/run_span_experiment.py
python experiments/write_html_brief.py
```

The scripts use only local files and do not download models or call external APIs at runtime.

## Validation Benchmark Suite

The validation suite expands the six-query toy demo into a larger synthetic benchmark for retrieval-time control tradeoffs. It is exploratory and synthetic; it should be used to inspect where methods help, tie, or fail.

Benchmark commands:

```bash
python experiments/generate_benchmark.py --config configs/benchmark_default.yaml
python experiments/run_benchmark_suite.py --config configs/benchmark_default.yaml
python experiments/run_attack_suite.py --config configs/benchmark_default.yaml
python experiments/run_ablation_suite.py --config configs/benchmark_fast.yaml
python experiments/run_label_noise_suite.py --config configs/benchmark_default.yaml
python experiments/run_seed_stability.py --config configs/benchmark_fast.yaml
python experiments/run_negative_vector_baseline.py --config configs/benchmark_default.yaml
python experiments/make_validation_report.py
```

For laptop smoke tests, replace `benchmark_default.yaml` with `benchmark_fast.yaml`.

The report is written to `docs/validation_report.html`. Open it in a browser to inspect main results, attack-type breakdowns, label-noise behavior, bootstrap intervals, plots, and mined failure cases.

## Hard Negative Pair Suite

The hard negative pair suite targets a narrower question: does the negative-aware forget penalty contribute beyond Positive-only RAG, Query Rewrite RAG, and the arithmetic Negative Vector Baseline?

Run:

```bash
python experiments/run_hard_negative_pair_suite.py
```

Outputs:

- `results/hard_negative_pair_summary.csv`
- `results/hard_negative_pair_cases.csv`
- `results/hard_negative_pair_pairwise_deltas.csv`
- `results/label_consistency_check.csv`
- `docs/hard_negative_pair_report.html`
- `paper/hard_negative_pair_appendix.md`

The suite generates 8 domains, 10 scenarios per domain, and 4 query variants per scenario. Each scenario includes a safe public document, a forbidden internal near-duplicate, a safe near-duplicate that explicitly avoids the restricted attribute, an obsolete document, and a distractor.

Conservative result summary from the current run:

- `GoodForget-RAG gamma=0` often leaks badly in hard negative settings, especially under `tfidf_word` and `tfidf_char`.
- `gamma > 0` improves the safe-forbidden margin and substantially reduces leakage relative to `gamma=0` in the near-duplicate settings.
- GoodForget-RAG improves over Positive-only and Query Rewrite on leakage in several representations, but it does not improve utility in this suite.
- Metadata Filter remains strongest when labels are reliable and should be treated as the oracle baseline.
- Negative Vector Baseline can be close to GoodForget-RAG in some settings, so the current suite does not consistently separate forget-intent penalty from an arithmetic negative-vector baseline across all representations.
- Higher gamma can hurt utility by suppressing safe near-duplicate documents that mention excluded attributes.
- The report now includes pairwise delta tables and a label consistency check. Role-like names such as `forbidden_internal_doc` and `stale_or_obsolete_doc` are generator descriptors; the consistency file verifies whether those descriptors match `is_forbidden`.

Conservative claim for citation: The hard negative pair suite shows that `gamma > 0` forget penalties can substantially reduce forbidden retrieval relative to `gamma=0` and Positive-only retrieval in near-duplicate settings. However, GoodForget-RAG does not beat oracle metadata filtering, and the current suite does not consistently separate forget-intent penalty from the arithmetic negative-vector baseline across all representations.

## Expected Outputs

The main experiment writes:

- `results/summary.csv`
- `results/split_summary.csv`
- `results/toy_results.csv`

The sensitivity run writes:

- `results/sensitivity.csv`

The limitation-mitigation checks write:

- `results/auto_intent_summary.csv`
- `results/auto_intent_results.csv`
- `results/representation_summary.csv`
- `results/representation_results.csv`
- `results/span_summary.csv`
- `results/span_results.csv`
- `docs/experiment_brief.html`

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

## Limitation-Mitigation Checks

The repository now includes additional checks that make some limitations measurable rather than merely stated:

- **Heuristic intent check**: compares oracle positive/forget intents against simple query-derived intents. This reduces reliance on an untested oracle assumption, but the heuristic is intentionally weak.
- **Representation sensitivity**: compares raw TF-IDF with a local TF-IDF + truncated SVD LSA representation. This is not a downloaded dense embedding model, but it checks whether the toy result depends on one lexical feature space.
- **Span-level check**: splits documents into sentence-like spans to test whether mixed-evidence cases can preserve safe spans while suppressing forbidden spans.
- **HTML brief**: `docs/experiment_brief.html` summarizes intermediate results for human review.

## Red-Team Limitations

- Oracle positive and forget intents are assumed.
- A heuristic intent comparison is included, but it is not a production query rewriter.
- The dataset is synthetic and may favor the proposed method.
- TF-IDF is a lexical proxy, not a dense semantic embedding.
- Dense embedding performance is not validated here; Local LSA is only a local sensitivity check.
- The experiment evaluates retrieval-context leakage, not full LLM answer leakage.
- `answer_leakage_rate` is only a deterministic proxy.
- The method operates at document level, not span level.
- A span-level experiment is included, but span labels are derived heuristically.
- Metadata filtering can outperform GoodForget-RAG when reliable labels are available.
- GoodForget-RAG is not a substitute for privacy review, policy enforcement, or model unlearning.

## What Not To Claim

- Do not claim model unlearning.
- Do not claim guaranteed forgetting.
- Do not claim SOTA.
- Do not claim semantic forgetting proof.
- Do not hide failures or cases where simpler baselines win.

Current conservative validation summary: on the fast synthetic benchmark, GoodForget-RAG reduces leakage relative to Vanilla RAG, but it often trades away utility. Metadata Filter is better when labels are reliable. Positive-only and Query Rewrite baselines can tie or beat GoodForget-RAG on utility because the benchmark provides positive intents.

## LinkedIn-Friendly Positioning

GoodForget-RAG frames forgetting in RAG as **selective non-use of evidence**. The practical question is not only "what should we retrieve?" but also "what should we avoid using?" This repository is a reproducible technical note for exploring that question with local TF-IDF retrieval, conservative baselines, and explicit red-team limitations.

## Suggested Next Steps

- Replace TF-IDF/Local LSA with vetted local dense embeddings and rerun the same evaluation.
- Replace heuristic span labels with human-reviewed span labels.
- Add adversarial paraphrase generation and human-reviewed labels.
- Evaluate with real corpora where policy labels are incomplete or noisy.
- Add tests for metric definitions and ranking stability.
