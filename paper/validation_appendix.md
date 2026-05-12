# Validation Appendix

## 1. Benchmark Generation Procedure

The benchmark generator creates deterministic synthetic corpora from a configuration file. Each scenario contains allowed public documents, forbidden internal or obsolete documents, mixed-evidence documents, and noisy public background documents. Queries are generated from attack templates and include positive intents, forget sets, expected relevant document IDs, and forbidden document IDs.

## 2. Splits and Attack Types

Splits include literal, paraphrase, mixed evidence, stale policy, near duplicate, codeword, noisy forget set, and incomplete forget set. Attack types include direct negative, paraphrased negative, codeword attack, contrastive attack, indirect attack, stale-policy confusion, noisy forget, and incomplete forget.

## 3. Baselines

The suite evaluates Vanilla RAG, Positive-only RAG, Query Rewrite RAG, Keyword Blocklist, Metadata Filter, GoodForget-RAG, GoodForget-RAG with hard thresholding, GoodForget-RAG span-level, and a Negative Vector Baseline.

## 4. Metrics

Metrics include leakage rate, answer leakage proxy, utility recall, safe context recall, over-filter rate, forbidden removal precision and recall, context purity, context sufficiency, false suppression rate, rank of first forbidden document, and a convenience tradeoff score.

The tradeoff score is not universal:

```text
tradeoff_score = utility_recall - leakage_rate - 0.5 * over_filter_rate
```

## 5. Statistical Analysis

Bootstrap confidence intervals are computed over query-level rows for leakage rate, utility recall, over-filter rate, and tradeoff score. Seed stability is computed by regenerating the synthetic benchmark under multiple deterministic seeds.

## 6. Main Results

The fast benchmark shows that GoodForget-RAG can reduce leakage relative to Vanilla RAG, but it may lose utility. Metadata filtering performs best when labels are reliable. Positive-only and Query Rewrite baselines can be competitive because positive intents are available.

## 7. Ablation Results

Ablations vary alpha, beta, gamma, top-k, candidate-k, and forget threshold. Increasing gamma or lowering thresholds can suppress forbidden evidence but can also reduce utility and context sufficiency.

## 8. Label Noise Study

Metadata filtering is an oracle baseline under clean labels. Label-noise experiments remove forbidden labels or add false positive forbidden labels. This tests why retrieval-time controls can be relevant when labels are incomplete, dynamic, or too coarse.

## 9. Span-Level Study

Span-level retrieval splits documents into sentence-like spans. This reduces document-level mixed-evidence limitations in the synthetic setting, but span labels are still synthetic and should not be treated as human-reviewed annotations.

## 10. Negative Vector Baseline

The Negative Vector Baseline tests the flawed arithmetic idea of using `-v` as an opposite forbidden vector. It is not equivalent to explicit forget-intent suppression. Results should be read honestly: it may tie some settings, but it is representation-sensitive and not recommended as a principled forgetting mechanism.

## 11. Failure Analysis

Failure cases are mined to `results/failure_cases.csv` and summarized in `docs/failure_cases.md`. Likely causes include lexical overlap failure, codeword mismatch, overbroad forget sets, incomplete forget sets, near-duplicate confusion, mixed-evidence granularity, representation sensitivity, and threshold settings.

## 12. Limitations

This benchmark is synthetic. It is not model unlearning, not a safety guarantee, not SOTA, and not proof of semantic forgetting. It evaluates retrieval-time behavior under local deterministic representations. Real corpora, human labels, dense embedding validation, adversarial prompt behavior, and full LLM answer leakage remain outside the scope of this appendix.
