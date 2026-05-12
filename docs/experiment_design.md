# Experiment Design

## Benchmark Shape

The generator creates domains, scenarios, documents, and query variants from deterministic templates. The default configuration targets 8 domains, 6 scenarios per domain, 5 documents per scenario, and 4 query variants per scenario. The fast configuration is smaller for quick local validation.

## Splits

- `literal`
- `paraphrase`
- `mixed_evidence`
- `stale_policy`
- `near_duplicate`
- `codeword`
- `noisy_forget_set`
- `incomplete_forget_set`

## Baselines

- Vanilla RAG
- Positive-only RAG
- Query Rewrite RAG
- Keyword Blocklist
- Metadata Filter
- GoodForget-RAG
- GoodForget-RAG hard threshold
- GoodForget-RAG span-level
- Negative Vector Baseline

## Metrics

The suite reports leakage, answer leakage proxy, utility recall, safe context recall, over-filter rate, forbidden removal precision/recall, context purity, context sufficiency, false suppression rate, rank of first forbidden document, and a convenience tradeoff score.

The tradeoff score is:

```text
utility_recall - leakage_rate - 0.5 * over_filter_rate
```

It is not a universal metric.
