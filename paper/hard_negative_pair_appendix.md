# Hard Negative Pair Appendix

## Purpose

The hard negative pair suite is designed to isolate whether the negative-aware forget penalty contributes beyond Positive-only RAG, Query Rewrite RAG, and the Negative Vector Baseline.

This appendix does not claim model unlearning, guaranteed forgetting, SOTA, or proof of semantic forgetting.

## Design

Each synthetic scenario includes:

- `safe_public_doc`
- `forbidden_internal_doc`
- `near_duplicate_safe_doc`
- `stale_or_obsolete_doc`
- `distractor_doc`

The safe and forbidden documents intentionally share many surface terms. The forbidden document contains a restricted attribute such as layoffs, closure targets, non-public acquisition targets, unreleased vulnerability workarounds, recalled medication instructions, obsolete compliance rules, confidential pricing plans, or undisclosed vendor scores.

The near-duplicate safe document is close to the forbidden document but explicitly excludes the restricted attribute. This makes the test harder: a forget penalty can suppress the forbidden document, but it may also penalize safe near-duplicate text.

## Methods

The suite compares Vanilla RAG, Positive-only RAG, Query Rewrite RAG, Metadata Filter, Keyword Blocklist, Negative Vector Baseline, and GoodForget-RAG at `gamma = 0`, `0.5`, `1.0`, and `1.5`.

## Critical Margin Metric

For each query, the suite computes:

```text
margin_before = score(safe_expected_doc) - score(forbidden_doc)
margin_after = score(safe_expected_doc) - score(forbidden_doc)
margin_improvement = margin_after - margin_before
```

This is necessary because leakage alone can hide whether GoodForget-RAG actually changed the ranking. If `gamma > 0` has no margin effect relative to `gamma = 0`, the suite does not isolate a contribution from the forget penalty for that case.

## Conservative Interpretation

The hard negative pair suite shows that `gamma > 0` forget penalties can substantially reduce forbidden retrieval relative to `gamma=0` and Positive-only retrieval in near-duplicate settings. However, GoodForget-RAG does not beat oracle metadata filtering, and the current suite does not consistently separate forget-intent penalty from the arithmetic negative-vector baseline across all representations.

If GoodForget-RAG does not outperform Positive-only RAG, the report says so. If `gamma = 0` performs as well as `gamma > 0`, the negative penalty is not isolated. If the Negative Vector Baseline performs similarly, this benchmark slice does not distinguish the arithmetic baseline from forget-intent suppression.

## Pairwise Deltas and Label Consistency

The suite writes `results/hard_negative_pair_pairwise_deltas.csv` with GoodForget-RAG `gamma=0.5` compared against `gamma=0`, Positive-only RAG, Query Rewrite RAG, Negative Vector Baseline, and Metadata Filter. It also writes `results/label_consistency_check.csv`.

Role-like document IDs such as `forbidden_internal_doc` and `stale_or_obsolete_doc` are generator descriptors. They are not independent external labels. The consistency check verifies whether those descriptors match the generated `is_forbidden` field before the results are cited.

## Limitations

The suite is synthetic. It uses local deterministic representations and fixed templates. It is intended for red-team analysis of retrieval-time control tradeoffs, not for claims about LLM parameter changes or production safety.
