# Red-Team Notes

## What This Project Demonstrates

GoodForget-RAG demonstrates a small retrieval-time control mechanism for RAG. It shows how a retriever can combine query relevance, positive intent relevance, and a forget-intent penalty under a deterministic TF-IDF representation.

The project also demonstrates a more careful evaluation setup than a single vanilla baseline:

- positive-only retrieval;
- query rewrite retrieval;
- keyword blocklist filtering;
- label-aware metadata filtering;
- split-level reporting;
- retrieval-context leakage metrics;
- a deterministic answer leakage proxy;
- sensitivity analysis over penalty and threshold values.

## What This Project Does Not Demonstrate

This project does not demonstrate model unlearning. It does not show that an LLM has forgotten anything internally. It does not prove semantic forgetting, safety, privacy compliance, or robustness against adversarial prompts.

It also does not validate dense embedding retrieval. The current implementation uses TF-IDF as a lexical proxy. Dense embedding behavior may differ substantially.

## Likely Reviewer Criticisms

1. **Synthetic bias**: the dataset may be designed in a way that favors the proposed method.
2. **Oracle inputs**: positive intents and forget sets are provided directly.
3. **Weak semantic coverage**: TF-IDF may miss paraphrases and code words.
4. **Document-level granularity**: mixed documents may contain both allowed and forbidden spans.
5. **Metadata baseline**: if reliable forbidden labels are available, metadata filtering may be stronger.
6. **Answer evaluation gap**: selected context is not the same as actual generated answers.
7. **Small sample size**: six queries are not enough for broad claims.

## How This Version Addresses Some Criticisms

The project includes multiple baselines so vanilla RAG is not the only comparison. Query rewrite and positive-only retrieval are included because raw negative phrasing can unfairly disadvantage vanilla retrieval.

The dataset has three splits. Literal examples test easy keyword cases. Paraphrase examples show where keyword blocklists and lexical retrieval can fail. Mixed-evidence examples show document-level limitations.

The metrics include over-filtering and safe context recall, not only leakage. This makes it harder to hide poor utility behind aggressive filtering.

The technical note explicitly states that the method is retrieval-time control, not model unlearning, and that TF-IDF is a lexical proxy.

## Remaining Weaknesses

The dataset is still synthetic and small. Real corpora would contain noisier labels, near duplicates, stale policies, partial restrictions, and contradictory evidence.

The forget set is manually specified. A real deployment would need a process for generating, reviewing, and updating forget intents.

The answer leakage metric is deterministic and evidence-based. It does not capture hallucinated leakage, paraphrased leakage, prompt injection, or model priors.

The method works at document level. A span-level retriever or redactor would be more appropriate for mixed evidence.

Metadata filtering remains a strong oracle baseline. GoodForget-RAG is most relevant when labels are unavailable, incomplete, or too coarse, but this repository does not solve label governance.
