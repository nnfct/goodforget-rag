# LinkedIn Post Draft (EN)

I built a small technical-note project called **GoodForget-RAG**.

The idea is to frame forgetting in RAG not as removal of knowledge from model parameters, but as **selective non-use of retrievable evidence**.

In RAG, the key question is not only "what should we retrieve?" It is also "what should we avoid using?"

GoodForget-RAG is a toy implementation of **negative-aware retrieval**. It scores candidate documents using the original query, a positive intent, and a penalty for documents that are semantically close to a forget set:

```text
S(d) = alpha * sim(q, d) + beta * sim(p, d) - gamma * max_i sim(f_i, d)
```

The forget items are explicit forget-intent texts, not arithmetic negative vectors.

This is not model unlearning and not a safety guarantee. It is a retrieval-time control mechanism that explores **behavioral forgetting in RAG**: suppress forbidden evidence while preserving useful adjacent knowledge.

The repository includes a deterministic TF-IDF implementation, a small synthetic corpus, a toy evaluation, result CSVs, and a short technical note.

The practical takeaway: RAG systems should be designed not only to retrieve relevant evidence, but also to reason about which evidence should not enter the context.
