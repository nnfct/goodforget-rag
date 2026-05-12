# LinkedIn Post Draft (EN)

I built a small technical-note project called **GoodForget-RAG**.

The idea is to frame forgetting in RAG not as deletion of knowledge from model parameters, but as **selective non-use of retrievable evidence**.

In RAG, the key question is not only "what should we retrieve?" It is also "what should we avoid using?"

GoodForget-RAG is a toy implementation of **negative-aware retrieval**. It scores candidate documents using the original query, a positive intent, and a penalty for documents close to a forget set:

```text
S(d) = alpha * sim(q, d) + beta * sim(p, d) - gamma * max_i sim(f_i, d)
```

The forget-intent vector is not the arithmetic negative vector `-v`. It is a separate representation of what should not be used.

The current version is a TF-IDF-based toy evaluation. It is not model unlearning, not a safety guarantee, and not proof of semantic forgetting.

The project includes stronger red-team framing than a simple demo:

- synthetic data may favor the method;
- positive and forget intents are assumed as oracle inputs;
- TF-IDF is only a lexical proxy;
- answer leakage is measured with a deterministic proxy, not an LLM;
- metadata filtering can outperform GoodForget-RAG when reliable labels are available.

The practical takeaway: RAG systems should be designed not only to retrieve relevant evidence, but also to reason about which evidence should not enter the context.
