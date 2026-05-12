# GoodForget-RAG: Negative-Aware Retrieval for Selective Non-Use of Knowledge

## Abstract

Retrieval-augmented generation (RAG) systems usually optimize for retrieving evidence that is relevant to a user query. In many practical settings, however, a system also needs to avoid using certain evidence: obsolete policy, recalled instructions, deprecated internal procedures, or records that should no longer influence generated answers. This note introduces **GoodForget-RAG**, a toy implementation of negative-aware retrieval for **retrieval-time forgetting**. The method does not modify language model parameters. Instead, it changes retrieval behavior by penalizing candidate documents that are semantically close to an explicit forget set while preserving useful evidence through a positive intent term. In a small synthetic evaluation, this mechanism reduces forbidden-document retrieval compared with vanilla and positive-only retrieval baselines.

## 1. Motivation

RAG systems make language models more useful by supplying external context. That same mechanism can also amplify outdated or restricted evidence when the retriever ranks it highly. A conventional retriever asks: which documents are most similar to the query? In a knowledge governance setting, a second question is equally important: which documents should not be used even if they are semantically close?

This note treats forgetting as a retrieval behavior, not as model erasure. The goal is to explore how a retriever can avoid selected evidence at inference time while still preserving adjacent, useful knowledge.

**Good forgetting is the ability to suppress forbidden evidence while preserving utility on adjacent, non-forbidden knowledge.**

This framing is useful for RAG systems because many undesirable uses of information occur at the context-selection layer. If a forbidden document is not placed in the prompt context, it has less opportunity to influence the answer. This is still a limited control: it does not prove that a model lacks internal knowledge, and it does not guarantee safe behavior under all prompts.

## 2. Problem Framing

Assume a corpus of candidate documents, a user query, a positive intent, and one or more forget-intent descriptions. The retriever must select a top-k context that is useful for the query while avoiding documents that are close to the forget set.

The project uses the following terms:

- **Retrieval-time forgetting**: changing retrieval behavior at inference time to avoid selected evidence.
- **Negative-aware retrieval**: scoring candidate documents with an explicit negative or forget-aware term.
- **Selective non-use of evidence**: excluding or demoting evidence that should not be used.
- **Behavioral forgetting in RAG**: producing a retrieval behavior that avoids forbidden evidence without claiming parameter-level erasure.

The task is not model unlearning. It is a small technical note and toy evaluation for an inference-time control mechanism.

## 3. Method: GoodForget-RAG

GoodForget-RAG scores each candidate document with:

```text
S(d) = alpha * sim(q, d) + beta * sim(p, d) - gamma * max_i sim(f_i, d)
```

Where:

- `q` is the original user query.
- `p` is the positive intent, or the useful information intent.
- `f_i` is a forget-intent text.
- `d` is a candidate document.
- `sim` is cosine similarity.
- `alpha`, `beta`, and `gamma` control the strength of query relevance, positive relevance, and forget penalty.

The forget term is not created by subtracting vectors or by constructing an arithmetic negative embedding. Each forget item is encoded as a normal text description, and the maximum similarity to any forget item becomes the penalty. This makes the method easy to inspect and keeps the toy implementation simple.

The implementation in this repository uses scikit-learn TF-IDF vectors with cosine similarity. This choice is intentionally modest: it avoids API keys and paid services, runs on a laptop, and makes the experiment deterministic. The same scoring interface could be used with other local encoders, but this note does not evaluate those variants.

## 4. Toy Evaluation

The toy corpus contains synthetic RAG examples where useful and forbidden documents are close in topic:

- current EV tax-credit guidance versus an obsolete flat-credit brochure;
- modern password reset controls versus a deprecated phone-support memo;
- general medication safety guidance versus a recalled dosage card;
- current privacy and retention guidance versus an archived privacy statement.

Each query includes:

- an original query;
- a positive intent;
- a forget set;
- expected relevant document IDs;
- forbidden document IDs.

The experiment compares three methods:

1. **Vanilla RAG** retrieves by similarity to the original query only.
2. **Positive-only RAG** retrieves by similarity to the positive intent only.
3. **GoodForget-RAG** retrieves with positive relevance and a negative-aware forget penalty.

The top-k value is 2. The GoodForget-RAG toy configuration uses `alpha = 0.4`, `beta = 0.8`, and `gamma = 1.2`.

The evaluation reports:

- `leakage_rate`: fraction of queries where a forbidden document appears in the selected top-k context;
- `utility_recall`: mean fraction of expected useful documents retrieved;
- `empty_context_rate`: fraction of queries where no context remains.

## 5. Results

The deterministic run produces:

```text
           method  leakage_rate  utility_recall  empty_context_rate  num_queries
      Vanilla RAG         0.750           0.625               0.000            4
Positive-only RAG         0.500           0.750               0.000            4
   GoodForget-RAG         0.000           1.000               0.000            4
```

The result shows the intended behavior on this small dataset. Vanilla RAG retrieves forbidden documents for most queries because the forbidden documents are on-topic. Positive-only RAG improves utility but still retrieves forbidden documents when the forbidden evidence is close to the positive intent. GoodForget-RAG demotes documents close to the forget set and retrieves the expected useful documents in this toy configuration.

These results should be interpreted narrowly. They show that a negative-aware scoring term can change retrieval behavior in a controlled synthetic setting. They do not show broad robustness, model unlearning, or comprehensive protection against all uses of forbidden knowledge.

## 6. Limitations

This note has several important limitations.

First, the dataset is synthetic and small. It is designed to make the retrieval behavior easy to inspect, not to represent the complexity of real enterprise or safety-critical corpora.

Second, TF-IDF captures lexical overlap rather than deeper semantic meaning. This is useful for reproducibility but limits the conclusions that can be drawn about embedding-based production systems.

Third, the forget set is manually specified. In real deployments, defining what should not be used is a policy and data-governance problem as much as a retrieval problem.

Fourth, suppressing retrieved evidence is not equivalent to deleting knowledge from a language model. A model may still produce content from its parameters, from the user prompt, or from other retrieved documents. This technique should therefore be treated as one possible layer in a larger system.

Finally, the chosen weights are illustrative. Different corpora and policies would require calibration, validation, and monitoring.

## 7. Conclusion

GoodForget-RAG is a small demonstration of negative-aware retrieval for selective non-use of evidence in RAG systems. It reframes forgetting as a retrieval-time behavior: preserve useful adjacent knowledge while demoting evidence that matches an explicit forget intent. The approach is simple, local, and reproducible, making it suitable as a technical note and starting point for further experiments.

The main contribution is conceptual and practical: RAG design should ask not only what evidence to retrieve, but also what evidence should be avoided.
