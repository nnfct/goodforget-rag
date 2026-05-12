# GoodForget-RAG

GoodForget-RAG is a lightweight technical-note project for **negative-aware retrieval** in retrieval-augmented generation (RAG). It demonstrates a small, deterministic, local toy implementation of **retrieval-time forgetting**: retrieve useful evidence while penalizing candidate documents that are semantically close to an explicit forget set.

This is a toy technical note, not model unlearning, and it does not remove knowledge from LLM parameters. It is not a safety guarantee. The project studies an inference-time / retrieval-time control mechanism for **selective non-use of evidence**.

## Method

The retrieval score combines the original query, a positive information intent, and an explicit forget-intent penalty:

```text
S(d) = alpha * sim(q, d) + beta * sim(p, d) - gamma * max_i sim(f_i, d)
```

Where:

- `q` is the original user query.
- `p` is the positive intent, meaning the useful information the system should preserve.
- `f_i` is a forget-intent text, not an arithmetic negative vector.
- `d` is a candidate document.
- `sim` is cosine similarity over local TF-IDF vectors.
- `alpha`, `beta`, and `gamma` are configurable weights.

The implementation compares three retrieval modes:

- **Vanilla RAG**: retrieves by original query similarity only.
- **Positive-only RAG**: retrieves by positive intent similarity only.
- **GoodForget-RAG**: retrieves with positive relevance and a negative-aware forget penalty.

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
│   └── eval.py
├── experiments/
│   ├── run_experiment.py
│   ├── toy_corpus.jsonl
│   └── toy_queries.jsonl
├── results/
│   ├── summary.csv
│   └── toy_results.csv
├── paper/
│   └── technical_note.md
├── docs/
│   ├── linkedin_post_kr.md
│   ├── linkedin_post_en.md
│   └── github_upload_steps_kr.md
└── notebooks/
    └── goodforget_demo.ipynb
```

## Quickstart

```bash
git clone https://github.com/nnfct/goodforget-rag.git
cd goodforget-rag
python -m pip install -e .
python experiments/run_experiment.py
```

If your Linux distribution blocks system-wide `pip` installs, create a virtual environment first or use your preferred Python environment manager.

## Expected Result Table

Running the toy experiment writes `results/summary.csv` and `results/toy_results.csv`, and prints a table similar to:

```text
           method  leakage_rate  utility_recall  empty_context_rate  num_queries
      Vanilla RAG         0.750           0.625               0.000            4
Positive-only RAG         0.500           0.750               0.000            4
   GoodForget-RAG         0.000           1.000               0.000            4
```

## Interpretation

In this synthetic dataset, several forbidden documents are semantically close to useful evidence: obsolete tax-credit rules, deprecated security procedures, recalled medication notes, and archived privacy claims. Vanilla retrieval often selects those documents because they match the query. Positive-only retrieval improves utility in some cases but can still retrieve forbidden evidence when the forbidden document is close to the useful intent.

GoodForget-RAG uses a negative-aware retrieval penalty to reduce selection of documents close to the forget set. In this toy run, it reduces `leakage_rate` while preserving the expected useful documents. This should be read as a controlled demonstration of **behavioral forgetting in RAG**, not as a claim about model parameter changes.

## Limitations

- The corpus and queries are synthetic and small.
- TF-IDF is deterministic and transparent, but it is weaker than modern embedding models.
- The forget set must be written explicitly and carefully.
- The method controls retrieved context; it does not erase knowledge from a language model.
- The result should not be treated as production validation or broad safety assurance.
- Real systems need broader evaluation, adversarial testing, and policy/process design around what should not be used.

## Suggested Next Steps

- Evaluate on larger corpora with realistic near-duplicate and policy-conflict cases.
- Compare TF-IDF with local embedding models while keeping the same scoring interface.
- Add thresholding and abstention behavior when all candidates are too close to the forget set.
- Study how forget-intent wording changes retrieval behavior.
- Add tests for ranking stability and metric calculations.

## LinkedIn-Friendly Positioning

GoodForget-RAG frames forgetting in RAG as **selective non-use of evidence**: not deleting model parameters, but controlling what evidence enters the context window. The practical question is not only "what should we retrieve?" but also "what should we avoid using?"
