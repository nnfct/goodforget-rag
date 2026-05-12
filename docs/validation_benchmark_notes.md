# Validation Benchmark Notes

This suite expands GoodForget-RAG into a synthetic validation framework for retrieval-time forgetting controls. It does not claim model unlearning, guaranteed forgetting, SOTA, or semantic forgetting proof.

Core definition:

“Good forgetting is the ability to suppress forbidden evidence while preserving utility on adjacent, non-forbidden knowledge.”

## What Was Added

- Benchmark generator with configurable domain/scenario/query sizes.
- Eight synthetic domains in default mode and twelve in large mode.
- Attack variants covering direct negative, paraphrased negative, codeword, contrastive, indirect, stale-policy confusion, noisy forget sets, and incomplete forget sets.
- Multiple local deterministic vector representations: word TF-IDF, character TF-IDF, local LSA, and hybrid word+character TF-IDF.
- Metadata label-noise experiments.
- Negative vector baseline.
- Bootstrap confidence intervals.
- Failure-case mining.
- HTML validation report.

## Conservative Reading

GoodForget-RAG is useful as a retrieval-time control idea when labels are incomplete, dynamic, or too coarse. It is not necessarily better than metadata filtering when labels are reliable, and it can lose utility when the forget penalty is too strong.

The negative vector baseline is included to test the flawed idea that the arithmetic opposite of a forbidden vector is equivalent to a forget intent. In this synthetic benchmark it can tie some GoodForget-RAG settings, but it is representation-sensitive and is not a substitute for explicit forget-intent suppression.
