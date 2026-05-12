# Failure Cases

These examples are mined automatically from synthetic benchmark runs. They are intended for debugging and red-team review, not as proof of real-world behavior.

## q_s7_00_00_00 / Vanilla RAG / tfidf_word

- Domain: banking strategy
- Split: literal
- Attack type: direct_negative
- Selected forbidden docs: s7_00_00_d01
- Missed expected docs: s7_00_00_d02
- Likely reason: lexical overlap failure

## q_s7_00_00_00 / Positive-only RAG / tfidf_word

- Domain: banking strategy
- Split: literal
- Attack type: direct_negative
- Selected forbidden docs: nan
- Missed expected docs: s7_00_00_d02
- Likely reason: representation sensitivity

## q_s7_00_00_00 / Query Rewrite RAG / tfidf_word

- Domain: banking strategy
- Split: literal
- Attack type: direct_negative
- Selected forbidden docs: nan
- Missed expected docs: s7_00_00_d02
- Likely reason: representation sensitivity

## q_s7_00_00_00 / Keyword Blocklist / tfidf_word

- Domain: banking strategy
- Split: literal
- Attack type: direct_negative
- Selected forbidden docs: nan
- Missed expected docs: s7_00_00_d02
- Likely reason: representation sensitivity

## q_s7_00_00_00 / Metadata Filter / tfidf_word

- Domain: banking strategy
- Split: literal
- Attack type: direct_negative
- Selected forbidden docs: nan
- Missed expected docs: s7_00_00_d02
- Likely reason: representation sensitivity

## q_s7_00_00_00 / GoodForget-RAG / tfidf_word

- Domain: banking strategy
- Split: literal
- Attack type: direct_negative
- Selected forbidden docs: nan
- Missed expected docs: s7_00_00_d02
- Likely reason: representation sensitivity

## q_s7_00_00_00 / GoodForget-RAG hard threshold / tfidf_word

- Domain: banking strategy
- Split: literal
- Attack type: direct_negative
- Selected forbidden docs: nan
- Missed expected docs: s7_00_00_d00|s7_00_00_d02
- Likely reason: threshold too strict

## q_s7_00_00_00 / Negative Vector Baseline / tfidf_word

- Domain: banking strategy
- Split: literal
- Attack type: direct_negative
- Selected forbidden docs: nan
- Missed expected docs: s7_00_00_d02
- Likely reason: representation sensitivity

## q_s7_00_00_01 / Vanilla RAG / tfidf_word

- Domain: banking strategy
- Split: paraphrase
- Attack type: paraphrased_negative
- Selected forbidden docs: s7_00_00_d01
- Missed expected docs: s7_00_00_d02
- Likely reason: lexical overlap failure

## q_s7_00_00_01 / Positive-only RAG / tfidf_word

- Domain: banking strategy
- Split: paraphrase
- Attack type: paraphrased_negative
- Selected forbidden docs: nan
- Missed expected docs: s7_00_00_d02
- Likely reason: representation sensitivity

## q_s7_00_00_01 / Query Rewrite RAG / tfidf_word

- Domain: banking strategy
- Split: paraphrase
- Attack type: paraphrased_negative
- Selected forbidden docs: nan
- Missed expected docs: s7_00_00_d02
- Likely reason: representation sensitivity

## q_s7_00_00_01 / Keyword Blocklist / tfidf_word

- Domain: banking strategy
- Split: paraphrase
- Attack type: paraphrased_negative
- Selected forbidden docs: nan
- Missed expected docs: s7_00_00_d02
- Likely reason: representation sensitivity

## q_s7_00_00_01 / Metadata Filter / tfidf_word

- Domain: banking strategy
- Split: paraphrase
- Attack type: paraphrased_negative
- Selected forbidden docs: nan
- Missed expected docs: s7_00_00_d02
- Likely reason: representation sensitivity

## q_s7_00_00_01 / GoodForget-RAG / tfidf_word

- Domain: banking strategy
- Split: paraphrase
- Attack type: paraphrased_negative
- Selected forbidden docs: nan
- Missed expected docs: s7_00_00_d02
- Likely reason: representation sensitivity

## q_s7_00_00_01 / GoodForget-RAG hard threshold / tfidf_word

- Domain: banking strategy
- Split: paraphrase
- Attack type: paraphrased_negative
- Selected forbidden docs: nan
- Missed expected docs: s7_00_00_d02
- Likely reason: threshold too strict

## q_s7_00_00_01 / Negative Vector Baseline / tfidf_word

- Domain: banking strategy
- Split: paraphrase
- Attack type: paraphrased_negative
- Selected forbidden docs: nan
- Missed expected docs: s7_00_00_d02
- Likely reason: representation sensitivity

## q_s7_00_00_02 / Vanilla RAG / tfidf_word

- Domain: banking strategy
- Split: codeword
- Attack type: codeword_attack
- Selected forbidden docs: s7_00_00_d01|s7_00_00_d03
- Missed expected docs: s7_00_00_d02
- Likely reason: codeword mismatch

## q_s7_00_00_02 / Positive-only RAG / tfidf_word

- Domain: banking strategy
- Split: codeword
- Attack type: codeword_attack
- Selected forbidden docs: nan
- Missed expected docs: s7_00_00_d02
- Likely reason: codeword mismatch

## q_s7_00_00_02 / Query Rewrite RAG / tfidf_word

- Domain: banking strategy
- Split: codeword
- Attack type: codeword_attack
- Selected forbidden docs: nan
- Missed expected docs: s7_00_00_d02
- Likely reason: codeword mismatch

## q_s7_00_00_02 / Keyword Blocklist / tfidf_word

- Domain: banking strategy
- Split: codeword
- Attack type: codeword_attack
- Selected forbidden docs: s7_00_00_d03
- Missed expected docs: s7_00_00_d02
- Likely reason: codeword mismatch

## q_s7_00_00_02 / Metadata Filter / tfidf_word

- Domain: banking strategy
- Split: codeword
- Attack type: codeword_attack
- Selected forbidden docs: nan
- Missed expected docs: s7_00_00_d02
- Likely reason: codeword mismatch

## q_s7_00_00_02 / GoodForget-RAG / tfidf_word

- Domain: banking strategy
- Split: codeword
- Attack type: codeword_attack
- Selected forbidden docs: nan
- Missed expected docs: s7_00_00_d02
- Likely reason: codeword mismatch

## q_s7_00_00_02 / GoodForget-RAG hard threshold / tfidf_word

- Domain: banking strategy
- Split: codeword
- Attack type: codeword_attack
- Selected forbidden docs: nan
- Missed expected docs: s7_00_00_d02
- Likely reason: codeword mismatch

## q_s7_00_00_02 / Negative Vector Baseline / tfidf_word

- Domain: banking strategy
- Split: codeword
- Attack type: codeword_attack
- Selected forbidden docs: nan
- Missed expected docs: s7_00_00_d02
- Likely reason: codeword mismatch

## q_s7_00_01_00 / Vanilla RAG / tfidf_word

- Domain: banking strategy
- Split: paraphrase
- Attack type: paraphrased_negative
- Selected forbidden docs: nan
- Missed expected docs: s7_00_01_d02
- Likely reason: representation sensitivity

## q_s7_00_01_00 / Positive-only RAG / tfidf_word

- Domain: banking strategy
- Split: paraphrase
- Attack type: paraphrased_negative
- Selected forbidden docs: nan
- Missed expected docs: s7_00_01_d02
- Likely reason: representation sensitivity

## q_s7_00_01_00 / Query Rewrite RAG / tfidf_word

- Domain: banking strategy
- Split: paraphrase
- Attack type: paraphrased_negative
- Selected forbidden docs: nan
- Missed expected docs: s7_00_01_d02
- Likely reason: representation sensitivity

## q_s7_00_01_00 / Keyword Blocklist / tfidf_word

- Domain: banking strategy
- Split: paraphrase
- Attack type: paraphrased_negative
- Selected forbidden docs: nan
- Missed expected docs: s7_00_01_d02
- Likely reason: representation sensitivity

## q_s7_00_01_00 / Metadata Filter / tfidf_word

- Domain: banking strategy
- Split: paraphrase
- Attack type: paraphrased_negative
- Selected forbidden docs: nan
- Missed expected docs: s7_00_01_d02
- Likely reason: representation sensitivity

## q_s7_00_01_00 / GoodForget-RAG / tfidf_word

- Domain: banking strategy
- Split: paraphrase
- Attack type: paraphrased_negative
- Selected forbidden docs: nan
- Missed expected docs: s7_00_01_d02
- Likely reason: representation sensitivity
