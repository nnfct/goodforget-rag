# LinkedIn Post Draft: Validation Benchmark (KR)

GoodForget-RAG를 단순 toy demo에서 synthetic validation benchmark suite로 확장했습니다.

핵심은 여전히 같습니다.

RAG에서 중요한 질문은 "무엇을 가져올 것인가?"뿐 아니라 "무엇을 사용하지 않을 것인가?"입니다.

이번 버전에서는 GoodForget-RAG가 항상 이기도록 만드는 것이 아니라, 오히려 어디서 도움이 되고, 어디서 baseline과 비슷하고, 어디서 실패하는지를 드러내는 데 초점을 뒀습니다.

추가한 것:

- 여러 synthetic domain과 scenario
- direct/paraphrase/codeword/contrastive/stale/noisy/incomplete forget 공격 변형
- TF-IDF word, TF-IDF char, local LSA, hybrid lexical representation 비교
- metadata filter label noise 실험
- negative vector baseline
- bootstrap confidence interval
- failure case mining
- HTML validation report

중요한 제한도 그대로 명시했습니다.

이건 model unlearning이 아닙니다. safety guarantee도 아닙니다. semantic forgetting을 증명하지도 않습니다. reliable label이 있으면 metadata filtering이 더 낫습니다.

GoodForget-RAG는 label이 불완전하거나, 동적으로 변하거나, 너무 coarse할 때 고려할 수 있는 retrieval-time control 아이디어로 보는 것이 맞습니다.
