# LinkedIn Post Draft (KR)

LLM의 "망각"을 모델 내부 지식 삭제가 아니라, **검색 가능한 지식의 선택적 비사용**으로 정의해보았습니다.

RAG에서 중요한 질문은 "무엇을 가져올 것인가?"뿐 아니라 "무엇을 사용하지 않을 것인가?"이기도 합니다.

이번에 만든 **GoodForget-RAG**는 candidate document를 점수화할 때 원래 query와 positive intent는 반영하되, forget set과 가까운 후보 문서를 감점하는 **negative-aware retrieval** 방식입니다.

핵심 점수식은 다음과 같습니다.

```text
S(d) = alpha * sim(q, d) + beta * sim(p, d) - gamma * max_i sim(f_i, d)
```

여기서 forget-intent vector는 산술적인 negative vector `-v`가 아니라, "사용하지 말아야 할 근거"를 별도로 표현한 벡터입니다.

현재 버전은 TF-IDF 기반 toy evaluation입니다. 따라서 model unlearning이 아니고, safety guarantee도 아니며, semantic forgetting을 증명하지도 않습니다.

대신 이 프로젝트는 RAG의 context selection 단계에서 forbidden evidence를 어떻게 덜 가져오게 만들 수 있는지, 그리고 그 과정에서 utility를 얼마나 보존할 수 있는지를 작은 실험으로 보여줍니다.

특히 red-team 관점에서 몇 가지 한계를 명시했습니다.

- synthetic dataset이 방법에 유리할 수 있음
- positive intent와 forget intent를 oracle처럼 가정함
- TF-IDF는 lexical proxy일 뿐 dense semantic retrieval 검증이 아님
- answer leakage는 실제 LLM generation이 아니라 deterministic proxy임
- reliable metadata label이 있으면 metadata filtering이 더 강할 수 있음

핵심 메시지는 단순합니다.

RAG 시스템에서 좋은 검색은 관련 문서를 잘 찾는 것만이 아니라, 사용하지 말아야 할 근거를 잘 피하는 것도 포함해야 합니다.
