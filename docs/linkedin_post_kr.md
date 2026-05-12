# LinkedIn Post Draft (KR)

LLM의 "망각"을 꼭 모델 내부 지식 삭제로만 볼 필요는 없다고 생각했습니다.

이번에 만든 **GoodForget-RAG**는 RAG에서 망각을 "검색 가능한 지식의 선택적 비사용"으로 정의한 작은 기술 노트입니다.

RAG에서 중요한 질문은 "무엇을 가져올 것인가?"뿐 아니라 "무엇을 사용하지 않을 것인가?"이기도 합니다.

GoodForget-RAG는 candidate document를 점수화할 때 원래 query와 positive intent는 반영하되, forget set과 의미적으로 가까운 문서는 감점하는 **negative-aware retrieval** 방식입니다.

핵심 점수식은 다음과 같습니다.

```text
S(d) = alpha * sim(q, d) + beta * sim(p, d) - gamma * max_i sim(f_i, d)
```

여기서 `f_i`는 산술적인 negative vector가 아니라, "사용하지 말아야 할 근거"를 설명하는 forget-intent text입니다.

중요한 점은 이 방식이 **model unlearning이 아니라 retrieval-time control mechanism**이라는 것입니다. 모델 파라미터에서 지식을 삭제했다고 주장하지 않습니다. 대신 RAG의 context selection 단계에서 forbidden evidence가 들어오지 않도록 제어하는 toy implementation입니다.

작은 synthetic toy evaluation에서는 Vanilla RAG와 Positive-only RAG가 일부 forbidden evidence를 top-k에 포함하는 반면, GoodForget-RAG는 forget set과 가까운 문서를 감점해 leakage를 줄이는 모습을 보였습니다.

이 프로젝트의 메시지는 단순합니다.

RAG 시스템에서 "좋은 검색"은 관련 문서를 잘 찾는 것뿐 아니라, 사용하지 말아야 할 근거를 잘 피하는 것이기도 합니다.
