# ADR 0001: 도메인 타입에 pydantic을 쓴다

- 상태: 채택 (2026-11-17, U1)
- 결정한 사람: 석한나

## 맥락

U1에서 부품끼리 주고받는 도메인 타입(Document, Chunk, Hit, Question, Usage, Answer, Verdict와 보조 타입)을 정해야 했다. 같은 타입이 세 곳에서 쓰인다.

- 부품 사이의 값
- 실행 기록 JSON(U6)
- 바깥 입력 검증: 문항 파일(U5)과 구성 파일(U2)

U1 메모 초안은 표준 라이브러리 frozen dataclass를 제안했다. 이 경우 문항 파일 검증은 U5에서 pydantic 모델로 따로 정의해야 한다.

## 결정

도메인 타입을 **pydantic `BaseModel(frozen=True)`**로 정의하고, U1에서 런타임 의존성 `pydantic`(2.x)을 추가한다.

- 같은 타입을 두 번 정의하지 않는다. 문항 파일 검증(U5)과 도메인 타입이 한 정의를 쓴다.
- 공통 바탕 `Frozen`의 설정:
  - `extra="forbid"`, `allow_inf_nan=False`, `validate_default=True`
  - 별칭으로 읽고 쓰기(`type`↔`qtype`)
- 깊은 불변: Mapping 칸은 검증 뒤 `MappingProxyType`, list는 tuple로 바꾼다. 해시는 JSON 표현으로 계산한다.
- 설계 §9.1이 이미 설정 검증에 pydantic을 쓰기로 정해 두었으므로, 새 도구가 생기는 것은 아니다.

## 결과

- 좋은 점:
  - 검증과 JSON 직렬화를 정의 하나로 처리한다.
  - U2의 설정 스키마, R1의 설정 폼 자동 생성(§8.2)과 도구가 같다.
- 나쁜 점:
  - 객체를 만들 때마다 검증 비용이 든다(실행 기록 한 줄마다).
  - 해시를 JSON 직렬화로 계산하므로 dataclass보다 느리다. R0 규모(문항 수십 개 × arm × 반복)에서는 문제가 되지 않는다고 본다. 느려지면 U6에서 다시 본다.
- 버전: pydantic 2.14.3(`uv.lock`). mypy에는 `pydantic.mypy` 플러그인을 켠다.
