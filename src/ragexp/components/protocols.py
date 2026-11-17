"""부품 인터페이스(Protocol) 9개.

모든 부품은 다음을 갖는다.

- 클래스 속성 `version: ClassVar[str]`(필수). 구현 동작이 바뀌면 올린다. 구성·채점 버전에 들어간다.
- 메서드 `fingerprint() -> Fingerprint`. 출력을 바꾸는 설정만 담는다(`ragexp.fingerprint`).

선택 클래스 속성(Protocol 멤버는 아니고 레지스트리가 읽는다):

- `fingerprint_allow: ClassVar[frozenset[str]]`: fingerprint 금지 이름 규칙의 예외(정상 파라미터)
- `config_model: ClassVar[type | None]`: 설정 스키마(U2에서 쓰임)
- `needs_gold: ClassVar[bool]`(Retriever만): 정답 근거가 필요한 검색기(oracle). 기본 False
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import ClassVar, Literal, Protocol, runtime_checkable

from ragexp.fingerprint import Fingerprint
from ragexp.types import (
    Answer,
    Chunk,
    Document,
    GenerationInput,
    Hit,
    JudgeInput,
    JudgeQuestion,
    Question,
    RetrievalQuery,
    Verdict,
)


@runtime_checkable
class Component(Protocol):
    """모든 부품의 공통 약속."""

    version: ClassVar[str]

    def fingerprint(self) -> Fingerprint: ...


@runtime_checkable
class Corpus(Component, Protocol):
    """원문 문서를 읽는다."""

    def documents(self) -> Iterator[Document]: ...


@runtime_checkable
class Chunker(Component, Protocol):
    """문서를 덩어리로 나눈다."""

    def chunk(self, doc: Document) -> list[Chunk]: ...


@runtime_checkable
class Embedder(Component, Protocol):
    """덩어리·질문을 벡터로 바꾼다. 질문과 문서에 다른 접두어를 쓰는 모델을 위해 kind를 받는다."""

    def embed(
        self, texts: Sequence[str], *, kind: Literal["query", "document"]
    ) -> list[list[float]]: ...

    def dimension(self) -> int: ...


@runtime_checkable
class VectorStore(Component, Protocol):
    """벡터를 저장하고 찾는다."""

    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None: ...

    def search(self, vector: Sequence[float], k: int) -> list[Hit]: ...


@runtime_checkable
class Retriever(Component, Protocol):
    """검색 전략. 몇 개를 돌려줄지(k)는 설정값이다."""

    def retrieve(self, query: RetrievalQuery) -> list[Hit]: ...


@runtime_checkable
class Reranker(Component, Protocol):
    """후보를 다시 정렬한다."""

    def rerank(self, query: str, hits: Sequence[Hit]) -> list[Hit]: ...


@runtime_checkable
class Generator(Component, Protocol):
    """근거로 답을 만든다. 입력에는 정답 정보가 없다."""

    def generate(self, inp: GenerationInput, context: Sequence[Chunk]) -> Answer: ...


@runtime_checkable
class Judge(Component, Protocol):
    """답을 채점한다. 블라인드 보기만 받고, 정오는 내지 않는다(코드가 계산)."""

    def judge(self, question: JudgeQuestion, response: JudgeInput) -> Verdict: ...


@runtime_checkable
class QuestionGenerator(Component, Protocol):
    """문항 초안을 만든다(R0는 Protocol만). 초안은 split=None이다. R3에서 다시 정한다."""

    def draft(self, chunks: Sequence[Chunk], qtype: str, n: int) -> list[Question]: ...
