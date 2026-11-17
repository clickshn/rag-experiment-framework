"""시험용 외부 플러그인: 검색기 하나와 생성기 하나를 제공한다."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

from ragexp.registry import register
from ragexp.types import Answer, Chunk, GenerationInput, Hit, RetrievalQuery, Usage


@register("retriever", "fake_bm25")
class FakeBm25Retriever:
    version: ClassVar[str] = "1"

    def __init__(self, k: int = 10) -> None:
        self.k = k

    def retrieve(self, query: RetrievalQuery) -> list[Hit]:
        return []

    def fingerprint(self) -> dict[str, Any]:
        return {"k": self.k}


@register("generator", "fake_echo")
class FakeEchoGenerator:
    version: ClassVar[str] = "1"

    def generate(self, inp: GenerationInput, context: Sequence[Chunk]) -> Answer:
        return Answer(text=inp.question, finish_reason="stop", usage=Usage(latency_ms=0))

    def fingerprint(self) -> dict[str, Any]:
        return {}


@register("retriever", "marked_other")
class MismatchedRetriever(FakeBm25Retriever):
    """entry point 이름과 표시가 다른 경우를 시험하려고 둔 클래스."""
