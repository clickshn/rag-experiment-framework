"""도메인 타입: 부품끼리 주고받는 값.

모두 pydantic `BaseModel(frozen=True)`이다. 같은 정의로 바깥 입력(문항 파일 등)을 검증하고
실행 기록(JSON)으로 왕복한다. 원칙 몇 가지는 타입 자체로 강제한다.

- 정답 정보(gold·must·fail)는 생성기 입력(`GenerationInput`)에 칸이 없다.
- 판정 입력(`JudgeQuestion`, `JudgeInput`)에는 모델·arm·provenance 칸이 없다(블라인드).
- 판정 출력(`Verdict`)에는 정오 칸이 없다. 정오는 코드가 계산한다.
- 오류 메시지 칸에는 주소(`://`)를 넣을 수 없다.

Mapping 칸은 생성 시 재귀적으로 읽기 전용(`MappingProxyType`, list→tuple)으로 바뀌어,
넘긴 dict를 나중에 고쳐도 객체 값이 바뀌지 않는다.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Annotated, Any, Literal, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    JsonValue,
    PlainSerializer,
    model_validator,
)

from ragexp.fingerprint import has_address

# --- 읽기 전용 JSON 값 ---------------------------------------------------------


def _thaw(value: Any) -> Any:
    """읽기 전용 값을 일반 dict·list로 되돌린다(검증 입력·직렬화용)."""
    if isinstance(value, Mapping):
        return {k: _thaw(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(v) for v in value]
    return value


def _freeze(value: Any) -> Any:
    """dict·list를 재귀적으로 읽기 전용으로 바꾼다. NaN·inf는 거부한다."""
    if isinstance(value, Mapping):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("NaN·inf 값은 쓸 수 없습니다")
    return value


FrozenJson = Annotated[
    Mapping[str, JsonValue],
    BeforeValidator(_thaw),
    AfterValidator(_freeze),
    PlainSerializer(_thaw),
]
"""읽기 전용 JSON 객체 칸."""

# 판정 지표 이름으로 정오를 들여오지 못하게 막는다(정오는 코드가 계산, 설계 §7.2).
FORBIDDEN_METRIC_NAMES = frozenset({"correct", "is_correct", "pass", "passed", "accuracy"})


def _check_metric_names(value: Mapping[str, Any]) -> Mapping[str, Any]:
    bad = sorted(k for k in value if k.lower() in FORBIDDEN_METRIC_NAMES)
    if bad:
        raise ValueError(
            f"판정 지표에 정오를 넣을 수 없습니다({', '.join(bad)}). 정오는 코드가 계산합니다"
        )
    return value


FrozenMetrics = Annotated[
    Mapping[str, float | None],
    BeforeValidator(_thaw),
    AfterValidator(_check_metric_names),
    AfterValidator(_freeze),
    PlainSerializer(_thaw),
]
"""지표 이름 → 값. None은 '정의되지 않음'(예: gold가 빈 문항의 인용 재현율)."""


def _no_url(value: str) -> str:
    if has_address(value):
        raise ValueError("주소를 넣을 수 없습니다. 주소를 지운 짧은 문장으로 쓰세요")
    return value


def _hash_repr(value: Any) -> Any:
    """해시용 표현. 같다고 비교되는 값(키 순서만 다른 Mapping, 1과 1.0)이 같은 표현이 되게 한다."""
    if isinstance(value, Mapping):
        return tuple(sorted((str(k), _hash_repr(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_hash_repr(v) for v in value)
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    return value


NonEmptyStr = Annotated[str, Field(min_length=1)]
NoUrlStr = Annotated[str, AfterValidator(_no_url)]

FinishReason = Literal["stop", "length", "error", "other"]
ErrorKind = Literal[
    "connect", "timeout", "http_status", "client_error", "config_error", "parse", "other"
]
"""모델 호출 실패 종류. 서버 사정·모델·문항 사정·설정 오류 분류는 `client.failure_class`."""
VerdictStatus = Literal["ok", "truncated", "error", "not_applicable"]
Split = Literal["dev", "val", "test"]


class Frozen(BaseModel):
    """모든 도메인 타입의 바탕: 불변, 모르는 칸 거부, NaN·inf 거부, 해시 가능, JSON 왕복."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        allow_inf_nan=False,
        validate_default=True,
        validate_by_name=True,
        validate_by_alias=True,
        serialize_by_alias=True,
    )

    def __hash__(self) -> int:
        # Mapping 칸(MappingProxyType)은 해시할 수 없으므로 정규화한 표현으로 해시한다.
        return hash((type(self).__qualname__, _hash_repr(self.model_dump())))

    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> Self:
        """복사본. pydantic 기본 동작과 달리 update 값도 다시 검증한다(주소 금지 등 우회 방지).

        `model_construct`는 검증을 건너뛰므로 쓰지 않는다.
        """
        if not update:
            return super().model_copy(deep=deep)
        return self.model_validate({**self.model_dump(by_alias=False), **update})

    def to_dict(self) -> dict[str, Any]:
        """JSON으로 쓸 수 있는 dict(별칭 키 사용, tuple은 list)."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        return cls.model_validate(data)


# --- 핵심 타입 7개 -------------------------------------------------------------


class Section(Frozen):
    """(보조) 구조가 있는 문서의 나눔 단위. 법령이면 조문 하나(`section_id="제27조의2"`)."""

    section_id: NonEmptyStr
    heading: str = ""
    text: str
    meta: FrozenJson = {}


class Document(Frozen):
    """말뭉치의 원문 한 건(법령이면 법령 하나). `meta`에는 출처 정보만 둔다."""

    doc_id: NonEmptyStr
    title: str = ""
    text: str = ""
    sections: tuple[Section, ...] = ()
    meta: FrozenJson = {}


class Chunk(Frozen):
    """검색·근거의 단위. `chunk_id`는 문항 `gold`와 같은 체계다."""

    chunk_id: NonEmptyStr
    doc_id: NonEmptyStr
    text: str
    heading: str = ""
    meta: FrozenJson = {}


class Hit(Frozen):
    """검색·리랭크 결과 한 줄. 정답 여부 칸은 일부러 없다(채점 층에서 gold와 대조)."""

    chunk: Chunk
    score: float
    rank: int = Field(ge=1)
    source: NonEmptyStr


class Usage(Frozen):
    """모델 호출 한 번의 비용 재료. 서버가 토큰 수를 주지 않으면 None(0과 구분).

    `latency_ms`는 마지막 시도 한 번의 시간, `total_ms`는 재시도·대기를 포함한 전체 시간이다
    (U3). `total_ms`는 예전 기록과 맞추려고 선택 칸이다.
    """

    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    latency_ms: float = Field(ge=0)
    total_ms: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _total_covers_latency(self) -> Self:
        if self.total_ms is not None and self.total_ms < self.latency_ms:
            raise ValueError(
                "total_ms는 latency_ms보다 작을 수 없습니다(재시도·대기를 포함한 시간)"
            )
        return self


class Answer(Frozen):
    """생성기 출력. `served_model`은 응답에 실제로 찍힌 모델 이름이다.

    `attempts`는 호출 안 시도 횟수, `warnings`는 클라이언트 경고(모델 이름 불일치 등)다.
    """

    text: str
    finish_reason: FinishReason
    usage: Usage
    served_model: NoUrlStr | None = None
    error_kind: ErrorKind | None = None
    error: NoUrlStr = ""
    attempts: int = Field(default=1, ge=1)
    warnings: tuple[NoUrlStr, ...] = ()

    @model_validator(mode="after")
    def _error_consistent(self) -> Self:
        if (self.error_kind is not None) != (self.finish_reason == "error"):
            raise ValueError("error_kind는 finish_reason이 'error'일 때만, 그리고 반드시 채운다")
        return self


class Question(Frozen):
    """문항(설계 §6.3). JSON 키 `type`은 `qtype`으로 읽고 다시 `type`으로 쓴다.

    `split=None`은 아직 나누지 않은 초안이다. 답이 없는 문항은 `gold=()`이다.
    """

    qid: NonEmptyStr
    qtype: NonEmptyStr = Field(alias="type")
    split: Split | None = None
    question: NonEmptyStr
    gold: tuple[NonEmptyStr, ...] = ()
    must: tuple[str, ...] = ()
    must_min: int = Field(ge=0)
    fail: tuple[str, ...] = ()
    optional: tuple[str, ...] = ()
    scoring_note: str = ""
    expected_tool_calls: tuple[FrozenJson, ...] = ()
    answer_schema: FrozenJson | None = None
    provenance: FrozenJson = {}

    @model_validator(mode="after")
    def _must_min_in_range(self) -> Self:
        if self.must_min > len(self.must):
            raise ValueError(f"must_min({self.must_min})이 must 개수({len(self.must)})보다 큽니다")
        return self

    def for_generation(self) -> GenerationInput:
        """생성기에 넘길 보기. 정답 정보는 들어가지 않는다."""
        return GenerationInput(
            question=self.question, qtype=self.qtype, answer_schema=self.answer_schema
        )

    def for_judge(self) -> JudgeQuestion:
        """판정에 넘길 보기. qid·split·provenance는 들어가지 않는다."""
        return JudgeQuestion(
            qtype=self.qtype,
            question=self.question,
            gold=self.gold,
            must=self.must,
            must_min=self.must_min,
            fail=self.fail,
            optional=self.optional,
            scoring_note=self.scoring_note,
            answer_schema=self.answer_schema,
        )


class Verdict(Frozen):
    """판정 부품 하나의 출력. 정오(correct) 칸은 일부러 없다. 정오는 코드가 계산한다(U7).

    `judge_id`는 판정 인스턴스 ID(같은 `llm` 판정 둘을 구분), `judge_type`은 등록 이름이다.
    """

    judge_id: NonEmptyStr
    judge_type: NonEmptyStr
    status: VerdictStatus
    must: tuple[bool | None, ...] = ()
    fail: tuple[bool | None, ...] = ()
    metrics: FrozenMetrics = {}
    usage: Usage | None = None
    note: NoUrlStr = ""


# --- 보조 타입: 원칙을 타입으로 강제 -------------------------------------------


class RetrievalQuery(Frozen):
    """검색기 입력.

    `gold`는 정답 근거가 필요한 검색기(`needs_gold`)에만, 생성만 수준에서만 채운다(U6이 강제).
    """

    text: NonEmptyStr
    gold: tuple[NonEmptyStr, ...] | None = None


class GenerationInput(Frozen):
    """생성기 입력. gold·must·fail 칸이 없다."""

    question: NonEmptyStr
    qtype: NonEmptyStr
    answer_schema: FrozenJson | None = None


class JudgeQuestion(Frozen):
    """판정에 넘기는 문항 보기. qid·split·provenance 칸이 없다."""

    qtype: NonEmptyStr
    question: NonEmptyStr
    gold: tuple[NonEmptyStr, ...] = ()
    must: tuple[str, ...] = ()
    must_min: int = Field(ge=0)
    fail: tuple[str, ...] = ()
    optional: tuple[str, ...] = ()
    scoring_note: str = ""
    answer_schema: FrozenJson | None = None


class JudgeInput(Frozen):
    """판정에 넘기는 답 보기. served_model·usage·arm 칸이 없다(블라인드)."""

    answer_text: str
    finish_reason: FinishReason
    context: tuple[Chunk, ...] = ()

    @classmethod
    def from_answer(cls, answer: Answer, context: Sequence[Chunk]) -> JudgeInput:
        return cls(
            answer_text=answer.text, finish_reason=answer.finish_reason, context=tuple(context)
        )


__all__ = [
    "Answer",
    "Chunk",
    "Document",
    "ErrorKind",
    "FinishReason",
    "Frozen",
    "FrozenJson",
    "GenerationInput",
    "Hit",
    "JudgeInput",
    "JudgeQuestion",
    "Question",
    "RetrievalQuery",
    "Section",
    "Split",
    "Usage",
    "Verdict",
    "VerdictStatus",
]
