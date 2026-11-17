"""도메인 타입 시험: 불변·해시·검증·JSON 왕복·원칙 타입."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from ragexp.types import (
    Answer,
    Chunk,
    Document,
    Frozen,
    GenerationInput,
    Hit,
    JudgeInput,
    JudgeQuestion,
    Question,
    RetrievalQuery,
    Section,
    Usage,
    Verdict,
)

# 설계 §6.3 문항 예시(그대로)
DESIGN_EXAMPLE = json.loads(
    """{"qid": "M001", "type": "multi_article", "split": "dev",
 "question": "...",
 "gold": ["law:000729:제27조", "law:000729:제28조"],
 "must": ["...", "..."], "must_min": 2, "fail": ["..."],
 "optional": ["..."], "scoring_note": "",
 "expected_tool_calls": [], "answer_schema": null,
 "provenance": {"drafted_by": "llm:model-x", "reviewed_by": "human",
                "checks": ["gold_exists", "quote_match", "ambiguity_ok"]}}"""
)


def chunk(cid: str = "law:1:제1조", **kw: Any) -> Chunk:
    return Chunk(chunk_id=cid, doc_id="law:1", text="본문", **kw)


def usage() -> Usage:
    return Usage(input_tokens=10, output_tokens=5, latency_ms=120.5)


def answer(**kw: Any) -> Answer:
    base: dict[str, Any] = {
        "text": "답 [1]",
        "finish_reason": "stop",
        "usage": usage(),
        "served_model": "model-a",
    }
    base.update(kw)
    return Answer(**base)


def question(**kw: Any) -> Question:
    data = dict(DESIGN_EXAMPLE)
    data.update(kw)
    return Question.from_dict(data)


SAMPLES: dict[str, Callable[[], Frozen]] = {
    "Section": lambda: Section(section_id="제27조의2", heading="목적", text="…", meta={"n": 1}),
    "Document": lambda: Document(
        doc_id="law:1",
        title="보조금법",
        sections=(Section(section_id="제1조", text="t"),),
        meta={"시행일": "2026-01-01", "tags": ["a", {"b": 1}]},
    ),
    "Chunk": lambda: chunk(meta={"article": "제1조"}),
    "Hit": lambda: Hit(chunk=chunk(), score=0.75, rank=1, source="oracle"),
    "Question": question,
    "Usage": usage,
    "Answer": answer,
    "Verdict": lambda: Verdict(
        judge_id="judges[1]",
        judge_type="llm",
        status="ok",
        must=(True, None),
        fail=(False,),
        metrics={"citation_recall": 0.5, "citation_precision": None},
        usage=usage(),
    ),
    "RetrievalQuery": lambda: RetrievalQuery(text="질문", gold=("law:1:제1조",)),
    "GenerationInput": lambda: question().for_generation(),
    "JudgeQuestion": lambda: question().for_judge(),
    "JudgeInput": lambda: JudgeInput.from_answer(answer(), [chunk()]),
}


# --- 불변·해시·왕복 -------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(SAMPLES))
def test_roundtrip_json(name: str) -> None:
    obj = SAMPLES[name]()
    data = obj.to_dict()
    text = json.dumps(data, ensure_ascii=False)  # JSON으로 쓸 수 있어야 함
    again = type(obj).from_dict(json.loads(text))
    assert again == obj
    assert hash(again) == hash(obj)


@pytest.mark.parametrize("name", sorted(SAMPLES))
def test_frozen_assignment(name: str) -> None:
    obj = SAMPLES[name]()
    field = next(iter(type(obj).model_fields))
    with pytest.raises(ValidationError):
        setattr(obj, field, getattr(obj, field))


def test_mapping_fields_are_deeply_readonly() -> None:
    meta: dict[str, Any] = {"tags": ["a"], "nested": {"k": 1}}
    c = chunk(meta=meta)
    meta["tags"].append("b")
    meta["nested"]["k"] = 2
    meta["new"] = 3
    assert c.to_dict()["meta"] == {"tags": ["a"], "nested": {"k": 1}}
    with pytest.raises(TypeError):
        c.meta["x"] = 1  # type: ignore[index]
    with pytest.raises(TypeError):
        c.meta["nested"]["k"] = 9  # type: ignore[index,call-overload]


def test_default_meta_is_readonly_and_not_shared() -> None:
    a, b = Section(section_id="a", text="t"), Section(section_id="b", text="t")
    assert a.meta is not b.meta
    with pytest.raises(TypeError):
        a.meta["x"] = 1  # type: ignore[index]


def test_hashable_in_sets() -> None:
    c1, c2 = chunk(meta={"a": 1}), chunk(meta={"a": 1})
    assert {c1, c2} == {c1}
    assert len({Hit(chunk=c1, score=1.0, rank=1, source="v"), question(), question()}) == 2


def test_equal_objects_have_equal_hash() -> None:
    a = chunk(meta={"a": 1, "b": 2.0})
    b = chunk(meta={"b": 2, "a": 1.0})  # 키 순서와 1/1.0 표기만 다름
    assert a == b
    assert hash(a) == hash(b)
    assert len({a, b}) == 1


def test_model_copy_revalidates_update() -> None:
    a = answer(finish_reason="error", error_kind="timeout", error="시간 초과")
    assert a.model_copy(update={"error": "30초 초과"}).error == "30초 초과"
    with pytest.raises(ValidationError):
        a.model_copy(update={"error": "see http://10.0.0.1/v1"})
    with pytest.raises(ValidationError):
        question().model_copy(update={"must_min": 9})
    assert question().model_copy() == question()


def test_extra_fields_rejected() -> None:
    with pytest.raises(ValidationError):
        Chunk.from_dict({"chunk_id": "c", "doc_id": "d", "text": "t", "is_gold": True})


# --- 생성 시 검증 ---------------------------------------------------------------

INVALID: list[tuple[str, Callable[[], object]]] = [
    ("빈 chunk_id", lambda: Chunk(chunk_id="", doc_id="d", text="t")),
    ("빈 doc_id", lambda: Document(doc_id="")),
    ("rank 0", lambda: Hit(chunk=chunk(), score=1.0, rank=0, source="v")),
    ("score NaN", lambda: Hit(chunk=chunk(), score=float("nan"), rank=1, source="v")),
    ("score inf", lambda: Hit(chunk=chunk(), score=float("inf"), rank=1, source="v")),
    ("meta 속 NaN", lambda: chunk(meta={"x": [1.0, float("nan")]})),
    ("음수 토큰", lambda: Usage(input_tokens=-1, latency_ms=1)),
    ("음수 지연", lambda: Usage(latency_ms=-0.1)),
    ("must_min > must", lambda: question(must=["a"], must_min=2)),
    ("음수 must_min", lambda: question(must_min=-1)),
    ("모르는 split", lambda: question(split="train")),
    ("빈 gold 항목", lambda: question(gold=[""])),
    (
        "error에 주소",
        lambda: answer(
            finish_reason="error", error_kind="http_status", error="500 at http://10.0.0.1/v1"
        ),
    ),
    ("error_kind 없이 error 종료", lambda: answer(finish_reason="error")),
    ("정상 종료인데 error_kind", lambda: answer(error_kind="timeout")),
    ("모르는 finish_reason", lambda: answer(finish_reason="cut")),
    (
        "error에 IP:포트",
        lambda: answer(
            finish_reason="error", error_kind="connect", error="connect 10.0.0.5:8000 refused"
        ),
    ),
    (
        "metrics에 정오",
        lambda: Verdict(judge_id="j", judge_type="llm", status="ok", metrics={"correct": 1.0}),
    ),
    (
        "metrics에 accuracy",
        lambda: Verdict(judge_id="j", judge_type="llm", status="ok", metrics={"Accuracy": 1.0}),
    ),
    (
        "note에 주소",
        lambda: Verdict(judge_id="j", judge_type="llm", status="ok", note="see https://x.y"),
    ),
    (
        "metrics NaN",
        lambda: Verdict(judge_id="j", judge_type="llm", status="ok", metrics={"r": float("nan")}),
    ),
]


@pytest.mark.parametrize("case", INVALID, ids=[c[0] for c in INVALID])
def test_invalid_values_rejected(case: tuple[str, Callable[[], object]]) -> None:
    with pytest.raises(ValidationError):
        case[1]()


def test_error_answer_ok() -> None:
    a = answer(text="", finish_reason="error", error_kind="timeout", error="시간 초과(30초)")
    assert a.error_kind == "timeout"


def test_tokens_may_be_unknown() -> None:
    u = Usage(latency_ms=3.0)
    assert u.input_tokens is None and u.output_tokens is None


# --- 문항 -----------------------------------------------------------------------


def test_question_design_example_roundtrip() -> None:
    q = Question.from_dict(DESIGN_EXAMPLE)
    assert q.qtype == "multi_article"
    assert q.to_dict() == DESIGN_EXAMPLE  # `type` 키 유지, 같은 JSON


def test_question_accepts_field_name_too() -> None:
    data = dict(DESIGN_EXAMPLE)
    data["qtype"] = data.pop("type")
    assert Question.from_dict(data).to_dict()["type"] == "multi_article"


def test_draft_question_has_no_split() -> None:
    data = {k: v for k, v in DESIGN_EXAMPLE.items() if k != "split"}
    assert Question.from_dict(data).split is None


def test_answerless_question() -> None:
    q = question(gold=[], must=[], must_min=0)
    assert q.gold == ()


# --- 원칙을 타입으로 강제 --------------------------------------------------------


def field_names(model: type[BaseModel]) -> set[str]:
    names = set(model.model_fields)
    names |= {f.alias for f in model.model_fields.values() if f.alias}
    return names


IDENTITY_FIELDS = {"model", "served_model", "usage", "arm", "provenance", "qid", "split"}
ANSWER_FIELDS = {"gold", "must", "fail", "optional", "must_min", "scoring_note"}


def test_judge_inputs_are_blind() -> None:
    assert field_names(JudgeInput).isdisjoint(IDENTITY_FIELDS)
    assert field_names(JudgeQuestion).isdisjoint(IDENTITY_FIELDS)


def test_generation_input_has_no_answer_info() -> None:
    assert field_names(GenerationInput).isdisjoint(ANSWER_FIELDS)
    gi = question().for_generation()
    assert gi.question == DESIGN_EXAMPLE["question"]
    assert "law:000729" not in gi.model_dump_json()


def test_verdict_has_no_correct_field() -> None:
    assert "correct" not in field_names(Verdict)


def test_from_answer_drops_model_identity() -> None:
    ji = JudgeInput.from_answer(answer(served_model="secret-model-x"), [chunk()])
    dumped = ji.model_dump_json()
    assert "secret-model-x" not in dumped
    assert ji.answer_text == "답 [1]" and ji.context == (chunk(),)


def test_for_judge_drops_provenance_and_ids() -> None:
    jq = question().for_judge()
    dumped = jq.model_dump_json()
    assert "llm:model-x" not in dumped and "M001" not in dumped and "dev" not in dumped
    assert jq.gold == tuple(DESIGN_EXAMPLE["gold"]) and jq.must_min == 2


def test_retrieval_query_gold_defaults_to_none() -> None:
    assert RetrievalQuery(text="질문").gold is None
