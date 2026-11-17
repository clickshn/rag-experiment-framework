"""버전 해시 시험: 같은 실험은 같은 해시, 다른 실험은 정해진 해시만 바뀜, 원재료로 다시 계산."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml
from conftest import CONFIGS, ENV, SECRET_VALUES

from ragexp.config import load_config
from ragexp.datasets.questions import load_questions
from ragexp.registry import Registry
from ragexp.types import Question
from ragexp.versioning import (
    SCHEME,
    VersionError,
    Versions,
    arm_preimage,
    canonical_json,
    compute_versions,
    drop_none,
    duplicate_arm_warnings,
    make_version,
    normalize_text,
    question_set_version_for,
    verify_versions,
)

BASE_TEXT = (CONFIGS / "base.yaml").read_text(encoding="utf-8")
PROMPT = (CONFIGS / "prompt.md").read_text(encoding="utf-8")
QUESTIONS = (CONFIGS / "questions.jsonl").read_text(encoding="utf-8")


def base_data() -> dict[str, Any]:
    data = yaml.safe_load(BASE_TEXT)
    assert isinstance(data, dict)
    return data


def write(
    tmp_path: Path,
    config: dict[str, Any] | str,
    *,
    prompt: str = PROMPT,
    questions: str = QUESTIONS,
    prompt_name: str = "prompt.md",
    questions_name: str = "questions.jsonl",
) -> Path:
    if isinstance(config, dict):
        config = copy.deepcopy(config)
        config["prompt"], config["questions"] = prompt_name, questions_name
        text = yaml.safe_dump(config, allow_unicode=True, sort_keys=True)
    else:
        text = config
    (tmp_path / prompt_name).write_bytes(prompt.encode("utf-8"))
    (tmp_path / questions_name).write_bytes(questions.encode("utf-8"))
    path = tmp_path / "config.yaml"
    path.write_bytes(text.encode("utf-8"))
    return path


def versions_of(path: Path, reg: Registry, env: dict[str, str] = ENV) -> Versions:
    loaded = load_config(path, reg, env=env)
    return compute_versions(loaded, reg, load_questions(loaded.questions_path))


def summary(v: Versions) -> dict[str, Any]:
    return {
        "arms": {k: x.sha256 for k, x in v.arms.items()},
        "judging": v.judging.sha256,
        "question_set": v.question_set.sha256,
        "run_set": v.run_set.sha256,
    }


# --- 정규화 기본 -------------------------------------------------------------------


def test_canonical_json_rules() -> None:
    assert canonical_json({"b": 1, "a": [1.0, 2.5, True]}) == '{"a":[1,2.5,true],"b":1}'
    assert canonical_json({"t": 0.0}) == canonical_json({"t": 0})
    assert canonical_json({"한": "글"}) == '{"한":"글"}'
    with pytest.raises(VersionError):
        canonical_json({"x": float("nan")})
    with pytest.raises(VersionError):
        canonical_json({1: "x"})


def test_drop_none_only_drops_none() -> None:
    assert drop_none({"a": None, "b": 0, "c": "", "d": [], "e": False, "f": {"g": None}}) == {
        "b": 0,
        "c": "",
        "d": [],
        "e": False,
        "f": {},
    }


def test_normalize_text() -> None:
    assert normalize_text("﻿a\r\nb\rc\n") == "a\nb\nc\n"


def test_level_is_in_arm_version() -> None:
    a = make_version("cfg", arm_preimage("generation_only", {}, "p", {"model": "m"}))
    b = make_version("cfg", arm_preimage("full", {}, "p", {"model": "m"}))
    assert a != b


# --- 같은 실험 → 같은 해시(기준 1) ---------------------------------------------------

Variant = Callable[[Path], Path]


def _text(fn: Callable[[str], str]) -> Variant:
    return lambda d: write(d, fn(BASE_TEXT))


def _data(fn: Callable[[dict[str, Any]], object], **kw: Any) -> Variant:
    def make(d: Path) -> Path:
        data = base_data()
        fn(data)
        return write(d, data, **kw)

    return make


def _set(path: str, value: Any) -> Callable[[dict[str, Any]], None]:
    def fn(data: dict[str, Any]) -> None:
        *head, last = path.split(".")
        node: Any = data
        for part in head:
            node = node[int(part)] if part.isdigit() else node[part]
        if value is _DELETE:
            del node[last]
        else:
            node[last] = value

    return fn


_DELETE = object()

SAME: dict[str, Variant] = {
    "주석 추가": _text(lambda t: "# 맨 위 주석\n" + t.replace("repeats: 3", "repeats: 3  # 반복")),
    "키 순서·흐름 스타일(다시 덤프)": _data(lambda d: None),
    "CRLF": _text(lambda t: t.replace("\n", "\r\n")),
    "meta 값 변경": _data(_set("generators.A.meta.price_per_1m_input", 99.0)),
    "meta 삭제": _data(_set("generators.B.meta", _DELETE)),
    "run 변경": _data(_set("generators.A.run", {"concurrency": 1, "timeout_s": 30})),
    "최상위 run": _data(_set("run", {"invalid_ratio_max": 0.2})),
    "revision: null": _data(_set("generators.A.revision", None)),
    "temperature 0.0": _data(_set("generators.A.params.temperature", 0.0)),
    "name 변경": _data(_set("name", "other-name")),
    "판정 순서": _data(lambda d: d["judges"].reverse()),
    "판정 ID 이름": _data(_set("judges.1.id", "main_renamed")),
    "프롬프트 CRLF·BOM": _data(lambda d: None, prompt="﻿" + PROMPT.replace("\n", "\r\n")),
    "프롬프트·문항 파일 이름": _data(
        lambda d: None, prompt_name="p2.md", questions_name="q2.jsonl"
    ),
    "문항 줄 순서·키 순서·CRLF": _data(
        lambda d: None,
        questions="\r\n".join(
            json.dumps(dict(reversed(list(json.loads(line).items()))), ensure_ascii=False)
            for line in reversed(QUESTIONS.strip().splitlines())
        ),
    ),
    "문항 기본값 칸을 직접 적기": _data(
        lambda d: None,
        questions=QUESTIONS.replace(
            '"must_min": 0,', '"must_min": 0, "optional": [], "scoring_note": "",'
        ),
    ),
}


@pytest.mark.parametrize("name", list(SAME))
def test_same_experiment_same_versions(name: str, tmp_path: Path, fake_registry: Registry) -> None:
    (tmp_path / "base").mkdir()
    (tmp_path / "v").mkdir()
    base = versions_of(write(tmp_path / "base", BASE_TEXT), fake_registry)
    other = versions_of(SAME[name](tmp_path / "v"), fake_registry)
    assert summary(other) == summary(base)


def test_env_values_do_not_matter(tmp_path: Path, fake_registry: Registry) -> None:
    path = write(tmp_path, BASE_TEXT)
    env2 = {k: v + "9" for k, v in ENV.items()}
    assert summary(versions_of(path, fake_registry, env2)) == summary(
        versions_of(path, fake_registry)
    )


def test_explicit_default_equals_omitted(tmp_path: Path, fake_registry: Registry) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    da, db = base_data(), base_data()
    da["chunker"] = {"type": "fake_chunker"}
    db["chunker"] = {"type": "fake_chunker", "max_chars": 1000}  # 부품 기본값
    assert summary(versions_of(write(a, da), fake_registry)) == summary(
        versions_of(write(b, db), fake_registry)
    )


# --- 다른 실험 → 정해진 해시만 바뀜(기준 2) -------------------------------------------

ALL = ("A", "B", "C")
# (변경 이름, 변형, 바뀌는 arm, 채점 버전 바뀜, 문항 세트 바뀜, 실행 묶음 바뀜)
CHANGES: list[tuple[str, Variant, tuple[str, ...], bool, bool, bool]] = [
    ("chunker 설정", _data(_set("chunker.max_chars", 900)), ALL, False, False, True),
    ("corpus 스냅샷", _data(_set("corpus.snapshot_sha256", "bbbb")), ALL, False, False, True),
    ("retriever 설정", _data(_set("retriever.seed", 8)), ALL, False, False, True),
    (
        "프롬프트 내용",
        _data(lambda d: None, prompt=PROMPT + "추가 지시\n"),
        ALL,
        False,
        False,
        True,
    ),
    ("A 모델", _data(_set("generators.A.model", "model-a2")), ("A",), False, False, True),
    ("A revision", _data(_set("generators.A.revision", "r2")), ("A",), False, False, True),
    ("B revision 변경", _data(_set("generators.B.revision", "def456")), ("B",), False, False, True),
    ("A params", _data(_set("generators.A.params.max_tokens", 4096)), ("A",), False, False, True),
    ("주 판정 모델", _data(_set("judges.1.model", "judge-x2")), (), True, False, False),
    ("주 판정 revision", _data(_set("judges.1.revision", "j1")), (), True, False, False),
    (
        "주 판정 바꾸기",
        _data(
            lambda d: (
                d["judges"][1].update(role="secondary"),
                d["judges"][2].update(role="primary"),
            )
        ),
        (),
        True,
        False,
        False,
    ),
    (
        "보조 판정 모델(채점 버전 그대로)",
        _data(_set("judges.2.model", "judge-z")),
        (),
        False,
        False,
        False,
    ),
    (
        "보조 판정 추가(채점 버전 그대로)",
        _data(
            lambda d: d["judges"].append(
                {
                    "id": "judge_3",
                    "type": "fake_llm",
                    "role": "secondary",
                    "base_url": "${JUDGE2_URL}",
                    "model": "j3",
                }
            )
        ),
        (),
        False,
        False,
        False,
    ),
    (
        "보조 판정 삭제(채점 버전 그대로)",
        _data(lambda d: d["judges"].pop(2)),
        (),
        False,
        False,
        False,
    ),
    (
        "문항 내용",
        _data(lambda d: None, questions=QUESTIONS.replace("누가 하나", "누가 해야 하나")),
        (),
        False,
        True,
        True,
    ),
    (
        "문항 provenance",
        _data(
            lambda d: None,
            questions=QUESTIONS.replace('"drafted_by": "human"}}', '"drafted_by": "llm"}}', 1),
        ),
        (),
        False,
        True,
        True,
    ),
    ("repeats", _data(_set("repeats", 5)), (), False, False, True),
]


@pytest.mark.parametrize("case", CHANGES, ids=[c[0] for c in CHANGES])
def test_change_table(
    case: tuple[str, Variant, tuple[str, ...], bool, bool, bool],
    tmp_path: Path,
    fake_registry: Registry,
) -> None:
    _, variant, arms, judging, qs, run = case
    (tmp_path / "base").mkdir()
    (tmp_path / "v").mkdir()
    base = summary(versions_of(write(tmp_path / "base", BASE_TEXT), fake_registry))
    other = summary(versions_of(variant(tmp_path / "v"), fake_registry))
    changed_arms = tuple(a for a in ALL if base["arms"][a] != other["arms"][a])
    assert changed_arms == arms
    assert (base["judging"] != other["judging"]) is judging
    assert (base["question_set"] != other["question_set"]) is qs
    assert (base["run_set"] != other["run_set"]) is run


def test_component_version_bump(
    tmp_path: Path, fake_registry: Registry, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write(tmp_path, BASE_TEXT)
    base = summary(versions_of(path, fake_registry))
    chunker = fake_registry.get("chunker", "fake_chunker")
    monkeypatch.setattr(chunker, "version", "2")
    assert all(
        base["arms"][a] != x for a, x in summary(versions_of(path, fake_registry))["arms"].items()
    )
    monkeypatch.setattr(chunker, "version", "1")
    rules = fake_registry.get("judge", "fake_rules")
    monkeypatch.setattr(rules, "version", "2")
    after = summary(versions_of(path, fake_registry))
    assert after["judging"] != base["judging"] and after["arms"] == base["arms"]


# --- arm 독립(기준 3) ---------------------------------------------------------------


def test_arm_versions_independent(tmp_path: Path, fake_registry: Registry) -> None:
    for d in ("base", "add", "rename"):
        (tmp_path / d).mkdir()
    base = summary(versions_of(write(tmp_path / "base", BASE_TEXT), fake_registry))

    added = base_data()
    added["generators"]["D"] = {"type": "fake_chat", "base_url": "${LOCAL_URL}", "model": "d"}
    after_add = summary(versions_of(write(tmp_path / "add", added), fake_registry))
    assert {a: after_add["arms"][a] for a in ALL} == base["arms"]
    assert after_add["run_set"] != base["run_set"]

    renamed = base_data()
    renamed["generators"]["Z"] = renamed["generators"].pop("A")
    after_rename = summary(versions_of(write(tmp_path / "rename", renamed), fake_registry))
    assert after_rename["arms"]["Z"] == base["arms"]["A"]
    assert after_rename["run_set"] != base["run_set"]


def test_duplicate_arm_config_warns(tmp_path: Path, fake_registry: Registry) -> None:
    data = base_data()
    data["generators"]["A2"] = copy.deepcopy(data["generators"]["A"])
    loaded = load_config(write(tmp_path, data), fake_registry, env=ENV)
    v = compute_versions(loaded, fake_registry, load_questions(loaded.questions_path))
    assert v.arms["A"] == v.arms["A2"]
    assert any("A, A2" in w for w in duplicate_arm_warnings(v))


# --- 골든(기준 4) -------------------------------------------------------------------

GOLDEN = {
    "arms": {"A": "cfg-05f276175657", "B": "cfg-0454ac6dbaba", "C": "cfg-0d287d8ad3f8"},
    "judging": "jdg-90cbad2aa87f",
    "question_set": "qs-c1f26fd7e51b",
    "run_set": "run-adc3fc60a094",
}


def test_golden_versions(tmp_path: Path, fake_registry: Registry) -> None:
    """정규화 규칙이 실수로 바뀌면 깨진다. 의도적으로 바꿀 때는 SCHEME을 올리고 값을 갱신한다."""
    assert SCHEME == 1
    v = versions_of(write(tmp_path, BASE_TEXT), fake_registry)
    assert {
        "arms": {k: x.short for k, x in v.arms.items()},
        "judging": v.judging.short,
        "question_set": v.question_set.short,
        "run_set": v.run_set.short,
    } == GOLDEN


# --- 원재료로 다시 계산(기준 4) -------------------------------------------------------


def test_verify_versions_roundtrip(tmp_path: Path, fake_registry: Registry) -> None:
    v = versions_of(write(tmp_path, BASE_TEXT), fake_registry)
    again = Versions.from_dict(json.loads(json.dumps(v.to_dict())))
    verify_versions(again)
    assert again == v
    assert "judge_side" in again.secondary_judges


def test_preimages_have_no_secrets(tmp_path: Path, fake_registry: Registry) -> None:
    v = versions_of(write(tmp_path, BASE_TEXT), fake_registry)
    text = json.dumps(v.to_dict(), ensure_ascii=False)
    for secret in SECRET_VALUES:
        assert secret not in text
    assert "${" not in text and "prompt.md" not in text


@pytest.mark.parametrize(
    "tamper",
    [
        lambda d: d["preimages"]["arms"]["A"]["generator"].update(model="evil"),
        lambda d: d["preimages"]["judging"].update(merge={"method": "vote"}),
        lambda d: d["preimages"]["run_set"].update(repeats=9),
        lambda d: d["preimages"]["question_set"]["questions"].pop(),
        lambda d: d["arms"].pop("C"),
        lambda d: d["secondary_judges"]["judge_side"]["fingerprint"].update(model="z"),
    ],
)
def test_verify_versions_detects_tampering(
    tamper: Callable[[dict[str, Any]], None], tmp_path: Path, fake_registry: Registry
) -> None:
    data = versions_of(write(tmp_path, BASE_TEXT), fake_registry).to_dict()
    tamper(data)
    with pytest.raises(VersionError):
        verify_versions(Versions.from_dict(data))


# --- 문항 세트(기준 10) -------------------------------------------------------------


def _q(qid: str, **kw: Any) -> Question:
    base: dict[str, Any] = {"qid": qid, "type": "t", "question": "질문", "must_min": 0}
    base.update(kw)
    return Question.from_dict(base)


def test_question_set_order_free() -> None:
    a = question_set_version_for([_q("Q1"), _q("Q2")])
    b = question_set_version_for([_q("Q2"), _q("Q1")])
    assert a == b


def test_question_set_defaults_do_not_matter() -> None:
    a = question_set_version_for([_q("Q1")])
    b = question_set_version_for([_q("Q1", optional=[], scoring_note="", provenance={})])
    assert a == b


def test_question_set_content_matters() -> None:
    a = question_set_version_for([_q("Q1")])
    assert a != question_set_version_for([_q("Q1", question="다른 질문")])
    assert a != question_set_version_for([_q("Q1", provenance={"reviewed_by": "human"})])


def test_question_set_duplicate_and_empty() -> None:
    with pytest.raises(VersionError, match="중복"):
        question_set_version_for([_q("Q1"), _q("Q1")])
    with pytest.raises(VersionError):
        question_set_version_for([])


# --- 구현 검토 지적 반영 ------------------------------------------------------------


def _pair(tmp_path: Path, fn: Callable[[dict[str, Any]], object] | None = None) -> Path:
    d = tmp_path / ("v" if fn else "base")
    d.mkdir(exist_ok=True)
    data = base_data()
    if fn:
        fn(data)
    return write(d, data)


def test_one_arm_generator_version(
    tmp_path: Path, fake_registry: Registry, monkeypatch: pytest.MonkeyPatch
) -> None:
    """지적 8: 한 arm의 generator version만 올리면 그 arm만 바뀐다."""
    path = _pair(tmp_path, lambda d: d["generators"]["C"].update(type="fake_chat2"))
    before = summary(versions_of(path, fake_registry))
    monkeypatch.setattr(fake_registry.get("generator", "fake_chat2"), "version", "2")
    after = summary(versions_of(path, fake_registry))
    assert [a for a in ALL if before["arms"][a] != after["arms"][a]] == ["C"]
    assert after["judging"] == before["judging"]


def test_primary_judge_version(
    tmp_path: Path, fake_registry: Registry, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _pair(tmp_path)
    before = summary(versions_of(path, fake_registry))
    monkeypatch.setattr(fake_registry.get("judge", "fake_llm"), "version", "2")
    after = summary(versions_of(path, fake_registry))
    assert after["judging"] != before["judging"] and after["arms"] == before["arms"]
    assert after["run_set"] == before["run_set"]


def _add_rules2(d: dict[str, Any]) -> None:
    d["judges"].append({"id": "rules2", "type": "fake_rules2", "role": "rules"})


def test_rules_judges_add_remove_and_order(tmp_path: Path, fake_registry: Registry) -> None:
    """지적 8: 코드 규칙 판정을 더하거나 빼면 채점 버전이 바뀌고, 순서는 영향이 없다."""
    base = summary(versions_of(_pair(tmp_path), fake_registry))
    added = summary(versions_of(_pair(tmp_path, _add_rules2), fake_registry))
    assert added["judging"] != base["judging"]

    def reorder(d: dict[str, Any]) -> None:
        _add_rules2(d)
        d["judges"].insert(0, d["judges"].pop())

    (tmp_path / "r").mkdir()
    data = base_data()
    reorder(data)
    reordered = summary(versions_of(write(tmp_path / "r", data), fake_registry))
    assert reordered["judging"] == added["judging"]

    (tmp_path / "x").mkdir()
    removed = base_data()
    removed["judges"].pop(0)  # 코드 규칙 판정 삭제
    assert (
        summary(versions_of(write(tmp_path / "x", removed), fake_registry))["judging"]
        != base["judging"]
    )


def test_unused_part_does_not_change_versions(tmp_path: Path, fake_registry: Registry) -> None:
    """지적 8(기준 5): 안 쓰는 부품을 더해도 해시가 그대로다."""
    base = summary(versions_of(_pair(tmp_path), fake_registry))
    with_unused = summary(
        versions_of(
            _pair(tmp_path, lambda d: d.update(embedder={"type": "fake_embedder"})), fake_registry
        )
    )
    assert with_unused == base


def test_exponent_float_same_as_decimal(tmp_path: Path, fake_registry: Registry) -> None:
    """지적 17: temperature 1e-3과 0.001은 같은 실험이다."""
    a = _pair(tmp_path, lambda d: d["generators"]["C"]["params"].update(temperature=0.001))
    text = a.read_text(encoding="utf-8").replace("temperature: 0.001", "temperature: 1e-3")
    assert "1e-3" in text
    (tmp_path / "e").mkdir()
    b = write(tmp_path / "e", text)
    assert summary(versions_of(a, fake_registry)) == summary(versions_of(b, fake_registry))


def test_verify_rejects_scheme_mismatch(tmp_path: Path, fake_registry: Registry) -> None:
    """지적 16: 기록의 scheme과 원재료의 scheme이 같아야 한다."""
    data = versions_of(write(tmp_path, BASE_TEXT), fake_registry).to_dict()
    data["scheme"] = SCHEME + 1
    with pytest.raises(VersionError, match="해시 규칙 버전"):
        verify_versions(Versions.from_dict(data))
    data["scheme"] = SCHEME
    data["preimages"]["judging"]["scheme"] = SCHEME + 1
    with pytest.raises(VersionError):
        verify_versions(Versions.from_dict(data))
