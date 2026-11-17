"""버전 해시: 구성(arm별)·채점·문항 세트·실행 묶음.

- 정규 JSON(키 정렬, 정수로 떨어지는 float는 int, NaN 거부)을 sha256으로 해시한다.
- 해시마다 원재료(preimage)를 함께 남겨, 실행 기록만으로 다시 계산·검증할 수 있다(설계 §8.1).
- 해시 규칙 자체를 바꾸면 `SCHEME`을 올린다.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field

from ragexp import __version__
from ragexp.types import Frozen, FrozenJson, Question

if TYPE_CHECKING:
    from ragexp.config import LoadedConfig
    from ragexp.registry import Registry

SCHEME = 1
"""해시 규칙 버전. 정규화 규칙을 바꾸면 올린다(모든 해시가 의도적으로 바뀜)."""

MERGE_RULE_R0: dict[str, Any] = {"method": "primary_only"}
"""판정 합치기 규칙. R0는 주 판정 하나로 점수를 낸다. 교차 판정은 R4."""

VersionKind = Literal["cfg", "jdg", "qs", "run"]


class VersionError(ValueError):
    """해시를 만들 수 없거나 다시 계산한 값이 다름."""


# --- 정규화 -----------------------------------------------------------------------


def _normalize(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        out: dict[str, Any] = {}
        for key, value in obj.items():
            if not isinstance(key, str):
                raise VersionError(f"해시할 키는 문자열이어야 합니다({key!r})")
            out[key] = _normalize(value)
        return out
    if isinstance(obj, (list, tuple)):
        return [_normalize(v) for v in obj]
    if isinstance(obj, bool) or obj is None or isinstance(obj, (str, int)):
        return obj
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise VersionError("NaN·inf는 해시할 수 없습니다")
        return int(obj) if obj.is_integer() else obj
    raise VersionError(f"해시할 수 없는 값입니다({type(obj).__name__})")


def canonical_json(obj: Any) -> str:
    """정규 JSON: 키 정렬, 공백 없는 구분자, UTF-8, 정수로 떨어지는 float는 int."""
    return json.dumps(
        _normalize(obj), sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _drop_none_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return drop_none(value)
    if isinstance(value, (list, tuple)):
        return [_drop_none_value(v) for v in value]
    return value


def drop_none(mapping: Mapping[str, Any]) -> dict[str, Any]:
    """None인 칸을 뺀다(중첩 dict, 리스트 안의 dict 포함). 리스트 항목인 None은 그대로 둔다.

    모든 부품 fingerprint에 적용된다(registry.component_fingerprint). 새 선택 칸(기본 None)을
    더해도 기존 해시가 그대로이게 한다.
    """
    return {k: _drop_none_value(v) for k, v in mapping.items() if v is not None}


def normalize_text(text: str) -> str:
    """텍스트 파일 내용 정규화: BOM 제거, CRLF·CR → LF."""
    return text.removeprefix("﻿").replace("\r\n", "\n").replace("\r", "\n")


def text_sha256(text: str) -> str:
    return sha256_hex(normalize_text(text))


# --- 버전 값 ----------------------------------------------------------------------


class Version(Frozen):
    """해시 하나. 표시는 `cfg-<앞 12자리>`."""

    kind: VersionKind
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @property
    def short(self) -> str:
        return f"{self.kind}-{self.sha256[:12]}"

    def __str__(self) -> str:
        return self.short


def make_version(kind: VersionKind, preimage: Mapping[str, Any]) -> Version:
    return Version(kind=kind, sha256=sha256_hex(canonical_json(preimage)))


# --- 원재료(preimage) --------------------------------------------------------------


def arm_preimage(
    level: str,
    parts: Mapping[str, Mapping[str, Any]],
    prompt_sha256: str,
    generator: Mapping[str, Any],
) -> dict[str, Any]:
    """arm 구성 버전의 원재료. arm 이름은 넣지 않는다."""
    return {
        "scheme": SCHEME,
        "level": level,
        "parts": dict(parts),
        "prompt_sha256": prompt_sha256,
        "generator": dict(generator),
    }


def judging_preimage(
    rules: Sequence[Mapping[str, Any]],
    primary: Mapping[str, Any],
    merge: Mapping[str, Any] = MERGE_RULE_R0,
) -> dict[str, Any]:
    """채점 버전의 원재료: 코드 규칙 판정 + 주 판정 + 합치기 규칙.

    판정 ID와 YAML 순서는 넣지 않는다(코드 규칙은 정규 JSON으로 정렬).
    보조 판정은 넣지 않는다(점수에 영향이 없으므로). 각자의 fingerprint로 따로 기록한다.
    """
    return {
        "scheme": SCHEME,
        "rules": sorted((dict(r) for r in rules), key=canonical_json),
        "primary": dict(primary),
        "merge": dict(merge),
    }


def question_digest(question: Question) -> str:
    """문항 하나의 해시. 기본값(빈 칸)은 빼서, 문항 칸이 늘어도 기존 해시가 그대로다."""
    return sha256_hex(canonical_json(question.model_dump(mode="json", exclude_defaults=True)))


def question_set_preimage(questions: Iterable[Question]) -> dict[str, Any]:
    """문항 세트 버전의 원재료: qid 순으로 정렬한 (qid, 문항 해시) 목록."""
    seen: dict[str, str] = {}
    for q in questions:
        if q.qid in seen:
            raise VersionError(f"문항 qid가 중복됩니다: {q.qid}")
        seen[q.qid] = question_digest(q)
    if not seen:
        raise VersionError("문항이 하나도 없습니다")
    return {"scheme": SCHEME, "questions": [[qid, seen[qid]] for qid in sorted(seen)]}


def question_set_version_for(questions: Iterable[Question]) -> Version:
    """문항 세트 버전."""
    return make_version("qs", question_set_preimage(questions))


def run_set_preimage(
    arms: Mapping[str, Version], question_set: Version, repeats: int
) -> dict[str, Any]:
    """실행 묶음 해시의 원재료: 생성 쪽만(채점 버전은 넣지 않음, 다시 채점 가능)."""
    return {
        "scheme": SCHEME,
        "arms": {name: v.sha256 for name, v in arms.items()},
        "question_set": question_set.sha256,
        "repeats": repeats,
    }


# --- 묶음 ------------------------------------------------------------------------


class Versions(Frozen):
    """한 실행의 버전 묶음. U6이 versions.json으로 쓴다."""

    scheme: int
    arms: dict[str, Version]
    judging: Version
    question_set: Version
    run_set: Version
    preimages: FrozenJson
    secondary_judges: FrozenJson = Field(default_factory=dict)
    """보조 판정: {판정 ID: {"type", "fingerprint", "sha256"}}. 어떤 해시에도 들어가지 않는다."""
    framework_version: str = __version__
    """기록만(해시 밖)."""


def compute_versions(
    loaded: LoadedConfig, registry: Registry, questions: Sequence[Question]
) -> Versions:
    """읽은 구성과 문항으로 버전 묶음을 만든다."""
    fp = registry.component_fingerprint
    parts = {kind: fp(obj) for kind, obj in sorted(loaded.parts.items())}

    arm_pre: dict[str, dict[str, Any]] = {}
    arms: dict[str, Version] = {}
    for name, arm in sorted(loaded.arms.items()):
        pre = arm_preimage(loaded.level, parts, loaded.arm_prompt(name).sha256, fp(arm.generator))
        arm_pre[name] = pre
        arms[name] = make_version("cfg", pre)

    rules = [fp(j.component) for j in loaded.judges if j.role == "rules"]
    primary = next(j for j in loaded.judges if j.role == "primary")
    judging_pre = judging_preimage(rules, fp(primary.component))
    secondary = {}
    for j in loaded.judges:
        if j.role == "secondary":
            jfp = fp(j.component)
            secondary[j.judge_id] = {
                "type": j.judge_type,
                "fingerprint": jfp,
                "sha256": sha256_hex(canonical_json(jfp)),
            }

    qs_pre = question_set_preimage(questions)
    question_set = make_version("qs", qs_pre)
    run_pre = run_set_preimage(arms, question_set, loaded.repeats)

    return Versions(
        scheme=SCHEME,
        arms=arms,
        judging=make_version("jdg", judging_pre),
        question_set=question_set,
        run_set=make_version("run", run_pre),
        preimages=_normalize(
            {"arms": arm_pre, "judging": judging_pre, "question_set": qs_pre, "run_set": run_pre}
        ),
        secondary_judges=_normalize(secondary),
    )


def verify_versions(versions: Versions) -> None:
    """원재료만으로 해시를 다시 계산해 대조한다(환경 변수·원본 파일·부품 없이).

    다르면 VersionError를 낸다.
    """
    if versions.scheme != SCHEME:
        raise VersionError(
            f"해시 규칙 버전이 다릅니다(기록 {versions.scheme}, 이 코드 {SCHEME}). "
            "이 코드로는 검증할 수 없습니다"
        )
    pre: Any = versions.preimages
    try:
        arm_pre = pre["arms"]
        checks: list[tuple[str, Version, Mapping[str, Any]]] = [
            (f"arms.{name}", versions.arms[name], arm_pre[name]) for name in versions.arms
        ]
        checks += [
            ("judging", versions.judging, pre["judging"]),
            ("question_set", versions.question_set, pre["question_set"]),
            ("run_set", versions.run_set, pre["run_set"]),
        ]
    except KeyError as exc:
        raise VersionError(f"원재료가 빠져 있습니다: {exc}") from None
    for label, _, preimage in checks:
        if preimage.get("scheme") != versions.scheme:
            raise VersionError(f"{label}: 원재료의 해시 규칙 버전이 기록과 다릅니다")
    if set(arm_pre) != set(versions.arms):
        raise VersionError("arm 목록과 원재료의 arm 목록이 다릅니다")
    for label, version, preimage in checks:
        again = make_version(version.kind, preimage)
        if again.sha256 != version.sha256:
            raise VersionError(f"{label}: 원재료로 다시 계산한 해시가 다릅니다")
    run = pre["run_set"]
    if dict(run["arms"]) != {n: v.sha256 for n, v in versions.arms.items()}:
        raise VersionError("실행 묶음의 arm 해시가 arm 구성 버전과 다릅니다")
    if run["question_set"] != versions.question_set.sha256:
        raise VersionError("실행 묶음의 문항 세트 해시가 다릅니다")
    secondary: Any = versions.secondary_judges
    for jid, rec in secondary.items():
        if sha256_hex(canonical_json(rec["fingerprint"])) != rec["sha256"]:
            raise VersionError(f"보조 판정 {jid}: fingerprint 해시가 다릅니다")


def duplicate_arm_warnings(versions: Versions) -> list[str]:
    """구성 버전이 같은 arm(같은 설정을 두 번 적음)에 대한 경고."""
    by_hash: dict[str, list[str]] = {}
    for name, v in sorted(versions.arms.items()):
        by_hash.setdefault(v.sha256, []).append(name)
    return [
        f"arm {', '.join(names)}의 구성 버전이 같습니다(같은 설정을 두 번 적었나요?)"
        for names in by_hash.values()
        if len(names) > 1
    ]
