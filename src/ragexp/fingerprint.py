"""부품 fingerprint 검사.

fingerprint는 부품의 "출력을 바꾸는 것"만 담는다(모델·revision·파라미터·템플릿 내용 해시·
스냅샷 내용 해시). 구성·채점 버전 해시에 들어가므로 다음은 들어가면 안 된다(설계 §5).

- 서버 주소·API 키·토큰, `${VAR}` 치환 자리
- 모델 meta(가격, GPU 메모리, 라이선스, 폐쇄망 반입, 종료 예정일, deploy)
- 절대 로컬 경로(같은 실험이 기계마다 다른 해시가 됨). 스냅샷은 경로가 아니라 내용 해시로 넣는다.

이름 검사는 어느 깊이에서든, 대소문자 무시로 한다. 부품은 클래스 속성 `fingerprint_allow`로
정상 파라미터의 예외를 선언할 수 있다(프레임워크를 고치지 않고).
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from typing import Any

Fingerprint = Mapping[str, Any]
"""부품이 내놓는 fingerprint(JSON 값만)."""

FORBIDDEN_NAMES = frozenset(
    {
        # 주소·비밀값
        "base_url",
        "api_base",
        "url",
        "endpoint",
        "host",
        "api_key",
        "access_token",
        "auth_token",
        "secret",
        "password",
        "token",
        "key",
        "apikey",
        "hostname",
        "server",
        "address",
        "oc",  # 국가법령정보센터 Open API 인증키(설계 §6.1)
        # 모델 meta(설계 §4·§5)
        "meta",
        "gpu_cost_per_hour",
        "vram_gb",
        "license",
        "deploy",
        "air_gapped",
        "end_of_life",
    }
)
FORBIDDEN_SUFFIXES = ("_key", "_secret", "_password", "_token")
FORBIDDEN_PREFIXES = ("price", "auth")

# Windows 드라이브(C:\, d:/), 문자열 어디에 있든. 앞이 글자·숫자면 드라이브가 아님(예: "ab:/").
_DRIVE_RE = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]")
# POSIX 절대 경로: "/이름/..."로 시작(예: /home/a, /Users/b, /data/snap).
_POSIX_ABS_RE = re.compile(r"^/[^/\s]+/")
# 문자열 중간의 흔한 POSIX 절대 경로(예: "copied from /home/a", "file:/Users/b").
_POSIX_ROOTS_RE = re.compile(
    r"(?<![\w.~/])/(home|Users|root|mnt|data|tmp|var|opt|srv|media|Volumes|private|workspace|models)/"
)
# 스킴 없는 주소를 문자열 **어디서든** 찾는다(문장 속, 조사·마침표가 붙어도).
# IPv4: 뒤에 문장 마침표는 허용("10.0.0.5."), 숫자가 이어지면 아님.
_IPV4_RE = re.compile(r"(?<![\d.])\d{1,3}(\.\d{1,3}){3}(?!\.?\d)")
# 호스트:포트: 호스트는 영문으로 시작("12:30" 시각은 아님), 포트 뒤에 영문·숫자가 이어지면 아님
# ("qwen3:32b" 같은 모델 태그는 주소가 아님). 한국어 조사·문장 부호는 붙어도 주소로 본다.
_HOST_PORT_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])(localhost|[A-Za-z][A-Za-z0-9-]*(\.[A-Za-z0-9-]+)*):\d{2,5}(?![A-Za-z0-9_])"
)


class FingerprintError(ValueError):
    """fingerprint에 들어가면 안 되는 이름·값이 있음."""


def normalize_name(name: str) -> str:
    """이름 비교용: 소문자, 하이픈→밑줄(api-key, X-Api-Key도 같은 규칙으로)."""
    return name.lower().replace("-", "_")


def _bad_name(name: str, allow: frozenset[str]) -> bool:
    low = normalize_name(name)
    if low in FORBIDDEN_NAMES:
        return True  # 정확한 금지 이름은 fingerprint_allow로도 풀 수 없다
    if low in allow:
        return False
    return low.endswith(FORBIDDEN_SUFFIXES) or low.startswith(FORBIDDEN_PREFIXES)


def has_address(value: str) -> bool:
    """문자열 어디에든 주소(스킴 있는 URL, IPv4, 호스트:포트)가 있는지 본다."""
    return "://" in value or bool(_IPV4_RE.search(value) or _HOST_PORT_RE.search(value))


# 흔한 비밀값 꼴(Bearer 토큰, OpenAI·HF 형식 키). 구성 원문 어디에 있든 거부한다.
#
# 이 목록은 확정이다(2026-11-17). 새 패턴은 **실제로 새는 사례가 나왔을 때만** 더한다.
# 그 밖의 미탐은 design.md "알려진 남는 위험"에 적고,
# 두 번째 겹(gitleaks pre-commit·CI 전체 이력 검사)에 맡긴다.
SECRET_PATTERNS: dict[str, str] = {
    "Bearer 토큰": r"\bBearer\s+[A-Za-z0-9._~+/=-]{8,}",
    "OpenAI 형식 키(sk-)": r"\bsk-[A-Za-z0-9_-]{16,}",
    "Hugging Face 토큰(hf_)": r"\bhf_[A-Za-z0-9]{20,}",
    "GitHub 토큰(ghp_)": r"\bghp_[A-Za-z0-9]{36}",
    "GitHub 세분 토큰(github_pat_)": r"\bgithub_pat_[A-Za-z0-9_]{22,}",
    "AWS 액세스 키(AKIA)": r"\bAKIA[0-9A-Z]{16}",
    "Google API 키(AIza)": r"\bAIza[0-9A-Za-z_-]{35}",
}
_SECRET_RE = re.compile("|".join(f"(?:{p})" for p in SECRET_PATTERNS.values()))


def has_secret_pattern(value: str) -> bool:
    """문자열에 흔한 비밀값 꼴(`SECRET_PATTERNS`)이 있는지 본다."""
    return bool(_SECRET_RE.search(value))


def _is_model_key(path: str) -> bool:
    """경로의 마지막 키가 모델 이름 칸(model, judge_model, model_name 등)인지 본다."""
    last = normalize_name(path.rsplit(".", 1)[-1].split("[", 1)[0])
    return last == "model" or last.endswith("_model") or last == "model_name"


def _bad_string(value: str) -> str | None:
    """문자열 값이 금지 대상이면 이유를, 아니면 None을 돌려준다."""
    if has_address(value):
        return "주소"
    if value.startswith("${"):
        return "환경 변수 치환 자리(${...})"
    if _DRIVE_RE.search(value) or value.startswith("\\\\"):
        return "절대 로컬 경로(드라이브·UNC)"
    if (
        value.startswith(("~/", "~\\"))
        or _POSIX_ABS_RE.match(value)
        or _POSIX_ROOTS_RE.search(value)
    ):
        return "절대 로컬 경로"
    return None


def _check(value: Any, path: str, allow: frozenset[str]) -> Any:
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise FingerprintError(f"{path}: 키는 문자열이어야 합니다({key!r})")
            child = f"{path}.{key}" if path else key
            if _bad_name(key, allow):
                raise FingerprintError(
                    f"{child}: fingerprint에 넣을 수 없는 이름입니다"
                    "(주소·비밀값·meta는 해시에서 뺍니다. 정상 파라미터면 fingerprint_allow에 선언)"
                )
            out[key] = _check(item, child, allow)
        return out
    if isinstance(value, (list, tuple)):
        return [_check(item, f"{path}[{i}]", allow) for i, item in enumerate(value)]
    if isinstance(value, str):
        reason = _bad_string(value)
        if reason:
            if "경로" in reason and _is_model_key(path):
                raise FingerprintError(
                    f"{path}: 모델 이름이 경로입니다({reason}). "
                    "vLLM은 --served-model-name으로 이름을 정하세요. "
                    "모델 경로는 PC마다 달라 구성 버전에 넣지 않습니다"
                )
            raise FingerprintError(
                f"{path}: 값에 {reason}이(가) 들어 있습니다. "
                "주소·경로는 해시에 넣지 않습니다(스냅샷·파일은 내용 해시로)"
            )
        return value
    if isinstance(value, bool) or value is None or isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise FingerprintError(f"{path}: NaN·inf는 쓸 수 없습니다")
        return value
    raise FingerprintError(f"{path}: JSON 값이 아닙니다({type(value).__name__})")


def check_fingerprint(fp: Mapping[str, Any], allow: Iterable[str] = ()) -> dict[str, Any]:
    """fingerprint를 검사하고 정규화한 사본을 돌려준다.

    정규화는 `tuple→list`, `Mapping→dict`만 한다(숫자·키 순서는 U2 해시 규칙이 정함).
    """
    if not isinstance(fp, Mapping):
        raise FingerprintError(f"fingerprint는 Mapping이어야 합니다({type(fp).__name__})")
    allowed = frozenset(normalize_name(a) for a in allow)
    result = _check(fp, "", allowed)
    assert isinstance(result, dict)
    return result
