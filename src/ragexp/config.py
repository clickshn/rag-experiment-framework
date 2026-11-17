"""실험 구성 YAML 읽기·검증.

순서(`load_config`):
1. 원문을 읽는다.
2. 원문 텍스트(주석 포함)를 줄·토큰 단위로 검사한다. 주소, 주석 속 비밀값을 본다.
3. YAML을 파싱하고 최상위 스키마를 검증한다. R0는 generation_only만 받는다.
4. 쓰는 부품과 안 쓰는 부품을 가른다. 안 쓰는 부품은 이름만 확인하고 경고한다.
5. level과 검색기 짝을 검사한다.
6. 쓰는 부품의 값·키를 검사한다: `${VAR}` 위치, 접속 칸 리터럴, 주소.
7. 프롬프트를 읽는다.
8. 환경 변수를 치환한다.
9. 부품을 만든다(registry.create, config_model 검증은 여기서 한 번).

치환한 값(주소·키)은 객체 안에만 있다. 오류 메시지·repr·파일에는 나가지 않는다.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from typing import Annotated, Any, Literal, get_args
from urllib.parse import urlsplit

import httpx
import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    SecretStr,
    ValidationError,
    field_validator,
)
from pydantic_core import PydanticCustomError

from ragexp.fingerprint import FingerprintError, has_address, has_secret_pattern, normalize_name
from ragexp.registry import ComponentKind, Registry, RegistryError
from ragexp.versioning import drop_none, normalize_text, sha256_hex

CONNECTION_FIELDS = frozenset(
    {
        "base_url",
        "api_base",
        "endpoint",
        "url",
        "host",
        "hostname",
        "api_key",
        "apikey",
        "access_token",
        "auth_token",
        "token",
        "key",
        "secret",
        "password",
        "oc",
    }
)
"""접속 칸(정확한 이름). `${VAR}`만 쓸 수 있고, 부품 설정 맨 위에만 둔다.

meta 칸은 접속 칸이 아니다.
"""
CONNECTION_SUFFIXES = ("_url", "_key", "_token", "_secret", "_password")
CONNECTION_PREFIXES = ("auth",)  # authorization, auth_header …

SUPPORTED_LEVELS: tuple[str, ...] = ("generation_only",)
ALL_LEVELS = ("generation_only", "retrieval_only", "full")
OPTIONAL_PARTS: tuple[ComponentKind, ...] = ("embedder", "vector_store", "reranker")
USED_PARTS: dict[str, tuple[ComponentKind, ...]] = {
    "generation_only": ("corpus", "chunker", "retriever"),
    "retrieval_only": ("corpus", "chunker", "embedder", "vector_store", "retriever", "reranker"),
    "full": ("corpus", "chunker", "embedder", "vector_store", "retriever", "reranker"),
}

_VAR_RE = re.compile(r"^\$\{([A-Z][A-Z0-9_]*)\}$")
# 줄 안의 `이름: 값`·`이름=값` 짝(주석·본문 모두). 값은 따옴표·쉼표·괄호를 떼고 본다.
_PAIR_RE = re.compile(r"([A-Za-z0-9_-]+)\s*[:=]\s*(\S+)")
# 키처럼 생긴 값: 영문·숫자·기호 8자 이상, 숫자나 -·_가 섞임(예: sk-live-abc, abcd1234efgh).
_TOKEN_LIKE_RE = re.compile(r"^(?=.*[0-9_-])[A-Za-z0-9._~+/=-]{8,}$")
_ARM_RE = re.compile(r"^[A-Za-z0-9_-]+$")
UNSET_PLACEHOLDER = "__ragexp_unset__"
"""`--no-env`로 읽을 때 빈 접속 칸에 넣는 자리표시 값. 모델 클라이언트는 이 값이면 만들지 않는다."""

RESERVED_KEYS = frozenset({"type", "meta", "run", "id", "role", "prompt_template"})
"""부품 항목의 예약 칸(부품 설정으로 넘기지 않음). generator의 `prompt`는 R4까지 따로 막는다."""


class ConfigError(ValueError):
    """구성 오류. 메시지에는 칸 경로·줄 번호만 넣고 값은 넣지 않는다."""

    def __init__(self, messages: list[str] | str) -> None:
        self.messages = [messages] if isinstance(messages, str) else list(messages)
        super().__init__("\n".join(self.messages))


def is_connection_field(name: str, allow: frozenset[str] = frozenset()) -> bool:
    """`${VAR}`만 쓸 수 있는 접속 칸인지 본다. 부품이 fingerprint_allow로 연 이름은 아니다."""
    n = normalize_name(name)
    if n in allow:
        return False
    return (
        n in CONNECTION_FIELDS
        or n.endswith(CONNECTION_SUFFIXES)
        or n.startswith(CONNECTION_PREFIXES)
    )


# --- YAML 로더 -------------------------------------------------------------------


class _Loader(yaml.SafeLoader):
    """SafeLoader + 중복 키 오류 + 날짜 자동 변환 끔 + 소수점 없는 지수 표기(1e-3)도 float."""


_Loader.yaml_implicit_resolvers = {
    first: [(tag, rx) for tag, rx in resolvers if tag != "tag:yaml.org,2002:timestamp"]
    for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
# YAML 1.1(PyYAML)은 `1e-3`을 문자열로 읽는다. YAML 1.2처럼 float로 읽게 한다.
_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"^[-+]?[0-9][0-9_]*[eE][-+]?[0-9]+$"),
    list("-+0123456789"),
)


def _construct_mapping(loader: _Loader, node: yaml.MappingNode) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    out: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        if not isinstance(key, str):
            raise ConfigError(f"{key_node.start_mark.line + 1}행: 키는 문자열이어야 합니다")
        if key in out:
            raise ConfigError(f"{key_node.start_mark.line + 1}행: 키 '{key}'가 중복됩니다")
        out[key] = loader.construct_object(value_node, deep=True)
    return out


_Loader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


def _check_json_types(obj: Any, path: str) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            _check_json_types(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _check_json_types(v, f"{path}[{i}]")
    elif not (obj is None or isinstance(obj, (str, int, float, bool))):
        raise ConfigError(f"{path}: JSON 값이 아닙니다({type(obj).__name__})")


def parse_yaml(text: str) -> Any:
    """YAML을 읽는다(SafeLoader 기반). 오류 메시지에 원문 조각은 넣지 않는다."""
    try:
        data = yaml.load(text, Loader=_Loader)
    except yaml.MarkedYAMLError as exc:
        mark = exc.problem_mark
        where = f"{mark.line + 1}행 {mark.column + 1}열" if mark else "위치 모름"
        raise ConfigError(f"YAML 문법 오류({where})") from None
    except yaml.YAMLError:
        raise ConfigError("YAML 문법 오류") from None
    _check_json_types(data, "")
    return data


# --- 원문 텍스트 검사 --------------------------------------------------------------


def scan_raw_text(text: str) -> list[str]:
    """원문(주석 포함)을 **줄 전체** 단위로 검사한다. 파서는 주석을 버리므로 텍스트로 본다.

    주석을 따로 찾지 않는다(따옴표·이스케이프 규칙에 흔들리지 않게). 줄 어디에든 아래가 있으면 오류:

    - 주소(URL·IPv4·호스트:포트). 문장 속·조사·문장 부호가 붙어도 찾는다.
    - 흔한 비밀값 꼴(`Bearer …`, `sk-…`, `hf_…`)
    - `접속칸: 값` 짝인데 값이 `${VAR}`가 아니고 키처럼 생김(주석 속 `# api_key: sk-live-…` 등)
    """
    errors: list[str] = []
    for lineno, line in enumerate(normalize_text(text).split("\n"), start=1):
        if has_address(line):
            errors.append(f"{lineno}행: 주소가 있습니다(주석 포함). 접속 칸에 ${{VAR}}로 쓰세요")
        elif has_secret_pattern(line):
            errors.append(f"{lineno}행: 비밀값으로 보이는 값이 있습니다(주석 포함). 지우세요")
        else:
            for m in _PAIR_RE.finditer(line):
                value = m.group(2).strip("\"'`,;)]}")
                if (
                    is_connection_field(m.group(1))
                    and not _VAR_RE.match(value)
                    and _TOKEN_LIKE_RE.match(value)
                ):
                    errors.append(
                        f"{lineno}행: 접속 칸 값이 직접 적혀 있습니다(주석 포함). 지우세요"
                    )
                    break
    return errors


# --- 스키마 -----------------------------------------------------------------------

NonEmpty = Annotated[str, Field(min_length=1)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RunSettings(_Strict):
    """generator·judge의 실행 설정(해시 밖). U3(시간 제한·재시도)·U6(동시 처리)이 읽는다."""

    concurrency: int = Field(default=4, ge=1)
    timeout_s: float = Field(default=180.0, gt=0)
    max_retries: int = Field(default=2, ge=0)


class TopRunSettings(_Strict):
    """실행 전체 설정(해시 밖). U6이 읽는다."""

    invalid_ratio_max: float = Field(default=0.05, ge=0, le=1)


class ModelMeta(_Strict):
    """모델 선택용 정보(설계 §4). 해시에 넣지 않고 비교·비용 계산(U8)에만 쓴다."""

    price_per_1m_input: float | None = Field(default=None, ge=0)
    price_per_1m_output: float | None = Field(default=None, ge=0)
    gpu_cost_per_hour: float | None = Field(default=None, ge=0)
    vram_gb: float | None = Field(default=None, gt=0)
    license: NonEmpty | None = None
    deploy: Literal["cloud", "on_prem"] | None = None
    air_gapped: bool | None = None
    end_of_life: Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2}$")] | None = None


RESERVED_PARAMS = frozenset({"model", "messages", "stream", "response_format"})
"""`params`에 쓸 수 없는 키. 클라이언트가 채우거나(model·messages), 쓰지 않거나(stream),
문항 데이터에서만 받는(response_format) 값이다(U3)."""


def normalize_base_url(value: str) -> str:
    """`base_url`을 검사하고 끝의 `/` 하나를 뗀다. `/v1`까지 포함한 주소로 받는다.

    틀리면 ValueError를 낸다. 메시지에는 이유만 넣고 값은 넣지 않는다.
    """
    if any(c.isspace() for c in value):
        raise ValueError("공백이 있습니다")
    if "?" in value or "#" in value:
        raise ValueError("쿼리(?)·조각(#)은 쓰지 않습니다")
    # urlsplit보다 엄격한 httpx 해석(IPv4 범위·IDNA 등)도 통과해야 한다. 아니면 첫 호출에서
    # httpx.InvalidURL(메시지에 호스트 값)이 난다.
    # except 안에서 다시 내지 않는다(원래 예외가 __context__로 이어지지 않게).
    readable = True
    try:
        parts = urlsplit(value)
        _ = parts.port  # 포트가 숫자가 아니면 여기서 ValueError
        httpx.URL(value)
    except (ValueError, httpx.InvalidURL):
        readable = False
    if not readable:
        raise ValueError("주소로 읽을 수 없습니다")
    if parts.scheme not in ("http", "https"):
        raise ValueError("스킴은 http·https만 씁니다")
    if "@" in parts.netloc:
        raise ValueError("주소 안에 계정(user:pw@)을 넣지 않습니다. 키는 api_key 칸에 씁니다")
    if not parts.hostname:
        raise ValueError("호스트가 없습니다")
    return value[:-1] if value.endswith("/") else value


class ModelSettings(_Strict):
    """생성·판정 모델 설정의 공통 바탕. U3·U7이 상속해 부품의 config_model로 쓴다.

    `run`은 로더가 YAML의 `run`(없으면 기본값)으로 넣어 준다(해시 밖).
    검증 오류 문구에는 입력값을 넣지 않는다(`hide_input_in_errors`, 주소가 새지 않게).
    """

    model_config = ConfigDict(hide_input_in_errors=True)

    model: NonEmpty
    revision: NonEmpty | None = None
    params: dict[str, JsonValue] = Field(default_factory=dict)
    base_url: SecretStr
    api_key: SecretStr | None = None
    run: RunSettings = Field(default_factory=RunSettings)

    @field_validator("base_url")
    @classmethod
    def _check_base_url(cls, value: SecretStr) -> SecretStr:
        raw = value.get_secret_value()
        if raw == UNSET_PLACEHOLDER:  # --no-env: 해시만 낸다. 클라이언트가 만들 때 거부한다
            return value
        try:
            return SecretStr(normalize_base_url(raw))
        except ValueError as exc:
            raise PydanticCustomError(
                "base_url_format", "base_url 형식 오류: {reason}", {"reason": str(exc)}
            ) from None

    @field_validator("params")
    @classmethod
    def _no_reserved_params(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        bad = sorted(RESERVED_PARAMS & set(value))
        if bad:
            raise PydanticCustomError(
                "params_reserved_key",
                "params에 쓸 수 없는 키입니다({keys})",
                {"keys": ", ".join(bad)},
            )
        return value

    def fingerprint_items(self) -> dict[str, Any]:
        """fingerprint에 넣을 값: model·revision·params(None은 뺌). 접속 칸은 넣지 않는다."""
        return drop_none({"model": self.model, "revision": self.revision, "params": self.params})


class GeneratorSettings(ModelSettings):
    """생성기 설정. `prompt_template`은 로더가 넣는다(YAML에 쓰면 오류)."""

    prompt_template: str


class _RawConfig(_Strict):
    name: NonEmpty
    level: NonEmpty
    corpus: dict[str, Any]
    chunker: dict[str, Any]
    retriever: dict[str, Any]
    embedder: dict[str, Any] | None = None
    vector_store: dict[str, Any] | None = None
    reranker: dict[str, Any] | None = None
    generators: dict[str, dict[str, Any]] = Field(min_length=1)
    prompt: NonEmpty
    judges: list[dict[str, Any]] = Field(min_length=1)
    questions: NonEmpty
    repeats: int = Field(ge=1)
    run: TopRunSettings = Field(default_factory=TopRunSettings)


def _pydantic_errors(exc: ValidationError, prefix: str = "") -> list[str]:
    """pydantic 오류를 칸 경로·종류로만 바꾼다(입력값은 넣지 않음)."""
    out = []
    for e in exc.errors(include_input=False, include_url=False, include_context=False):
        loc = ".".join(str(p) for p in e["loc"])
        path = f"{prefix}.{loc}" if prefix and loc else (prefix or loc or "(구성)")
        out.append(f"{path}: {e['type']}")
    return out


# --- 읽은 결과 --------------------------------------------------------------------


@dataclass(frozen=True)
class PromptText:
    text: str = field(repr=False)
    sha256: str


@dataclass(frozen=True)
class ArmConfig:
    name: str
    generator: Any
    meta: ModelMeta | None
    run: RunSettings


JudgeRole = Literal["primary", "rules", "secondary"]


@dataclass(frozen=True)
class JudgeConfig:
    judge_id: str
    judge_type: str
    role: JudgeRole
    component: Any
    meta: ModelMeta | None
    run: RunSettings


@dataclass
class LoadedConfig:
    """읽고 검증한 구성. 접속 값은 부품 인스턴스 안(SecretStr)에만 있다."""

    path: Path
    raw_bytes: bytes = field(repr=False)
    """원문 바이트 그대로(BOM·줄바꿈 포함). 실행 기록(U6)은 이것을 바이트로 쓴다."""
    raw_text: str = field(repr=False)
    """검사·파싱용 텍스트(BOM 제외)."""
    raw: dict[str, Any] = field(repr=False)
    name: str
    level: str
    parts: dict[str, Any]
    arms: dict[str, ArmConfig]
    judges: list[JudgeConfig]
    prompt: PromptText
    questions_path: Path
    repeats: int
    run: TopRunSettings
    warnings: list[str]

    def arm_prompt(self, arm: str) -> PromptText:
        """arm이 쓰는 프롬프트. R0는 모든 arm이 top-level 하나를 같이 쓴다(R4에서 arm별)."""
        if arm not in self.arms:
            raise KeyError(arm)
        return self.prompt


# --- 부품 항목 검사·치환 ------------------------------------------------------------


@dataclass
class _Spec:
    path: str
    kind: ComponentKind
    type: str
    settings: dict[str, Any]
    meta: ModelMeta | None
    run: RunSettings | None
    extra: dict[str, Any]


def _split_spec(raw: Any, path: str, kind: ComponentKind, errors: list[str]) -> _Spec | None:
    if not isinstance(raw, dict):
        errors.append(f"{path}: 부품 항목은 매핑이어야 합니다")
        return None
    ctype = raw.get("type")
    if not isinstance(ctype, str) or not ctype:
        errors.append(f"{path}.type: 등록 이름이 필요합니다")
        return None
    allowed_reserved = {"type", "meta"}
    if kind in ("generator", "judge"):
        allowed_reserved.add("run")
    if kind == "judge":
        allowed_reserved |= {"id", "role"}
    if kind == "generator" and "prompt" in raw:
        errors.append(f"{path}.prompt: arm별 프롬프트는 R4에서 지원합니다")
    for key in raw:
        if key in RESERVED_KEYS and key not in allowed_reserved:
            if key == "prompt_template":
                errors.append(
                    f"{path}.prompt_template: 로더가 넣는 예약 칸입니다. YAML에 쓰지 마세요"
                )
            else:
                errors.append(f"{path}.{key}: 이 부품에서 쓸 수 없는 예약 칸입니다")
    meta = run = None
    try:
        if raw.get("meta") is not None:
            meta = ModelMeta.model_validate(raw["meta"])
        elif "meta" in raw:
            errors.append(f"{path}.meta: 비어 있습니다")
    except ValidationError as exc:
        errors += _pydantic_errors(exc, f"{path}.meta")
    if "run" in allowed_reserved:
        try:
            run = RunSettings.model_validate(raw.get("run") or {})
        except ValidationError as exc:
            errors += _pydantic_errors(exc, f"{path}.run")
    skip = RESERVED_KEYS | ({"prompt"} if kind == "generator" else set())
    settings = {k: v for k, v in raw.items() if k not in skip}
    extra = {k: raw[k] for k in ("id", "role") if k in raw}
    return _Spec(path, kind, ctype, settings, meta, run, extra)


def _check_values(
    obj: Any, path: str, allow: frozenset[str], errors: list[str], *, top: bool = True
) -> None:
    """`${VAR}` 위치, 접속 칸 리터럴, 주소, 비밀값 꼴, 빈 값을 검사한다.

    접속 칸은 부품 설정 **맨 위**에만 둘 수 있다(중첩된 칸의 치환 값은 일반 문자열로 남아
    repr로 샐 수 있으므로). `top`은 지금 보는 dict가 부품 설정 맨 위인지다.
    """
    if isinstance(obj, dict):
        for key, value in obj.items():
            child = f"{path}.{key}"
            if has_address(key) or has_secret_pattern(key):
                errors.append(f"{path}: 키에 주소·비밀값이 있습니다")
                continue
            if is_connection_field(key, allow):
                if not top:
                    errors.append(f"{child}: 접속 칸은 부품 설정 맨 위에만 둡니다")
                elif not (isinstance(value, str) and _VAR_RE.match(value)):
                    errors.append(f"{child}: 접속 칸에는 ${{VAR}}만 씁니다(직접 쓴 값은 거부)")
                continue
            _check_values(value, child, allow, errors, top=False)
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            _check_values(value, f"{path}[{i}]", allow, errors, top=False)
    elif isinstance(obj, str):
        if obj == "":
            errors.append(f"{path}: 비어 있습니다")
        elif "${" in obj:
            errors.append(f"{path}: ${{VAR}}는 접속 칸(base_url·api_key 등)에만 씁니다")
        elif has_address(obj):
            errors.append(f"{path}: 주소가 있습니다. 접속 칸에 ${{VAR}}로 쓰세요")
        elif has_secret_pattern(obj):
            errors.append(f"{path}: 비밀값으로 보이는 값이 있습니다. 접속 칸에 ${{VAR}}로 쓰세요")


def _is_secret_field(model: Any, name: str) -> bool:
    """config_model의 칸이 SecretStr(또는 SecretStr | None)인지 본다."""
    info = getattr(model, "model_fields", {}).get(name)
    if info is None:
        return False
    ann = info.annotation
    return ann is SecretStr or SecretStr in get_args(ann)


def _substitute(obj: Any, env: Mapping[str, str], allow_missing: bool, missing: set[str]) -> Any:
    if isinstance(obj, dict):
        return {k: _substitute(v, env, allow_missing, missing) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_substitute(v, env, allow_missing, missing) for v in obj]
    if isinstance(obj, str):
        m = _VAR_RE.match(obj)
        if m:
            value = env.get(m.group(1), "")
            if not value:
                missing.add(m.group(1))
                return UNSET_PLACEHOLDER if allow_missing else ""
            return value
    return obj


def _relative_file(base: Path, value: str, path: str, errors: list[str]) -> Path | None:
    p = PurePath(value)
    if p.is_absolute() or re.match(r"^[A-Za-z]:", value) or value.startswith(("~", "\\")):
        errors.append(f"{path}: YAML 파일 기준 상대 경로만 씁니다")
        return None
    full = base / value
    if not full.is_file():
        errors.append(f"{path}: 파일이 없습니다({value})")
        return None
    return full


# --- 진입점 ----------------------------------------------------------------------


def load_config(
    path: Path,
    registry: Registry,
    *,
    env: Mapping[str, str] | None = None,
    allow_missing_env: bool = False,
) -> LoadedConfig:
    """구성 YAML을 읽고 검증해 부품을 만든다. 오류는 모두 모아 ConfigError 하나로 낸다."""
    import os

    env = os.environ if env is None else env
    raw_bytes = path.read_bytes()
    try:
        raw_text = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ConfigError("구성 파일이 UTF-8이 아닙니다") from None

    # 2. 원문 검사
    errors = scan_raw_text(raw_text)
    if errors:
        raise ConfigError(errors)

    # 3. 파싱·최상위 스키마
    data = parse_yaml(raw_text)
    if not isinstance(data, dict):
        raise ConfigError("구성 최상위는 매핑이어야 합니다")
    try:
        top = _RawConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(_pydantic_errors(exc)) from None
    if top.level not in ALL_LEVELS:
        raise ConfigError(f"level: 모르는 값입니다. 가능한 값: {', '.join(ALL_LEVELS)}")
    if top.level not in SUPPORTED_LEVELS:
        raise ConfigError(f"level: {top.level}은(는) R2 이후 지원합니다. R0는 generation_only")
    warnings: list[str] = []
    _check_values({"name": top.name}, "", frozenset(), errors)

    # 4. 쓰는 부품·안 쓰는 부품
    used = USED_PARTS[top.level]
    specs: list[_Spec] = []
    for kind in ("corpus", "chunker", "embedder", "vector_store", "retriever", "reranker"):
        raw_part = getattr(top, kind)
        if kind in used:
            spec = _split_spec(raw_part, kind, kind, errors)
            if spec:
                specs.append(spec)
        elif raw_part is not None:
            ctype = raw_part.get("type") if isinstance(raw_part, dict) else None
            if not isinstance(ctype, str) or ctype not in registry.available(kind):
                errors.append(f"{kind}.type: 모르는 {kind} 부품입니다")
            else:
                warnings.append(f"{kind}: {top.level}에서는 쓰지 않아 만들지도 해시하지도 않습니다")
                # 만들지는 않지만 원문에 남으므로 리터럴 키·주소·비밀값은 검사한다(allow 없음).
                # `${VAR}`가 정의돼 있는지는 확인하지 않는다.
                unused = {k: v for k, v in raw_part.items() if k not in RESERVED_KEYS}
                _check_values(unused, kind, frozenset(), errors)

    arm_specs: dict[str, _Spec] = {}
    for arm, raw_gen in top.generators.items():
        if not _ARM_RE.match(arm):
            errors.append(f"generators.{arm}: arm 이름은 영문·숫자·_·-만 씁니다")
            continue
        spec = _split_spec(raw_gen, f"generators.{arm}", "generator", errors)
        if spec:
            arm_specs[arm] = spec

    judge_specs: list[tuple[str, JudgeRole, _Spec]] = []
    for i, raw_judge in enumerate(top.judges):
        spec = _split_spec(raw_judge, f"judges[{i}]", "judge", errors)
        if not spec:
            continue
        jid = spec.extra.get("id", spec.type)
        role = spec.extra.get("role")
        if not isinstance(jid, str) or not _ARM_RE.match(jid):
            errors.append(f"judges[{i}].id: 영문·숫자·_·-만 씁니다")
            continue
        if role not in ("primary", "rules", "secondary"):
            errors.append(f"judges[{i}].role: 필수입니다. primary·rules·secondary 중 하나")
            continue
        judge_specs.append((jid, role, spec))
    ids = [jid for jid, _, _ in judge_specs]
    for dup in sorted({j for j in ids if ids.count(j) > 1}):
        errors.append(f"judges: 판정 ID '{dup}'가 겹칩니다. 같은 type이 둘 이상이면 id를 적으세요")
    primaries = [jid for jid, role, _ in judge_specs if role == "primary"]
    if len(primaries) != 1 and len(judge_specs) == len(top.judges):
        errors.append(
            f"judges: 주 판정(role: primary)은 정확히 하나여야 합니다({len(primaries)}개)"
        )

    # 등록 확인(쓰는 부품만 불러옴)
    all_specs = specs + list(arm_specs.values()) + [s for _, _, s in judge_specs]
    classes: dict[int, type] = {}
    for spec in all_specs:
        try:
            classes[id(spec)] = registry.get(spec.kind, spec.type)
        except RegistryError as exc:
            errors.append(f"{spec.path}.type: {exc}")
    if errors:
        raise ConfigError(errors)

    # 5. level과 검색기 짝
    retriever_spec = next(s for s in specs if s.kind == "retriever")
    if not getattr(classes[id(retriever_spec)], "needs_gold", False):
        errors.append(
            "retriever: generation_only에는 정답 근거 주입 검색기(needs_gold)가 필요합니다"
        )

    # 판정 성격(judge_kind: code·model)을 밝힌 부품은 role과 맞는지 본다.
    for _jid, role, spec in judge_specs:
        judge_kind = getattr(classes[id(spec)], "judge_kind", None)
        if judge_kind == "code" and role != "rules":
            errors.append(f"{spec.path}.role: 코드 규칙 판정은 role: rules여야 합니다")
        elif judge_kind == "model" and role == "rules":
            errors.append(f"{spec.path}.role: 모델 판정은 role: rules가 될 수 없습니다")

    # 6. 값·키 검사
    for spec in all_specs:
        cls = classes[id(spec)]
        allow = frozenset(normalize_name(a) for a in getattr(cls, "fingerprint_allow", frozenset()))
        _check_values(spec.settings, spec.path, allow, errors)
        # 접속 칸을 받는 부품은 config_model에서 SecretStr로 받아야 한다(repr로 새지 않게).
        model = getattr(cls, "config_model", None)
        for key in spec.settings:
            if is_connection_field(key, allow) and not _is_secret_field(model, key):
                errors.append(
                    f"{spec.path}.{key}: 이 부품은 접속 칸을 SecretStr로 받지 않습니다"
                    "(부품의 config_model에서 SecretStr로 선언해야 함)"
                )
        if spec.meta is not None:
            _check_values(
                spec.meta.model_dump(exclude_none=True), f"{spec.path}.meta", allow, errors
            )

    # 7. 프롬프트·문항 경로
    base = path.parent
    prompt_path = _relative_file(base, top.prompt, "prompt", errors)
    questions_path = _relative_file(base, top.questions, "questions", errors)
    if errors:
        raise ConfigError(errors)
    assert prompt_path is not None and questions_path is not None
    prompt_text = normalize_text(prompt_path.read_text(encoding="utf-8-sig"))
    prompt = PromptText(text=prompt_text, sha256=sha256_hex(prompt_text))

    # 8. 치환
    missing: set[str] = set()
    resolved = {id(s): _substitute(s.settings, env, allow_missing_env, missing) for s in all_specs}
    if missing and not allow_missing_env:
        raise ConfigError(
            f"환경 변수가 없거나 비어 있습니다: {', '.join(sorted(missing))}"
            "(값을 넣거나, 해시만 보려면 --no-env)"
        )
    if missing:
        warnings.append(f"환경 변수 없이 자리표시 값으로 채웠습니다: {', '.join(sorted(missing))}")

    # 9. 부품 생성
    def create(spec: _Spec, extra_settings: dict[str, Any] | None = None) -> Any:
        settings = {**resolved[id(spec)], **(extra_settings or {})}
        # config_model에 run 칸이 있는 부품(ModelSettings 상속)에만 실행 설정을 넣는다(해시 밖).
        model = getattr(classes[id(spec)], "config_model", None)
        if spec.run is not None and "run" in getattr(model, "model_fields", {}):
            settings["run"] = spec.run
        try:
            return registry.create(spec.kind, spec.type, settings)
        except RegistryError as exc:
            errors.append(f"{spec.path}: {exc}")
            return None

    parts: dict[str, Any] = {s.kind: create(s) for s in specs}
    arms = {
        arm: ArmConfig(
            name=arm,
            generator=create(spec, {"prompt_template": prompt.text}),
            meta=spec.meta,
            run=spec.run or RunSettings(),
        )
        for arm, spec in arm_specs.items()
    }
    judges = [
        JudgeConfig(
            judge_id=jid,
            judge_type=spec.type,
            role=role,
            component=create(spec),
            meta=spec.meta,
            run=spec.run or RunSettings(),
        )
        for jid, role, spec in judge_specs
    ]
    if errors:
        raise ConfigError(errors)

    # 10. fingerprint 사전 검사(경로형 모델 이름 등은 실행 전에 칸 경로와 함께 알린다)
    built: list[tuple[str, Any]] = [(s.path, parts[s.kind]) for s in specs]
    built += [(spec.path, arms[arm].generator) for arm, spec in arm_specs.items()]
    built += [(spec.path, j.component) for (_, _, spec), j in zip(judge_specs, judges, strict=True)]
    for spec_path, obj in built:
        try:
            registry.component_fingerprint(obj)
        except (FingerprintError, RegistryError) as exc:
            errors.append(f"{spec_path}: {exc}")
    if errors:
        raise ConfigError(errors)

    return LoadedConfig(
        path=path,
        raw_bytes=raw_bytes,
        raw_text=raw_text,
        raw=data,
        name=top.name,
        level=top.level,
        parts=parts,
        arms=arms,
        judges=judges,
        prompt=prompt,
        questions_path=questions_path,
        repeats=top.repeats,
        run=top.run,
        warnings=warnings,
    )


__all__ = [
    "CONNECTION_FIELDS",
    "CONNECTION_PREFIXES",
    "CONNECTION_SUFFIXES",
    "RESERVED_PARAMS",
    "UNSET_PLACEHOLDER",
    "ArmConfig",
    "ConfigError",
    "GeneratorSettings",
    "JudgeConfig",
    "LoadedConfig",
    "ModelMeta",
    "ModelSettings",
    "PromptText",
    "RunSettings",
    "TopRunSettings",
    "is_connection_field",
    "load_config",
    "normalize_base_url",
    "parse_yaml",
    "scan_raw_text",
]
