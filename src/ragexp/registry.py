"""부품 등록·생성.

- `@register(kind, name)`은 클래스에 표시만 붙이고 Protocol 충족을 검사한다.
  전역 상태는 바꾸지 않는다.
- `Registry`는 표시된 클래스를 직접 받거나(`add`), entry point로 외부 부품을 찾는다.

entry point 규칙(그룹 `ragexp.components`):

    [ragexp.components]
    retriever.bm25 = my_pkg.retrievers:Bm25Retriever

이름이 `<kind>.<name>`이라서 **불러오지 않고도** 어떤 부품이 있는지 안다(`available`).
실제로 쓰는 부품만 그때 불러오고(`get`·`create`), 실패하면 플러그인 이름을 담아 멈춘다.
쓰지 않는 플러그인의 문제(이름 형식 오류, 불러오기 실패)는 경고만 낸다.
"""

from __future__ import annotations

import difflib
import re
import warnings
from collections.abc import Callable, Iterable, Mapping
from importlib import metadata
from types import ModuleType
from typing import Any, Final, Literal, TypeVar, cast, get_args

from ragexp.components.protocols import (
    Chunker,
    Corpus,
    Embedder,
    Generator,
    Judge,
    QuestionGenerator,
    Reranker,
    Retriever,
    VectorStore,
)
from ragexp.fingerprint import FORBIDDEN_NAMES, check_fingerprint, normalize_name

ComponentKind = Literal[
    "corpus",
    "chunker",
    "embedder",
    "vector_store",
    "retriever",
    "reranker",
    "generator",
    "judge",
    "question_generator",
]
KINDS: Final[tuple[ComponentKind, ...]] = get_args(ComponentKind)

PROTOCOLS: Final[dict[ComponentKind, type]] = {
    "corpus": Corpus,
    "chunker": Chunker,
    "embedder": Embedder,
    "vector_store": VectorStore,
    "retriever": Retriever,
    "reranker": Reranker,
    "generator": Generator,
    "judge": Judge,
    "question_generator": QuestionGenerator,
}

# Protocol마다 필요한 메서드(3.12에는 Protocol 멤버를 얻는 공개 API가 없어 직접 적는다).
REQUIRED_METHODS: Final[dict[ComponentKind, tuple[str, ...]]] = {
    "corpus": ("documents", "fingerprint"),
    "chunker": ("chunk", "fingerprint"),
    "embedder": ("embed", "dimension", "fingerprint"),
    "vector_store": ("add", "search", "fingerprint"),
    "retriever": ("retrieve", "fingerprint"),
    "reranker": ("rerank", "fingerprint"),
    "generator": ("generate", "fingerprint"),
    "judge": ("judge", "fingerprint"),
    "question_generator": ("draft", "fingerprint"),
}

ENTRY_POINT_GROUP: Final = "ragexp.components"
MARKER: Final = "__ragexp_component__"
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_RESERVED_FP_KEYS = frozenset({"type", "version"})

T = TypeVar("T", bound=type)


class RegistryError(Exception):
    """등록·조회·생성 오류. 실행 전에 멈추게 한다."""


class RegistryWarning(UserWarning):
    """쓰지 않는 플러그인의 문제. 실행은 계속된다."""


def _check_kind(kind: str) -> ComponentKind:
    if kind not in KINDS:
        raise RegistryError(f"모르는 부품 종류입니다: {kind!r}. 가능한 종류: {', '.join(KINDS)}")
    return kind


def _check_name(name: str) -> None:
    if not _NAME_RE.match(name):
        raise RegistryError(f"부품 이름은 소문자·숫자·밑줄만 씁니다(첫 글자는 소문자): {name!r}")


def check_component_class(cls: type, kind: ComponentKind) -> None:
    """클래스가 kind의 Protocol을 충족하는지 검사한다(메서드 표, version 필수, 선택 속성 형식)."""
    missing = [m for m in REQUIRED_METHODS[kind] if not callable(getattr(cls, m, None))]
    if missing:
        raise RegistryError(
            f"{cls.__qualname__}은(는) {kind} 부품에 필요한 메서드가 없습니다: {', '.join(missing)}"
        )
    version = getattr(cls, "version", None)
    if not isinstance(version, str) or not version:
        raise RegistryError(
            f"{cls.__qualname__}에 클래스 속성 version(비어 있지 않은 문자열)이 필요합니다"
        )
    allow: object = getattr(cls, "fingerprint_allow", frozenset())
    if not isinstance(allow, frozenset) or not all(isinstance(a, str) for a in allow):
        raise RegistryError(f"{cls.__qualname__}.fingerprint_allow는 frozenset[str]이어야 합니다")
    locked = sorted(a for a in allow if normalize_name(a) in FORBIDDEN_NAMES)
    if locked:
        raise RegistryError(
            f"{cls.__qualname__}.fingerprint_allow로 풀 수 없는 이름입니다: {', '.join(locked)}"
            "(주소·비밀값·meta)"
        )
    if kind == "retriever" and not isinstance(getattr(cls, "needs_gold", False), bool):
        raise RegistryError(f"{cls.__qualname__}.needs_gold는 bool이어야 합니다")


def register(kind: ComponentKind, name: str) -> Callable[[T], T]:
    """부품 클래스에 (kind, name) 표시를 붙인다. 등록은 Registry가 한다."""
    checked_kind = _check_kind(kind)
    _check_name(name)

    def deco(cls: T) -> T:
        check_component_class(cls, checked_kind)
        setattr(cls, MARKER, (checked_kind, name))
        return cls

    return deco


def marker_of(cls: type) -> tuple[ComponentKind, str] | None:
    """클래스 자신에 붙은 표시(상속된 표시는 무시)."""
    value = cls.__dict__.get(MARKER)
    return cast("tuple[ComponentKind, str] | None", value)


def _dist_name(ep: metadata.EntryPoint) -> str:
    dist = getattr(ep, "dist", None)
    return dist.name if dist is not None else ep.value.split(":", 1)[0]


class Registry:
    """부품 등록부. 시험은 새 Registry를 만들어 서로 섞이지 않게 한다."""

    def __init__(self, *, use_entry_points: bool = True) -> None:
        self._use_entry_points = use_entry_points
        self._classes: dict[tuple[ComponentKind, str], type] = {}
        self._entry_points: dict[tuple[ComponentKind, str], list[metadata.EntryPoint]] | None = None

    # --- 직접 등록 -----------------------------------------------------------

    def add(self, cls: type) -> None:
        """표시된 클래스를 등록한다."""
        mark = marker_of(cls)
        if mark is None:
            raise RegistryError(f"{cls.__qualname__}에 @register 표시가 없습니다")
        kind, name = mark
        check_component_class(cls, kind)
        existing = self._classes.get(mark)
        if existing is cls:
            return
        if existing is not None:
            raise RegistryError(
                f"{kind} '{name}'이(가) 이미 등록돼 있습니다"
                f"({existing.__module__}.{existing.__qualname__})"
            )
        if mark in self._discover():
            providers = ", ".join(_dist_name(ep) for ep in self._discover()[mark])
            raise RegistryError(
                f"{kind} '{name}'은(는) 플러그인({providers})이 이미 제공합니다. 이름이 겹칩니다"
            )
        self._classes[mark] = cls

    def add_module(self, module: ModuleType) -> None:
        """모듈 안에서 표시된 클래스를 모두 등록한다."""
        for obj in vars(module).values():
            if isinstance(obj, type) and marker_of(obj) is not None:
                self.add(obj)

    # --- entry point ---------------------------------------------------------

    def _discover(self) -> dict[tuple[ComponentKind, str], list[metadata.EntryPoint]]:
        """entry point를 이름으로만 찾는다(불러오지 않음). 레지스트리마다 한 번."""
        if self._entry_points is not None:
            return self._entry_points
        found: dict[tuple[ComponentKind, str], list[metadata.EntryPoint]] = {}
        if self._use_entry_points:
            for ep in metadata.entry_points(group=ENTRY_POINT_GROUP):
                kind, _, name = ep.name.partition(".")
                if kind not in KINDS or not _NAME_RE.match(name):
                    warnings.warn(
                        f"플러그인 '{_dist_name(ep)}'의 entry point 이름 '{ep.name}'이 "
                        "'<kind>.<name>' 형식이 아니라 건너뜁니다",
                        RegistryWarning,
                        stacklevel=2,
                    )
                    continue
                found.setdefault((kind, name), []).append(ep)
        self._entry_points = found
        return found

    def _load(self, key: tuple[ComponentKind, str], ep: metadata.EntryPoint) -> type:
        kind = key[0]
        plugin = _dist_name(ep)
        try:
            obj = ep.load()
        except Exception as exc:
            raise RegistryError(
                f"플러그인 '{plugin}'의 부품 '{ep.name}'을(를) 불러오지 못했습니다: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(obj, type):
            raise RegistryError(f"플러그인 '{plugin}'의 '{ep.name}'이(가) 클래스가 아닙니다")
        if marker_of(obj) != key:
            raise RegistryError(
                f"플러그인 '{plugin}'의 '{ep.name}'이(가) 가리키는 클래스의 @register 표시"
                f"({marker_of(obj)})가 entry point 이름과 다릅니다"
            )
        try:
            check_component_class(obj, kind)
        except RegistryError as exc:
            raise RegistryError(f"플러그인 '{plugin}': {exc}") from exc
        self._classes[key] = obj
        return obj

    def load_all(self) -> list[str]:
        """모든 플러그인을 불러와 본다(점검용).

        실패는 경고로 알리고 멈추지 않는다. 실패한 이름 목록을 돌려준다.
        """
        failed: list[str] = []
        for kind, name in sorted(self._discover()):
            try:
                self.get(kind, name)
            except RegistryError as exc:
                warnings.warn(str(exc), RegistryWarning, stacklevel=2)
                failed.append(f"{kind}.{name}")
        return failed

    # --- 조회·생성 -----------------------------------------------------------

    def available(self, kind: ComponentKind) -> list[str]:
        """kind의 부품 이름 목록(플러그인은 불러오지 않고 이름만)."""
        k = _check_kind(kind)
        names = {n for (kk, n) in self._classes if kk == k}
        names |= {n for (kk, n) in self._discover() if kk == k}
        return sorted(names)

    def get(self, kind: ComponentKind, name: str) -> type:
        """부품 클래스를 돌려준다. 플러그인이면 이때 불러온다."""
        k = _check_kind(kind)
        key = (k, name)
        eps = self._discover().get(key, [])
        if len(eps) > 1:
            providers = ", ".join(sorted(_dist_name(ep) for ep in eps))
            raise RegistryError(f"{k} '{name}'을(를) 여러 플러그인이 제공합니다: {providers}")
        cls = self._classes.get(key)
        if cls is not None:
            return cls
        if eps:
            return self._load(key, eps[0])
        known = self.available(k)
        hint = difflib.get_close_matches(name, known, n=3)
        msg = f"모르는 {k} 부품입니다: {name!r}. 등록된 이름: {', '.join(known) or '(없음)'}"
        if hint:
            msg += f". 혹시: {', '.join(hint)}"
        raise RegistryError(msg)

    def create(self, kind: ComponentKind, name: str, config: Mapping[str, Any]) -> Any:
        """설정으로 부품을 만든다. meta는 받지 않는다(U2가 떼어 낸 뒤 넘김).

        생성자 오류 메시지에는 설정 값(주소·키가 있을 수 있음)을 넣지 않고 키 이름만 넣는다.
        """
        cls = self.get(kind, name)
        if "meta" in config:
            raise RegistryError(
                f"{kind} '{name}': 부품 설정에 meta를 넘기지 않습니다(해시 제외 정보)"
            )
        try:
            return cls(**config)
        except Exception as exc:
            keys = ", ".join(sorted(config)) or "(없음)"
            raise RegistryError(
                f"{kind} '{name}'을(를) 만들지 못했습니다({type(exc).__name__}). 설정 키: {keys}"
            ) from None

    def component_fingerprint(self, obj: object) -> dict[str, Any]:
        """부품 객체의 fingerprint에 type·version을 붙여 검사한 값을 돌려준다."""
        cls = type(obj)
        mark = marker_of(cls)
        if mark is None or self._classes.get(mark) is not cls:
            raise RegistryError(f"{cls.__qualname__}은(는) 이 레지스트리에 등록된 부품이 아닙니다")
        kind, name = mark
        fp = obj.fingerprint()  # type: ignore[attr-defined]
        reserved = _RESERVED_FP_KEYS & set(fp)
        if reserved:
            raise RegistryError(
                f"{kind} '{name}': fingerprint에 {sorted(reserved)}를 직접 넣지 않습니다"
                "(레지스트리가 붙임)"
            )
        allow: Iterable[str] = getattr(cls, "fingerprint_allow", frozenset())
        return {"type": name, "version": cls.version, **check_fingerprint(fp, allow)}  # type: ignore[attr-defined]


_default: Registry | None = None


def default_registry() -> Registry:
    """앱이 쓰는 기본 레지스트리(하나)."""
    global _default
    if _default is None:
        _default = Registry()
    return _default
