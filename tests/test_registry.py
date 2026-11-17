"""레지스트리 시험(entry point 없이): 표시·등록·검사·생성·fingerprint."""

from __future__ import annotations

import types
from collections.abc import Iterator, Sequence
from typing import Any, ClassVar, Literal

import pytest

from ragexp.components import protocols as P
from ragexp.fingerprint import FingerprintError
from ragexp.registry import (
    KINDS,
    PROTOCOLS,
    REQUIRED_METHODS,
    ComponentKind,
    Registry,
    RegistryError,
    check_component_class,
    default_registry,
    marker_of,
    register,
)
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
    Usage,
    Verdict,
)

# --- 9개 종류의 최소 가짜 구현 -----------------------------------------------------


class _Base:
    version: ClassVar[str] = "1"

    def __init__(self, **config: Any) -> None:
        self.config = config

    def fingerprint(self) -> dict[str, Any]:
        return dict(self.config)


@register("corpus", "fake")
class FakeCorpus(_Base):
    def documents(self) -> Iterator[Document]:
        yield Document(doc_id="d1")


@register("chunker", "fake")
class FakeChunker(_Base):
    def chunk(self, doc: Document) -> list[Chunk]:
        return [Chunk(chunk_id=f"{doc.doc_id}:1", doc_id=doc.doc_id, text=doc.text)]


@register("embedder", "fake")
class FakeEmbedder(_Base):
    def embed(
        self, texts: Sequence[str], *, kind: Literal["query", "document"]
    ) -> list[list[float]]:
        return [[float(len(t))] for t in texts]

    def dimension(self) -> int:
        return 1


@register("vector_store", "fake")
class FakeVectorStore(_Base):
    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        return None

    def search(self, vector: Sequence[float], k: int) -> list[Hit]:
        return []


@register("retriever", "fake")
class FakeRetriever(_Base):
    needs_gold: ClassVar[bool] = True

    def retrieve(self, query: RetrievalQuery) -> list[Hit]:
        return []


@register("reranker", "fake")
class FakeReranker(_Base):
    def rerank(self, query: str, hits: Sequence[Hit]) -> list[Hit]:
        return list(hits)


@register("generator", "fake")
class FakeGenerator(_Base):
    def generate(self, inp: GenerationInput, context: Sequence[Chunk]) -> Answer:
        return Answer(text="a", finish_reason="stop", usage=Usage(latency_ms=0))


@register("judge", "fake")
class FakeJudge(_Base):
    def judge(self, question: JudgeQuestion, response: JudgeInput) -> Verdict:
        return Verdict(judge_id="j", judge_type="fake", status="ok")


@register("question_generator", "fake")
class FakeQuestionGenerator(_Base):
    def draft(self, chunks: Sequence[Chunk], qtype: str, n: int) -> list[Question]:
        return []


FAKES: dict[ComponentKind, type] = {
    "corpus": FakeCorpus,
    "chunker": FakeChunker,
    "embedder": FakeEmbedder,
    "vector_store": FakeVectorStore,
    "retriever": FakeRetriever,
    "reranker": FakeReranker,
    "generator": FakeGenerator,
    "judge": FakeJudge,
    "question_generator": FakeQuestionGenerator,
}


@pytest.fixture
def reg() -> Registry:
    r = Registry(use_entry_points=False)
    for cls in FAKES.values():
        r.add(cls)
    return r


# --- Protocol 충족 ----------------------------------------------------------------


def test_fakes_cover_all_kinds() -> None:
    assert set(FAKES) == set(KINDS) == set(PROTOCOLS)


@pytest.mark.parametrize("kind", KINDS)
def test_fake_satisfies_protocol(kind: ComponentKind, reg: Registry) -> None:
    obj = reg.create(kind, "fake", {})
    assert isinstance(obj, PROTOCOLS[kind])
    assert isinstance(obj, P.Component)
    assert reg.available(kind) == ["fake"]


MISSING_CASES = [(k, m) for k in KINDS for m in REQUIRED_METHODS[k]]


@pytest.mark.parametrize(
    ("kind", "method"), MISSING_CASES, ids=[f"{k}-{m}" for k, m in MISSING_CASES]
)
def test_missing_method_rejected(kind: ComponentKind, method: str) -> None:
    broken = type("Broken", (FAKES[kind],), {method: None})
    with pytest.raises(RegistryError, match=method):
        register(kind, "broken")(broken)


@pytest.mark.parametrize("version", [None, "", 1])
def test_version_required(version: object) -> None:
    attrs: dict[str, Any] = {"retrieve": lambda self, q: [], "fingerprint": lambda self: {}}
    if version is not None:
        attrs["version"] = version
    with pytest.raises(RegistryError, match="version"):
        register("retriever", "noversion")(type("NoVersion", (), attrs))


def test_bad_optional_attributes_rejected() -> None:
    with pytest.raises(RegistryError, match="fingerprint_allow"):
        register("generator", "x")(
            type("X", (FakeGenerator,), {"fingerprint_allow": {"stop_token"}})
        )
    with pytest.raises(RegistryError, match="풀 수 없는"):
        register("generator", "y")(
            type("Y", (FakeGenerator,), {"fingerprint_allow": frozenset({"api_key"})})
        )
    with pytest.raises(RegistryError, match="needs_gold"):
        check_component_class(type("Y", (FakeRetriever,), {"needs_gold": "yes"}), "retriever")


@pytest.mark.parametrize("name", ["", "Fake", "fake-x", "1fake", "a.b"])
def test_bad_names_rejected(name: str) -> None:
    with pytest.raises(RegistryError):
        register("retriever", name)


def test_unknown_kind_rejected() -> None:
    with pytest.raises(RegistryError, match="모르는 부품 종류"):
        register("index", "x")  # type: ignore[arg-type]
    with pytest.raises(RegistryError, match="모르는 부품 종류"):
        Registry(use_entry_points=False).available("index")  # type: ignore[arg-type]


# --- 표시와 등록 -------------------------------------------------------------------


def test_decorator_only_marks() -> None:
    @register("retriever", "only_marked")
    class Only(FakeRetriever):
        pass

    assert marker_of(Only) == ("retriever", "only_marked")
    with pytest.raises(RegistryError, match="only_marked"):
        Registry(use_entry_points=False).get("retriever", "only_marked")


def test_inherited_marker_is_ignored() -> None:
    class Child(FakeRetriever):
        pass

    assert marker_of(Child) is None
    with pytest.raises(RegistryError, match="@register"):
        Registry(use_entry_points=False).add(Child)


def test_add_module() -> None:
    mod = types.ModuleType("fake_mod")
    mod.FakeRetriever = FakeRetriever  # type: ignore[attr-defined]
    mod.NotComponent = int  # type: ignore[attr-defined]
    r = Registry(use_entry_points=False)
    r.add_module(mod)
    assert r.get("retriever", "fake") is FakeRetriever


def test_duplicate_rejected_same_class_ignored(reg: Registry) -> None:
    reg.add(FakeRetriever)  # 같은 클래스: 무시

    @register("retriever", "fake")
    class Other(FakeRetriever):
        pass

    with pytest.raises(RegistryError, match="이미 등록"):
        reg.add(Other)


def test_unknown_name_lists_and_suggests(reg: Registry) -> None:
    with pytest.raises(RegistryError) as info:
        reg.get("retriever", "fak")
    msg = str(info.value)
    assert "등록된 이름: fake" in msg and "혹시: fake" in msg


def test_registries_are_independent() -> None:
    a, b = Registry(use_entry_points=False), Registry(use_entry_points=False)
    a.add(FakeRetriever)
    assert a.available("retriever") == ["fake"] and b.available("retriever") == []


def test_default_registry_is_single() -> None:
    assert default_registry() is default_registry()


# --- 생성 ------------------------------------------------------------------------


def test_create_passes_config(reg: Registry) -> None:
    obj = reg.create("generator", "fake", {"model": "m", "temperature": 0})
    assert obj.config == {"model": "m", "temperature": 0}


def test_create_rejects_meta(reg: Registry) -> None:
    with pytest.raises(RegistryError, match="meta"):
        reg.create("generator", "fake", {"model": "m", "meta": {"vram_gb": 31}})


def test_create_error_hides_config_values() -> None:
    @register("generator", "strict")
    class Strict(FakeGenerator):
        def __init__(self, model: str) -> None:
            raise ValueError(f"cannot use {model}")

    r = Registry(use_entry_points=False)
    r.add(Strict)
    config = {"model": "m", "base_url": "http://localhost:1", "api_key": "dummy-key"}
    with pytest.raises(RegistryError) as info:
        r.create("generator", "strict", config)
    msg = str(info.value)
    assert "http://localhost:1" not in msg and "dummy-key" not in msg
    assert "api_key" in msg and "base_url" in msg  # 키 이름만
    assert info.value.__cause__ is None and info.value.__suppress_context__


# --- fingerprint ------------------------------------------------------------------


def test_component_fingerprint_adds_type_and_version(reg: Registry) -> None:
    obj = reg.create("embedder", "fake", {"model": "kure", "dimension": 1024})
    assert reg.component_fingerprint(obj) == {
        "type": "fake",
        "version": "1",
        "model": "kure",
        "dimension": 1024,
    }


def test_same_config_same_fingerprint(reg: Registry) -> None:
    a = reg.create("generator", "fake", {"model": "m", "params": {"temperature": 0}})
    b = reg.create("generator", "fake", {"model": "m", "params": {"temperature": 0}})
    assert reg.component_fingerprint(a) == reg.component_fingerprint(b)


@pytest.mark.parametrize("key", ["type", "version"])
def test_component_cannot_set_reserved_keys(reg: Registry, key: str) -> None:
    obj = reg.create("generator", "fake", {key: "x"})
    with pytest.raises(RegistryError, match=key):
        reg.component_fingerprint(obj)


def test_fingerprint_rules_apply(reg: Registry) -> None:
    obj = reg.create("generator", "fake", {"base_url": "dummy"})
    with pytest.raises(FingerprintError):
        reg.component_fingerprint(obj)


def test_fingerprint_allow_declared_by_component() -> None:
    @register("generator", "with_stop")
    class WithStop(FakeGenerator):
        fingerprint_allow: ClassVar[frozenset[str]] = frozenset({"stop_token"})

    r = Registry(use_entry_points=False)
    r.add(WithStop)
    obj = r.create("generator", "with_stop", {"stop_token": "</s>"})
    assert r.component_fingerprint(obj)["stop_token"] == "</s>"


def test_unregistered_object_rejected(reg: Registry) -> None:
    @register("generator", "elsewhere")
    class Elsewhere(FakeGenerator):
        pass

    with pytest.raises(RegistryError, match="등록된 부품이 아닙니다"):
        reg.component_fingerprint(Elsewhere())
    with pytest.raises(RegistryError, match="등록된 부품이 아닙니다"):
        reg.component_fingerprint(object())


def test_component_fingerprint_drops_none(reg: Registry) -> None:
    """None 칸은 모든 부품 fingerprint에서 뺀다(U2 구현 검토 지적 3)."""
    obj = reg.create("generator", "fake", {"a": 1, "new_opt": None, "c": [{"d": None, "e": 2}]})
    assert reg.component_fingerprint(obj) == {
        "type": "fake",
        "version": "1",
        "a": 1,
        "c": [{"e": 2}],
    }
    plain = reg.create("generator", "fake", {"a": 1, "c": [{"e": 2}]})
    assert reg.component_fingerprint(obj) == reg.component_fingerprint(plain)
