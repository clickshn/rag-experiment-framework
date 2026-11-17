"""entry point(외부 부품) 시험.

`tmp_path`에 가짜 배포 패키지(dist-info + 모듈)를 만들고 `sys.path`에 넣어, 실제
`importlib.metadata` 탐색으로 확인한다. 플러그인 모듈 원본은 tests/fixtures/plugins에 있다.
"""

from __future__ import annotations

import importlib
import shutil
import sys
import warnings
from collections.abc import Iterator, Mapping, Sequence
from importlib import metadata
from pathlib import Path
from typing import Any, ClassVar, Protocol

import pytest

from ragexp.registry import Registry, RegistryError, RegistryWarning, register
from ragexp.types import Hit, RetrievalQuery

PLUGINS = Path(__file__).parent / "fixtures" / "plugins"
FAKE = "fake_ragexp_plugin"
BROKEN = "broken_ragexp_plugin"
GOOD_EPS = {
    "retriever.fake_bm25": f"{FAKE}:FakeBm25Retriever",
    "generator.fake_echo": f"{FAKE}:FakeEchoGenerator",
}


class Install(Protocol):
    def __call__(self, dist: str, eps: Mapping[str, str], modules: Sequence[str] = ()) -> None: ...


@pytest.fixture
def install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Install]:
    """가짜 배포 패키지를 tmp_path에 설치하는 함수. 시험 뒤 sys.path·sys.modules를 정리한다."""
    monkeypatch.syspath_prepend(str(tmp_path))

    def _install(dist: str, eps: Mapping[str, str], modules: Sequence[str] = ()) -> None:
        info = tmp_path / f"{dist.replace('-', '_')}-0.1.dist-info"
        info.mkdir()
        (info / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: {dist}\nVersion: 0.1\n", encoding="utf-8"
        )
        body = "[ragexp.components]\n" + "".join(f"{k} = {v}\n" for k, v in eps.items())
        (info / "entry_points.txt").write_text(body, encoding="utf-8")
        for mod in modules:
            shutil.copy(PLUGINS / f"{mod}.py", tmp_path / f"{mod}.py")
        importlib.invalidate_caches()

    yield _install
    for mod in (FAKE, BROKEN):
        sys.modules.pop(mod, None)
    importlib.invalidate_caches()


# (a) 외부 부품이 보이고, 쓸 때만 불러온다 ------------------------------------------


def test_plugin_listed_without_import_then_used(install: Install) -> None:
    install("fake-plugin", GOOD_EPS, [FAKE])
    r = Registry()
    assert "fake_bm25" in r.available("retriever")
    assert "fake_echo" in r.available("generator")
    assert FAKE not in sys.modules  # 이름만 보고 불러오지 않음

    cls = r.get("retriever", "fake_bm25")
    assert cls.__name__ == "FakeBm25Retriever"
    obj = r.create("retriever", "fake_bm25", {"k": 3})
    assert obj.retrieve(RetrievalQuery(text="q")) == []
    assert r.component_fingerprint(obj) == {"type": "fake_bm25", "version": "1", "k": 3}


# (b) 새 Registry 두 개가 같은 플러그인을 각각 본다 -----------------------------------


def test_two_registries_each_see_plugin(install: Install) -> None:
    install("fake-plugin", GOOD_EPS, [FAKE])
    r1, r2 = Registry(), Registry()
    c1 = r1.get("retriever", "fake_bm25")
    assert FAKE in sys.modules  # 이미 import된 뒤에도
    c2 = r2.get("retriever", "fake_bm25")
    assert c1 is c2
    obj = r2.create("retriever", "fake_bm25", {})
    assert r2.component_fingerprint(obj)["type"] == "fake_bm25"


# (c) 쓰는 플러그인이 깨졌으면 플러그인 이름을 담아 멈춘다 ------------------------------


def test_used_broken_plugin_stops_with_name(install: Install) -> None:
    install("broken-plugin", {"retriever.broken": f"{BROKEN}:Anything"}, [BROKEN])
    with pytest.raises(RegistryError) as info:
        Registry().get("retriever", "broken")
    msg = str(info.value)
    assert "broken-plugin" in msg and "retriever.broken" in msg and "ImportError" in msg


def test_missing_module_plugin_stops(install: Install) -> None:
    install("ghost-plugin", {"retriever.ghost": "no_such_module_xyz:Ghost"})
    with pytest.raises(RegistryError, match="ghost-plugin"):
        Registry().create("retriever", "ghost", {})


# 안 쓰는 깨진 플러그인이 있어도 다른 부품은 동작한다 ---------------------------------


def test_unused_broken_plugin_does_not_block_others(install: Install) -> None:
    install("fake-plugin", GOOD_EPS, [FAKE])
    install("broken-plugin", {"retriever.broken": f"{BROKEN}:Anything"}, [BROKEN])
    r = Registry()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # 쓰는 부품만 다루면 경고도 없다
        assert {"broken", "fake_bm25"} <= set(r.available("retriever"))
        obj = r.create("retriever", "fake_bm25", {"k": 1})
        assert isinstance(obj.retrieve(RetrievalQuery(text="q")), list)
        r.create("generator", "fake_echo", {})
    assert BROKEN not in sys.modules  # 깨진 플러그인은 불러오지도 않음


def test_load_all_warns_but_does_not_stop(install: Install) -> None:
    install("fake-plugin", GOOD_EPS, [FAKE])
    install("broken-plugin", {"retriever.broken": f"{BROKEN}:Anything"}, [BROKEN])
    r = Registry()
    with pytest.warns(RegistryWarning, match="broken-plugin"):
        failed = r.load_all()
    assert failed == ["retriever.broken"]
    assert r.get("retriever", "fake_bm25").__name__ == "FakeBm25Retriever"


def test_malformed_entry_point_name_warns_and_is_skipped(install: Install) -> None:
    install("fake-plugin", GOOD_EPS, [FAKE])
    install("odd-plugin", {"badname": f"{FAKE}:FakeBm25Retriever", "index.x": f"{FAKE}:X"})
    r = Registry()
    with pytest.warns(RegistryWarning, match="odd-plugin"):
        names = r.available("retriever")
    assert "fake_bm25" in names and "badname" not in names


# (d) 이름이 겹치면 오류 ------------------------------------------------------------


def test_same_name_from_two_plugins_is_ambiguous(install: Install) -> None:
    eps = {"retriever.fake_bm25": f"{FAKE}:FakeBm25Retriever"}
    install("plugin-a", eps, [FAKE])
    install("plugin-b", eps)
    r = Registry()
    assert r.available("retriever").count("fake_bm25") == 1
    with pytest.raises(RegistryError) as info:
        r.get("retriever", "fake_bm25")
    assert "plugin-a" in str(info.value) and "plugin-b" in str(info.value)


def test_in_process_add_conflicts_with_plugin(install: Install) -> None:
    install("fake-plugin", GOOD_EPS, [FAKE])

    @register("retriever", "fake_bm25")
    class Local:
        version: ClassVar[str] = "1"

        def retrieve(self, query: RetrievalQuery) -> list[Hit]:
            return []

        def fingerprint(self) -> dict[str, Any]:
            return {}

    with pytest.raises(RegistryError, match="fake-plugin"):
        Registry().add(Local)


def test_entry_point_marker_mismatch(install: Install) -> None:
    install("fake-plugin", {"retriever.renamed": f"{FAKE}:MismatchedRetriever"}, [FAKE])
    with pytest.raises(RegistryError, match="표시"):
        Registry().get("retriever", "renamed")


def test_entry_point_not_a_class(install: Install) -> None:
    install("fake-plugin", {"retriever.notcls": f"{FAKE}:register"}, [FAKE])
    with pytest.raises(RegistryError, match="클래스가 아닙니다"):
        Registry().get("retriever", "notcls")


# (e) 한 레지스트리 안에서는 한 번만 찾고 한 번만 불러온다 -------------------------------


def test_discover_and_load_once(install: Install, monkeypatch: pytest.MonkeyPatch) -> None:
    install("fake-plugin", GOOD_EPS, [FAKE])
    calls = {"discover": 0, "load": 0}
    real_entry_points = metadata.entry_points
    real_load = metadata.EntryPoint.load

    def counting_entry_points(**kw: Any) -> metadata.EntryPoints:
        calls["discover"] += 1
        return real_entry_points(**kw)

    def counting_load(self: metadata.EntryPoint) -> Any:
        calls["load"] += 1
        return real_load(self)

    monkeypatch.setattr("ragexp.registry.metadata.entry_points", counting_entry_points)
    monkeypatch.setattr(metadata.EntryPoint, "load", counting_load)

    r = Registry()
    for _ in range(3):
        r.available("retriever")
        r.get("retriever", "fake_bm25")
        r.create("retriever", "fake_bm25", {})
    assert calls == {"discover": 1, "load": 1}

    Registry().get("retriever", "fake_bm25")  # 새 레지스트리는 따로 찾는다
    assert calls == {"discover": 2, "load": 2}
