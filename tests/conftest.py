"""공용 시험 도구."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from ragexp.registry import Registry

CONFIGS = Path(__file__).parent / "fixtures" / "configs"
ENV = {
    "GEN_A_URL": "http://localhost:1",
    "GEN_A_KEY": "dummy-key",
    "LOCAL_URL": "http://10.0.0.5:8000/v1",
    "JUDGE1_URL": "http://localhost:2",
    "JUDGE2_URL": "http://localhost:3",
}
"""시험용 치환 값. 출력·오류·repr에 이 값들이 나오면 안 된다."""
SECRET_VALUES = ("http://localhost:1", "dummy-key", "10.0.0.5:8000", "localhost:2", "localhost:3")


def _fake_module() -> ModuleType:
    name = "ragexp_test_fake_components"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, CONFIGS / "fake_components.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fake_registry() -> Registry:
    """U2 시험용 가짜 부품을 담은 레지스트리(entry point는 읽지 않음)."""
    r = Registry(use_entry_points=False)
    r.add_module(_fake_module())
    return r
