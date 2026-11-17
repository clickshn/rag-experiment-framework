"""`ragexp config check` 시험."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import CONFIGS, ENV, SECRET_VALUES
from typer.testing import CliRunner

from ragexp.cli import app
from ragexp.registry import Registry

runner = CliRunner()
GOLDEN_ARMS = {"A": "cfg-05f276175657", "B": "cfg-0454ac6dbaba", "C": "cfg-0d287d8ad3f8"}


@pytest.fixture(autouse=True)
def use_fakes(fake_registry: Registry, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ragexp.cli.default_registry", lambda: fake_registry)


@pytest.fixture
def with_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)


def no_secrets(text: str) -> None:
    for secret in SECRET_VALUES:
        assert secret not in text


@pytest.mark.usefixtures("with_env")
def test_check_prints_versions() -> None:
    result = runner.invoke(app, ["config", "check", str(CONFIGS / "base.yaml")])
    assert result.exit_code == 0, result.output
    for arm, short in GOLDEN_ARMS.items():
        assert f"arm {arm}: {short}" in result.output
    assert "채점 버전: jdg-" in result.output and "실행 묶음 해시: run-" in result.output
    no_secrets(result.output)


@pytest.mark.usefixtures("with_env")
def test_check_json() -> None:
    result = runner.invoke(app, ["config", "check", str(CONFIGS / "base.yaml"), "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["ok"] is True and data["versions"]["arms"] == GOLDEN_ARMS
    assert "preimages" in data["versions_full"]
    no_secrets(result.output)


def test_check_no_env_gives_same_versions(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in ENV:
        monkeypatch.delenv(k, raising=False)
    failed = runner.invoke(app, ["config", "check", str(CONFIGS / "base.yaml")])
    assert failed.exit_code == 1 and "GEN_A_URL" in failed.output
    result = runner.invoke(
        app, ["config", "check", str(CONFIGS / "base.yaml"), "--no-env", "--json"]
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["versions"]["arms"] == GOLDEN_ARMS
    assert any("자리표시" in w for w in data["warnings"])


@pytest.mark.usefixtures("with_env")
def test_check_error_exit_code(tmp_path: Path) -> None:
    text = (CONFIGS / "base.yaml").read_text(encoding="utf-8").replace("${GEN_A_KEY}", "dummy-key")
    for name in ("prompt.md", "questions.jsonl"):
        (tmp_path / name).write_text((CONFIGS / name).read_text(encoding="utf-8"), encoding="utf-8")
    bad = tmp_path / "bad.yaml"
    bad.write_text(text, encoding="utf-8")
    result = runner.invoke(app, ["config", "check", str(bad)])
    assert result.exit_code == 1
    assert "접속 칸 값이 직접 적혀 있습니다" in result.output  # 원문 검사가 줄 번호로 잡음
    no_secrets(result.output)
    as_json = runner.invoke(app, ["config", "check", str(bad), "--json"])
    assert as_json.exit_code == 1 and json.loads(as_json.output)["ok"] is False


def test_missing_file() -> None:
    result = runner.invoke(app, ["config", "check", "no_such.yaml"])
    assert result.exit_code == 1


def test_pretty_exceptions_hide_locals() -> None:
    assert app.pretty_exceptions_show_locals is False


@pytest.mark.usefixtures("with_env")
def test_path_like_model_name_cli(tmp_path: Path) -> None:
    """구현 검토 지적 7: traceback 없이 칸 경로·해결 방법을 담은 오류, --json도 JSON."""
    text = (
        (CONFIGS / "base.yaml")
        .read_text(encoding="utf-8")
        .replace("model: local-27b", "model: /models/local-27b")
    )
    for name in ("prompt.md", "questions.jsonl"):
        (tmp_path / name).write_text((CONFIGS / name).read_text(encoding="utf-8"), encoding="utf-8")
    bad = tmp_path / "bad.yaml"
    bad.write_text(text, encoding="utf-8")
    result = runner.invoke(app, ["config", "check", str(bad)])
    # 정상 종료 경로(typer.Exit → SystemExit)로 끝나야 한다. 다른 예외면 traceback이 나온 것.
    assert result.exit_code == 1 and isinstance(result.exception, SystemExit)
    assert "Traceback" not in result.output
    assert "generators.B" in result.output and "--served-model-name" in result.output
    as_json = runner.invoke(app, ["config", "check", str(bad), "--json"])
    assert json.loads(as_json.output)["ok"] is False
