"""Claude Code 훅 스크립트(.claude/hooks) 시험."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

HOOKS = Path(__file__).resolve().parents[1] / ".claude" / "hooks"
sys.path.insert(0, str(HOOKS))

import block_env_access  # noqa: E402
import gitleaks_before_commit  # noqa: E402
import ruff_on_edit  # noqa: E402


def run_hook(script: str, payload: dict[str, Any]) -> subprocess.CompletedProcess[bytes]:
    """실제 훅처럼 stdin에 JSON을 넣어 스크립트를 실행한다."""
    return subprocess.run(
        [sys.executable, str(HOOKS / script)],
        input=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        capture_output=True,
        check=False,
    )


# --- .env 접근 차단 ---------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [".env", "C:\\repo\\.env", "/repo/.env.local", "sub/.env.production"],
)
def test_env_blocked(path: str) -> None:
    code, msg = block_env_access.decide({"tool_input": {"file_path": path}})
    assert code == 2
    assert "차단" in msg


@pytest.mark.parametrize("path", [".env.example", "src/ragexp/env.py", "docs/.envrc.md", ""])
def test_env_allowed(path: str) -> None:
    assert block_env_access.decide({"tool_input": {"file_path": path}}) == (0, "")


def test_env_hook_process_exit_code() -> None:
    proc = run_hook(
        "block_env_access.py", {"tool_name": "Read", "tool_input": {"file_path": ".env"}}
    )
    assert proc.returncode == 2
    assert "차단" in proc.stderr.decode("utf-8")


# --- 커밋 전 gitleaks -------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        "git commit -m 'x'",
        "git add -A && git commit -F -",
        "git -c user.name=a commit --amend",
        "cd repo; git commit",
    ],
)
def test_commit_detected(command: str) -> None:
    assert gitleaks_before_commit.is_commit(command)


@pytest.mark.parametrize(
    "command",
    ["git status", "git log --grep commit", "echo commit", "uv run pytest", "git commit-tree x"],
)
def test_non_commit_ignored(command: str) -> None:
    assert not gitleaks_before_commit.is_commit(command)


def test_non_commit_passes_without_gitleaks() -> None:
    assert gitleaks_before_commit.decide({"tool_input": {"command": "git status"}}) == (0, "")


def test_commit_blocked_when_gitleaks_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gitleaks_before_commit.shutil, "which", lambda _: None)
    code, msg = gitleaks_before_commit.decide({"tool_input": {"command": "git commit"}})
    assert code == 2
    assert "gitleaks" in msg


@pytest.mark.skipif(shutil.which("gitleaks") is None, reason="gitleaks 없음")
def test_commit_blocked_on_staged_secret(tmp_path: Path) -> None:
    git = ["git", "-C", str(tmp_path)]
    subprocess.run([*git, "init", "-q"], check=True)
    # 시험용 가짜 키. 실제 키가 아니다(gitleaks 기본 규칙에 걸리는 형식만 흉내 냄).
    fake = "AKIA" + "QYLPMN5HHHFPZAM2"
    (tmp_path / "config.py").write_text(f'aws_key = "{fake}"\n', encoding="utf-8")
    subprocess.run([*git, "add", "config.py"], check=True)
    code, msg = gitleaks_before_commit.decide(
        {"tool_input": {"command": "git commit -m x"}, "cwd": str(tmp_path)}
    )
    assert code == 2
    assert fake not in msg  # --redact


@pytest.mark.skipif(shutil.which("gitleaks") is None, reason="gitleaks 없음")
def test_commit_allowed_when_clean(tmp_path: Path) -> None:
    git = ["git", "-C", str(tmp_path)]
    subprocess.run([*git, "init", "-q"], check=True)
    (tmp_path / "ok.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.run([*git, "add", "ok.py"], check=True)
    payload = {"tool_input": {"command": "git commit -m x"}, "cwd": str(tmp_path)}
    assert gitleaks_before_commit.decide(payload) == (0, "")


# --- .py 편집 뒤 ruff -------------------------------------------------------


def test_ruff_formats_python_file(tmp_path: Path) -> None:
    f = tmp_path / "messy.py"
    f.write_text("import os\nx=( 1,2 )\n", encoding="utf-8")
    code, _ = ruff_on_edit.decide({"tool_input": {"file_path": str(f)}})
    assert code == 0
    assert f.read_text(encoding="utf-8") == "x = (1, 2)\n"  # 포맷 + 안 쓰는 import 제거


def test_ruff_reports_unfixable(tmp_path: Path) -> None:
    f = tmp_path / "bad.py"
    f.write_text("print(undefined_name)\n", encoding="utf-8")
    code, msg = ruff_on_edit.decide({"tool_input": {"file_path": str(f)}})
    assert code == 2
    assert "F821" in msg


def test_ruff_ignores_non_python(tmp_path: Path) -> None:
    f = tmp_path / "notes.md"
    f.write_text("x=( 1,2 )\n", encoding="utf-8")
    assert ruff_on_edit.decide({"tool_input": {"file_path": str(f)}}) == (0, "")
    assert f.read_text(encoding="utf-8") == "x=( 1,2 )\n"
