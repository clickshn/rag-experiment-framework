"""PreToolUse(Bash): git commit 전에 스테이징된 변경을 gitleaks로 검사한다.

다음 경우 종료 코드 2로 커밋을 막는다.
- `git commit -a`·`--all`: 커밋하는 순간 스테이징되는 변경은 staged 검사에 보이지 않는다.
  먼저 `git add`로 올린 뒤 커밋하게 안내한다.
- gitleaks가 비밀값을 찾거나, gitleaks가 없음.
pre-commit을 설치하지 않은 상태에서도 막기 위한 이중 장치다.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import sys
from typing import Any

from _common import emit, read_payload

COMMIT_RE = re.compile(r"(^|[;&|(\s])git(\s+-[cC]\s+\S+)*\s+commit(?![\w-])")
GITLEAKS_ARGS = ["git", "--pre-commit", "--redact", "--staged", "--no-banner", "--verbose"]


def is_commit(command: str) -> bool:
    """셸 명령에 git commit이 들어 있는지 본다."""
    return bool(COMMIT_RE.search(command))


# git commit에서 값을 따로 받는 옵션(다음 토큰이 값이다).
_SHORT_WITH_VALUE = set("mFCct")
# 값을 붙여서만 받는 짧은 옵션(-S<keyid>, -u<mode>): 뒤 글자는 값이다.
_SHORT_OPTIONAL_VALUE = set("Su")
_LONG_WITH_VALUE = {
    "--message",
    "--file",
    "--reuse-message",
    "--reedit-message",
    "--template",
    "--author",
    "--date",
    "--fixup",
    "--squash",
    "--cleanup",
    "--trailer",
    "--pathspec-from-file",
}
_PUNCTUATION = set("();<>|&")  # 셸 구분자·리다이렉션: 여기서 commit 인자가 끝난다


def _commit_args(command: str) -> list[list[str]]:
    """명령 속 git commit마다 그 뒤 인자 목록을 돌려준다.

    같은 줄의 셸 구분자·리다이렉션 전까지만 본다(heredoc 본문의 글자를 옵션으로 읽지 않게).
    """
    result: list[list[str]] = []
    for m in COMMIT_RE.finditer(command):
        rest = command[m.end() :].split("\n", 1)[0]
        lexer = shlex.shlex(rest, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        args: list[str] = []
        try:
            for tok in lexer:
                if set(tok) <= _PUNCTUATION:
                    break
                args.append(tok)
        except ValueError:  # 따옴표가 닫히지 않음 등: 읽은 데까지만 본다
            pass
        result.append(args)
    return result


def _has_all_flag(args: list[str]) -> bool:
    skip_next = False
    for arg in args:
        if skip_next:
            skip_next = False
            continue
        if arg == "--":
            return False
        if arg == "--all":
            return True
        if arg.startswith("--"):
            skip_next = arg in _LONG_WITH_VALUE
            continue
        if arg.startswith("-") and len(arg) > 1:
            for i, ch in enumerate(arg[1:]):
                if ch == "a":
                    return True
                if ch in _SHORT_WITH_VALUE:
                    skip_next = i == len(arg) - 2  # 값이 붙어 있지 않으면 다음 토큰이 값
                    break
                if ch in _SHORT_OPTIONAL_VALUE:
                    break
    return False


def uses_commit_all(command: str) -> bool:
    """git commit에 -a/--all(묶음 옵션 -am 등 포함)이 쓰였는지 본다."""
    return any(_has_all_flag(args) for args in _commit_args(command))


def run_gitleaks(cwd: str) -> tuple[int, str]:
    exe = shutil.which("gitleaks")
    if exe is None:
        return 2, "차단: gitleaks를 찾지 못했습니다. 설치해 PATH에 넣은 뒤 다시 커밋하세요."
    proc = subprocess.run(
        [exe, *GITLEAKS_ARGS],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if proc.returncode != 0:
        detail = (proc.stdout + proc.stderr).strip()
        return 2, f"차단: gitleaks가 스테이징된 변경에서 비밀값을 찾았습니다.\n{detail}"
    return 0, ""


def decide(payload: dict[str, Any]) -> tuple[int, str]:
    command = str((payload.get("tool_input") or {}).get("command") or "")
    if not is_commit(command):
        return 0, ""
    if uses_commit_all(command):
        return 2, (
            "차단: `git commit -a`/`--all`은 쓰지 않습니다. 커밋 순간 스테이징되는 변경은 "
            "gitleaks staged 검사에 보이지 않기 때문입니다. "
            "`git add <파일>`(또는 `git add -A`)로 먼저 올린 뒤 `git commit`하세요."
        )
    cwd = str(payload.get("cwd") or os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd())
    return run_gitleaks(cwd)


def main() -> int:
    return emit(*decide(read_payload()))


if __name__ == "__main__":
    sys.exit(main())
