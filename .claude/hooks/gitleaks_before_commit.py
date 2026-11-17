"""PreToolUse(Bash): git commit 전에 스테이징된 변경을 gitleaks로 검사한다.

비밀값을 찾거나 gitleaks가 없으면 종료 코드 2로 커밋을 막는다.
pre-commit을 설치하지 않은 상태에서도 막기 위한 이중 장치다.
"""

from __future__ import annotations

import os
import re
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
    cwd = str(payload.get("cwd") or os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd())
    return run_gitleaks(cwd)


def main() -> int:
    return emit(*decide(read_payload()))


if __name__ == "__main__":
    sys.exit(main())
