"""PostToolUse(Edit|Write): 고친 .py 파일에 ruff check --fix와 ruff format을 돌린다.

편집을 되돌리지는 않는다. 고칠 수 없는 린트 오류가 남으면 종료 코드 2로 Claude에게 알린다.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from _common import emit, read_payload


def ruff(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "ruff", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def decide(payload: dict[str, Any]) -> tuple[int, str]:
    path = str((payload.get("tool_input") or {}).get("file_path") or "")
    if not path.endswith(".py") or not Path(path).is_file():
        return 0, ""
    # 고칠 수 있는 것을 먼저 고치고 포맷한 뒤, 남은 문제를 확인한다.
    # 안 쓰는 import(F401)는 지우지 않고 알리기만 한다. 여러 번에 나눠 고치는 중간에
    # 아직 쓰기 전인 import가 사라지는 것을 막는다.
    ruff("check", "--fix", "--unfixable", "F401", "--quiet", path)
    ruff("format", "--quiet", path)
    check = ruff("check", "--quiet", path)
    if check.returncode != 0:
        return 2, f"ruff가 고치지 못한 문제가 남았습니다({path}):\n{check.stdout.strip()}"
    return 0, ""


def main() -> int:
    return emit(*decide(read_payload()))


if __name__ == "__main__":
    sys.exit(main())
