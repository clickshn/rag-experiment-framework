"""PreToolUse(Read|Edit|Write): .env 계열 파일을 읽거나 쓰지 못하게 막는다.

.env.example(값 없는 목록)만 허용한다. 막으면 종료 코드 2.
"""

from __future__ import annotations

import sys
from pathlib import PurePath
from typing import Any

from _common import emit, read_payload

ALLOWED = {".env.example"}


def is_secret_env(path: str) -> bool:
    """경로가 .env 또는 .env.* 파일인지 본다(.env.example 제외)."""
    name = PurePath(path.replace("\\", "/")).name
    if name in ALLOWED:
        return False
    return name == ".env" or name.startswith(".env.")


def decide(payload: dict[str, Any]) -> tuple[int, str]:
    tool_input = payload.get("tool_input") or {}
    path = str(tool_input.get("file_path") or tool_input.get("path") or "")
    if path and is_secret_env(path):
        return 2, (
            f"차단: 비밀값 파일({PurePath(path).name})은 읽거나 쓰지 않습니다. "
            "필요한 변수 이름은 .env.example에 적고, 값은 사용자가 직접 넣습니다."
        )
    return 0, ""


def main() -> int:
    return emit(*decide(read_payload()))


if __name__ == "__main__":
    sys.exit(main())
