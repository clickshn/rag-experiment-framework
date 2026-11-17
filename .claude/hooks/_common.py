"""Claude Code 훅 스크립트의 공통 도구(표준 라이브러리만 쓴다)."""

from __future__ import annotations

import json
import sys
from typing import Any


def read_payload() -> dict[str, Any]:
    """stdin의 훅 입력 JSON을 UTF-8로 읽는다(Windows 기본 인코딩을 피함)."""
    raw = sys.stdin.buffer.read().decode("utf-8")
    data = json.loads(raw) if raw.strip() else {}
    return data if isinstance(data, dict) else {}


def emit(code: int, message: str) -> int:
    """메시지를 stderr로 내보내고 종료 코드를 돌려준다. 코드 2면 Claude에게 전달된다."""
    if message:
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        print(message, file=sys.stderr)
    return code
