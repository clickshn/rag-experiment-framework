"""문항 파일 읽기(JSONL). 로더는 이 한 곳이다.

U2는 JSONL → `Question` 파싱만 한다. 스냅샷 대조 같은 내용 검증은 U5가 이 파일에 더한다.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from ragexp.types import Question


class QuestionFileError(ValueError):
    """문항 파일을 읽을 수 없음. 메시지에는 줄 번호와 칸 위치만 넣는다(문항 원문은 넣지 않음)."""


def load_questions(path: Path) -> list[Question]:
    """JSONL 문항 파일을 읽는다. 빈 줄은 무시한다."""
    questions: list[Question] = []
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        raise QuestionFileError(f"{path.name}: UTF-8이 아닙니다") from None
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise QuestionFileError(f"{path.name}:{lineno}: JSON 형식 오류({exc.msg})") from None
        if not isinstance(data, dict):
            raise QuestionFileError(f"{path.name}:{lineno}: 한 줄에 JSON 객체 하나여야 합니다")
        try:
            questions.append(Question.from_dict(data))
        except ValidationError as exc:
            where = "; ".join(
                f"{'.'.join(str(p) for p in e['loc']) or '(문항)'}: {e['type']}"
                for e in exc.errors(include_input=False, include_url=False)
            )
            raise QuestionFileError(f"{path.name}:{lineno}: 문항 형식 오류({where})") from None
    return questions
