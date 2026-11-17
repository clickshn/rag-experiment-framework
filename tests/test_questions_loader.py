"""문항 로더(datasets/questions.py) 시험."""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import CONFIGS

from ragexp.datasets.questions import QuestionFileError, load_questions


def test_loads_fixture() -> None:
    qs = load_questions(CONFIGS / "questions.jsonl")
    assert [q.qid for q in qs] == ["Q001", "Q002"]
    assert qs[1].gold == ()


def test_blank_lines_bom_crlf(tmp_path: Path) -> None:
    text = (CONFIGS / "questions.jsonl").read_text(encoding="utf-8")
    path = tmp_path / "q.jsonl"
    path.write_bytes(("﻿\n" + text.replace("\n", "\r\n\r\n")).encode("utf-8"))
    assert len(load_questions(path)) == 2


def test_bad_json_line_number(tmp_path: Path) -> None:
    path = tmp_path / "q.jsonl"
    path.write_text(
        '{"qid": "Q1", "type": "t", "question": "q", "must_min": 0}\n{oops\n', encoding="utf-8"
    )
    with pytest.raises(QuestionFileError, match=r"q\.jsonl:2: JSON"):
        load_questions(path)


def test_not_an_object(tmp_path: Path) -> None:
    path = tmp_path / "q.jsonl"
    path.write_text("[1, 2]\n", encoding="utf-8")
    with pytest.raises(QuestionFileError, match="객체"):
        load_questions(path)


def test_validation_error_has_location_not_content(tmp_path: Path) -> None:
    path = tmp_path / "q.jsonl"
    path.write_text(
        '{"qid": "Q1", "type": "t", "question": "비밀스러운 질문 본문", '
        '"must": ["a"], "must_min": 5}\n',
        encoding="utf-8",
    )
    with pytest.raises(QuestionFileError) as info:
        load_questions(path)
    msg = str(info.value)
    assert "q.jsonl:1" in msg and "비밀스러운" not in msg
