"""패키지 import와 CLI 기본 동작을 확인한다."""

from typer.testing import CliRunner

import ragexp
from ragexp.cli import app

runner = CliRunner()


def test_version_string() -> None:
    assert ragexp.__version__ == "0.0.0"


def test_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "ragexp" in result.output or "RAG" in result.output


def test_version_option() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == f"ragexp {ragexp.__version__}"
