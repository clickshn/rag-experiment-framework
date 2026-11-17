"""ragexp 명령줄 진입점."""

import json
from pathlib import Path
from typing import Annotated, Any, NoReturn

import typer

from ragexp import __version__
from ragexp.config import ConfigError, load_config
from ragexp.datasets.questions import load_questions
from ragexp.registry import RegistryError, default_registry
from ragexp.versioning import compute_versions, duplicate_arm_warnings

app = typer.Typer(
    name="ragexp",
    help="RAG 부품을 바꿔 끼우며 공정하게 비교하는 실험 프레임워크.",
    no_args_is_help=True,
    # 예외 화면에 지역 변수(치환한 주소·키가 있을 수 있음)를 보이지 않는다.
    pretty_exceptions_show_locals=False,
)
config_app = typer.Typer(
    help="실험 구성(YAML) 다루기.", no_args_is_help=True, pretty_exceptions_show_locals=False
)
app.add_typer(config_app, name="config")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"ragexp {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="버전을 출력하고 끝낸다.",
        ),
    ] = False,
) -> None:
    """RAG 실험 도구."""


@config_app.command("check")
def config_check(
    path: Annotated[Path, typer.Argument(help="실험 구성 YAML 파일")],
    as_json: Annotated[bool, typer.Option("--json", help="기계용 JSON으로 출력")] = False,
    no_env: Annotated[
        bool,
        typer.Option("--no-env", help="환경 변수 없이 해시만 본다(접속 값은 해시에 들어가지 않음)"),
    ] = False,
) -> None:
    """구성을 실행 전에 검사하고 버전 해시(구성·채점·문항 세트·실행 묶음)를 출력한다."""
    registry = default_registry()
    try:
        loaded = load_config(path, registry, allow_missing_env=no_env)
        questions = load_questions(loaded.questions_path)
        versions = compute_versions(loaded, registry, questions)
    except ConfigError as exc:
        _fail(exc.messages, as_json)
    except (ValueError, RegistryError, OSError) as exc:
        # QuestionFileError·VersionError·FingerprintError·UnicodeDecodeError는 ValueError다.
        _fail([f"{type(exc).__name__}: {exc}"], as_json)

    warnings = loaded.warnings + duplicate_arm_warnings(versions)
    if as_json:
        out: dict[str, Any] = {
            "ok": True,
            "warnings": warnings,
            "versions": {
                "arms": {name: v.short for name, v in versions.arms.items()},
                "judging": versions.judging.short,
                "question_set": versions.question_set.short,
                "run_set": versions.run_set.short,
            },
            "versions_full": versions.to_dict(),
        }
        typer.echo(json.dumps(out, ensure_ascii=False, indent=2))
        return
    for w in warnings:
        typer.echo(f"경고: {w}")
    typer.echo(f"구성: {loaded.name} (level={loaded.level}, 문항 {len(questions)}개)")
    for name, v in versions.arms.items():
        typer.echo(f"  arm {name}: {v.short}")
    typer.echo(f"채점 버전: {versions.judging.short}")
    typer.echo(f"문항 세트 버전: {versions.question_set.short}")
    typer.echo(f"실행 묶음 해시: {versions.run_set.short}")


def _fail(messages: list[str], as_json: bool) -> NoReturn:
    if as_json:
        typer.echo(json.dumps({"ok": False, "errors": messages}, ensure_ascii=False, indent=2))
    else:
        for m in messages:
            typer.echo(f"오류: {m}", err=True)
    raise typer.Exit(code=1)
