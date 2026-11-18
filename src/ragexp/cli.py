"""ragexp 명령줄 진입점."""

import json
import os
import re
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, NoReturn

import typer

from ragexp import __version__
from ragexp.config import ConfigError, load_config
from ragexp.datasets import law_api
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
corpus_app = typer.Typer(
    help="말뭉치 수집·스냅샷.", no_args_is_help=True, pretty_exceptions_show_locals=False
)
app.add_typer(corpus_app, name="corpus")

DEFAULT_SNAPSHOT_ROOT = Path("workspace") / "snapshots"


class ProxyOption(StrEnum):
    none = "none"
    env = "env"


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


@corpus_app.command("fetch-law")
def corpus_fetch_law(
    laws: Annotated[
        list[str], typer.Option("--law", help="법령명(정확히 같게). 여러 번 줄 수 있다")
    ],
    as_of: Annotated[str, typer.Option("--as-of", help="기준일 YYYY-MM-DD(그날 시행 중인 판)")],
    out: Annotated[
        Path | None,
        typer.Option(
            "--out", help="스냅샷 폴더(기본 workspace/snapshots/law-ko-<기준일 YYYYMMDD>)"
        ),
    ] = None,
    proxy: Annotated[
        ProxyOption,
        typer.Option("--proxy", help="none: 프록시 안 씀(기본), env: 환경 변수 프록시를 따름"),
    ] = ProxyOption.none,
) -> None:
    """국가법령정보센터에서 법령의 기준일 판을 받아 스냅샷을 만든다.

    인증키는 환경 변수 LAW_API_OC로만 받는다.
    """
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", as_of):
        _fail(["--as-of는 YYYY-MM-DD 꼴이어야 합니다"], False)
    try:
        as_of_date = date.fromisoformat(as_of)
    except ValueError:
        _fail(["--as-of가 올바른 날짜가 아닙니다"], False)
    oc = law_api.oc_from_env(os.environ)
    if oc is None:
        _fail([law_api.MISSING_OC], False)
    out_dir = out if out is not None else DEFAULT_SNAPSHOT_ROOT / f"law-ko-{as_of_date:%Y%m%d}"
    try:
        result = law_api.fetch_law_snapshot(laws, as_of_date, out_dir, oc=oc, proxy=proxy.value)
    except law_api.LawFetchError as exc:
        _fail([str(exc)], False)
    except OSError as exc:
        _fail([f"스냅샷 폴더를 쓸 수 없습니다({type(exc).__name__})"], False)

    m = result.manifest
    typer.echo(f"스냅샷: {result.path} (기준일 {m.as_of}, 받은 시각 {m.fetched_at})")
    for e in m.laws:
        typer.echo(
            f"  {e.name}: 법령ID {e.law_id}, MST {e.mst}, 시행일자 {e.effective_date}, "
            f"{e.bytes:,}바이트"
        )
    typer.echo(f"내용 해시: {m.content_hash}")
    echoed = "예(검색 응답은 저장하지 않음)" if result.search_echoes_oc else "아니오"
    typer.echo(f"참고: 검색 응답에 인증키가 들어 있었나: {echoed}")
