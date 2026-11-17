"""ragexp 명령줄 진입점."""

from typing import Annotated

import typer

from ragexp import __version__

app = typer.Typer(
    name="ragexp",
    help="RAG 부품을 바꿔 끼우며 공정하게 비교하는 실험 프레임워크.",
    no_args_is_help=True,
)


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
