"""인증키(OC) 누출 시험(U4 4절): 스냅샷 파일·오류(예외 연쇄)·로그·CLI 출력·고정 자료·repr."""

from __future__ import annotations

import logging
import re
import traceback
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import quote

import httpx
import pytest
from fakes.fake_law_api import (
    FAKE_OC,
    FIXTURES,
    HTML_ERROR,
    PROXY_ENV,
    Fail,
    FakeLaw,
    FakeLawApi,
    FakeVersion,
    html_error_with_oc,
    law_body,
    run_fetch,
)
from pydantic import SecretStr
from typer.testing import CliRunner

from ragexp.cli import app
from ragexp.datasets import law_api
from ragexp.datasets.law_api import MISSING_OC, LawFetchError, _LawApi, make_http_client

FORBIDDEN = (FAKE_OC, quote(FAKE_OC, safe=""), "law.go.kr", "lawSearch.do?", "lawService.do?")
"""오류·로그·출력에 나오면 안 되는 것: OC 값과 요청 주소."""


@pytest.fixture(autouse=True)
def no_proxy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in PROXY_ENV:
        monkeypatch.delenv(name, raising=False)


def assert_clean(text: str) -> None:
    for needle in FORBIDDEN:
        assert needle not in text, needle


def chain(exc: BaseException) -> Iterator[BaseException]:
    """`__cause__`·`__context__`를 따라 연쇄 전체를 돈다."""
    seen: set[int] = set()
    stack = [exc]
    while stack:
        cur = stack.pop()
        if id(cur) in seen:
            continue
        seen.add(id(cur))
        yield cur
        stack += [e for e in (cur.__cause__, cur.__context__) if e is not None]


def one_version(body: bytes) -> FakeLawApi:
    return FakeLawApi(
        [FakeLaw("시험법", "009999", "법률", (FakeVersion("100002", "20260101", body=body),))]
    )


# --- 스냅샷 파일 --------------------------------------------------------------------------


def test_oc_not_in_snapshot_files(tmp_path: Path) -> None:
    out = tmp_path / "out"
    api = FakeLawApi(echo_oc_in_search=True)  # 검색 응답의 법령상세링크에 OC가 되돌아온다
    run = run_fetch(out, api, names=["시험법", "시험법 시행령"])
    assert run.result is not None, run.error
    assert run.result.search_echoes_oc is True
    files = [p for p in out.rglob("*") if p.is_file()]
    assert len(files) == 4
    for path in files:
        data = path.read_bytes()
        assert FAKE_OC.encode() not in data
        assert quote(FAKE_OC, safe="").encode() not in data
        assert b"lawSearch.do" not in data


def test_search_echo_flag_false(tmp_path: Path) -> None:
    run = run_fetch(tmp_path / "out", FakeLawApi(echo_oc_in_search=False))
    assert run.result is not None
    assert run.result.search_echoes_oc is False


def test_response_with_oc_refused(tmp_path: Path) -> None:
    out = tmp_path / "out"
    run = run_fetch(out, FakeLawApi(echo_oc_in_body=True))
    assert isinstance(run.error, LawFetchError)
    message = str(run.error)
    assert "응답에 인증키(OC)와 같은 문자열이 있어 저장하지 않습니다" in message
    assert "잘못 걸릴 수 있습니다" in message
    assert_clean(message)
    assert not out.exists()
    assert list(tmp_path.iterdir()) == []


def test_percent_encoded_oc_refused(tmp_path: Path) -> None:
    oc = "oc-test-SECRET/7f3a"  # 퍼센트 인코딩하면 모양이 바뀌는 값
    body = law_body("009999", "시험법", marker=quote(oc, safe=""))
    api = FakeLawApi(
        [FakeLaw("시험법", "009999", "법률", (FakeVersion("100002", "20260101", body=body),))],
        oc=oc,
    )
    run = run_fetch(tmp_path / "out", api, oc=oc)
    assert "저장하지 않습니다" in str(run.error)


# --- 오류·예외 연쇄 -------------------------------------------------------------------------


def _cases() -> list[tuple[str, FakeLawApi, list[str]]]:
    other = FakeLaw("시험법", "008888", "법률", (FakeVersion("500001", "20200101"),))
    return [
        ("connect", FakeLawApi(script=[Fail(exc=httpx.ConnectError)] * 3), ["시험법"]),
        ("connect-timeout", FakeLawApi(script=[Fail(exc=httpx.ConnectTimeout)] * 3), ["시험법"]),
        ("read-timeout", FakeLawApi(script=[Fail(exc=httpx.ReadTimeout)] * 3), ["시험법"]),
        (
            "remote-protocol",
            FakeLawApi(script=[Fail(exc=httpx.RemoteProtocolError)] * 3),
            ["시험법"],
        ),
        (
            "unsupported-protocol",
            FakeLawApi(script=[Fail(exc=httpx.UnsupportedProtocol)]),
            ["시험법"],
        ),
        ("4xx", FakeLawApi(script=[Fail(status=404)]), ["시험법"]),
        ("5xx", FakeLawApi(script=[Fail(status=502)] * 3), ["시험법"]),
        ("429", FakeLawApi(script=[Fail(status=429)] * 3), ["시험법"]),
        ("redirect", FakeLawApi(script=[Fail(status=302)]), ["시험법"]),
        (
            "html-search",
            FakeLawApi(script=[Fail(status=200, raw=html_error_with_oc(FAKE_OC))]),
            ["시험법"],
        ),
        ("html-body", one_version(html_error_with_oc(FAKE_OC)), ["시험법"]),
        ("xml-error", one_version(b"<broken"), ["시험법"]),
        ("law-id-mismatch", one_version(law_body("001111", "시험법")), ["시험법"]),
        ("name-not-found", FakeLawApi(), ["없는법"]),
        ("name-ambiguous", FakeLawApi(extra_search_rows=[other]), ["시험법"]),
        ("oc-in-body", FakeLawApi(echo_oc_in_body=True), ["시험법"]),
    ]


@pytest.mark.parametrize(
    ("case", "api", "names"), _cases(), ids=lambda v: v if isinstance(v, str) else ""
)
def test_oc_not_in_errors(tmp_path: Path, case: str, api: FakeLawApi, names: list[str]) -> None:
    run = run_fetch(tmp_path / "out", api, names=names)
    exc = run.error
    assert isinstance(exc, LawFetchError), case
    texts = [str(exc), repr(exc), "".join(traceback.format_exception(exc))]
    for link in chain(exc):
        assert not isinstance(link, httpx.HTTPError), f"{case}: httpx 예외가 연쇄에 남음"
        texts += [str(link), repr(link), repr(link.args)]
    assert exc.__cause__ is None and exc.__context__ is None
    for text in texts:
        assert_clean(text)
    assert not (tmp_path / "out").exists()


# --- 로그 --------------------------------------------------------------------------------


def test_httpx_logs_url_without_guard(caplog: pytest.LogCaptureFixture) -> None:
    """시험이 의미가 있는지 확인: 막지 않으면 httpx가 INFO에서 OC가 든 주소를 찍는다."""
    caplog.set_level(logging.DEBUG)
    with make_http_client("none", FakeLawApi()) as client:
        client.get("lawSearch.do", params={"OC": FAKE_OC, "target": "eflaw", "type": "XML"})
    assert FAKE_OC in caplog.text


@pytest.mark.parametrize("names", [["시험법"], ["없는법"]], ids=["success", "failure"])
def test_oc_not_in_logs(tmp_path: Path, caplog: pytest.LogCaptureFixture, names: list[str]) -> None:
    caplog.set_level(logging.DEBUG)
    for name in ("httpx", "httpcore"):
        caplog.set_level(logging.DEBUG, logger=name)
    run_fetch(tmp_path / "out", names=names)
    run_fetch(tmp_path / "out2", FakeLawApi(script=[Fail(exc=httpx.ConnectError)] * 3))
    for record in caplog.records:
        assert_clean(record.getMessage())
    assert_clean(caplog.text)
    # 끝나면 로거 수준을 되돌린다.
    assert logging.getLogger("httpx").level == logging.DEBUG
    assert logging.getLogger("httpcore").level == logging.DEBUG


# --- CLI 출력 ------------------------------------------------------------------------------


@pytest.fixture
def cli_with_fake(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[FakeLawApi]]:
    """CLI의 수집 함수에 가짜 API를 끼운다. 쓰인 가짜 API를 목록에 남긴다."""
    used: list[FakeLawApi] = []
    original = law_api.fetch_law_snapshot

    def with_fake(*args: object, **kwargs: object) -> object:
        api = used[-1] if used else FakeLawApi()
        if not used:
            used.append(api)
        return original(*args, **kwargs, transport=api, sleep=lambda s: None)  # type: ignore[arg-type]

    monkeypatch.setattr(law_api, "fetch_law_snapshot", with_fake)
    yield used


def _invoke(out: Path, *laws: str) -> tuple[int, str]:
    args = ["corpus", "fetch-law", "--as-of", "2026-10-29", "--out", str(out)]
    for law in laws:
        args += ["--law", law]
    result = CliRunner().invoke(app, args)
    return result.exit_code, result.output + (result.stderr if result.stderr_bytes else "")


@pytest.mark.parametrize(
    ("api", "laws", "code"),
    [
        (FakeLawApi(), ["시험법", "시험법 시행령"], 0),
        (FakeLawApi(), ["없는법"], 1),
        (FakeLawApi(script=[Fail(exc=httpx.ConnectError)] * 3), ["시험법"], 1),
        (FakeLawApi(echo_oc_in_body=True), ["시험법"], 1),
        (one_version(HTML_ERROR), ["시험법"], 1),
    ],
    ids=["success", "not-found", "connect", "oc-in-body", "html"],
)
def test_oc_not_in_cli_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cli_with_fake: list[FakeLawApi],
    api: FakeLawApi,
    laws: list[str],
    code: int,
) -> None:
    monkeypatch.setenv("LAW_API_OC", FAKE_OC)
    cli_with_fake.append(api)
    exit_code, output = _invoke(tmp_path / "out", *laws)
    assert exit_code == code, output
    assert_clean(output)
    if code:
        assert output.startswith("오류: ")


@pytest.mark.parametrize("value", [None, "", "   "])
def test_missing_oc_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cli_with_fake: list[FakeLawApi],
    value: str | None,
) -> None:
    if value is None:
        monkeypatch.delenv("LAW_API_OC", raising=False)
    else:
        monkeypatch.setenv("LAW_API_OC", value)
    exit_code, output = _invoke(tmp_path / "out", "시험법")
    assert exit_code == 1
    assert MISSING_OC in output
    assert cli_with_fake == []  # 수집 함수를 부르지 않았다(호출 0번)
    assert not (tmp_path / "out").exists()


def test_missing_oc_in_function(tmp_path: Path) -> None:
    run = run_fetch(tmp_path / "out", oc="  ")
    assert str(run.error) == MISSING_OC
    assert run.api.calls == 0


# --- 고정 자료·repr ------------------------------------------------------------------------


def test_fixtures_only_fake_oc() -> None:
    """레포의 법령 고정 자료에는 시험용 가짜 OC만 있다(이 폴더만 본다)."""
    pattern = re.compile(r"OC=([^&\"'<>\s\]]+)")
    found: list[str] = []
    files = [p for p in FIXTURES.rglob("*") if p.is_file()]
    assert files
    for path in files:
        found += pattern.findall(path.read_text(encoding="utf-8"))
    assert found, "OC 자리가 있는 고정 자료가 하나는 있어야 합니다"
    assert set(found) == {FAKE_OC}


def test_oc_secretstr_repr() -> None:
    oc = SecretStr(FAKE_OC)
    api = _LawApi(oc, "none", FakeLawApi(), lambda s: None)
    try:
        for text in (repr(oc), str(oc), repr(api), str(api), repr(vars(api).get("_oc"))):
            assert_clean(text)
    finally:
        api.__exit__(None, None, None)
