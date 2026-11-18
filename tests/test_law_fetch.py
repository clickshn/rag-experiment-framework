"""법령 수집 시험(U4): 판 고르기·호출 인자·쪽 읽기·재시도·프록시·폴더 처리·manifest."""

from __future__ import annotations

import hashlib
import inspect
import itertools
import json
import os
from datetime import date
from pathlib import Path

import httpx
import pytest
from fakes.fake_law_api import (
    AS_OF,
    FAKE_OC,
    FIXTURES,
    HTML_ERROR,
    PROXY_ENV,
    Fail,
    FakeLaw,
    FakeLawApi,
    FakeVersion,
    law_body,
    run_fetch,
    standard_laws,
)
from typer.testing import CliRunner

from ragexp.cli import app
from ragexp.datasets import law_api
from ragexp.datasets.law_api import (
    API_HINT,
    CALL_INTERVAL_S,
    SOURCE_NOTICE,
    SearchRow,
    fetch_law_snapshot,
    load_snapshot,
    make_http_client,
    normalize_law_id,
    parse_search,
    pick_version,
)


@pytest.fixture(autouse=True)
def no_proxy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """사용자 환경의 프록시 변수가 시험에 끼어들지 않게 지운다."""
    for name in PROXY_ENV:
        monkeypatch.delenv(name, raising=False)


def one_version_api(body: bytes, *, law_id: str = "009999") -> FakeLawApi:
    """시험법 판 하나(MST 100002, 2026-01-01)만 있는 가짜 API. 본문을 정해 준다."""
    version = FakeVersion("100002", "20260101", "20251201", "00001", body=body)
    return FakeLawApi([FakeLaw("시험법", law_id, "법률", (version,))])


def fullwidth(digits: str) -> str:
    """전각 숫자 문자열(ASCII 숫자가 아님)."""
    return "".join(chr(0xFF10 + int(c)) for c in digits)


def error_text(run_error: Exception | None) -> str:
    assert run_error is not None, "실패해야 합니다"
    return str(run_error)


# --- 판 고르기·호출 인자 ---------------------------------------------------------------


def test_picks_version_in_force_on_as_of(tmp_path: Path) -> None:
    run = run_fetch(tmp_path / "snap")
    assert run.result is not None, run.error
    law = run.result.manifest.laws[0]
    assert (law.mst, law.effective_date) == ("100002", "2026-01-01")  # 지난·현행·시행예정 중 현행

    assert run.api.paths() == ["lawSearch.do", "lawSearch.do", "lawService.do"]
    first, second, third = (run.api.params(i) for i in range(3))
    assert first | {"OC": "-"} == {
        "target": "eflaw",
        "type": "XML",
        "query": "시험법",
        "display": "100",
        "page": "1",
        "OC": "-",
    }
    assert second["LID"] == "009999" and second["nw"] == "1,2,3"
    assert "query" not in second
    assert third == {
        "target": "eflaw",
        "type": "XML",
        "MST": "100002",
        "efYd": "20260101",
        "OC": FAKE_OC,
    }
    assert "ID" not in third and "JO" not in third
    assert all(r.url.scheme == "https" for r in run.api.requests)


@pytest.mark.parametrize(
    ("as_of", "mst"),
    [(date(2026, 1, 1), "100002"), (date(2025, 12, 31), "100001"), (date(2027, 1, 1), "100003")],
)
def test_as_of_boundary_is_inclusive(tmp_path: Path, as_of: date, mst: str) -> None:
    run = run_fetch(tmp_path / "snap", as_of=as_of)
    assert run.result is not None, run.error
    assert run.result.manifest.laws[0].mst == mst
    assert run.result.manifest.as_of == as_of.isoformat()


def _row(mst: str, eff: str, prom: str, no: str) -> SearchRow:
    return SearchRow("시험법", "009999", mst, eff, prom, no, "법률")


_A = ("300", "20260101", "20251201", "00009")


@pytest.mark.parametrize(
    ("other", "expected"),
    [
        (("299", "20260101", "20251215", "00003"), "299"),  # 시행일자가 같으면 공포일자가 늦은 판
        (("299", "20260101", "20251201", "00010"), "299"),  # 공포일자도 같으면 공포번호가 큰 판
        (("301", "20260101", "20251201", "00009"), "301"),  # 공포번호도 같으면 MST가 큰 판
    ],
)
def test_same_effective_date_tiebreak(other: tuple[str, str, str, str], expected: str) -> None:
    rows = [_row(*_A), _row(*other), _row("100", "20200101", "20191201", "00001")]
    rows.append(_row("900", "20270101", "", ""))  # 시행예정 판은 고르지 않는다
    for perm in itertools.permutations(rows):
        picked = pick_version(list(perm), "009999", AS_OF)
        assert not isinstance(picked, str)
        assert picked.row.mst == expected


def test_same_effective_date_tiebreak_through_fetch(tmp_path: Path) -> None:
    versions = (
        FakeVersion("100010", "20260101", "20251201", "00001"),
        FakeVersion("100011", "20260101", "20251215", "00002"),
    )
    for i, order in enumerate((versions, versions[::-1])):
        api = FakeLawApi([FakeLaw("시험법", "009999", "법률", order)])
        run = run_fetch(tmp_path / f"snap{i}", api)
        assert run.result is not None, run.error
        assert run.result.manifest.laws[0].mst == "100011"


def test_reads_all_pages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(law_api, "DISPLAY", 1)
    run = run_fetch(tmp_path / "snap")
    assert run.result is not None, run.error
    assert run.result.manifest.laws[0].mst == "100002"  # 현행 판은 판 목록 2쪽에 있다
    searches = [p for p in (run.api.params(i) for i in range(run.api.calls)) if "page" in p]
    name_pages = [p["page"] for p in searches if "query" in p]
    lid_pages = [p["page"] for p in searches if "LID" in p]
    assert name_pages == ["1", "2", "3", "4"]  # 시험법 3행 + 시험법 시행령 1행
    assert lid_pages == ["1", "2", "3"]
    assert all(p["display"] == "1" for p in searches)


def test_name_not_found(tmp_path: Path) -> None:
    run = run_fetch(tmp_path / "snap", names=["없는법"])
    assert "이름이 정확히 같은 법령이 없습니다" in error_text(run.error)
    assert "검색 결과의 이름: 없음" in error_text(run.error)

    run = run_fetch(tmp_path / "snap", names=["시험"])  # 부분 일치만 있음
    assert "검색 결과의 이름: 시험법, 시험법 시행령" in error_text(run.error)
    assert "lawService.do" not in run.api.paths()


def test_name_ambiguous(tmp_path: Path) -> None:
    other = FakeLaw("시험법", "008888", "법률", (FakeVersion("500001", "20200101"),))
    run = run_fetch(tmp_path / "snap", FakeLawApi(extra_search_rows=[other]))
    assert "이름이 같은 법령이 여럿입니다(법령ID 008888, 009999)" in error_text(run.error)
    assert run.api.calls == 1


def test_no_version_before_as_of(tmp_path: Path) -> None:
    run = run_fetch(tmp_path / "snap", as_of=date(2019, 1, 1))
    assert "기준일(2019-01-01) 이전에 시행된 판이 없습니다" in error_text(run.error)
    assert "lawService.do" not in run.api.paths()


@pytest.mark.parametrize("raw", [fullwidth("009999"), "٠٠٩٩٩٩", "1234567", "abc"])
def test_law_id_must_be_ascii_digits(raw: str) -> None:
    assert normalize_law_id(raw) is None


def test_law_id_normalized_to_six_digits(tmp_path: Path) -> None:
    api = one_version_api(law_body("9999", "시험법"), law_id="9999")
    run = run_fetch(tmp_path / "snap", api)
    assert run.result is not None, run.error
    assert run.result.manifest.laws[0].law_id == "009999"
    assert run.result.manifest.laws[0].file == "laws/009999.xml"
    assert run.api.params(1)["LID"] == "9999"  # 검색 응답의 원문 값으로 부른다


# --- 응답 검사 ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        HTML_ERROR,
        b"<html><body>error</body></html>",
        b"not xml at all",
        b'<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "b">]><x>&a;</x>',
    ],
    ids=["doctype-html", "html", "not-xml", "entity"],
)
def test_html_or_doctype_response(tmp_path: Path, body: bytes) -> None:
    api = one_version_api(body)
    run = run_fetch(tmp_path / "snap", api)
    message = error_text(run.error)
    assert message.startswith("시험법: 본문 — XML 응답이 아닙니다")
    assert API_HINT in message
    assert not (tmp_path / "snap").exists()


def test_html_search_response(tmp_path: Path) -> None:
    run = run_fetch(tmp_path / "snap", FakeLawApi(script=[Fail(status=200, raw=HTML_ERROR)]))
    message = error_text(run.error)
    assert message.startswith("시험법: 법령 검색 — XML 검색 응답이 아닙니다")
    assert API_HINT in message


def test_law_id_mismatch_in_body(tmp_path: Path) -> None:
    body = law_body("001111", "시험법")
    api = one_version_api(body)
    run = run_fetch(tmp_path / "snap", api)
    assert "본문의 법령ID가 검색한 법령ID(009999)와 다릅니다" in error_text(run.error)


def test_parse_search_fixture() -> None:
    parsed = parse_search((FIXTURES / "search_page.xml").read_bytes())
    assert parsed is not None
    total, rows = parsed
    assert total == 3
    assert [(r.name, r.law_id, r.mst, r.effective_date, r.kind) for r in rows] == [
        ("시험법", "009999", "100002", "20260101", "법률"),
        ("시험법", "009999", "100001", "20200701", "법률"),
        ("시험법 시행령", "009998", "200001", "20260101", "대통령령"),
    ]


# --- 재시도·간격 -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "fail",
    [Fail(status=429), Fail(status=503), Fail(exc=httpx.ReadTimeout), Fail(exc=httpx.ConnectError)],
    ids=["429", "503", "timeout", "connect"],
)
def test_retries_server_failures_three_times(tmp_path: Path, fail: Fail) -> None:
    run = run_fetch(tmp_path / "snap", FakeLawApi(script=[fail] * 3))
    assert run.error is not None
    assert run.api.calls == 3
    assert run.sleeps == [1.0, 2.0]


def test_retry_then_success(tmp_path: Path) -> None:
    run = run_fetch(
        tmp_path / "snap", FakeLawApi(script=[Fail(status=500), Fail(exc=httpx.ReadTimeout)])
    )
    assert run.result is not None, run.error
    assert run.api.calls == 5  # 검색 3번(실패 2번) + 판 목록 + 본문
    assert run.sleeps == [1.0, 2.0, CALL_INTERVAL_S, CALL_INTERVAL_S]


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_other_4xx_not_retried(tmp_path: Path, status: int) -> None:
    run = run_fetch(tmp_path / "snap", FakeLawApi(script=[Fail(status=status)]))
    assert f"HTTP {status}" in error_text(run.error)
    assert run.api.calls == 1
    assert run.sleeps == []


def test_call_interval(tmp_path: Path) -> None:
    run = run_fetch(tmp_path / "snap", names=["시험법", "시험법 시행령"])
    assert run.result is not None, run.error
    assert run.api.calls == 6
    assert run.sleeps == [CALL_INTERVAL_S] * 5
    assert [e.name for e in run.result.manifest.laws] == ["시험법", "시험법 시행령"]


# --- 프록시·리다이렉트 --------------------------------------------------------------------


def test_proxy_default_none(monkeypatch: pytest.MonkeyPatch) -> None:
    assert inspect.signature(fetch_law_snapshot).parameters["proxy"].default == "none"
    # httpx는 transport를 주면 환경 변수 프록시를 아예 쓰지 않는다. 그래서 transport 없이 만든
    # 클라이언트(실제 수집과 같음)에 프록시 transport가 붙는지로 확인한다.
    monkeypatch.setenv("HTTPS_PROXY", "http://localhost:9")
    monkeypatch.setenv("HTTP_PROXY", "http://localhost:9")
    with make_http_client("none") as client:
        assert client.trust_env is False
        assert client._mounts == {}
    with make_http_client("env") as client:
        assert client.trust_env is True
        transports = list(client._mounts.values())
        assert transports and all(isinstance(t, httpx.HTTPTransport) for t in transports)


def test_cli_proxy_default_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[object] = []

    def fake_fetch(*args: object, **kwargs: object) -> None:
        seen.append(kwargs["proxy"])
        raise law_api.LawFetchError("멈춤")

    monkeypatch.setattr(law_api, "fetch_law_snapshot", fake_fetch)
    monkeypatch.setenv("LAW_API_OC", FAKE_OC)
    args = ["corpus", "fetch-law", "--law", "시험법", "--as-of", "2026-10-29"]
    CliRunner().invoke(app, [*args, "--out", str(tmp_path / "s")])
    CliRunner().invoke(app, [*args, "--out", str(tmp_path / "s"), "--proxy", "env"])
    assert seen == ["none", "env"]


@pytest.mark.parametrize("exc", [httpx.ConnectError, httpx.ConnectTimeout])
def test_connect_failure_hints_proxy(tmp_path: Path, exc: type[httpx.RequestError]) -> None:
    run = run_fetch(tmp_path / "a", FakeLawApi(script=[Fail(exc=exc)] * 3))
    message = error_text(run.error)
    assert message.endswith("회사 프록시가 필요하면 --proxy env")
    assert FAKE_OC not in message and "law.go.kr" not in message

    run = run_fetch(tmp_path / "b", FakeLawApi(script=[Fail(exc=exc)] * 3), proxy="env")
    assert "--proxy env" not in error_text(run.error)


def test_no_proxy_hint_for_read_timeout(tmp_path: Path) -> None:
    run = run_fetch(tmp_path / "a", FakeLawApi(script=[Fail(exc=httpx.ReadTimeout)] * 3))
    assert "--proxy env" not in error_text(run.error)


def test_no_redirect_follow(tmp_path: Path) -> None:
    fail = Fail(status=302, headers={"Location": "https://localhost:9/elsewhere"})
    run = run_fetch(tmp_path / "snap", FakeLawApi(script=[fail]))
    assert "HTTP 302(리다이렉트는 따르지 않습니다)" in error_text(run.error)
    assert run.api.calls == 1


# --- 폴더 처리 ------------------------------------------------------------------------


def test_out_dir_exists_and_partial(tmp_path: Path) -> None:
    out = tmp_path / "snap"
    out.mkdir()
    run = run_fetch(out)
    assert "출력 폴더가 이미 있습니다(snap)" in error_text(run.error)
    assert run.api.calls == 0

    # 실패하면 임시 폴더가 남지 않는다.
    out2 = tmp_path / "snap2"
    run = run_fetch(out2, names=["없는법"])
    assert run.error is not None
    assert not out2.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["snap"]

    # 강제 종료로 남은 임시 폴더가 있으면 지우지 않고 멈춘다.
    leftover = tmp_path / ".snap3.partial"
    leftover.mkdir()
    run = run_fetch(tmp_path / "snap3")
    assert "남은 임시 폴더(.snap3.partial)를 지우고 다시 돌리세요" in error_text(run.error)
    assert run.api.calls == 0
    assert leftover.exists()


def test_bad_names_rejected_before_calls(tmp_path: Path) -> None:
    for names in ([], ["시험법", "시험법"], [" 시험법"], [""]):
        run = run_fetch(tmp_path / "snap", names=names)
        assert run.error is not None
        assert run.api.calls == 0


def test_creates_parent_dirs(tmp_path: Path) -> None:
    run = run_fetch(tmp_path / "workspace" / "snapshots" / "law-ko-20261029")
    assert run.result is not None, run.error
    assert (tmp_path / "workspace" / "snapshots" / "law-ko-20261029" / "manifest.json").is_file()


# --- manifest ---------------------------------------------------------------------------


def test_manifest_fields(tmp_path: Path) -> None:
    out = tmp_path / "snap"
    run = run_fetch(out, names=["시험법", "시험법 시행령"])
    assert run.result is not None, run.error
    assert sorted(str(p.relative_to(out).as_posix()) for p in out.rglob("*") if p.is_file()) == [
        "SOURCE.md",
        "laws/009998.xml",
        "laws/009999.xml",
        "manifest.json",
    ]
    raw = (out / "manifest.json").read_bytes()
    data = json.loads(raw)
    assert list(data) == [
        "schema",
        "source",
        "api",
        "as_of",
        "fetched_at",
        "ragexp_version",
        "laws",
        "content_hash",
    ]
    assert data["schema"] == "ragexp.law_snapshot/1"
    assert data["api"] == {"service": "lawService.do", "target": "eflaw", "type": "XML"}
    assert data["as_of"] == "2026-10-29"
    assert data["fetched_at"] == "2026-11-18T05:12:00Z"
    assert data["laws"][0] == {
        "name": "시험법",
        "law_id": "009999",
        "mst": "100002",
        "kind": "법률",
        "effective_date": "2026-01-01",
        "promulgation_date": "2025-12-01",
        "promulgation_no": "00001",
        "file": "laws/009999.xml",
        "sha256": hashlib.sha256((out / "laws/009999.xml").read_bytes()).hexdigest(),
        "bytes": len((out / "laws/009999.xml").read_bytes()),
    }
    assert data["laws"][1]["kind"] == "대통령령"
    assert data["content_hash"].startswith("snap-")
    # 본문 응답 바이트를 그대로 저장한다.
    assert (out / "laws/009999.xml").read_bytes() == law_body("009999", "시험법")
    # 요청 주소는 남기지 않는다.
    for needle in (b"http", b"lawSearch", b"lawService.do?", b"OC", b"query", b"LID"):
        assert needle not in raw.replace(SOURCE_NOTICE.encode(), b"")
    assert load_snapshot(out).content_hash == data["content_hash"]
    assert b"\r\n" not in raw


def test_source_notice_written(tmp_path: Path) -> None:
    out = tmp_path / "snap"
    run = run_fetch(out)
    assert run.result is not None, run.error
    source = (out / "SOURCE.md").read_text(encoding="utf-8")
    assert SOURCE_NOTICE in source
    assert "| 시험법 | 009999 | 100002 | 2026-01-01 |" in source
    assert "기준일: 2026-10-29" in source
    assert run.result.manifest.source == SOURCE_NOTICE


def test_cli_success_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    api = FakeLawApi(standard_laws())
    original = law_api.fetch_law_snapshot

    def with_fake(*args: object, **kwargs: object) -> object:
        return original(*args, **kwargs, transport=api, sleep=lambda s: None)  # type: ignore[arg-type]

    monkeypatch.setattr(law_api, "fetch_law_snapshot", with_fake)
    monkeypatch.setenv("LAW_API_OC", FAKE_OC)
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        app, ["corpus", "fetch-law", "--law", "시험법", "--as-of", "2026-10-29"]
    )
    assert result.exit_code == 0, result.output
    assert "시험법: 법령ID 009999, MST 100002, 시행일자 2026-01-01" in result.output
    assert "내용 해시: snap-" in result.output
    assert "검색 응답에 인증키가 들어 있었나: 예(검색 응답은 저장하지 않음)" in result.output
    assert (tmp_path / "workspace" / "snapshots" / "law-ko-20261029" / "manifest.json").is_file()


@pytest.mark.parametrize("as_of", ["2026-1-29", "20261029", "2026-02-30"])
def test_cli_bad_as_of(as_of: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LAW_API_OC", FAKE_OC)
    result = CliRunner().invoke(app, ["corpus", "fetch-law", "--law", "시험법", "--as-of", as_of])
    assert result.exit_code == 1
    assert "--as-of" in result.output


# --- 실제 API(기본 제외) -------------------------------------------------------------------


@pytest.mark.network
def test_fetch_real_one_law(tmp_path: Path) -> None:
    if not os.environ.get("LAW_API_OC", "").strip():
        pytest.skip("LAW_API_OC가 없어 건너뜀")
    out = tmp_path / "snap"
    result = CliRunner().invoke(
        app,
        ["corpus", "fetch-law", "--law", "국가재정법", "--as-of", "2026-10-29", "--out", str(out)],
    )
    assert result.exit_code == 0, result.output
    snapshot = load_snapshot(out)
    assert [e.name for e in snapshot.manifest.laws] == ["국가재정법"]
    assert snapshot.manifest.laws[0].effective_date <= "2026-10-29"
