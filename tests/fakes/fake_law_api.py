"""가짜 국가법령정보센터 Open API(httpx `MockTransport`, 소켓 없음).

- `lawSearch.do`(target=eflaw): `query`(법령명 부분 일치)나 `LID`(법령ID)로 판 행을 돌려준다.
  `display`·`page`로 쪽을 나눈다. `법령상세링크`에 요청한 OC를 되돌려 넣는다(실제 API처럼).
- `lawService.do`(target=eflaw): `MST`+`efYd` 짝이 맞으면 본문 XML, 아니면 HTML 오류 페이지.
  `ID`를 주면 HTML 오류 페이지(실측 기록: `ID`+`efYd`는 쓰지 않음).
- `script`: 호출마다 정해 둔 실패(상태 코드·httpx 예외·임의 바이트)를 먼저 돌려준다.
  `None`이면 정상.

응답 형식은 합성 고정 자료 `tests/fixtures/law/`와 같은 칸 이름을 쓴다.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from xml.sax.saxutils import escape

import httpx
from pydantic import SecretStr

from ragexp.datasets.law_api import FetchResult, LawFetchError, ProxyMode, fetch_law_snapshot

FAKE_OC = "oc-test-SECRET-7f3a"
"""시험용 가짜 인증키. 레포의 고정 자료에는 이 값만 쓴다."""
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "law"
BODY_FIXTURE = FIXTURES / "body_009999.xml"

HTML_ERROR = (
    b"<!DOCTYPE html>\n<html><head><title>error</title></head><body>request failed</body></html>\n"
)


def html_error_with_oc(oc: str) -> bytes:
    """오류 페이지에 OC가 되돌아오는 경우(새지 않는지 시험)."""
    return HTML_ERROR.replace(b"request failed", f"request failed OC={oc}".encode())


@dataclass(frozen=True)
class FakeVersion:
    mst: str
    effective: str  # YYYYMMDD
    promulgation_date: str = "20250101"
    promulgation_no: str = "00001"
    body: bytes | None = None


@dataclass(frozen=True)
class FakeLaw:
    name: str
    law_id: str
    kind: str
    versions: tuple[FakeVersion, ...]


def law_body(law_id: str, name: str, *, marker: str = "") -> bytes:
    """합성 본문(시험법)의 법령ID·법령명을 바꾼 본문. `marker`를 넣으면 바이트가 달라진다."""
    text = BODY_FIXTURE.read_text(encoding="utf-8")
    text = text.replace("<법령ID>009999</법령ID>", f"<법령ID>{law_id}</법령ID>")
    text = text.replace("<![CDATA[시험법]]></법령명_한글>", f"<![CDATA[{name}]]></법령명_한글>")
    if marker:
        text = text.replace("</법령>", f"<!-- {marker} -->\n</법령>")
    return text.encode("utf-8")


def standard_laws() -> list[FakeLaw]:
    """시험법(판 셋: 지난·현행·시행예정)과 시험법 시행령(판 하나)."""
    return [
        FakeLaw(
            "시험법",
            "009999",
            "법률",
            (
                FakeVersion("100001", "20200701", "20200101", "00000"),
                FakeVersion("100002", "20260101", "20251201", "00001"),
                FakeVersion("100003", "20270101", "20261101", "00009"),
            ),
        ),
        FakeLaw(
            "시험법 시행령",
            "009998",
            "대통령령",
            (FakeVersion("200001", "20260101", "20251215", "00002"),),
        ),
    ]


@dataclass(frozen=True)
class Fail:
    """한 번의 실패. `exc`가 있으면 그 예외(메시지에 요청 주소·OC를 넣음), 아니면 `status` 응답."""

    status: int = 500
    exc: type[httpx.RequestError] | None = None
    raw: bytes | None = None
    headers: dict[str, str] = field(default_factory=dict)


def _row_xml(law: FakeLaw, v: FakeVersion, index: int, oc: str | None) -> str:
    link = f"/DRF/lawService.do?OC={oc}&amp;" if oc is not None else "/DRF/lawService.do?"
    link += f"target=eflaw&amp;MST={v.mst}&amp;type=HTML&amp;efYd={v.effective}"
    return (
        f'  <law id="{index}">\n'
        f"    <법령일련번호>{v.mst}</법령일련번호>\n"
        f"    <법령명한글>{escape(law.name)}</법령명한글>\n"
        f"    <법령ID>{law.law_id}</법령ID>\n"
        f"    <공포일자>{v.promulgation_date}</공포일자>\n"
        f"    <공포번호>{v.promulgation_no}</공포번호>\n"
        f"    <법령구분명>{escape(law.kind)}</법령구분명>\n"
        f"    <시행일자>{v.effective}</시행일자>\n"
        f"    <법령상세링크>{link}</법령상세링크>\n"
        f"  </law>\n"
    )


class FakeLawApi(httpx.MockTransport):
    """가짜 법령 API. 받은 요청을 `requests`에 남긴다."""

    def __init__(
        self,
        laws: Sequence[FakeLaw] | None = None,
        *,
        script: Sequence[Fail | None] = (),
        oc: str = FAKE_OC,
        echo_oc_in_search: bool = True,
        echo_oc_in_body: bool = False,
        extra_search_rows: Sequence[FakeLaw] = (),
    ) -> None:
        self.laws = list(laws) if laws is not None else standard_laws()
        self.extra_search_rows = list(extra_search_rows)
        self.script = list(script)
        self.oc = oc
        self.echo_oc_in_search = echo_oc_in_search
        self.echo_oc_in_body = echo_oc_in_body
        self.requests: list[httpx.Request] = []
        super().__init__(self._handle)

    # --- 기록 -------------------------------------------------------------------------

    @property
    def calls(self) -> int:
        return len(self.requests)

    def params(self, i: int) -> dict[str, str]:
        return dict(self.requests[i].url.params)

    def paths(self) -> list[str]:
        return [r.url.path.rsplit("/", 1)[-1] for r in self.requests]

    def service_params(self) -> list[dict[str, str]]:
        return [dict(r.url.params) for r in self.requests if r.url.path.endswith("lawService.do")]

    # --- 응답 -------------------------------------------------------------------------

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.script:
            step = self.script.pop(0)
            if step is not None:
                return self._fail(step, request)
        params = request.url.params
        if params.get("OC") != self.oc:
            return httpx.Response(200, content=HTML_ERROR)
        path = request.url.path
        if path.endswith("/lawSearch.do"):
            return self._search(request)
        if path.endswith("/lawService.do"):
            return self._service(request)
        return httpx.Response(404, content=HTML_ERROR)

    @staticmethod
    def _fail(step: Fail, request: httpx.Request) -> httpx.Response:
        if step.exc is not None:
            raise step.exc(f"failed for {request.url}", request=request)
        content = step.raw if step.raw is not None else f"error at {request.url}".encode()
        return httpx.Response(step.status, content=content, headers=step.headers)

    def _search(self, request: httpx.Request) -> httpx.Response:
        params = request.url.params
        if params.get("target") != "eflaw" or params.get("type") != "XML":
            return httpx.Response(200, content=HTML_ERROR)
        display = int(params.get("display", "20"))
        page = int(params.get("page", "1"))
        rows: list[tuple[FakeLaw, FakeVersion]] = []
        if "query" in params:
            query = params["query"]
            for law in [*self.laws, *self.extra_search_rows]:
                if query in law.name:
                    rows += [(law, v) for v in law.versions]
        elif "LID" in params:
            for law in self.laws:
                if law.law_id == params["LID"]:
                    rows += [(law, v) for v in law.versions]
        chunk = rows[(page - 1) * display : page * display]
        oc = self.oc if self.echo_oc_in_search else None
        body = (
            '<?xml version="1.0" encoding="UTF-8"?>\n<LawSearch>\n'
            "  <target>eflaw</target>\n"
            f"  <totalCnt>{len(rows)}</totalCnt>\n  <page>{page}</page>\n"
            + "".join(
                _row_xml(law, v, (page - 1) * display + i + 1, oc)
                for i, (law, v) in enumerate(chunk)
            )
            + "</LawSearch>\n"
        )
        return httpx.Response(200, content=body.encode("utf-8"))

    def _service(self, request: httpx.Request) -> httpx.Response:
        params = request.url.params
        if "ID" in params or params.get("target") != "eflaw" or params.get("type") != "XML":
            return httpx.Response(200, content=HTML_ERROR)
        for law in self.laws:
            for v in law.versions:
                if v.mst == params.get("MST") and v.effective == params.get("efYd"):
                    body = v.body if v.body is not None else law_body(law.law_id, law.name)
                    if self.echo_oc_in_body:
                        body = body.replace("</법령>".encode(), b"")
                        body += f"<!-- OC={self.oc} -->\n</법령>\n".encode()
                    return httpx.Response(200, content=body)
        return httpx.Response(200, content=HTML_ERROR)


# --- 시험 도우미 --------------------------------------------------------------------

PROXY_ENV = (
    "HTTP_PROXY",
    "http_proxy",
    "HTTPS_PROXY",
    "https_proxy",
    "ALL_PROXY",
    "all_proxy",
    "NO_PROXY",
    "no_proxy",
)
FIXED_NOW = datetime(2026, 11, 18, 5, 12, 0, tzinfo=UTC)
AS_OF = date(2026, 10, 29)


@dataclass
class FetchRun:
    """`fetch_law_snapshot`을 가짜 API로 돌린 결과. 실패면 `error`에 예외가 있다."""

    api: FakeLawApi
    sleeps: list[float]
    result: FetchResult | None = None
    error: Exception | None = None


def run_fetch(
    out_dir: Path,
    api: FakeLawApi | None = None,
    names: Sequence[str] = ("시험법",),
    *,
    as_of: date = AS_OF,
    now: datetime = FIXED_NOW,
    oc: str = FAKE_OC,
    proxy: ProxyMode = "none",
) -> FetchRun:
    """가짜 API로 수집한다. 대기는 기록만 하고 기다리지 않는다. `LawFetchError`는 잡아 돌려준다."""
    run = FetchRun(api=api if api is not None else FakeLawApi(), sleeps=[])
    try:
        run.result = fetch_law_snapshot(
            names,
            as_of,
            out_dir,
            oc=SecretStr(oc),
            transport=run.api,
            sleep=run.sleeps.append,
            now=lambda: now,
            proxy=proxy,
        )
    except LawFetchError as exc:
        run.error = exc
    return run
