"""법령 수집·스냅샷(U4, 설계 §6.1): 국가법령정보센터 Open API(eflaw)에서 기준일 판을 받는다.

순서(docs/units/U4.md 2절):
1. `lawSearch.do?target=eflaw&query=<법령명>`: 법령명이 정확히 같은 행의 법령ID를 찾는다.
2. `lawSearch.do?target=eflaw&LID=<법령ID>&nw=1,2,3`: 판 목록(모든 쪽)에서
   기준일에 시행 중인 판을 고른다.
3. `lawService.do?target=eflaw&MST=<법령일련번호>&efYd=<그 판의 시행일자>`:
   본문 XML을 그대로 저장한다.

인증키(OC)가 새지 않게 하는 규칙(4절):
- 오류 문구는 고정 문구다(법령명·단계·오류 종류·상태 코드). 주소·응답 본문을 넣지 않는다.
- 예외는 `except` 밖에서 낸다. 그래서 `__context__`에 httpx 예외(요청 주소 포함)가 남지 않는다.
- 수집하는 동안 httpx·httpcore 로거를 WARNING 이상으로 올린다(INFO에서 요청 주소를 찍음).
- 검색 응답은 저장하지 않는다. 본문 응답에 OC 문자열이 있으면 저장을 거부한다.
- 리다이렉트를 따르지 않는다. 프록시는 기본 `none`(환경 변수·시스템 프록시를 읽지 않음).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from types import TracebackType
from typing import Literal, Self
from urllib.parse import quote

import httpx
from pydantic import Field, SecretStr, ValidationError

from ragexp import __version__
from ragexp.types import Frozen
from ragexp.versioning import canonical_json, sha256_hex

LAW_API_BASE = "https://www.law.go.kr/DRF/"
"""API 주소(코드에 고정, 구성 YAML에는 없음). https만 쓴다."""
SEARCH_PATH = "lawSearch.do"
SERVICE_PATH = "lawService.do"
OC_ENV = "LAW_API_OC"

SNAPSHOT_SCHEMA = "ragexp.law_snapshot/1"
SOURCE_NOTICE = (
    "출처: 국가법령정보센터(법제처), https://www.law.go.kr — 공공누리 제1유형. "
    "법적 효력이 없는 참고자료입니다."
)
CONTENT_HASH_PREFIX = "snap-"

TIMEOUT_S = 30.0
MAX_ATTEMPTS = 3
RETRY_WAITS_S = (1.0, 2.0)
"""다시 시도하기 전 대기(첫 실패 뒤 1초, 두 번째 실패 뒤 2초)."""
CALL_INTERVAL_S = 0.5
"""호출과 호출 사이 쉬는 시간(과도한 호출 제한 대비)."""
DISPLAY = 100
"""검색 한 쪽의 행 수."""
MAX_PAGES = 50

ProxyMode = Literal["none", "env"]

API_HINT = "인증키 승인과 신청한 법령 종류, MST·시행일 짝을 확인하세요"
MISSING_OC = f"{OC_ENV} 환경 변수를 넣으세요"
_PROXY_HINT = ". 회사 프록시가 필요하면 --proxy env"
_LAW_FILE_RE = re.compile(r"laws/([0-9]{6})\.xml")
_YMD_RE = re.compile(r"[0-9]{8}")


class LawFetchError(ValueError):
    """수집 실패. 메시지는 고정 문구(법령명·단계·오류 종류·상태 코드)뿐이다."""


class SnapshotError(ValueError):
    """스냅샷 검증 실패. 메시지에는 스냅샷 안의 파일 이름만 넣는다."""


# --- manifest -----------------------------------------------------------------------


class ApiInfo(Frozen):
    """본문을 받은 호출 종류. 주소·OC는 남기지 않는다."""

    service: str
    target: str
    type: str


class LawEntry(Frozen):
    """스냅샷의 법령 하나(판 하나)."""

    name: str = Field(min_length=1)
    law_id: str = Field(pattern=r"^[0-9]{6}$")
    mst: str = Field(pattern=r"^[0-9]+$")
    kind: str
    effective_date: str = Field(pattern=r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
    promulgation_date: str = Field(pattern=r"^([0-9]{4}-[0-9]{2}-[0-9]{2})?$")
    promulgation_no: str
    file: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: int = Field(ge=0)


class SnapshotManifest(Frozen):
    """`manifest.json`. `content_hash`는 파일 해시만으로 만든다(받은 날짜·기준일은 빠짐)."""

    schema_: Literal["ragexp.law_snapshot/1"] = Field(alias="schema")
    source: str
    api: ApiInfo
    as_of: str = Field(pattern=r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
    fetched_at: str
    ragexp_version: str
    laws: tuple[LawEntry, ...] = Field(min_length=1)
    content_hash: str = Field(pattern=r"^snap-[0-9a-f]{64}$")

    def to_json_bytes(self) -> bytes:
        data = self.model_dump(mode="json", by_alias=True)
        return (json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


@dataclass(frozen=True)
class Snapshot:
    """검증을 통과한 스냅샷. `content_hash`는 파일에서 **다시 계산한** 값이다."""

    path: Path
    manifest: SnapshotManifest
    content_hash: str

    def law_path(self, entry: LawEntry) -> Path:
        return self.path / entry.file


@dataclass(frozen=True)
class FetchResult:
    path: Path
    manifest: SnapshotManifest
    search_echoes_oc: bool
    """검색 응답(저장하지 않음)에 OC가 들어 있었는지. 구조 확인용(5절)."""


def snapshot_content_hash(entries: Iterable[tuple[str, str]]) -> str:
    """`(file, sha256)` 쌍을 file 순으로 정렬해 정규 JSON으로 만들고 sha256을 낸다."""
    pairs = sorted([file, digest] for file, digest in entries)
    return CONTENT_HASH_PREFIX + sha256_hex(canonical_json(pairs))


def law_file_name(law_id: str) -> str:
    return f"laws/{law_id}.xml"


# --- 스냅샷 읽기 ---------------------------------------------------------------------


def _read_manifest(path: Path) -> SnapshotManifest | str:
    """manifest를 읽어 검증한다. 실패하면 오류 문구를 돌려준다(예외는 부른 쪽이 밖에서 냄)."""
    try:
        raw = (path / "manifest.json").read_bytes()
    except OSError:
        return "manifest.json을 읽을 수 없습니다"
    try:
        data = json.loads(raw)
    except ValueError:
        return "manifest.json이 JSON이 아닙니다"
    if not isinstance(data, dict) or data.get("schema") != SNAPSHOT_SCHEMA:
        return f"manifest.json의 schema가 {SNAPSHOT_SCHEMA}가 아닙니다(알 수 없는 스냅샷 형식)"
    try:
        return SnapshotManifest.model_validate(data)
    except ValidationError as exc:
        # 값(input)은 넣지 않고 칸 위치만 넣는다.
        locs = sorted({".".join(str(p) for p in e["loc"]) for e in exc.errors()})
        return f"manifest.json 형식 오류({', '.join(locs)})"


def _check_entries(manifest: SnapshotManifest) -> str | None:
    seen: set[str] = set()
    for entry in manifest.laws:
        m = _LAW_FILE_RE.fullmatch(entry.file)
        if m is None:
            return f"manifest의 file 경로가 laws/<6자리 숫자>.xml 꼴이 아닙니다({entry.law_id})"
        if m.group(1) != entry.law_id:
            return f"manifest의 file({entry.file})이 법령ID({entry.law_id})와 맞지 않습니다"
        if entry.file in seen:
            return f"manifest에 같은 파일이 두 번 있습니다({entry.file})"
        seen.add(entry.file)
    return None


def load_snapshot(path: Path) -> Snapshot:
    """스냅샷을 검증해 읽는다: schema, 파일 경로, 파일별 sha256, 내용 해시를 모두 다시 확인한다."""
    path = Path(path)
    manifest = _read_manifest(path)
    if isinstance(manifest, str):
        raise SnapshotError(manifest)
    problem = _check_entries(manifest)
    if problem is not None:
        raise SnapshotError(problem)
    pairs: list[tuple[str, str]] = []
    for entry in manifest.laws:
        data = _read_bytes(path / entry.file)
        if data is None:
            raise SnapshotError(f"스냅샷 파일을 읽을 수 없습니다({entry.file})")
        digest = hashlib.sha256(data).hexdigest()
        if digest != entry.sha256 or len(data) != entry.bytes:
            raise SnapshotError(f"스냅샷 파일이 manifest의 해시와 다릅니다({entry.file})")
        pairs.append((entry.file, digest))
    recomputed = snapshot_content_hash(pairs)
    if recomputed != manifest.content_hash:
        raise SnapshotError("manifest.json의 content_hash가 파일에서 다시 계산한 값과 다릅니다")
    return Snapshot(path=path, manifest=manifest, content_hash=recomputed)


def _read_bytes(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except OSError:
        return None


# --- 응답 읽기 -----------------------------------------------------------------------


def _xml_root(content: bytes) -> ET.Element | None:
    """XML로 읽는다. `<!DOCTYPE`·`<!ENTITY`가 있거나(HTML 오류 페이지 등) 읽을 수 없으면 None."""
    lowered = content.lower()
    if b"<!doctype" in lowered or b"<!entity" in lowered:
        return None
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return None
    if root.tag.lower() == "html":
        return None
    return root


def _text(el: ET.Element, tag: str) -> str:
    return (el.findtext(tag) or "").strip()


def _ascii_digits(value: str) -> bool:
    """ASCII 숫자로만 된 비지 않은 문자열인지(전각·아라비아 숫자 등은 거부)."""
    return value.isascii() and value.isdigit()


def normalize_law_id(raw: str) -> str | None:
    """법령ID를 0으로 채운 6자리 문자열로 만든다. 6자리 숫자로 만들 수 없으면 None."""
    raw = raw.strip()
    if not _ascii_digits(raw) or len(raw) > 6:
        return None
    return raw.zfill(6)


def _ymd(raw: str) -> str | None:
    """`YYYYMMDD` → `YYYY-MM-DD`. 형식이 틀리면 None."""
    if not _YMD_RE.fullmatch(raw):
        return None
    try:
        return date(int(raw[:4]), int(raw[4:6]), int(raw[6:])).isoformat()
    except ValueError:
        return None


@dataclass(frozen=True)
class SearchRow:
    """검색 응답(①·②)의 행 하나. 원문 문자열 그대로."""

    name: str
    law_id: str
    mst: str
    effective_date: str
    promulgation_date: str
    promulgation_no: str
    kind: str


def parse_search(content: bytes) -> tuple[int, list[SearchRow]] | None:
    """검색 응답을 읽는다. XML이 아니거나 `totalCnt`가 없으면 None."""
    root = _xml_root(content)
    if root is None:
        return None
    total = _text(root, "totalCnt")
    if not _ascii_digits(total):
        return None
    rows = [
        SearchRow(
            name=_text(el, "법령명한글"),
            law_id=_text(el, "법령ID"),
            mst=_text(el, "법령일련번호"),
            effective_date=_text(el, "시행일자"),
            promulgation_date=_text(el, "공포일자"),
            promulgation_no=_text(el, "공포번호"),
            kind=_text(el, "법령구분명"),
        )
        for el in root.iter("law")
        if el is not root
    ]
    return int(total), rows


@dataclass(frozen=True)
class _Version:
    row: SearchRow
    effective: str  # YYYY-MM-DD
    promulgation: str  # YYYY-MM-DD 또는 빈 문자열


def _version_key(v: _Version) -> tuple[str, str, int, int]:
    no = v.row.promulgation_no
    return (v.effective, v.promulgation, int(no) if _ascii_digits(no) else -1, int(v.row.mst))


def pick_version(rows: Sequence[SearchRow], law_id: str, as_of: date) -> _Version | str:
    """기준일에 시행 중인 판: 시행일자 ≤ 기준일 중 가장 늦은 판. 같으면 공포일자→공포번호→MST.

    판 목록의 순서와 상관없이 같은 판을 고른다. 고를 수 없으면 오류 문구를 돌려준다.
    """
    versions: list[_Version] = []
    for row in rows:
        if normalize_law_id(row.law_id) != law_id:
            continue
        effective = _ymd(row.effective_date)
        promulgation = _ymd(row.promulgation_date) if row.promulgation_date else ""
        if effective is None or promulgation is None or not _ascii_digits(row.mst):
            return "판 목록 형식 오류(시행일자·공포일자·법령일련번호)"
        versions.append(_Version(row, effective, promulgation))
    if not versions:
        return "판 목록이 비어 있습니다"
    in_force = [v for v in versions if v.effective <= as_of.isoformat()]
    if not in_force:
        return f"기준일({as_of.isoformat()}) 이전에 시행된 판이 없습니다"
    return max(in_force, key=_version_key)


# --- 호출 ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Failure:
    message: str
    retry: bool
    connect: bool = False


@contextmanager
def quiet_http_logs() -> Iterator[None]:
    """httpx·httpcore 로거(하위 로거 포함)를 WARNING 이상으로 올리고, 끝나면 되돌린다."""
    names = {"httpx", "httpcore"}
    names.update(
        n for n in list(logging.root.manager.loggerDict) if n.startswith(("httpx.", "httpcore."))
    )
    saved: list[tuple[logging.Logger, int]] = []
    for name in sorted(names):
        logger = logging.getLogger(name)
        saved.append((logger, logger.level))
        if logger.getEffectiveLevel() < logging.WARNING:
            logger.setLevel(logging.WARNING)
    try:
        yield
    finally:
        for logger, level in saved:
            logger.setLevel(level)


def make_http_client(
    proxy: ProxyMode, transport: httpx.BaseTransport | None = None
) -> httpx.Client:
    """수집용 httpx 클라이언트.

    리다이렉트를 따르지 않고, `none`이면 환경 변수·시스템 프록시를 읽지 않는다.
    """
    return httpx.Client(
        base_url=LAW_API_BASE,
        transport=transport,
        timeout=TIMEOUT_S,
        follow_redirects=False,
        trust_env=proxy == "env",
    )


class _LawApi:
    """호출 한 곳. 재시도·호출 간격·OC 붙이기를 여기서만 한다."""

    def __init__(
        self,
        oc: SecretStr,
        proxy: ProxyMode,
        transport: httpx.BaseTransport | None,
        sleep: Callable[[float], None],
    ) -> None:
        self._oc = oc
        self._proxy = proxy
        self._sleep = sleep
        self._http = make_http_client(proxy, transport)
        self._calls = 0
        self.search_echoes_oc = False

    def __repr__(self) -> str:
        return f"_LawApi(proxy={self._proxy!r})"

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._http.close()

    def contains_oc(self, content: bytes) -> bool:
        value = self._oc.get_secret_value()
        needles = {value.encode("utf-8"), quote(value, safe="").encode("ascii")}
        return any(n in content for n in needles)

    def get(self, path: str, params: Mapping[str, str], *, law: str, step: str) -> bytes:
        """한 번 부른다(재시도 포함). 실패하면 고정 문구의 `LawFetchError`를 낸다."""
        if self._calls:
            self._sleep(CALL_INTERVAL_S)
        self._calls += 1
        attempt = 0
        while True:
            attempt += 1
            outcome = self._attempt(path, params)
            if isinstance(outcome, bytes):
                return outcome
            if outcome.retry and attempt < MAX_ATTEMPTS:
                self._sleep(RETRY_WAITS_S[attempt - 1])
                continue
            break
        hint = _PROXY_HINT if outcome.connect and self._proxy == "none" else ""
        raise LawFetchError(f"{law}: {step} 실패 — {outcome.message}{hint}")

    def _attempt(self, path: str, params: Mapping[str, str]) -> bytes | _Failure:
        """한 번 시도한다. httpx 예외는 종류만 남기고 문자열(요청 주소 포함)은 버린다."""
        query = {**params, "OC": self._oc.get_secret_value()}
        try:
            response = self._http.get(path, params=query)
        except httpx.TimeoutException as exc:
            connect = isinstance(exc, httpx.ConnectTimeout)
            what = "연결 시간 초과" if connect else "응답 시간 초과"
            return _Failure(f"{what}({TIMEOUT_S:g}초)", retry=True, connect=connect)
        except httpx.NetworkError as exc:
            connect = isinstance(exc, httpx.ConnectError)
            return _Failure(f"연결 실패({type(exc).__name__})", retry=True, connect=connect)
        except httpx.RemoteProtocolError:
            return _Failure("연결 실패(서버가 응답 중 연결을 끊음)", retry=True)
        except httpx.HTTPError as exc:
            return _Failure(f"호출 오류({type(exc).__name__})", retry=False)
        status = response.status_code
        if status == 200:
            return response.content
        if status == 429 or 500 <= status < 600:
            return _Failure(f"HTTP {status}", retry=True)
        if 300 <= status < 400:
            return _Failure(f"HTTP {status}(리다이렉트는 따르지 않습니다)", retry=False)
        return _Failure(f"HTTP {status}", retry=False)

    def search_all(self, params: Mapping[str, str], *, law: str, step: str) -> list[SearchRow]:
        """검색 응답을 모든 쪽 읽는다(`totalCnt` 기준)."""
        rows: list[SearchRow] = []
        page = 1
        while True:
            content = self.get(
                SEARCH_PATH,
                {**params, "display": str(DISPLAY), "page": str(page)},
                law=law,
                step=step,
            )
            if self.contains_oc(content):
                self.search_echoes_oc = True
            parsed = parse_search(content)
            if parsed is None:
                raise LawFetchError(
                    f"{law}: {step} — XML 검색 응답이 아닙니다(HTML 오류 페이지 등). {API_HINT}"
                )
            total, page_rows = parsed
            rows.extend(page_rows)
            if not page_rows or page * DISPLAY >= total:
                return rows
            page += 1
            if page > MAX_PAGES:
                raise LawFetchError(f"{law}: {step} — 검색 결과가 {MAX_PAGES}쪽을 넘습니다")


# --- 수집 ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _find_law_id(api: _LawApi, name: str) -> tuple[str, str]:
    """① 법령명이 정확히 같은 행의 법령ID. (정규화한 값, 원문 값)을 돌려준다."""
    rows = api.search_all(
        {"target": "eflaw", "type": "XML", "query": name}, law=name, step="법령 검색"
    )
    exact = [r for r in rows if r.name == name]
    ids: dict[str, str] = {}
    for row in exact:
        norm = normalize_law_id(row.law_id)
        if norm is None:
            raise LawFetchError(f"{name}: 법령 검색 — 법령ID가 6자리 숫자가 아닙니다")
        ids.setdefault(norm, row.law_id)
    if len(ids) == 1:
        ((norm, raw),) = ids.items()
        return norm, raw
    if not ids:
        names = sorted({r.name for r in rows if r.name})[:10]
        listed = ", ".join(names) if names else "없음"
        raise LawFetchError(
            f"{name}: 법령 검색 — 이름이 정확히 같은 법령이 없습니다. 검색 결과의 이름: {listed}"
        )
    raise LawFetchError(
        f"{name}: 법령 검색 — 이름이 같은 법령이 여럿입니다(법령ID {', '.join(sorted(ids))})"
    )


def _fetch_one(api: _LawApi, name: str, as_of: date, folder: Path) -> LawEntry:
    law_id, raw_id = _find_law_id(api, name)
    rows = api.search_all(
        {"target": "eflaw", "type": "XML", "LID": raw_id, "nw": "1,2,3"},
        law=name,
        step="판 목록",
    )
    picked = pick_version(rows, law_id, as_of)
    if isinstance(picked, str):
        raise LawFetchError(f"{name}: 판 목록 — {picked}")
    row = picked.row
    content = api.get(
        SERVICE_PATH,
        {"target": "eflaw", "type": "XML", "MST": row.mst, "efYd": row.effective_date},
        law=name,
        step="본문",
    )
    root = _xml_root(content)
    if root is None:
        raise LawFetchError(f"{name}: 본문 — XML 응답이 아닙니다(HTML 오류 페이지 등). {API_HINT}")
    body_id = root.findtext(".//법령ID")
    if body_id is None:
        raise LawFetchError(f"{name}: 본문 — 법령ID가 없습니다. {API_HINT}")
    if normalize_law_id(body_id) != law_id:
        raise LawFetchError(f"{name}: 본문 — 본문의 법령ID가 검색한 법령ID({law_id})와 다릅니다")
    if api.contains_oc(content):
        raise LawFetchError(
            f"{name}: 본문 — 응답에 인증키(OC)와 같은 문자열이 있어 저장하지 않습니다"
            "(OC가 아주 짧거나 흔한 문자열이면 잘못 걸릴 수 있습니다)"
        )
    file = law_file_name(law_id)
    target = folder / file
    if target.exists():
        raise LawFetchError(f"{name}: 법령ID({law_id})가 다른 법령명과 겹칩니다")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    return LawEntry(
        name=name,
        law_id=law_id,
        mst=row.mst,
        kind=row.kind,
        effective_date=picked.effective,
        promulgation_date=picked.promulgation,
        promulgation_no=row.promulgation_no,
        file=file,
        sha256=hashlib.sha256(content).hexdigest(),
        bytes=len(content),
    )


def _source_md(manifest: SnapshotManifest) -> str:
    lines = [
        "# 출처",
        "",
        manifest.source,
        "",
        f"- 기준일: {manifest.as_of} (그날 시행 중인 판)",
        f"- 받은 날짜: {manifest.fetched_at}",
        "- 별표 파일(HWP·PDF)은 받지 않았습니다.",
        "",
        "| 법령명 | 법령ID | MST | 시행일자 |",
        "| --- | --- | --- | --- |",
    ]
    lines += [f"| {e.name} | {e.law_id} | {e.mst} | {e.effective_date} |" for e in manifest.laws]
    return "\n".join(lines) + "\n"


def _check_names(names: Sequence[str]) -> str | None:
    if not names:
        return "법령명을 하나 이상 주세요(--law)"
    if any(not n.strip() or n != n.strip() for n in names):
        return "법령명이 비었거나 앞뒤에 공백이 있습니다"
    if len(set(names)) != len(names):
        return "같은 법령명이 두 번 있습니다"
    return None


def partial_dir(out_dir: Path) -> Path:
    return out_dir.parent / f".{out_dir.name}.partial"


def fetch_law_snapshot(
    names: Sequence[str],
    as_of: date,
    out_dir: Path,
    *,
    oc: SecretStr,
    transport: httpx.BaseTransport | None = None,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], datetime] = _utcnow,
    proxy: ProxyMode = "none",
) -> FetchResult:
    """법령들의 기준일 판을 받아 스냅샷 폴더(`laws/*.xml`·`manifest.json`·`SOURCE.md`)를 만든다.

    임시 폴더(`.<이름>.partial`)에 쓰고 모두 성공하면 이름을 바꾼다. 실패하면 임시 폴더를 지운다.
    """
    out_dir = Path(out_dir)
    problem = _check_names(names)
    if problem is not None:
        raise LawFetchError(problem)
    if not oc.get_secret_value().strip():
        raise LawFetchError(MISSING_OC)
    if out_dir.exists():
        raise LawFetchError(f"출력 폴더가 이미 있습니다({out_dir.name}). 스냅샷은 바꾸지 않습니다")
    partial = partial_dir(out_dir)
    if partial.exists():
        raise LawFetchError(f"남은 임시 폴더({partial.name})를 지우고 다시 돌리세요")

    partial.mkdir(parents=True)
    try:
        with quiet_http_logs(), _LawApi(oc, proxy, transport, sleep) as api:
            entries = [_fetch_one(api, name, as_of, partial) for name in names]
            echoes = api.search_echoes_oc
        manifest = SnapshotManifest.model_validate(
            {
                "schema": SNAPSHOT_SCHEMA,
                "source": SOURCE_NOTICE,
                "api": {"service": SERVICE_PATH, "target": "eflaw", "type": "XML"},
                "as_of": as_of.isoformat(),
                "fetched_at": now().astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "ragexp_version": __version__,
                "laws": [e.model_dump() for e in entries],
                "content_hash": snapshot_content_hash((e.file, e.sha256) for e in entries),
            }
        )
        (partial / "manifest.json").write_bytes(manifest.to_json_bytes())
        (partial / "SOURCE.md").write_bytes(_source_md(manifest).encode("utf-8"))
        partial.rename(out_dir)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return FetchResult(path=out_dir, manifest=manifest, search_echoes_oc=echoes)


def oc_from_env(environ: Mapping[str, str]) -> SecretStr | None:
    value = environ.get(OC_ENV, "").strip()
    return SecretStr(value) if value else None


__all__ = [
    "API_HINT",
    "CALL_INTERVAL_S",
    "DISPLAY",
    "LAW_API_BASE",
    "MAX_ATTEMPTS",
    "MISSING_OC",
    "OC_ENV",
    "RETRY_WAITS_S",
    "SNAPSHOT_SCHEMA",
    "SOURCE_NOTICE",
    "ApiInfo",
    "FetchResult",
    "LawEntry",
    "LawFetchError",
    "ProxyMode",
    "SearchRow",
    "Snapshot",
    "SnapshotError",
    "SnapshotManifest",
    "fetch_law_snapshot",
    "load_snapshot",
    "make_http_client",
    "normalize_law_id",
    "oc_from_env",
    "parse_search",
    "partial_dir",
    "pick_version",
    "quiet_http_logs",
    "snapshot_content_hash",
]
