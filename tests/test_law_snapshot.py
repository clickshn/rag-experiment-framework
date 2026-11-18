"""스냅샷 내용 해시·검증 시험(U4): 결정성, 고친 파일 거부, 이상한 경로 거부, 모르는 형식 거부."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest
from fakes.fake_law_api import FakeLaw, FakeLawApi, FakeVersion, law_body, run_fetch, standard_laws

from ragexp.datasets.law_api import SnapshotError, load_snapshot, snapshot_content_hash

NAMES = ["시험법", "시험법 시행령"]
FW_ID = "".join(chr(0xFF10 + int(c)) for c in "009999")
"""전각 숫자 법령ID(ASCII 숫자가 아니라 거부돼야 함)."""
FW_MST = "".join(chr(0xFF10 + int(c)) for c in "286449")


def make_snapshot(out: Path, **kw: Any) -> Path:
    run = run_fetch(out, names=NAMES, **kw)
    assert run.result is not None, run.error
    return out


@pytest.fixture
def snap(tmp_path: Path) -> Path:
    return make_snapshot(tmp_path / "zq-folder")


def read_manifest(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    return data


def write_manifest(path: Path, data: dict[str, Any]) -> None:
    text = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    (path / "manifest.json").write_bytes(text.encode("utf-8"))


def snapshot_error(path: Path) -> str:
    with pytest.raises(SnapshotError) as info:
        load_snapshot(path)
    message = str(info.value)
    # 파일 이름만 담는다(로컬 경로를 넣지 않음).
    assert str(path) not in message and path.name not in message
    return message


# --- 결정성 -----------------------------------------------------------------------------


def test_content_hash_deterministic(tmp_path: Path) -> None:
    a = make_snapshot(tmp_path / "a")
    # 받은 날짜·기준일(같은 판을 고르는 날)·폴더 위치가 달라도 같다.
    b = make_snapshot(
        tmp_path / "deeper" / "b",
        as_of=date(2026, 6, 1),
        now=datetime(2027, 3, 1, 0, 0, tzinfo=UTC),
    )
    ha = load_snapshot(a).content_hash
    assert ha == load_snapshot(b).content_hash
    assert read_manifest(a)["as_of"] != read_manifest(b)["as_of"]
    assert read_manifest(a)["fetched_at"] != read_manifest(b)["fetched_at"]

    # manifest 키 순서·laws 순서를 바꿔도 같다.
    data = read_manifest(a)
    reordered = {k: data[k] for k in reversed(list(data))}
    reordered["laws"] = [{k: e[k] for k in reversed(list(e))} for e in reversed(data["laws"])]
    write_manifest(a, reordered)
    assert load_snapshot(a).content_hash == ha

    # 폴더를 옮겨도 같다.
    moved = tmp_path / "moved"
    shutil.copytree(a, moved)
    assert load_snapshot(moved).content_hash == ha

    # 정해 둔 값(해시 규칙이 몰래 바뀌지 않게).
    pairs = [(e["file"], e["sha256"]) for e in data["laws"]]
    assert ha == snapshot_content_hash(pairs) == snapshot_content_hash(reversed(pairs))
    assert ha.startswith("snap-") and len(ha) == len("snap-") + 64


def test_content_hash_changes_with_one_byte(tmp_path: Path) -> None:
    base = load_snapshot(make_snapshot(tmp_path / "a")).content_hash
    laws = standard_laws()
    first = laws[0]
    original = law_body(first.law_id, first.name)
    old, new = "<공포번호>00001</공포번호>".encode(), "<공포번호>00002</공포번호>".encode()
    changed_body = original.replace(old, new, 1)
    assert len(changed_body) == len(original)
    assert sum(x != y for x, y in zip(original, changed_body, strict=True)) == 1  # 한 바이트만
    versions = tuple(
        FakeVersion(v.mst, v.effective, v.promulgation_date, v.promulgation_no, body=changed_body)
        for v in first.versions
    )
    api = FakeLawApi([FakeLaw(first.name, first.law_id, first.kind, versions), laws[1]])
    other = load_snapshot(make_snapshot(tmp_path / "b", api=api)).content_hash
    assert other != base


def test_content_hash_fixed_value() -> None:
    pairs = [("laws/000002.xml", "b" * 64), ("laws/000001.xml", "a" * 64)]
    expected = hashlib.sha256(
        f'[["laws/000001.xml","{"a" * 64}"],["laws/000002.xml","{"b" * 64}"]]'.encode()
    ).hexdigest()
    assert snapshot_content_hash(pairs) == "snap-" + expected


def test_load_returns_recomputed(snap: Path) -> None:
    snapshot = load_snapshot(snap)
    assert snapshot.content_hash == snapshot.manifest.content_hash
    assert [e.law_id for e in snapshot.manifest.laws] == ["009999", "009998"]
    assert snapshot.law_path(snapshot.manifest.laws[0]).read_bytes().startswith(b"<?xml")


# --- 고친 스냅샷 ---------------------------------------------------------------------------


def _flip_file_byte(path: Path) -> None:
    target = path / "laws" / "009999.xml"
    data = bytearray(target.read_bytes())
    data[-2] ^= 0x01
    target.write_bytes(bytes(data))


def _edit_entry_sha(path: Path) -> None:
    data = read_manifest(path)
    data["laws"][0]["sha256"] = "0" * 64
    write_manifest(path, data)


def _edit_content_hash(path: Path) -> None:
    data = read_manifest(path)
    data["content_hash"] = "snap-" + "0" * 64
    write_manifest(path, data)


def _edit_bytes(path: Path) -> None:
    data = read_manifest(path)
    data["laws"][0]["bytes"] += 1
    write_manifest(path, data)


def _delete_file(path: Path) -> None:
    (path / "laws" / "009998.xml").unlink()


@pytest.mark.parametrize(
    ("tamper", "expected"),
    [
        (_flip_file_byte, "스냅샷 파일이 manifest의 해시와 다릅니다(laws/009999.xml)"),
        (_edit_entry_sha, "스냅샷 파일이 manifest의 해시와 다릅니다(laws/009999.xml)"),
        (_edit_bytes, "스냅샷 파일이 manifest의 해시와 다릅니다(laws/009999.xml)"),
        (_edit_content_hash, "content_hash가 파일에서 다시 계산한 값과 다릅니다"),
        (_delete_file, "스냅샷 파일을 읽을 수 없습니다(laws/009998.xml)"),
    ],
    ids=["file-byte", "entry-sha", "entry-bytes", "content-hash", "missing-file"],
)
def test_tampered_rejected(snap: Path, tamper: Callable[[Path], None], expected: str) -> None:
    tamper(snap)
    assert expected in snapshot_error(snap)


# --- 이상한 경로 ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("file", "expected"),
    [
        ("../009999.xml", "laws/<6자리 숫자>.xml 꼴이 아닙니다"),
        ("laws/../../009999.xml", "laws/<6자리 숫자>.xml 꼴이 아닙니다"),
        ("laws/9999.xml", "laws/<6자리 숫자>.xml 꼴이 아닙니다"),
        ("laws\\009999.xml", "laws/<6자리 숫자>.xml 꼴이 아닙니다"),
        ("/laws/009999.xml", "laws/<6자리 숫자>.xml 꼴이 아닙니다"),
        ("laws/009998.xml", "법령ID(009999)와 맞지 않습니다"),
        (f"laws/{FW_ID}.xml", "laws/<6자리 숫자>.xml 꼴이 아닙니다"),  # 전각 숫자
    ],
)
def test_bad_file_path_rejected(snap: Path, file: str, expected: str) -> None:
    data = read_manifest(snap)
    data["laws"][0]["file"] = file
    write_manifest(snap, data)
    assert expected in snapshot_error(snap)


def test_duplicate_entry_rejected(snap: Path) -> None:
    data = read_manifest(snap)
    data["laws"][1] = data["laws"][0]
    write_manifest(snap, data)
    assert "같은 파일이 두 번 있습니다" in snapshot_error(snap)


# --- 형식 ---------------------------------------------------------------------------------


def test_unknown_schema_rejected(snap: Path) -> None:
    data = read_manifest(snap)
    data["schema"] = "ragexp.law_snapshot/2"
    write_manifest(snap, data)
    assert "알 수 없는 스냅샷 형식" in snapshot_error(snap)


@pytest.mark.parametrize(
    ("edit", "expected"),
    [
        (lambda d: d.pop("laws"), "manifest.json 형식 오류(laws)"),
        (lambda d: d.update(extra=1), "manifest.json 형식 오류(extra)"),
        (lambda d: d["laws"][0].update(law_id="9999"), "manifest.json 형식 오류(laws.0.law_id)"),
        (lambda d: d.update(laws=[]), "manifest.json 형식 오류(laws)"),
        (
            lambda d: d["laws"][0].update(law_id=FW_ID),
            "manifest.json 형식 오류(laws.0.law_id)",
        ),
        (lambda d: d["laws"][0].update(mst=FW_MST), "manifest.json 형식 오류(laws.0.mst)"),
    ],
    ids=[
        "missing-laws",
        "extra-field",
        "short-law-id",
        "empty-laws",
        "fullwidth-id",
        "fullwidth-mst",
    ],
)
def test_bad_manifest_fields(
    snap: Path, edit: Callable[[dict[str, Any]], object], expected: str
) -> None:
    data = read_manifest(snap)
    edit(data)
    write_manifest(snap, data)
    assert expected in snapshot_error(snap)


def test_missing_or_broken_manifest(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert "manifest.json을 읽을 수 없습니다" in snapshot_error(empty)
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "manifest.json").write_bytes(b"{not json")
    assert "manifest.json이 JSON이 아닙니다" in snapshot_error(broken)
