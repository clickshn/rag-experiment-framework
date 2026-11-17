"""fingerprint 검사 시험. 시험 값은 엔트로피가 낮은 문자열(dummy)만 쓴다."""

from __future__ import annotations

from typing import Any

import pytest

from ragexp.fingerprint import FingerprintError, check_fingerprint

ALLOWED: list[dict[str, Any]] = [
    {"model": "nlpai-lab/KURE-v1", "revision": "abc123", "dimension": 1024},
    {"params": {"temperature": 0, "max_tokens": 2048, "top_p": 1.0, "seed": 7}},
    {"chunk_tokens": 512, "overlap_tokens": 64, "max_context_tokens": 7800},
    {"keyword_k": 20, "keyword_weight": 0.3, "top_k": 5, "top_n": 5},
    {"prompt_sha256": "e3b0c44298fc1c149afbf4c8996fb924", "snapshot_sha256": "0" * 64},
    {"chunk_id_format": "law:<id>:제N조", "tokenizer": "kiwi", "stop": ["\n\n"]},
    {"ratio": "1/2", "note": "a/b/c", "law": "law:000729:제27조"},
    {"nested": [{"a": None}, [True, 1.5]]},
    {"model": "Qwen/Qwen3.5-27B-FP8", "revision": "v1.2.3", "dtype": "fp8"},
    {"ollama_tag": "qwen3:32b", "ratio": "3:2", "time": "12:30"},
]

REJECTED: list[tuple[str, dict[Any, Any]]] = [
    ("base_url", {"base_url": "dummy"}),
    ("대소문자 무시", {"API_KEY": "dummy"}),
    ("중첩 api_key", {"params": {"api_key": "dummy"}}),
    ("목록 속 이름", {"judges": [{"endpoint": "dummy"}]}),
    ("url", {"url": "dummy"}),
    ("host", {"host": "dummy"}),
    ("access_token", {"access_token": "dummy"}),
    ("끝말 _key", {"openai_key": "dummy"}),
    ("끝말 _token", {"hf_token": "dummy"}),
    ("끝말 _secret", {"client_secret": "dummy"}),
    ("접두어 price", {"price_per_1m_input": 0.0}),
    ("접두어 auth", {"authorization": "dummy"}),
    ("meta", {"meta": {"deploy": "cloud"}}),
    ("meta 칸 gpu_cost", {"gpu_cost_per_hour": 1.0}),
    ("meta 칸 vram", {"vram_gb": 31}),
    ("meta 칸 license", {"license": "Apache-2.0"}),
    ("meta 칸 deploy", {"deploy": "on_prem"}),
    ("meta 칸 eol", {"end_of_life": "2027-01-01"}),
    ("값에 주소", {"server": "http://localhost:8000/v1"}),
    ("값에 치환 자리", {"server": "${LOCAL_URL}"}),
    ("Windows 드라이브", {"snapshot": "C:\\data\\law-2026-11-17"}),
    ("Windows 드라이브 슬래시", {"snapshot": "d:/snapshots/law"}),
    ("문장 속 드라이브", {"note": "copied from E:\\work"}),
    ("UNC", {"snapshot": "\\\\fileserver\\share\\law"}),
    ("POSIX /home", {"snapshot": "/home/user/law"}),
    ("POSIX /Users", {"snapshot": "/Users/user/law"}),
    ("POSIX /data", {"snapshot": "/data/snapshots/law"}),
    ("홈 경로", {"snapshot": "~/snapshots/law"}),
    ("api_base", {"api_base": "dummy"}),
    ("auth_token", {"auth_token": "dummy"}),
    ("secret", {"secret": "dummy"}),
    ("password", {"password": "dummy"}),
    ("끝말 _password", {"db_password": "dummy"}),
    ("법령 OC", {"oc": "dummy"}),
    ("법령 OC 대문자", {"OC": "dummy"}),
    ("하이픈 api-key", {"api-key": "dummy"}),
    ("하이픈 x-api-key", {"headers": {"X-Api-Key": "dummy"}}),
    ("하이픈 구독 키", {"Ocp-Apim-Subscription-Key": "dummy"}),
    ("붙여 쓴 apikey", {"apikey": "dummy"}),
    ("맨 이름 token", {"token": "dummy"}),
    ("맨 이름 key", {"key": "dummy"}),
    ("hostname", {"hostname": "dummy"}),
    ("server", {"server": "dummy"}),
    ("address", {"address": "dummy"}),
    ("IPv4:포트", {"target": "10.0.0.5:8000"}),
    ("IPv4만", {"target": "192.168.0.10"}),
    ("문장 속 IPv4", {"note": "served from 10.1.2.3"}),
    ("호스트:포트", {"target": "gpu-01:8000"}),
    ("도메인:포트/경로", {"target": "llm.internal.example:443/v1"}),
    ("localhost:포트", {"target": "localhost:8000"}),
    ("문장 속 /home", {"note": "copied from /home/user/x"}),
    ("file:/ 경로", {"note": "file:/home/a/b"}),
    ("vLLM 모델 경로", {"model": "/models/Qwen3.5-27B-FP8"}),
    ("NaN", {"temperature": float("nan")}),
    ("inf", {"temperature": float("inf")}),
    ("bytes", {"blob": b"dummy"}),
    ("객체", {"obj": object()}),
    ("문자열 아닌 키", {1: "dummy"}),
]


@pytest.mark.parametrize("fp", ALLOWED)
def test_allowed(fp: dict[str, Any]) -> None:
    check_fingerprint(fp)


@pytest.mark.parametrize("case", REJECTED, ids=[c[0] for c in REJECTED])
def test_rejected(case: tuple[str, dict[Any, Any]]) -> None:
    with pytest.raises(FingerprintError):
        check_fingerprint(case[1])


def test_error_names_path() -> None:
    with pytest.raises(FingerprintError, match=r"judges\[0\]\.base_url"):
        check_fingerprint({"judges": [{"base_url": "dummy"}]})


def test_component_allow_list() -> None:
    with pytest.raises(FingerprintError):
        check_fingerprint({"stop_token": "</s>"})
    assert check_fingerprint({"stop_token": "</s>"}, allow={"Stop_Token"}) == {"stop_token": "</s>"}


@pytest.mark.parametrize("name", ["api_key", "API-KEY", "base_url", "meta", "oc", "license"])
def test_allow_cannot_unlock_forbidden_names(name: str) -> None:
    with pytest.raises(FingerprintError):
        check_fingerprint({name: "dummy"}, allow={name})


def test_allow_does_not_permit_bad_values() -> None:
    with pytest.raises(FingerprintError):
        check_fingerprint({"stop_token": "C:\\x"}, allow={"stop_token"})


def test_normalizes_tuples_and_mappings_only() -> None:
    from types import MappingProxyType

    fp = MappingProxyType({"a": (1, 2.0, MappingProxyType({"b": (True,)}))})
    out = check_fingerprint(fp)
    assert out == {"a": [1, 2.0, {"b": [True]}]}
    assert type(out["a"][0]) is int and type(out["a"][1]) is float  # 숫자는 그대로


def test_not_a_mapping() -> None:
    with pytest.raises(FingerprintError):
        check_fingerprint(["model"])  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "fp",
    [
        {"model": "/models/Qwen3.5-27B-FP8"},
        {"params": {"judge_model": "C:\models\gemma"}},
        {"model_name": "~/models/kure"},
    ],
)
def test_path_like_model_name_explains_fix(fp: dict[str, Any]) -> None:
    with pytest.raises(FingerprintError, match="--served-model-name") as info:
        check_fingerprint(fp)
    assert "PC마다 달라 구성 버전에 넣지 않습니다" in str(info.value)


def test_non_model_path_keeps_generic_message() -> None:
    with pytest.raises(FingerprintError, match="내용 해시") as info:
        check_fingerprint({"snapshot": "/home/user/law"})
    assert "served-model-name" not in str(info.value)
