"""모델 호출 통로 시험(U3): 정상 호출, finish_reason, 재시도·대응표, 값 누출, 모델 이름, 지연.

모든 시험은 네트워크 없이 돈다(httpx MockTransport, 127.0.0.1 가짜 서버).
"""

from __future__ import annotations

import email.utils
from typing import Any

import httpx
import pytest
from fakes.fake_openai import (
    BODY_MARKER,
    FakeClock,
    FakeOpenAIServer,
    ScriptedTransport,
    Step,
    chat_body,
    error_body,
    ok,
    scripted_transport,
)

from ragexp.client import (
    FAILURE_CLASSES,
    MAX_TIMEOUT_ATTEMPTS,
    ChatClient,
    ClientConfigError,
    failure_class,
)
from ragexp.config import UNSET_PLACEHOLDER, ModelSettings
from ragexp.types import Answer, ErrorKind, Usage

BASE = "http://localhost:9/v1"
KEY = "dummy-key"
LEAKS = (BASE, "localhost:9", KEY, BODY_MARKER, "/home/u", "/models/", "C:\\models")
"""클라이언트 출력(Answer·예외)에 나오면 안 되는 값."""
MSG = [{"role": "system", "content": "근거로 답하라"}, {"role": "user", "content": "질문"}]
NOW = 1_700_000_000.0


def settings(**kw: Any) -> ModelSettings:
    data: dict[str, Any] = {
        "model": "model-a",
        "base_url": BASE,
        "api_key": KEY,
        "params": {"temperature": 0, "max_tokens": 64},
    }
    data.update(kw)
    return ModelSettings.model_validate(data)


def make(steps: list[Step], **kw: Any) -> tuple[ChatClient, ScriptedTransport, FakeClock]:
    clock = FakeClock()
    transport = scripted_transport(steps, clock)
    client = ChatClient(
        settings(**kw), transport=transport, sleep=clock.sleep, clock=clock.clock, now=lambda: NOW
    )
    return client, transport, clock


def run(steps: list[Step], **kw: Any) -> tuple[Answer, ScriptedTransport, FakeClock]:
    client, transport, clock = make(steps, **kw)
    with client:
        answer = client.chat(MSG)
    return answer, transport, clock


def assert_no_leak(answer: Answer) -> None:
    texts = [answer.text, answer.error, *answer.warnings, str(answer.served_model), repr(answer)]
    for text in texts:
        for leak in LEAKS:
            assert leak not in text, (leak, text)


# --- 기준 1: 정상 호출 ---------------------------------------------------------------


def test_success_fields() -> None:
    step = Step(body=chat_body("답 [1]", usage={"prompt_tokens": 12, "completion_tokens": 5}))
    answer, transport, _ = run([Step(**{**step.__dict__, "elapsed_s": 0.25})])
    assert answer.text == "답 [1]" and answer.finish_reason == "stop"
    assert answer.usage == Usage(input_tokens=12, output_tokens=5, latency_ms=250, total_ms=250)
    assert answer.served_model == "model-a" and answer.warnings == ()
    assert answer.attempts == 1 and answer.error_kind is None and answer.error == ""
    assert transport.calls == 1 and failure_class(answer) is None


def test_success_request() -> None:
    _, transport, _ = run([ok()])
    request = transport.requests[0]
    assert request.method == "POST" and request.url.path == "/v1/chat/completions"
    assert request.headers["Authorization"] == f"Bearer {KEY}"
    assert transport.json_bodies()[0] == {
        "model": "model-a",
        "messages": MSG,
        "temperature": 0,
        "max_tokens": 64,
    }


def test_success_without_api_key_sends_no_auth_header() -> None:
    _, transport, _ = run([ok()], api_key=None)
    assert "Authorization" not in transport.requests[0].headers


def test_success_response_format_is_sent() -> None:
    schema = {"type": "json_schema", "json_schema": {"name": "a", "schema": {"type": "object"}}}
    client, transport, _ = make([ok()])
    client.chat(MSG, response_format=schema)
    assert transport.json_bodies()[0]["response_format"] == schema


def test_trailing_slash_base_url_does_not_double() -> None:
    _, transport, _ = run([ok()], base_url=BASE + "/")
    assert transport.requests[0].url.path == "/v1/chat/completions"


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        (None, (12, 5)),
        ({}, (None, None)),
        ({"prompt_tokens": -1, "completion_tokens": True}, (None, None)),
        ({"prompt_tokens": 3}, (3, None)),
    ],
    ids=["있음", "빈 usage", "음수·bool", "한쪽만"],
)
def test_success_usage_tokens(usage: dict[str, Any] | None, expected: tuple[Any, Any]) -> None:
    answer, _, _ = run([Step(body=chat_body(usage=usage))])
    assert (answer.usage.input_tokens, answer.usage.output_tokens) == expected


def test_success_usage_missing_is_none() -> None:
    body = chat_body()
    del body["usage"]
    answer, _, _ = run([Step(body=body)])
    assert answer.usage.input_tokens is None and answer.usage.output_tokens is None


def test_success_content_none_is_empty_text() -> None:
    body = chat_body()
    body["choices"][0]["message"]["content"] = None
    answer, _, _ = run([Step(body=body)])
    assert answer.text == "" and answer.finish_reason == "stop"


# --- 기준 2: finish_reason 매핑 --------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("stop", "stop"),
        ("length", "length"),
        ("content_filter", "other"),
        ("tool_calls", "other"),
        (None, "other"),
    ],
)
def test_finish_reason_mapping(raw: str | None, expected: str) -> None:
    answer, transport, _ = run([Step(body=chat_body(finish_reason=raw))])
    assert answer.finish_reason == expected and answer.error_kind is None
    assert transport.calls == 1  # 잘림도 재시도하지 않는다
    assert failure_class(answer) == ("item" if expected == "length" else None)


# --- 기준 3: 재시도와 대응표 --------------------------------------------------------------

EXCEPTIONS: list[tuple[type[httpx.RequestError], ErrorKind, int]] = [
    (httpx.ConnectTimeout, "timeout", MAX_TIMEOUT_ATTEMPTS),
    (httpx.ReadTimeout, "timeout", MAX_TIMEOUT_ATTEMPTS),
    (httpx.WriteTimeout, "timeout", MAX_TIMEOUT_ATTEMPTS),
    (httpx.PoolTimeout, "timeout", MAX_TIMEOUT_ATTEMPTS),
    (httpx.ConnectError, "connect", 3),
    (httpx.ReadError, "connect", 3),
    (httpx.WriteError, "connect", 3),
    (httpx.CloseError, "connect", 3),
    (httpx.RemoteProtocolError, "connect", 3),
    (httpx.LocalProtocolError, "other", 3),
    (httpx.ProxyError, "other", 3),
    (httpx.UnsupportedProtocol, "other", 3),
    (httpx.DecodingError, "other", 3),
    (httpx.TooManyRedirects, "other", 3),
]


@pytest.mark.parametrize(
    ("exc", "kind", "attempts"), EXCEPTIONS, ids=[e[0].__name__ for e in EXCEPTIONS]
)
def test_error_kind_exceptions(
    exc: type[httpx.RequestError], kind: ErrorKind, attempts: int
) -> None:
    answer, transport, _ = run([Step(exc=exc)] * 3)
    assert answer.finish_reason == "error" and answer.error_kind == kind
    assert answer.attempts == attempts == transport.calls
    assert answer.text == "" and failure_class(answer) == "server"
    assert_no_leak(answer)  # 예외 메시지에 주소·키를 넣었다


STATUSES: list[tuple[int, ErrorKind, int]] = [
    (408, "http_status", 3),
    (429, "http_status", 3),
    (500, "http_status", 3),
    (503, "http_status", 3),
    (302, "other", 3),
    (401, "config_error", 1),
    (403, "config_error", 1),
    (404, "config_error", 1),
    (400, "client_error", 1),
    (413, "client_error", 1),
    (422, "client_error", 1),
]


@pytest.mark.parametrize(
    ("status", "kind", "attempts"), STATUSES, ids=[str(s[0]) for s in STATUSES]
)
def test_error_kind_status(status: int, kind: ErrorKind, attempts: int) -> None:
    answer, transport, _ = run([Step(status=status)] * 3)
    assert answer.error_kind == kind and answer.attempts == attempts == transport.calls
    assert answer.text == "" and failure_class(answer) == FAILURE_CLASSES[kind]
    assert_no_leak(answer)  # 오류 본문에 주소·키·표시를 넣었다


PARSE_CASES: list[tuple[str, Step, str]] = [
    ("깨진 JSON", Step(raw=b"{not json " + BODY_MARKER.encode()), "JSON 아님"),
    ("UTF-8 아님", Step(raw=b"\xff\xfe\x00"), "JSON 아님"),
    ("choices 없음", Step(body={"detail": BODY_MARKER}), "choices 없음"),
    ("choices 빈 목록", Step(body={"choices": []}), "choices 없음"),
    ("최상위가 목록", Step(body=[1, 2]), "choices 없음"),
    ("message 없음", Step(body={"choices": [{"finish_reason": "stop"}]}), "message 없음"),
    ("content가 숫자", Step(body={"choices": [{"message": {"content": 3}}]}), "문자열이 아님"),
]


@pytest.mark.parametrize("case", PARSE_CASES, ids=[c[0] for c in PARSE_CASES])
def test_error_kind_parse(case: tuple[str, Step, str]) -> None:
    _, step, msg = case
    answer, transport, _ = run([step])
    assert answer.error_kind == "parse" and msg in answer.error
    assert transport.calls == 1 and failure_class(answer) == "item"
    assert_no_leak(answer)


def test_retry_then_success() -> None:
    answer, transport, clock = run([Step(status=503), Step(exc=httpx.ConnectError), ok()])
    assert answer.finish_reason == "stop" and answer.attempts == 3 == transport.calls
    assert clock.sleeps == [1.0, 2.0]  # 1초 * 2^(시도-1)


def test_retry_timeout_then_server_error_then_success() -> None:
    answer, _, _ = run([Step(exc=httpx.ReadTimeout), Step(status=503), ok()])
    assert answer.finish_reason == "stop" and answer.attempts == 3


def test_retry_second_timeout_stops_even_with_retries_left() -> None:
    steps = [Step(exc=httpx.ReadTimeout), Step(status=503), Step(exc=httpx.ReadTimeout), ok()]
    answer, transport, _ = run(steps, run={"max_retries": 5})
    assert answer.error_kind == "timeout" and answer.attempts == 3 == transport.calls


def test_retry_mixed_kinds_stop_at_total_limit() -> None:
    steps = [Step(status=503), Step(exc=httpx.ReadTimeout), Step(status=503), ok()]
    answer, transport, _ = run(steps)
    assert answer.error_kind == "http_status" and answer.attempts == 3 == transport.calls


def test_invalid_url_that_skipped_validation_is_config_error() -> None:
    """구성 검사가 막는 주소(IPv4 범위 밖)를 검증 없이 넣어도 예외 대신 설정 오류로 돌려준다."""
    from pydantic import SecretStr

    bad = ModelSettings.model_construct(
        model="model-a",
        base_url=SecretStr("http://10.0.0.256:8000/v1"),
        api_key=None,
        params={},
        revision=None,
        run=settings().run,
    )
    with ChatClient(bad, sleep=lambda s: None) as client:
        answer = client.chat(MSG)
    assert answer.error_kind == "config_error" and answer.attempts == 1
    assert "10.0.0.256" not in answer.error and failure_class(answer) == "config"


def test_retry_respects_max_retries() -> None:
    answer, transport, clock = run([Step(status=503)] * 6, run={"max_retries": 4})
    assert answer.attempts == 5 == transport.calls
    assert clock.sleeps == [1.0, 2.0, 4.0, 8.0]


def test_retry_backoff_is_capped() -> None:
    _, _, clock = run([Step(status=503)] * 8, run={"max_retries": 7})
    assert clock.sleeps == [1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 30.0]


def test_retry_zero_means_one_attempt() -> None:
    answer, transport, clock = run([Step(status=503)], run={"max_retries": 0})
    assert answer.attempts == 1 == transport.calls and clock.sleeps == []


def test_no_retry_for_item_and_config_errors_has_no_wait() -> None:
    for status in (400, 401):
        _, transport, clock = run([Step(status=status)])
        assert transport.calls == 1 and clock.sleeps == []


@pytest.mark.parametrize(
    ("value", "wait"),
    [
        ("5", 5.0),
        ("0", 0.0),
        ("1.5", 1.5),
        ("120", 60.0),
        (email.utils.formatdate(NOW + 7, usegmt=True), 7.0),
        (email.utils.formatdate(NOW - 30, usegmt=True), 0.0),
        (email.utils.formatdate(NOW + 600, usegmt=True), 60.0),
        ("soon", 1.0),
        ("-3", 1.0),
        ("", 1.0),
    ],
    ids=[
        "초",
        "0초",
        "소수",
        "초 상한",
        "날짜",
        "지난 날짜",
        "날짜 상한",
        "못 읽음",
        "음수",
        "빈 값",
    ],
)
def test_retry_after(value: str, wait: float) -> None:
    answer, _, clock = run([Step(status=429, headers={"Retry-After": value}), ok()])
    assert answer.finish_reason == "stop" and answer.attempts == 2
    assert clock.sleeps == [pytest.approx(wait)]


def test_retry_after_on_503_is_followed() -> None:
    _, _, clock = run([Step(status=503, headers={"Retry-After": "9"}), ok()])
    assert clock.sleeps == [9.0]


# --- 실패 분류 셋 ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("connect", "server"),
        ("timeout", "server"),
        ("http_status", "server"),
        ("other", "server"),
        ("client_error", "item"),
        ("parse", "item"),
        ("config_error", "config"),
    ],
)
def test_failure_class_table(kind: ErrorKind, expected: str) -> None:
    answer = Answer(
        text="", finish_reason="error", usage=Usage(latency_ms=1), error_kind=kind, error="x"
    )
    assert failure_class(answer) == expected


def test_failure_class_covers_every_error_kind() -> None:
    from typing import get_args

    assert set(FAILURE_CLASSES) == set(get_args(ErrorKind))


@pytest.mark.parametrize(
    ("status", "fields"),
    [(401, ("api_key",)), (403, ("api_key",)), (404, ("base_url", "model"))],
)
def test_config_error_message_names_fields(status: int, fields: tuple[str, ...]) -> None:
    answer, _, _ = run([Step(status=status)])
    assert answer.error.startswith(f"HTTP {status}")
    assert all(f in answer.error for f in fields)
    assert_no_leak(answer)


# --- 기준 4: 값이 새지 않음 ---------------------------------------------------------------


def test_no_leak_in_repr_of_client() -> None:
    client, _, _ = make([])
    for leak in (BASE, KEY):
        assert leak not in repr(client)


@pytest.mark.parametrize("field", ["base_url", "api_key"])
def test_no_leak_placeholder_refused(field: str) -> None:
    with pytest.raises(ClientConfigError) as info:
        ChatClient(settings(**{field: UNSET_PLACEHOLDER}))
    assert field in str(info.value) and "--no-env" in str(info.value)
    assert info.value.__context__ is None and info.value.__cause__ is None
    assert UNSET_PLACEHOLDER not in str(info.value)


def test_no_leak_error_body_is_not_text() -> None:
    answer, _, _ = run([Step(status=500, body=error_body())] * 3)
    assert answer.text == "" and answer.error == "HTTP 500"


# --- 기준 5: 모델 이름 -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("configured", "served", "recorded", "warn"),
    [
        ("model-a", "model-a", "model-a", None),
        ("model-a", "Model-A", "Model-A", "다릅니다"),
        ("Qwen/Qwen3-8B", "Qwen/Qwen3-8B", "Qwen/Qwen3-8B", None),
        ("Qwen3-8B", "/home/u/models/Qwen3-8B", "Qwen3-8B", "--served-model-name"),
        ("Qwen3-8B", "/models/Qwen3-8B", "Qwen3-8B", "--served-model-name"),
        ("X", "C:\\models\\X", "X", "--served-model-name"),
        ("model-a", None, None, None),
        ("model-a", "http://localhost:9/v1/model-a", None, "주소"),
    ],
    ids=["같음", "대소문자", "HF ID", "POSIX 경로", "/models 경로", "Windows 경로", "없음", "주소"],
)
def test_served_model(
    configured: str, served: str | None, recorded: str | None, warn: str | None
) -> None:
    answer, _, _ = run([Step(body=chat_body(model=served))], model=configured)
    assert answer.served_model == recorded
    if warn is None:
        assert answer.warnings == ()
    else:
        assert len(answer.warnings) == 1 and warn in answer.warnings[0]
    assert_no_leak(answer)  # 경로 전체·주소가 남지 않는다


def test_served_model_path_warning_shows_last_part_only() -> None:
    answer, _, _ = run([Step(body=chat_body(model="/home/u/models/Qwen3-8B"))], model="Qwen3-8B")
    assert "…/Qwen3-8B" in answer.warnings[0] and "home" not in answer.warnings[0]
    assert "PC마다 달라 구성 버전에 넣지 않습니다" in answer.warnings[0]


# --- 기준 9: 지연 -----------------------------------------------------------------------


def test_latency_and_total_on_success_after_retry() -> None:
    answer, _, _ = run(
        [Step(status=503, elapsed_s=0.1), Step(**{**ok().__dict__, "elapsed_s": 0.3})]
    )
    assert answer.usage.latency_ms == pytest.approx(300)
    assert answer.usage.total_ms == pytest.approx(100 + 1000 + 300)  # 실패 시도 + 대기 + 성공 시도


def test_latency_and_total_when_all_fail() -> None:
    answer, _, _ = run([Step(status=503, elapsed_s=0.1)] * 3)
    assert answer.usage.latency_ms == pytest.approx(100)
    assert answer.usage.total_ms == pytest.approx(100 + 1000 + 100 + 2000 + 100)


# --- 기준 10: 실제 소켓 ---------------------------------------------------------------


def _socket_client(server: FakeOpenAIServer) -> ChatClient:
    return ChatClient(settings(base_url=server.base_url), sleep=lambda s: None)


def test_real_socket_success() -> None:
    script = [(200, chat_body("소켓 답"))]
    with FakeOpenAIServer({"model-a": script}) as server, _socket_client(server) as client:
        answer = client.chat(MSG)
    assert answer.text == "소켓 답" and answer.attempts == 1
    assert answer.usage.latency_ms > 0
    assert server.requests[0]["path"] == "/v1/chat/completions"
    assert server.requests[0]["auth"] == f"Bearer {KEY}"
    assert server.requests[0]["body"]["model"] == "model-a"


def test_real_socket_503_then_success() -> None:
    script = [(503, error_body()), (200, chat_body("다시 답"))]
    with FakeOpenAIServer({"model-a": script}) as server, _socket_client(server) as client:
        answer = client.chat(MSG)
    assert answer.text == "다시 답" and answer.attempts == 2 and len(server.requests) == 2
    assert answer.usage.total_ms is not None and answer.usage.total_ms >= answer.usage.latency_ms
