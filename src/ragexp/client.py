"""모델 호출 통로(설계 §4): OpenAI 호환 `/chat/completions` 클라이언트.

모든 chat 호출이 이 한 곳을 지난다. 시간 제한·재시도·토큰·지연 기록·응답 모델 이름 확인·
잘림 구분을 여기서만 한다. 임베딩·리랭크 호출은 R2에서 더한다.

새지 않게 하는 규칙:
- 오류 문구는 고정 문구다. httpx 예외 문자열(주소가 들어 있음)과 응답 본문은 쓰지 않는다.
- 오류일 때 `Answer.text`는 빈 문자열이다.
- 응답 모델 이름이 경로면 마지막 조각만, 주소가 들어 있으면 아예 기록하지 않는다.
- 아무것도 출력하지 않는다.
"""

from __future__ import annotations

import email.utils
import json
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC
from types import MappingProxyType, TracebackType
from typing import Any, Literal, Self

import httpx

from ragexp.config import UNSET_PLACEHOLDER, ModelSettings
from ragexp.fingerprint import has_address, is_absolute_path
from ragexp.types import Answer, ErrorKind, FinishReason, Usage

CONNECT_TIMEOUT_S = 10.0
"""연결 시간 제한(고정). 응답을 기다리는 시간은 `run.timeout_s`."""
MAX_TIMEOUT_ATTEMPTS = 2
"""한 호출 안 시간 초과 상한. 두 번째 시간 초과가 나면 남은 횟수가 있어도 멈춘다."""
BACKOFF_BASE_S = 1.0
BACKOFF_MAX_S = 30.0
RETRY_AFTER_MAX_S = 60.0

FailureClass = Literal["server", "item", "config"]
"""실패 분류 셋: 서버 사정·모델·문항 사정·설정 오류(설계 §4 "실패 분류 셋")."""

FAILURE_CLASSES: Mapping[ErrorKind, FailureClass] = MappingProxyType(
    {
        "connect": "server",
        "timeout": "server",
        "http_status": "server",
        "other": "server",  # 모르는 오류가 조용히 오답이 되지 않게 서버 쪽에 둔다
        "client_error": "item",
        "parse": "item",
        "config_error": "config",
    }
)

_CONFIG_ERRORS: Mapping[int, str] = MappingProxyType(
    {
        401: "HTTP 401 인증 실패: api_key 칸(환경 변수 값)을 확인하세요",
        403: "HTTP 403 권한 없음: api_key 칸의 권한을 확인하세요",
        404: "HTTP 404 찾지 못함: base_url 칸(/v1까지 포함)과 model 칸을 확인하세요",
    }
)
_PATH_HINT = (
    "vLLM은 --served-model-name으로 이름을 정하세요. "
    "모델 경로는 PC마다 달라 구성 버전에 넣지 않습니다"
)
_DELTA_SECONDS_RE = re.compile(r"^\d+(\.\d+)?$")


def failure_class(answer: Answer) -> FailureClass | None:
    """답의 실패 분류. 정상 응답은 None, 잘림(`length`)은 오류가 아니지만 모델·문항 사정이다."""
    if answer.error_kind is not None:
        return FAILURE_CLASSES[answer.error_kind]
    if answer.finish_reason == "length":
        return "item"
    return None


class ClientConfigError(ValueError):
    """클라이언트를 만들 수 없는 설정(자리표시 값 등). 메시지에 값은 넣지 않는다."""


@dataclass(frozen=True)
class _Failure:
    kind: ErrorKind
    message: str
    retry_after_s: float | None = None


@dataclass(frozen=True)
class _Reply:
    text: str
    finish_reason: FinishReason
    input_tokens: int | None
    output_tokens: int | None
    model: str | None


def _token_count(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _parse_reply(content: bytes) -> _Reply | _Failure:
    """2xx 응답 본문을 읽는다. 형식이 틀리면 `parse` 실패(본문은 문구에 넣지 않음)."""
    try:
        data = json.loads(content)
    except ValueError:  # JSONDecodeError·UnicodeDecodeError
        return _Failure("parse", "응답 형식 오류(JSON 아님)")
    choices = data.get("choices") if isinstance(data, dict) else None
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return _Failure("parse", "응답 형식 오류(choices 없음)")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return _Failure("parse", "응답 형식 오류(message 없음)")
    text = message.get("content")
    if text is None:
        text = ""
    elif not isinstance(text, str):
        return _Failure("parse", "응답 형식 오류(content가 문자열이 아님)")
    raw_finish = choices[0].get("finish_reason")
    finish: FinishReason = raw_finish if raw_finish in ("stop", "length") else "other"
    usage = data.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    model = data.get("model")
    return _Reply(
        text=text,
        finish_reason=finish,
        input_tokens=_token_count(usage.get("prompt_tokens")),
        output_tokens=_token_count(usage.get("completion_tokens")),
        model=model if isinstance(model, str) and model else None,
    )


class ChatClient:
    """OpenAI 호환 chat 클라이언트. 부품은 처음 호출할 때 만든다(`--no-env` 구성도 해시는 냄).

    `transport`·`sleep`·`clock`·`now`는 시험용으로 바꿔 끼울 수 있다.
    """

    def __init__(
        self,
        settings: ModelSettings,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.perf_counter,
        now: Callable[[], float] = time.time,
    ) -> None:
        base_url = settings.base_url.get_secret_value()
        api_key = settings.api_key.get_secret_value() if settings.api_key is not None else None
        if base_url == UNSET_PLACEHOLDER:
            raise ClientConfigError(
                "base_url이 자리표시 값입니다. --no-env로 읽은 구성으로는 모델을 부를 수 없습니다"
                "(환경 변수를 넣고 다시 읽으세요)"
            )
        if api_key == UNSET_PLACEHOLDER:
            raise ClientConfigError(
                "api_key가 자리표시 값입니다. --no-env로 읽은 구성으로는 모델을 부를 수 없습니다"
                "(환경 변수를 넣고 다시 읽으세요)"
            )
        self._settings = settings
        self._sleep = sleep
        self._clock = clock
        self._now = now
        self._timeout_s = settings.run.timeout_s
        self._max_attempts = 1 + settings.run.max_retries
        self._path = base_url + "/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._proxy = settings.proxy
        self._http = httpx.Client(
            transport=transport,
            headers=headers,
            timeout=httpx.Timeout(self._timeout_s, connect=CONNECT_TIMEOUT_S),
            follow_redirects=False,
            # proxy: none이면 환경 변수·시스템 프록시를 읽지 않는다(사내망 직접 연결).
            trust_env=settings.proxy == "env",
        )

    def __repr__(self) -> str:
        return f"ChatClient(model={self._settings.model!r})"

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()

    # --- 호출 -------------------------------------------------------------------------

    def chat(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        response_format: Mapping[str, Any] | None = None,
    ) -> Answer:
        """한 번 부른다(호출 안 재시도 포함). 실패도 예외가 아니라 `Answer`로 돌려준다.

        `response_format`은 문항 데이터(`answer_schema`)에서 나온 값만 넘긴다. 그 밖의 출력
        조건은 모두 `settings.params`(해시에 들어감)로만 정한다.
        """
        body: dict[str, Any] = {
            "model": self._settings.model,
            "messages": [dict(m) for m in messages],
            **self._settings.params,
        }
        if response_format is not None:
            body["response_format"] = dict(response_format)

        start = self._clock()
        attempts = timeouts = 0
        while True:
            attempts += 1
            t0 = self._clock()
            outcome = self._attempt(body)
            latency_ms = (self._clock() - t0) * 1000
            if isinstance(outcome, _Reply):
                return self._answer(outcome, latency_ms, (self._clock() - start) * 1000, attempts)
            if outcome.kind == "timeout":
                timeouts += 1
            if (
                FAILURE_CLASSES[outcome.kind] == "server"
                and attempts < self._max_attempts
                and timeouts < MAX_TIMEOUT_ATTEMPTS
            ):
                self._sleep(self._wait_s(attempts, outcome.retry_after_s))
                continue
            total_ms = (self._clock() - start) * 1000
            return Answer(
                text="",
                finish_reason="error",
                usage=Usage(latency_ms=latency_ms, total_ms=max(total_ms, latency_ms)),
                error_kind=outcome.kind,
                error=outcome.message,
                attempts=attempts,
            )

    def _attempt(self, body: Mapping[str, Any]) -> _Reply | _Failure:
        """한 번 시도한다. httpx 예외는 종류만 남기고 문자열은 버린다."""
        try:
            response = self._http.post(self._path, json=body)
        except httpx.TimeoutException as exc:
            if isinstance(exc, httpx.ConnectTimeout):
                msg = f"연결 시간 초과({CONNECT_TIMEOUT_S:g}초)"
                return _Failure("timeout", msg + self._proxy_hint())
            return _Failure("timeout", f"응답 시간 초과({self._timeout_s:g}초)")
        except httpx.NetworkError as exc:
            hint = self._proxy_hint() if isinstance(exc, httpx.ConnectError) else ""
            return _Failure("connect", f"연결 실패({type(exc).__name__}){hint}")
        except httpx.RemoteProtocolError:
            return _Failure("connect", "연결 실패(서버가 응답 중 연결을 끊음)")
        except httpx.HTTPError as exc:
            return _Failure("other", f"호출 오류({type(exc).__name__})")
        except httpx.InvalidURL:  # HTTPError 밖. 구성 검사가 막지만 검증을 건너뛴 설정에 대비
            return _Failure("config_error", "base_url 칸을 주소로 읽을 수 없습니다")

        status = response.status_code
        if 200 <= status < 300:
            return _parse_reply(response.content)
        if status in _CONFIG_ERRORS:
            return _Failure("config_error", _CONFIG_ERRORS[status])
        retry_after = self._retry_after_s(response.headers.get("retry-after"))
        if status in (408, 429) or 500 <= status < 600:
            return _Failure("http_status", f"HTTP {status}", retry_after)
        if 400 <= status < 500:
            return _Failure("client_error", f"HTTP {status}")
        return _Failure("other", f"HTTP {status}", retry_after)  # 1xx·3xx(리다이렉트는 안 따름)

    def _proxy_hint(self) -> str:
        """프록시 없이 연결을 맺지 못했을 때 붙이는 안내(주소·키는 넣지 않음)."""
        if self._proxy == "none":
            return ". 회사 프록시가 필요하면 이 모델 설정에 proxy: env"
        return ""

    # --- 대기 -------------------------------------------------------------------------

    def _retry_after_s(self, value: str | None) -> float | None:
        """`Retry-After`를 초로 바꾼다. 초 형식·HTTP 날짜 형식을 읽고, 못 읽으면 None."""
        if value is None:
            return None
        value = value.strip()
        if _DELTA_SECONDS_RE.match(value):
            return float(value)
        try:
            when = email.utils.parsedate_to_datetime(value)
        except (TypeError, ValueError, IndexError):
            return None
        if when.tzinfo is None:  # "-0000"처럼 시간대가 없으면 UTC로 본다
            when = when.replace(tzinfo=UTC)
        return max(0.0, when.timestamp() - self._now())

    @staticmethod
    def _wait_s(attempt: int, retry_after_s: float | None) -> float:
        """다음 시도 전 대기. `Retry-After`가 있으면 그 값(60초 상한), 없으면 지수(30초 상한)."""
        if retry_after_s is not None:
            return min(retry_after_s, RETRY_AFTER_MAX_S)
        return float(min(BACKOFF_BASE_S * 2 ** (attempt - 1), BACKOFF_MAX_S))

    # --- 응답 -------------------------------------------------------------------------

    def _answer(self, reply: _Reply, latency_ms: float, total_ms: float, attempts: int) -> Answer:
        served, warnings = self._served_model(reply.model)
        return Answer(
            text=reply.text,
            finish_reason=reply.finish_reason,
            usage=Usage(
                input_tokens=reply.input_tokens,
                output_tokens=reply.output_tokens,
                latency_ms=latency_ms,
                total_ms=max(total_ms, latency_ms),
            ),
            served_model=served,
            attempts=attempts,
            warnings=tuple(warnings),
        )

    def _served_model(self, raw: str | None) -> tuple[str | None, list[str]]:
        """응답 모델 이름을 기록할 값과 경고로 바꾼다. 비교는 원래 값으로 정확히 한다."""
        if raw is None:
            return None, []
        if has_address(raw):
            return None, ["응답 모델 이름에 주소가 들어 있어 기록하지 않습니다"]
        configured = self._settings.model
        if is_absolute_path(raw):
            tail = re.split(r"[\\/]", raw.rstrip("\\/"))[-1] or "(이름 없음)"
            return tail, [
                f"응답 모델 이름(…/{tail})이 설정({configured})과 다릅니다. "
                f"응답 모델 이름이 경로입니다. {_PATH_HINT}"
            ]
        if raw != configured:
            return raw, [f"응답 모델 이름({raw})이 설정({configured})과 다릅니다"]
        return raw, []


__all__ = [
    "BACKOFF_BASE_S",
    "BACKOFF_MAX_S",
    "CONNECT_TIMEOUT_S",
    "FAILURE_CLASSES",
    "MAX_TIMEOUT_ATTEMPTS",
    "RETRY_AFTER_MAX_S",
    "ChatClient",
    "ClientConfigError",
    "FailureClass",
    "failure_class",
]
