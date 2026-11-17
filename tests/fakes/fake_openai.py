"""가짜 OpenAI 호환 서버 두 가지.

- `scripted_transport`: httpx `MockTransport`. 호출마다 정해 둔 응답·예외를 돌려준다(소켓 없음).
- `FakeOpenAIServer`: 표준 라이브러리 `http.server` 스레드 서버. `127.0.0.1:0`에서 뜨고 모델
  이름별로 정해 둔 답을 돌려준다(실제 소켓 시험, U9).
"""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import TracebackType
from typing import Any, Self
from urllib.parse import urlsplit

import httpx

BODY_MARKER = "SECRET-BODY-MARKER"
"""오류 응답 본문에 넣는 표시. 클라이언트 출력에 나오면 본문이 샌 것이다."""


def chat_body(
    text: str = "답 [1]",
    *,
    model: str | None = "model-a",
    finish_reason: str | None = "stop",
    usage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """정상 chat completion 응답 본문."""
    choice: dict[str, Any] = {"index": 0, "message": {"role": "assistant", "content": text}}
    if finish_reason is not None:
        choice["finish_reason"] = finish_reason
    body: dict[str, Any] = {"object": "chat.completion", "choices": [choice]}
    if model is not None:
        body["model"] = model
    body["usage"] = (
        dict(usage) if usage is not None else {"prompt_tokens": 12, "completion_tokens": 5}
    )
    return body


def error_body() -> dict[str, Any]:
    """오류 응답 본문. 주소·표시를 담아 새는지 시험한다."""
    return {"error": {"message": f"{BODY_MARKER} at http://localhost:9/v1 key dummy-key"}}


class FakeClock:
    """주입용 시계. `sleep`은 기다리지 않고 시간만 앞으로 돌린다."""

    def __init__(self) -> None:
        self.t = 1000.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += seconds


@dataclass(frozen=True)
class Step:
    """한 번의 응답. `exc`가 있으면 응답 대신 그 httpx 예외를 낸다(메시지에 주소를 넣음)."""

    status: int = 200
    body: Any = None
    raw: bytes | None = None
    headers: Mapping[str, str] = field(default_factory=dict)
    exc: type[httpx.RequestError] | None = None
    elapsed_s: float = 0.0


def ok(text: str = "답 [1]", **kw: Any) -> Step:
    return Step(body=chat_body(text, **kw))


class ScriptedTransport(httpx.MockTransport):
    """정해 둔 `Step`을 차례로 돌려주는 MockTransport. 받은 요청을 `requests`에 남긴다."""

    def __init__(self, steps: Sequence[Step], clock: FakeClock | None = None) -> None:
        self.steps = list(steps)
        self.requests: list[httpx.Request] = []
        self._clock = clock
        super().__init__(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if not self.steps:
            raise AssertionError("정해 둔 응답보다 많이 불렀습니다")
        step = self.steps.pop(0)
        if self._clock is not None:
            self._clock.t += step.elapsed_s
        if step.exc is not None:
            raise step.exc(f"failed for {request.url} with dummy-key", request=request)
        if step.raw is not None:
            return httpx.Response(step.status, content=step.raw, headers=dict(step.headers))
        body = step.body if step.body is not None else error_body()
        return httpx.Response(step.status, json=body, headers=dict(step.headers))

    @property
    def calls(self) -> int:
        return len(self.requests)

    def json_bodies(self) -> list[dict[str, Any]]:
        return [json.loads(r.content) for r in self.requests]


def scripted_transport(steps: Sequence[Step], clock: FakeClock | None = None) -> ScriptedTransport:
    return ScriptedTransport(steps, clock)


class FakeOpenAIServer:
    """실제 소켓으로 도는 가짜 OpenAI 호환 서버.

    `scripts`는 모델 이름 → (상태 코드, 본문) 목록이다. 차례로 돌려주고, 마지막 것은 되풀이한다.
    `base_url`은 `/v1`까지 포함한다. `/v1/chat/completions`가 아닌 경로는 404다.
    프록시 주소로 쓰면 절대 주소 꼴 요청도 받아 답한다(`requests[i]["path"]`에 그대로 남음).
    """

    def __init__(self, scripts: Mapping[str, Sequence[tuple[int, Mapping[str, Any]]]]) -> None:
        self._scripts = {name: list(steps) for name, steps in scripts.items()}
        self._lock = threading.Lock()
        self.requests: list[dict[str, Any]] = []
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler_class())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host!s}:{port}/v1"

    def _next(self, model: str) -> tuple[int, Mapping[str, Any]]:
        with self._lock:
            steps = self._scripts.get(model)
            if not steps:
                return 404, {"error": {"message": "model not found"}}
            return steps.pop(0) if len(steps) > 1 else steps[0]

    def _handler_class(self) -> type[BaseHTTPRequestHandler]:
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length) or b"{}")
                with server._lock:
                    server.requests.append(
                        {"path": self.path, "body": body, "auth": self.headers.get("Authorization")}
                    )
                payload: Mapping[str, Any]
                # 프록시로 쓰일 때는 경로가 절대 주소 꼴(http://대상/v1/...)로 온다.
                if urlsplit(self.path).path != "/v1/chat/completions":
                    status, payload = 404, {"error": {"message": "no route"}}
                else:
                    status, payload = server._next(str(body.get("model")))
                data = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, format: str, *args: Any) -> None:
                """표준 오류로 주소를 찍지 않는다."""

        return Handler

    def __enter__(self) -> Self:
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)
