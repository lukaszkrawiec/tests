"""Local HTTP fakes.

The two external surfaces this tool touches — Anthropic's count_tokens endpoint and the
GitHub REST API — are both reachable only with credentials this test suite does not
have. Standing up a real server on localhost and pointing the client at it exercises the
actual HTTP path, including headers, retries, and error handling, which a monkeypatched
function would skip.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable


class FakeServer:
    """A threaded HTTP server driven by a request handler function.

    The handler receives (method, path, body) and returns (status, payload).
    """

    def __init__(self, handler: Callable[[str, str, Any], tuple[int, Any]]) -> None:
        self._handler = handler
        self.requests: list[tuple[str, str, Any, dict[str, str]]] = []
        self._lock = threading.Lock()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            # HTTP/1.0 so each response closes its connection. Keep-alive would leave
            # sockets open past the test and surface as resource warnings.
            protocol_version = "HTTP/1.0"

            def _read_body(self) -> Any:
                length = int(self.headers.get("content-length") or 0)
                if not length:
                    return None
                raw = self.rfile.read(length)
                try:
                    return json.loads(raw)
                except ValueError:
                    return raw.decode("utf-8", "replace")

            def _respond(self, method: str) -> None:
                body = self._read_body()
                with outer._lock:
                    outer.requests.append(
                        (method, self.path, body, dict(self.headers.items()))
                    )
                try:
                    status, payload = outer._handler(method, self.path, body)
                except Exception as exc:  # pragma: no cover - test bug surfacing
                    status, payload = 500, {"error": repr(exc)}
                encoded = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def do_GET(self):  # noqa: N802 - http.server naming
                self._respond("GET")

            def do_POST(self):  # noqa: N802
                self._respond("POST")

            def do_PATCH(self):  # noqa: N802
                self._respond("PATCH")

            def log_message(self, *args):  # noqa: D102 - silence test output
                return

        # Threaded because warm() issues concurrent requests; a single-threaded server
        # would serialize them and hide any concurrency bug.
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def __enter__(self) -> "FakeServer":
        self._thread.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


def count_tokens_handler(
    *,
    base_overhead: int = 10,
    per_tool: int = 5,
    failures: list[int] | None = None,
) -> Callable[[str, str, Any], tuple[int, Any]]:
    """A count_tokens stand-in whose arithmetic is obvious by inspection.

    ``input_tokens`` is a fixed request overhead plus one token per whitespace-separated
    word in the system prompt and messages, plus a flat cost per tool. The fixed
    overhead is what makes marginal counting testable: a correct implementation
    subtracts it, an incorrect one reports it on every component.

    ``failures`` is a list of status codes to return before succeeding, for retry tests.
    """
    pending_failures = list(failures or [])

    def handler(method: str, path: str, body: Any) -> tuple[int, Any]:
        assert method == "POST", method
        assert path == "/v1/messages/count_tokens", path
        if pending_failures:
            return pending_failures.pop(0), {"error": {"message": "transient"}}
        assert isinstance(body, dict)
        total = base_overhead
        system = body.get("system") or ""
        total += len(system.split())
        for message in body.get("messages") or []:
            content = message.get("content")
            if isinstance(content, str):
                total += len(content.split())
        tools = body.get("tools") or []
        total += per_tool * len(tools)
        total += sum(len(json.dumps(tool).split()) for tool in tools)
        return 200, {"input_tokens": total}

    return handler
