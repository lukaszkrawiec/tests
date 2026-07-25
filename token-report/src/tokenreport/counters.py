"""Token counters.

The default counter calls Anthropic's ``/v1/messages/count_tokens``, which is free to
use (subject to its own requests-per-minute limits, independent of Messages API
limits). Counts are taken *marginally* against a cached baseline request: counting a
component in isolation would include fixed per-request overhead, inflating every
component and making the sum meaningless.

An offline approximation exists for one specific reason: pull requests from forks do
not receive repository secrets, so an API-only counter simply fails for outside
contributors. Approximate runs are labelled and refused entry into history.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterable, Mapping, Protocol, Sequence

from .contract import CounterInfo

DEFAULT_BASE_URL = "https://api.anthropic.com"
API_VERSION = "2023-06-01"
# A minimal user turn is required for a valid request; its cost is part of the
# baseline that every marginal count subtracts.
_PROBE_MESSAGES = [{"role": "user", "content": "."}]


class CounterError(RuntimeError):
    """Raised when a counter cannot produce a count."""


class Counter(Protocol):
    """Measures the marginal token cost of context."""

    @property
    def info(self) -> CounterInfo: ...

    def count_text(self, text: str) -> int: ...

    def count_tools(self, tools: Sequence[Mapping[str, Any]]) -> int: ...

    def warm(self, texts: Iterable[str]) -> None:
        """Optionally pre-populate a cache. Implementations may ignore this."""


class OfflineCounter:
    """Dependency-free approximation, or tiktoken when it happens to be installed.

    This is deliberately not presented as accurate. Claude's tokenizer is not
    available offline, and Claude 4.7 and later tokenize roughly 30% higher than
    earlier models for the same text, so no offline vocabulary tracks it. Useful for
    relative signal on fork PRs; never written to history.
    """

    #  Approximates a BPE pretokenizer: words with leading space, numbers in short
    #  runs, punctuation, and whitespace runs are all separate chunks.
    _PRETOKEN = re.compile(r"\s*[A-Za-z]+|\s*\d{1,3}|\s*[^\sA-Za-z\d]+|\s+")
    _CHARS_PER_TOKEN = 4.2

    def __init__(self, *, model: str | None = None) -> None:
        self._encoding = None
        self._detail = "heuristic pretokenizer; approximate"
        try:  # pragma: no cover - depends on optional install
            import tiktoken

            self._encoding = tiktoken.get_encoding("o200k_base")
            self._detail = "tiktoken o200k_base; approximate for Claude"
        except Exception:  # noqa: BLE001 - any failure means fall back to heuristic
            pass
        self._model = model

    @property
    def info(self) -> CounterInfo:
        return CounterInfo(name="offline", exact=False, detail=self._detail)

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        if self._encoding is not None:  # pragma: no cover - optional install
            return len(self._encoding.encode(text, disallowed_special=()))
        total = 0
        for chunk in self._PRETOKEN.findall(text):
            stripped = chunk.strip()
            if not stripped:
                # Whitespace runs mostly merge into adjacent tokens; a long run of
                # indentation still costs something.
                total += max(0, (len(chunk) - 1) // 4)
                continue
            total += max(1, math.ceil(len(chunk) / self._CHARS_PER_TOKEN))
        return total

    def count_tools(self, tools: Sequence[Mapping[str, Any]]) -> int:
        # A tool definition costs more than its serialized JSON: names, descriptions
        # and schema keys are re-rendered into a structured form. The per-tool
        # constant is a rough allowance for that framing.
        serialized = json.dumps(list(tools), sort_keys=True)
        return self.count_text(serialized) + 8 * len(tools)

    def warm(self, texts: Iterable[str]) -> None:
        return None


class LengthCounter:
    """Deterministic counter for tests: one token per whitespace-separated word.

    Exists so every stage downstream of counting can be tested without a network or a
    tokenizer, with counts that are obvious by inspection.
    """

    def __init__(self, *, exact: bool = True) -> None:
        self._exact = exact

    @property
    def info(self) -> CounterInfo:
        return CounterInfo(name="length", exact=self._exact, detail="one token per word")

    def count_text(self, text: str) -> int:
        return len(text.split())

    def count_tools(self, tools: Sequence[Mapping[str, Any]]) -> int:
        return sum(
            len(json.dumps(tool, sort_keys=True).split()) + 1 for tool in tools
        )

    def warm(self, texts: Iterable[str]) -> None:
        return None


class AnthropicCounter:
    """Exact counts via ``/v1/messages/count_tokens``.

    Uses urllib rather than the SDK so the tool has no runtime dependencies and can be
    pointed at a local fake in tests.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 4,
        concurrency: int = 8,
        sleep=time.sleep,
    ) -> None:
        if not api_key:
            raise CounterError("an API key is required for the Anthropic counter")
        self._model = model
        self._api_key = api_key
        self._base_url = (base_url or os.environ.get("ANTHROPIC_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries
        self._concurrency = max(1, concurrency)
        self._sleep = sleep
        self._baseline: int | None = None
        self._text_cache: dict[str, int] = {}
        self._requests = 0

    @property
    def info(self) -> CounterInfo:
        return CounterInfo(
            name="anthropic", exact=True, detail=f"count_tokens on {self._model}"
        )

    @property
    def request_count(self) -> int:
        return self._requests

    def _post(self, payload: Mapping[str, Any]) -> int:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self._base_url}/v1/messages/count_tokens",
            data=body,
            method="POST",
            headers={
                "content-type": "application/json",
                "anthropic-version": API_VERSION,
                "x-api-key": self._api_key,
            },
        )
        last_error: str | None = None
        for attempt in range(self._max_retries + 1):
            try:
                self._requests += 1
                with urllib.request.urlopen(request, timeout=self._timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                return int(data["input_tokens"])
            except urllib.error.HTTPError as exc:
                # An HTTPError owns an open response; read and close it rather than
                # leaving it for the garbage collector to trip over.
                try:
                    detail = exc.read().decode("utf-8", "replace")[:400]
                finally:
                    exc.close()
                # 429 and 5xx are transient; any other 4xx means a malformed request
                # that retrying will not fix.
                if exc.code != 429 and exc.code < 500:
                    raise CounterError(
                        f"count_tokens returned HTTP {exc.code}: {detail}"
                    ) from None
                last_error = f"HTTP {exc.code}: {detail}"
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            except (KeyError, ValueError) as exc:
                raise CounterError(f"unexpected count_tokens response: {exc}") from exc
            if attempt < self._max_retries:
                self._sleep(2**attempt)
        raise CounterError(
            f"count_tokens failed after {self._max_retries + 1} attempts: {last_error}"
        )

    def _get_baseline(self) -> int:
        if self._baseline is None:
            self._baseline = self._post(
                {"model": self._model, "messages": _PROBE_MESSAGES}
            )
        return self._baseline

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        cached = self._text_cache.get(text)
        if cached is not None:
            return cached
        baseline = self._get_baseline()
        total = self._post(
            {"model": self._model, "system": text, "messages": _PROBE_MESSAGES}
        )
        marginal = max(0, total - baseline)
        self._text_cache[text] = marginal
        return marginal

    def count_tools(self, tools: Sequence[Mapping[str, Any]]) -> int:
        if not tools:
            return 0
        baseline = self._get_baseline()
        total = self._post(
            {
                "model": self._model,
                "tools": list(tools),
                "messages": _PROBE_MESSAGES,
            }
        )
        return max(0, total - baseline)

    def warm(self, texts: Iterable[str]) -> None:
        """Fill the cache concurrently.

        A knowledge base of a few hundred documents would take minutes one request at
        a time; the endpoint's rate limits are in the thousands per minute, so modest
        concurrency is well within them.
        """
        pending = [t for t in dict.fromkeys(texts) if t and t not in self._text_cache]
        if not pending:
            return
        self._get_baseline()  # serialize the baseline before fanning out
        with ThreadPoolExecutor(max_workers=self._concurrency) as pool:
            list(pool.map(self.count_text, pending))


def resolve(
    *,
    model: str,
    prefer: str = "auto",
    api_key: str | None = None,
    base_url: str | None = None,
) -> Counter:
    """Pick a counter.

    ``auto`` uses the API when a key is available and falls back to the offline
    approximation otherwise, which is what keeps fork pull requests working.
    """
    key = api_key if api_key is not None else os.environ.get("ANTHROPIC_API_KEY", "")
    if prefer == "offline":
        return OfflineCounter(model=model)
    if prefer == "anthropic":
        if not key:
            raise CounterError(
                "counter 'anthropic' requested but ANTHROPIC_API_KEY is not set"
            )
        return AnthropicCounter(model=model, api_key=key, base_url=base_url)
    if prefer != "auto":
        raise CounterError(f"unknown counter {prefer!r} (expected auto/anthropic/offline)")
    if key:
        return AnthropicCounter(model=model, api_key=key, base_url=base_url)
    return OfflineCounter(model=model)
