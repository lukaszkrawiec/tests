"""Token counters.

The default is ``tiktoken``: no API key, no network at measurement time once its
vocabulary is cached, identical numbers on every machine, and it works on pull requests
from forks, which receive no repository secrets. Its counts are *not* Claude's — tiktoken
has no Claude vocabulary — so absolute figures are indicative and the trend is the signal.

``anthropic`` is opt-in and is the only counter whose absolute figures match what you are
billed. It calls ``/v1/messages/count_tokens``, which is free to use, subject to its own
requests-per-minute limits independent of the Messages API. Its counts are taken
*marginally* against a cached baseline request: measuring a component in isolation would
include fixed per-request overhead, inflating every component and making the sum
meaningless.

Which counter produced a report is recorded alongside it, and comparisons refuse to cross
a change of counter, because switching re-bases every number.
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
DEFAULT_COUNTER = "tiktoken"
# Allowance for the framing a provider adds around each tool definition, which costs
# more than the schema's serialized JSON. Offline counters cannot measure it.
TOOL_FRAMING_TOKENS = 8
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


class _OfflineBase:
    """Shared shape for counters that measure text locally. Subclasses set both."""

    name: str
    detail: str

    @property
    def info(self) -> CounterInfo:
        return CounterInfo(name=self.name, exact=False, detail=self.detail)

    def count_text(self, text: str) -> int:  # pragma: no cover - overridden
        raise NotImplementedError

    def count_tools(self, tools: Sequence[Mapping[str, Any]]) -> int:
        # A tool definition costs more than its serialized JSON: names, descriptions
        # and schema keys are re-rendered into a structured form. The per-tool constant
        # is a rough allowance for that framing.
        serialized = json.dumps(list(tools), sort_keys=True)
        return self.count_text(serialized) + TOOL_FRAMING_TOKENS * len(tools)

    def warm(self, texts: Iterable[str]) -> None:
        return None


class TiktokenCounter(_OfflineBase):
    """Counts with a tiktoken encoding — the default counter.

    ``exact`` is False, and that word is doing precise work: these counts are not
    Claude's. tiktoken has no Claude vocabulary, and Claude 4.7 and later tokenize
    roughly 30% higher than earlier models for identical text, so treat absolute
    figures as indicative and the trend as the signal.

    What it *is* is deterministic and reproducible: the same text yields the same count
    on every machine, forever, with no key and no network at measurement time. That is
    what makes it usable as a tracked series, which is a different property from being
    numerically right.
    """

    DEFAULT_ENCODING = "o200k_base"

    def __init__(self, *, encoding: str | None = None, model: str | None = None) -> None:
        requested = encoding or self.DEFAULT_ENCODING
        try:
            import tiktoken
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise CounterError(
                "the tiktoken counter needs the tiktoken package; install "
                "tokenreport with its default dependencies, or select the "
                "'heuristic' counter"
            ) from exc
        try:
            # Downloads and caches the vocabulary on first use. Set TIKTOKEN_CACHE_DIR
            # in CI to keep that off the critical path of every run.
            self._encoding = tiktoken.get_encoding(requested)
        except Exception as exc:  # noqa: BLE001 - unknown name or fetch failure
            raise CounterError(
                f"could not load tiktoken encoding {requested!r}: {exc}"
            ) from exc
        self.name = "tiktoken"
        self.encoding_name = requested
        self.detail = f"tiktoken {requested}; not Claude's tokenizer"
        self._model = model

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        # Special tokens are data here, not control sequences: a prompt that happens to
        # contain one must be counted, not rejected.
        return len(self._encoding.encode(text, disallowed_special=()))


class HeuristicCounter(_OfflineBase):
    """Pure-Python estimate, for when tiktoken cannot be installed or reached.

    Less accurate than tiktoken and it says so. Present because tiktoken fetches its
    vocabulary over the network on first use, and an air-gapped runner would otherwise
    have no counter at all.
    """

    #  Approximates a BPE pretokenizer: a word, a short run of digits, or a run of
    #  punctuation, each carrying any whitespace that precedes it. Every alternative
    #  requires at least one non-space character, so whitespace never forms a chunk of
    #  its own — it is absorbed into the chunk that follows, as a real BPE does.
    _PRETOKEN = re.compile(r"\s*[A-Za-z]+|\s*\d{1,3}|\s*[^\sA-Za-z\d]+")
    #  Calibrated against tiktoken o200k_base over this repository's prompts, markdown
    #  and source: mean ratio 1.00, and 0.96–1.01 on prose, which is what prompts are.
    #  Most chunks fall under the divisor and cost the floor of one token, which is why
    #  the constant is much larger than the familiar "four characters per token".
    _CHARS_PER_TOKEN = 10.0

    def __init__(self, *, model: str | None = None) -> None:
        self.name = "heuristic"
        self.detail = "pure-python estimate; least accurate counter"
        self._model = model

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        return sum(
            max(1, math.ceil(len(chunk) / self._CHARS_PER_TOKEN))
            for chunk in self._PRETOKEN.findall(text)
        )


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


COUNTER_NAMES = ("tiktoken", "heuristic", "anthropic")
# Only the provider's own endpoint reports the tokenization it will actually bill.
EXACT_COUNTERS = frozenset({"anthropic"})


def counter_is_exact(name: str) -> bool:
    """Whether counts from ``name`` match what the provider charges.

    History records the counter's name rather than this flag, so it is derived in one
    place instead of being stored — and cannot drift out of step with the counters.
    """
    return name in EXACT_COUNTERS


def resolve(
    *,
    model: str,
    prefer: str = DEFAULT_COUNTER,
    encoding: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> Counter:
    """Pick a counter by name.

    tiktoken is the default: no key, no per-run network dependency, identical numbers on
    every machine, and it works on fork pull requests, which receive no secrets. The
    Anthropic counter is opt-in for when exact Claude counts are wanted; it is the only
    one whose absolute figures match what you are billed.

    There is deliberately no "auto" mode. Silently switching counters based on whether a
    key happens to be present would change what the numbers mean between runs, and a
    trend whose tokenizer varies invisibly is worse than no trend.
    """
    if prefer == "tiktoken":
        try:
            return TiktokenCounter(encoding=encoding, model=model)
        except CounterError:
            # An air-gapped runner cannot fetch the vocabulary. Degrade rather than
            # fail, and let the report name the counter that actually ran.
            return HeuristicCounter(model=model)
    if prefer == "heuristic":
        return HeuristicCounter(model=model)
    if prefer == "anthropic":
        key = api_key if api_key is not None else os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise CounterError(
                "counter 'anthropic' requested but ANTHROPIC_API_KEY is not set; "
                "set the secret or use the default 'tiktoken' counter"
            )
        return AnthropicCounter(model=model, api_key=key, base_url=base_url)
    raise CounterError(
        f"unknown counter {prefer!r} (expected one of {', '.join(COUNTER_NAMES)})"
    )
