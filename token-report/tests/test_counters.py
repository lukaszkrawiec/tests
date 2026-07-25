import pytest

from tokenreport.counters import (
    DEFAULT_COUNTER,
    AnthropicCounter,
    CounterError,
    HeuristicCounter,
    LengthCounter,
    TiktokenCounter,
    counter_is_exact,
    resolve,
)

from .fakes import FakeServer, count_tokens_handler


class TestHeuristicCounter:
    def test_names_itself_and_admits_it_is_not_exact(self):
        # The name is what history stamps and what comparisons key on, so a counter must
        # identify itself specifically rather than as a vague "offline".
        info = HeuristicCounter().info
        assert info.exact is False
        assert info.name == "heuristic"

    def test_empty_text_is_zero(self):
        assert HeuristicCounter().count_text("") == 0

    def test_count_grows_with_text(self):
        counter = HeuristicCounter()
        short = counter.count_text("The quick brown fox.")
        long = counter.count_text("The quick brown fox. " * 20)
        assert 0 < short < long

    def test_estimate_tracks_tiktoken_within_a_quarter(self):
        # Calibrated against tiktoken rather than a chars-per-token rule of thumb: this
        # is the fallback when tiktoken is unavailable, so tiktoken is the target.
        tiktoken = pytest.importorskip("tiktoken")
        encoding = tiktoken.get_encoding(TiktokenCounter.DEFAULT_ENCODING)
        text = " ".join(
            ["Context engineering keeps prompts small and specific."] * 30
        )
        estimate = HeuristicCounter().count_text(text)
        truth = len(encoding.encode(text))
        assert 0.75 < estimate / truth < 1.25, f"{estimate} vs {truth}"

    def test_tools_cost_more_than_their_json(self):
        counter = HeuristicCounter()
        tool = {"name": "search", "description": "Search the web.", "input_schema": {}}
        assert counter.count_tools([tool]) > counter.count_text(str(tool)) / 2
        assert counter.count_tools([tool, tool]) > counter.count_tools([tool])


class TestLengthCounter:
    def test_one_token_per_word(self):
        assert LengthCounter().count_text("one two three") == 3


class TestAnthropicCounter:
    def make(self, server, **kwargs):
        return AnthropicCounter(
            model="claude-opus-5",
            api_key="test-key",
            base_url=server.base_url,
            sleep=lambda _: None,
            **kwargs,
        )

    def test_counts_are_marginal_not_absolute(self):
        # The fake adds 10 tokens of fixed request overhead. A component of three words
        # must report 3, not 13, or every component would be inflated and the totals
        # would not reconcile.
        with FakeServer(count_tokens_handler(base_overhead=10)) as server:
            counter = self.make(server)
            assert counter.count_text("one two three") == 3

    def test_baseline_is_requested_once_and_reused(self):
        with FakeServer(count_tokens_handler()) as server:
            counter = self.make(server)
            counter.count_text("alpha beta")
            counter.count_text("gamma delta epsilon")
            # One baseline request plus one per distinct text.
            assert counter.request_count == 3

    def test_identical_text_is_counted_once(self):
        with FakeServer(count_tokens_handler()) as server:
            counter = self.make(server)
            assert counter.count_text("same words here") == 3
            assert counter.count_text("same words here") == 3
            assert counter.request_count == 2  # baseline + one count

    def test_empty_text_makes_no_request(self):
        with FakeServer(count_tokens_handler()) as server:
            counter = self.make(server)
            assert counter.count_text("") == 0
            assert counter.request_count == 0

    def test_sends_text_as_system_prompt_with_the_configured_model(self):
        with FakeServer(count_tokens_handler()) as server:
            self.make(server).count_text("hello there")
            payloads = [body for _, _, body, _ in server.requests]
            assert payloads[-1]["system"] == "hello there"
            assert payloads[-1]["model"] == "claude-opus-5"
            assert payloads[-1]["messages"]  # a user turn is required

    def test_sends_authentication_and_version_headers(self):
        with FakeServer(count_tokens_handler()) as server:
            self.make(server).count_text("x")
            headers = server.requests[-1][3]
            lower = {k.lower(): v for k, v in headers.items()}
            assert lower["x-api-key"] == "test-key"
            assert lower["anthropic-version"] == "2023-06-01"

    def test_tools_are_sent_as_tools_not_serialized_text(self):
        with FakeServer(count_tokens_handler(per_tool=5)) as server:
            counter = self.make(server)
            tokens = counter.count_tools([{"name": "search", "description": "d"}])
            body = server.requests[-1][2]
            assert "tools" in body and body["tools"][0]["name"] == "search"
            assert tokens > 0

    def test_empty_tool_list_makes_no_request(self):
        with FakeServer(count_tokens_handler()) as server:
            counter = self.make(server)
            assert counter.count_tools([]) == 0
            assert counter.request_count == 0

    def test_retries_on_rate_limit_then_succeeds(self):
        with FakeServer(count_tokens_handler(failures=[429, 429])) as server:
            counter = self.make(server)
            assert counter.count_text("one two") == 2

    def test_retries_on_server_error(self):
        with FakeServer(count_tokens_handler(failures=[503])) as server:
            assert self.make(server).count_text("one") == 1

    def test_does_not_retry_a_malformed_request(self):
        # A 400 means the request shape is wrong; retrying wastes time and hides the bug.
        with FakeServer(count_tokens_handler(failures=[400])) as server:
            counter = self.make(server)
            with pytest.raises(CounterError, match="HTTP 400"):
                counter.count_text("one")
            assert counter.request_count == 1

    def test_gives_up_after_max_retries(self):
        with FakeServer(count_tokens_handler(failures=[500] * 10)) as server:
            counter = self.make(server, max_retries=2)
            with pytest.raises(CounterError, match="failed after 3 attempts"):
                counter.count_text("one")

    def test_warm_populates_cache_so_later_counts_are_free(self):
        with FakeServer(count_tokens_handler()) as server:
            counter = self.make(server)
            counter.warm(["alpha beta", "gamma", "alpha beta"])
            before = counter.request_count
            assert counter.count_text("alpha beta") == 2
            assert counter.count_text("gamma") == 1
            assert counter.request_count == before  # served from cache

    def test_requires_an_api_key(self):
        with pytest.raises(CounterError, match="API key is required"):
            AnthropicCounter(model="m", api_key="")


class TestTiktokenCounter:
    def test_is_the_default_counter(self):
        assert DEFAULT_COUNTER == "tiktoken"

    def test_names_itself_and_admits_it_is_not_claude(self):
        pytest.importorskip("tiktoken")
        info = TiktokenCounter().info
        assert info.name == "tiktoken"
        # Reproducible is not the same as correct, and the report must not imply it is.
        assert info.exact is False
        assert "not Claude" in info.detail

    def test_records_the_encoding_it_used(self):
        pytest.importorskip("tiktoken")
        assert TiktokenCounter().encoding_name == "o200k_base"
        assert TiktokenCounter(encoding="cl100k_base").encoding_name == "cl100k_base"

    def test_the_requested_encoding_is_the_one_actually_used(self):
        # Guards against silently falling back to the default, which would make the
        # configured encoding a lie and put mislabelled numbers into history.
        tiktoken = pytest.importorskip("tiktoken")
        text = "Progressive disclosure keeps the resident index small. 12345 — ok!"
        for name in ("o200k_base", "cl100k_base", "p50k_base"):
            expected = len(tiktoken.get_encoding(name).encode(text))
            assert TiktokenCounter(encoding=name).count_text(text) == expected, name

    def test_an_unknown_encoding_is_reported_clearly(self):
        pytest.importorskip("tiktoken")
        with pytest.raises(CounterError, match="could not load tiktoken encoding"):
            TiktokenCounter(encoding="not_a_real_encoding")

    def test_text_containing_a_special_token_is_counted_not_rejected(self):
        # A prompt discussing "<|endoftext|>" is data; tiktoken raises on it by default.
        pytest.importorskip("tiktoken")
        assert TiktokenCounter().count_text("the <|endoftext|> marker") > 0

    def test_empty_text_is_zero(self):
        pytest.importorskip("tiktoken")
        assert TiktokenCounter().count_text("") == 0


class TestCounterExactness:
    def test_only_the_provider_endpoint_is_exact(self):
        assert counter_is_exact("anthropic") is True
        assert counter_is_exact("tiktoken") is False
        assert counter_is_exact("heuristic") is False

    def test_an_unknown_counter_is_not_assumed_exact(self):
        assert counter_is_exact("something-new") is False


class TestResolve:
    def test_defaults_to_tiktoken_with_no_key_present(self):
        # No key, no network at measurement time, and it works on fork pull requests.
        counter = resolve(model="claude-opus-5", api_key="")
        assert counter.info.name in {"tiktoken", "heuristic"}

    def test_a_present_key_does_not_silently_change_the_counter(self):
        # Switching counter on the presence of a secret would change what the numbers
        # mean between runs, and a trend whose tokenizer varies invisibly is worse than
        # no trend at all.
        assert resolve(model="m", api_key="k").info.name != "anthropic"

    def test_anthropic_is_opt_in_by_name(self):
        assert resolve(model="m", prefer="anthropic", api_key="k").info.name == "anthropic"

    def test_explicit_anthropic_without_a_key_is_an_error(self):
        with pytest.raises(CounterError, match="ANTHROPIC_API_KEY is not set"):
            resolve(model="m", prefer="anthropic", api_key="")

    def test_heuristic_can_be_selected_explicitly(self):
        assert resolve(model="m", prefer="heuristic").info.name == "heuristic"

    def test_the_encoding_is_passed_through(self):
        pytest.importorskip("tiktoken")
        counter = resolve(model="m", prefer="tiktoken", encoding="cl100k_base")
        assert counter.encoding_name == "cl100k_base"

    def test_unknown_counter_name_is_an_error(self):
        with pytest.raises(CounterError, match="unknown counter 'magic'"):
            resolve(model="m", prefer="magic")
