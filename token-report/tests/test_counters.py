import pytest

from tokenreport.counters import (
    AnthropicCounter,
    CounterError,
    LengthCounter,
    OfflineCounter,
    resolve,
)

from .fakes import FakeServer, count_tokens_handler


class TestOfflineCounter:
    def test_reports_itself_as_inexact(self):
        # Exactness gates entry into history; an approximation must never claim it.
        info = OfflineCounter().info
        assert info.exact is False
        assert info.name == "offline"

    def test_empty_text_is_zero(self):
        assert OfflineCounter().count_text("") == 0

    def test_count_grows_with_text(self):
        counter = OfflineCounter()
        short = counter.count_text("The quick brown fox.")
        long = counter.count_text("The quick brown fox. " * 20)
        assert 0 < short < long

    def test_estimate_is_in_a_plausible_range_for_prose(self):
        # Roughly 4 characters per token is the accepted rule of thumb for English;
        # this guards against an estimator that is wrong by an order of magnitude.
        text = " ".join(["context engineering keeps prompts small"] * 40)
        estimate = OfflineCounter().count_text(text)
        assert len(text) / 8 < estimate < len(text) / 2

    def test_tools_cost_more_than_their_json(self):
        counter = OfflineCounter()
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


class TestResolve:
    def test_auto_falls_back_to_offline_without_a_key(self):
        # This is what keeps fork pull requests working, since they get no secrets.
        counter = resolve(model="claude-opus-5", prefer="auto", api_key="")
        assert counter.info.name == "offline"

    def test_auto_uses_the_api_when_a_key_is_present(self):
        counter = resolve(model="claude-opus-5", prefer="auto", api_key="k")
        assert counter.info.name == "anthropic"

    def test_explicit_anthropic_without_a_key_is_an_error(self):
        with pytest.raises(CounterError, match="ANTHROPIC_API_KEY is not set"):
            resolve(model="m", prefer="anthropic", api_key="")

    def test_explicit_offline_ignores_a_present_key(self):
        assert resolve(model="m", prefer="offline", api_key="k").info.name == "offline"

    def test_unknown_counter_name_is_an_error(self):
        with pytest.raises(CounterError, match="unknown counter 'magic'"):
            resolve(model="m", prefer="magic")
