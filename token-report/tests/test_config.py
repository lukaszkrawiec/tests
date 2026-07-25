"""Configuration validation.

Every case here was accepted before this review pass. The first is the one that matters:
`max_entries = 0` is an easy typo that silently deleted the entire recorded trend.
"""

from pathlib import Path

import pytest

from tokenreport.config import Config, ConfigError
from tokenreport.history import History, HistoryEntry, HistoryError


def cfg(**history):
    return Config.from_raw({"entrypoint": "a:b", "history": history}, root=Path("."))


class TestHistoryBounds:
    def test_zero_max_entries_is_rejected(self):
        # It used to be accepted, and prune then returned an empty history — wiping the
        # whole series on the next push to the default branch.
        with pytest.raises(ConfigError, match="max_entries must be a positive integer"):
            cfg(max_entries=0)

    def test_negative_retention_is_rejected(self):
        # A negative window put the cutoff in the future, so every entry counted as old
        # and all recent detail collapsed to one point per week.
        with pytest.raises(ConfigError, match="retention_days must be a positive integer"):
            cfg(retention_days=-5)

    def test_a_non_numeric_bound_is_a_config_error_not_a_traceback(self):
        # A bare ValueError escaped the CLI's handler and printed a stack trace.
        with pytest.raises(ConfigError):
            cfg(retention_days="ninety")

    def test_booleans_are_not_accepted_as_integers(self):
        with pytest.raises(ConfigError):
            cfg(max_entries=True)

    def test_valid_bounds_pass_through(self):
        assert cfg(retention_days=30, max_entries=100).retention_days == 30


class TestHistoryLocation:
    def test_branch_must_be_a_string(self):
        with pytest.raises(ConfigError, match="branch must be a non-empty string"):
            cfg(branch=123)

    def test_empty_branch_is_rejected(self):
        with pytest.raises(ConfigError):
            cfg(branch="   ")

    @pytest.mark.parametrize("bad", ["/etc/passwd", "../../escape.json", "a/../../b.json"])
    def test_path_must_stay_inside_the_data_branch(self, bad):
        # The value becomes a path in a git tree; absolute or traversing paths cannot be
        # represented and failed deep inside git plumbing with an opaque error.
        with pytest.raises(ConfigError, match="relative path"):
            cfg(path=bad)

    def test_a_nested_relative_path_is_allowed(self):
        # github-action-benchmark uses dev/bench/data.js, so nesting must keep working.
        assert cfg(path="dev/bench/data.json").history_path == "dev/bench/data.json"


class TestPruneGuard:
    def test_prune_refuses_non_positive_bounds_rather_than_emptying(self):
        history = History(
            entries=[
                HistoryEntry(commit="c", timestamp="2026-07-01T00:00:00Z", model="m",
                             counter="tiktoken", resident=1, on_demand=0)
            ]
        )
        with pytest.raises(HistoryError, match="positive bounds"):
            history.prune(max_entries=0)
        with pytest.raises(HistoryError, match="positive bounds"):
            history.prune(retention_days=0)
