import datetime as dt

import pytest

from tokenreport.contract import CountedComponent, CounterInfo, Kind, Report, Tier
from tokenreport.history import (
    History,
    HistoryEntry,
    HistoryError,
    build_history_files,
)

NOW = dt.datetime(2026, 7, 1, tzinfo=dt.timezone.utc)


def stamp(days_ago: int) -> str:
    return (NOW - dt.timedelta(days=days_ago)).isoformat().replace("+00:00", "Z")


def report(components, *, commit="abc123", exact=True, model="claude-opus-5",
           generated_at=None):
    return Report(
        model=model,
        counter=CounterInfo(name="anthropic" if exact else "tiktoken", exact=exact),
        commit=commit,
        generated_at=generated_at or stamp(0),
        components=[
            CountedComponent(
                id=cid,
                tokens=tokens,
                tier=Tier.ON_DEMAND if cid.startswith("doc") else Tier.RESIDENT,
                kind=Kind.OTHER,
            )
            for cid, tokens in components.items()
        ],
    )


def entry(commit, days_ago, *, resident=100, model="claude-opus-5", components=None,
          counter="anthropic"):
    return HistoryEntry(
        commit=commit,
        timestamp=stamp(days_ago),
        model=model,
        counter=counter,
        resident=resident,
        on_demand=0,
        components=components or {},
    )


class TestEntryFromReport:
    def test_captures_totals_and_per_component_counts(self):
        result = HistoryEntry.from_report(report({"a": 10, "doc.b": 90}))
        assert result.resident == 10
        assert result.on_demand == 90
        assert result.components == {"a": 10, "doc.b": 90}

    def test_records_an_approximate_counter_and_stamps_its_name(self):
        # tiktoken is the default counter and is not exact. Gating on exactness would
        # mean the trend never records a point and the dashboard stays empty forever.
        result = HistoryEntry.from_report(report({"a": 10}, exact=False))
        assert result.counter == "tiktoken"

    def test_refuses_a_report_with_no_commit(self):
        with pytest.raises(HistoryError, match="no commit sha"):
            HistoryEntry.from_report(report({"a": 10}, commit=None))


class TestSerialization:
    def test_round_trips(self):
        original = History(entries=[entry("a", 1), entry("b", 0)])
        restored = History.loads(original.dumps())
        assert [e.commit for e in restored.sorted_entries()] == ["a", "b"]

    def test_empty_and_missing_input_produce_empty_history(self):
        assert History.loads(None).entries == []
        assert History.loads("").entries == []
        assert History.loads("   ").entries == []

    def test_invalid_json_is_reported_clearly(self):
        with pytest.raises(HistoryError, match="not valid JSON"):
            History.loads("{not json")

    def test_future_schema_version_is_refused(self):
        with pytest.raises(HistoryError, match="schema_version 99 is not supported"):
            History.loads('{"schema_version": 99, "entries": []}')

    def test_output_ends_with_a_newline(self):
        assert History(entries=[entry("a", 0)]).dumps().endswith("\n")


class TestAppend:
    def test_adds_a_new_entry(self):
        result = History(entries=[entry("a", 1)]).append(entry("b", 0))
        assert [e.commit for e in result.sorted_entries()] == ["a", "b"]

    def test_replaces_an_entry_for_the_same_commit(self):
        # Re-running a workflow must not add a second point for one commit.
        history = History(entries=[entry("a", 1, resident=100)])
        result = history.append(entry("a", 0, resident=250))
        assert len(result.entries) == 1
        assert result.entries[0].resident == 250

    def test_does_not_mutate_the_original(self):
        history = History(entries=[entry("a", 1)])
        history.append(entry("b", 0))
        assert len(history.entries) == 1


class TestTokenizerChanges:
    def test_detects_a_tokenizer_change(self):
        history = History(
            entries=[
                entry("a", 3, model="claude-opus-5"),
                entry("b", 2, model="claude-opus-5"),
                entry("c", 1, model="claude-fable-5"),
            ]
        )
        assert history.tokenizer_changes() == ["c"]

    def test_no_change_when_the_model_is_stable(self):
        assert History(entries=[entry("a", 2), entry("b", 1)]).tokenizer_changes() == []

    def test_detects_a_counter_change_at_a_stable_model(self):
        # Switching from tiktoken to the API re-bases every number just as a model
        # change does, so it breaks the series the same way.
        history = History(
            entries=[
                entry("a", 3, counter="tiktoken"),
                entry("b", 2, counter="tiktoken"),
                entry("c", 1, counter="anthropic"),
            ]
        )
        assert history.tokenizer_changes() == ["c"]

    def test_detects_multiple_changes_including_a_revert(self):
        history = History(
            entries=[
                entry("a", 4, model="m1"),
                entry("b", 3, model="m2"),
                entry("c", 2, model="m1"),
            ]
        )
        assert history.tokenizer_changes() == ["b", "c"]


class TestPrune:
    def test_recent_entries_are_kept_at_full_resolution(self):
        entries = [entry(f"c{i}", i) for i in range(10)]
        result = History(entries=entries).prune(retention_days=90, now=NOW)
        assert len(result.entries) == 10

    def test_old_entries_are_downsampled_to_one_per_week(self):
        # Two entries in each of two old weeks collapse to one point each.
        entries = [
            entry("w1a", 200), entry("w1b", 199),
            entry("w2a", 193), entry("w2b", 192),
        ]
        result = History(entries=entries).prune(retention_days=90, now=NOW)
        assert len(result.entries) == 2

    def test_downsampling_keeps_the_latest_point_in_each_week(self):
        result = History(entries=[entry("older", 200), entry("newer", 199)]).prune(
            retention_days=90, now=NOW
        )
        assert [e.commit for e in result.entries] == ["newer"]

    def test_max_entries_caps_the_file_dropping_oldest_first(self):
        entries = [entry(f"c{i:03d}", i) for i in range(50)]
        result = History(entries=entries).prune(
            retention_days=90, max_entries=10, now=NOW
        )
        assert len(result.entries) == 10
        # The 10 most recent are days 0-9, i.e. the newest timestamps.
        assert result.sorted_entries()[-1].commit == "c000"

    def test_pruning_empty_history_is_safe(self):
        assert History().prune(now=NOW).entries == []

    def test_ordering_is_chronological_after_pruning(self):
        entries = [entry("c", 1), entry("a", 3), entry("b", 2)]
        result = History(entries=entries).prune(now=NOW)
        assert [e.commit for e in result.sorted_entries()] == ["a", "b", "c"]


class TestRebuildingAReport:
    """to_report() supplies the baseline every pull request is compared against."""

    def base(self):
        return Report(
            model="claude-opus-5",
            counter=CounterInfo(name="tiktoken", exact=False),
            commit="basecommit",
            generated_at=stamp(1),
            components=[
                CountedComponent(id="agent.p", tokens=100, tier=Tier.RESIDENT,
                                 kind=Kind.OTHER),
                CountedComponent(id="kb.doc", tokens=500, tier=Tier.ON_DEMAND,
                                 kind=Kind.OTHER),
            ],
        )

    def test_round_trips_the_totals_of_the_report_it_came_from(self):
        original = self.base()
        rebuilt = HistoryEntry.from_report(original).to_report(metadata_from=original)
        assert rebuilt.resident == original.resident
        assert rebuilt.on_demand == original.on_demand

    def test_a_component_removed_by_this_commit_keeps_its_recorded_tier(self):
        # The bug this guards: with the component gone from the head report, defaulting
        # it to resident made a deleted on-demand document look like a 500-token drop in
        # the total paid on every request.
        entry = HistoryEntry.from_report(self.base())
        head = Report(
            model="claude-opus-5",
            counter=CounterInfo(name="tiktoken", exact=False),
            commit="headcommit",
            components=[
                CountedComponent(id="agent.p", tokens=100, tier=Tier.RESIDENT,
                                 kind=Kind.OTHER)
            ],
        )
        rebuilt = entry.to_report(metadata_from=head)
        assert rebuilt.resident == 100
        assert rebuilt.on_demand == 500

    def test_only_non_resident_tiers_are_stored(self):
        # Resident is the default, so recording it for every component would bloat the
        # file for no information.
        assert HistoryEntry.from_report(self.base()).tiers == {"kb.doc": "on_demand"}

    def test_tiers_survive_a_serialization_round_trip(self):
        entry = HistoryEntry.from_report(self.base())
        restored = History.loads(History(entries=[entry]).dumps()).entries[0]
        assert restored.to_report().on_demand == 500

    def test_the_current_report_wins_when_a_component_changed_tier(self):
        # Promoting a document from on-demand to resident is a real change the current
        # report is authoritative about.
        entry = HistoryEntry.from_report(self.base())
        head = Report(
            model="claude-opus-5",
            counter=CounterInfo(name="tiktoken", exact=False),
            commit="headcommit",
            components=[
                CountedComponent(id="kb.doc", tokens=500, tier=Tier.RESIDENT,
                                 kind=Kind.OTHER)
            ],
        )
        assert entry.to_report(metadata_from=head).resident == 600

    def test_the_counter_is_reported_honestly_rather_than_assumed_exact(self):
        rebuilt = HistoryEntry.from_report(self.base()).to_report()
        assert rebuilt.counter.name == "tiktoken"
        assert rebuilt.counter.exact is False


class TestBuildHistoryFiles:
    def test_merges_a_report_into_existing_history(self):
        existing = History(entries=[entry("old", 5)]).dumps()
        history, text = build_history_files(
            existing=existing,
            report=report({"a": 42}, commit="new"),
            retention_days=90,
            max_entries=500,
            now=NOW,
        )
        assert [e.commit for e in history.sorted_entries()] == ["old", "new"]
        assert "42" in text

    def test_works_from_no_existing_history(self):
        history, _ = build_history_files(
            existing=None,
            report=report({"a": 1}, commit="first"),
            retention_days=90,
            max_entries=500,
            now=NOW,
        )
        assert len(history.entries) == 1

    def test_an_approximate_report_is_recorded_like_any_other(self):
        history, _ = build_history_files(
            existing=None,
            report=report({"a": 1}, exact=False),
            retention_days=90,
            max_entries=500,
            now=NOW,
        )
        assert history.entries[0].counter == "tiktoken"
