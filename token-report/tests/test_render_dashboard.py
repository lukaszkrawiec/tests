import datetime as dt
import re

import pytest

from tokenreport.contract import CountedComponent, CounterInfo, Kind, Report, Tier
from tokenreport.history import History, HistoryEntry
from tokenreport.render.dashboard import MAX_SERIES, render_dashboard

NOW = dt.datetime(2026, 7, 1, tzinfo=dt.timezone.utc)


def report(components, *, model="claude-opus-5", overhead=0):
    return Report(
        model=model,
        counter=CounterInfo(name="anthropic", exact=True),
        commit="deadbee",
        tool_set_overhead=overhead,
        components=[
            CountedComponent(
                id=cid,
                tokens=tokens,
                tier=Tier.ON_DEMAND if cid.startswith("doc") else Tier.RESIDENT,
                kind=Kind.OTHER,
                group=cid.split(".")[0],
                cache_prefix=not cid.startswith("doc"),
            )
            for cid, tokens in components.items()
        ],
    )


def entry(commit, days_ago, *, resident=100, on_demand=50, model="claude-opus-5",
          components=None):
    return HistoryEntry(
        commit=commit,
        timestamp=(NOW - dt.timedelta(days=days_ago)).isoformat().replace("+00:00", "Z"),
        model=model,
        counter="anthropic",
        resident=resident,
        on_demand=on_demand,
        cache_prefix=resident,
        components=components or {"agent.a": resident, "doc.b": on_demand},
    )


def render(history, rep=None, **kwargs):
    return render_dashboard(
        history, rep or report({"agent.a": 100, "doc.b": 50}),
        generated_at=NOW, **kwargs
    )


class TestDocument:
    def test_is_a_complete_standalone_document(self):
        html = render(History(entries=[entry("a", 2), entry("b", 1)]))
        assert html.startswith("<!doctype html>")
        assert "</html>" in html

    def test_has_no_external_requests(self):
        # It must render as a workflow artifact from disk and under a strict CSP.
        html = render(History(entries=[entry("a", 2), entry("b", 1)]))
        assert "http://" not in html.replace('lang="en"', "")
        assert "https://" not in html
        assert "<script" not in html
        assert "<link" not in html

    def test_declares_both_themes_including_the_toggle_scope(self):
        html = render(History(entries=[entry("a", 1)]))
        assert "prefers-color-scheme: dark" in html
        assert '[data-theme="dark"]' in html
        assert '[data-theme="light"]' in html

    def test_repo_name_appears_when_given(self):
        assert "owner/repo" in render(History(entries=[entry("a", 1)]), repo="owner/repo")

    def test_escapes_untrusted_component_ids(self):
        rep = report({"<script>alert(1)</script>": 10})
        html = render(History(entries=[entry("a", 1)]), rep)
        assert "<script>" not in html
        assert "&lt;script&gt;" in html


class TestStatTiles:
    def test_shows_both_tiers_as_separate_tiles(self):
        html = render(
            History(entries=[entry("a", 1)]), report({"agent.a": 2500, "doc.b": 9000})
        )
        assert "2,500" in html and "9,000" in html
        assert "every request" in html and "when retrieved" in html

    def test_delta_against_the_previous_recorded_commit(self):
        history = History(entries=[entry("a", 2, resident=100), entry("b", 1, resident=100)])
        html = render(history, report({"agent.a": 130}))
        assert "+30" in html

    def test_no_prior_point_is_stated_rather_than_shown_as_zero(self):
        # The dashboard is built just after the report is appended, so a history of one
        # entry is this report itself and there is nothing to compare against.
        html = render(History(entries=[entry("deadbee", 1)]))
        assert "no prior point" in html

    def test_baseline_skips_the_reports_own_entry(self):
        history = History(
            entries=[entry("older", 2, resident=100), entry("deadbee", 1, resident=999)]
        )
        html = render(history, report({"agent.a": 130}))
        # Compared against 'older' (100), not against its own recorded entry (999).
        assert "+30" in html

    def test_uses_the_latest_entry_when_the_report_is_not_recorded(self):
        # An approximate run is never appended, so the newest entry is its baseline.
        html = render(History(entries=[entry("other", 1, resident=100)]),
                      report({"agent.a": 120}))
        assert "+20" in html

    def test_hero_numbers_avoid_tabular_figures(self):
        # Equal-width digits make large standalone numbers look loose.
        html = render(History(entries=[entry("a", 1)]))
        tile_value = re.search(r"\.tile-value \{[^}]*\}", html).group(0)
        assert "tabular-nums" not in tile_value


class TestCharts:
    def test_renders_a_stacked_area_and_a_line_chart(self):
        html = render(History(entries=[entry("a", 3), entry("b", 2), entry("c", 1)]))
        assert html.count("<svg") == 2
        assert "<polygon" in html
        assert "<polyline" in html

    def test_charts_use_one_axis_each_never_dual(self):
        # Two measures of different meaning get two charts, never two y-scales.
        html = render(History(entries=[entry("a", 2), entry("b", 1)]))
        assert "Resident tokens over time" in html
        assert "On-demand tokens over time" in html

    def test_a_single_entry_says_a_trend_needs_more_points(self):
        html = render(History(entries=[entry("a", 1)]))
        assert "One data point so far" in html

    def test_empty_history_is_handled_without_a_chart(self):
        html = render(History())
        assert "No history recorded yet" in html
        assert "<polygon" not in html

    def test_legend_is_present_for_multiple_groups(self):
        rep = report({"agent.a": 100, "kb.b": 80, "doc.c": 10})
        html = render(History(entries=[entry("x", 2), entry("y", 1)]), rep)
        assert 'class="legend"' in html
        assert ">agent<" in html and ">kb<" in html

    def test_gridlines_are_solid_not_dashed(self):
        html = render(History(entries=[entry("a", 2), entry("b", 1)]))
        grid = re.search(r"\.grid \{[^}]*\}", html).group(0)
        assert "dash" not in grid

    def test_series_beyond_the_cap_fold_into_other(self):
        rep = report({f"g{i}.x": 10 for i in range(MAX_SERIES + 4)})
        html = render(History(entries=[entry("a", 2), entry("b", 1)]), rep)
        assert ">other<" in html
        # Never generate a ninth hue.
        assert f"s{MAX_SERIES}" not in re.search(
            r'<ul class="legend">.*?</ul>', html, re.DOTALL
        ).group(0)

    def test_every_charted_value_is_also_in_the_table(self):
        # Values must not be reachable only by hover.
        rep = report({"agent.a": 1234, "doc.b": 567})
        html = render(History(entries=[entry("x", 2), entry("y", 1)]), rep)
        table = re.search(r'<table class="data">.*?</table>', html, re.DOTALL).group(0)
        assert "1,234" in table and "567" in table

    def test_chart_scrolls_horizontally_rather_than_the_page(self):
        html = render(History(entries=[entry("a", 2), entry("b", 1)]))
        assert "overflow-x: auto" in html


class TestTokenizerChangeMarkers:
    def history(self):
        return History(
            entries=[
                entry("a", 4, model="claude-opus-5"),
                entry("b", 3, model="claude-opus-5"),
                entry("c", 2, model="claude-fable-5"),
                entry("d", 1, model="claude-fable-5"),
            ]
        )

    def test_marks_the_change(self):
        html = render(self.history())
        assert "tokenizer change" in html
        assert 'class="marker"' in html

    def test_the_series_is_broken_rather_than_connected_across(self):
        # Connecting across a ~30% tokenizer shift would draw an upgrade as a regression.
        broken = render(self.history())
        continuous = render(
            History(entries=[entry(c, 4 - i) for i, c in enumerate("abcd")])
        )
        assert broken.count("<polygon") > continuous.count("<polygon")

    def test_no_marker_when_the_model_is_stable(self):
        html = render(History(entries=[entry("a", 2), entry("b", 1)]))
        assert "tokenizer change" not in html


class TestComponentTable:
    def test_lists_components_largest_first(self):
        rep = report({"agent.small": 10, "agent.big": 900})
        html = render(History(entries=[entry("a", 1)]), rep)
        assert html.index("agent.big") < html.index("agent.small")

    def test_labels_the_tier_of_each_component(self):
        rep = report({"agent.a": 10, "doc.b": 20})
        html = render(History(entries=[entry("a", 1)]), rep)
        table = re.search(r'<table class="data">.*?</table>', html, re.DOTALL).group(0)
        assert "resident" in table and "on-demand" in table

    def test_table_numbers_use_tabular_figures(self):
        html = render(History(entries=[entry("a", 1)]))
        rule = re.search(r"table\.data \.num \{[^}]*\}", html).group(0)
        assert "tabular-nums" in rule


class TestFooter:
    def test_records_the_model_and_counter(self):
        html = render(History(entries=[entry("a", 1)]))
        assert "claude-opus-5" in html
        assert "anthropic" in html
