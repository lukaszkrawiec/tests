import pytest

from tokenreport.compare import compare, evaluate
from tokenreport.config import Budgets
from tokenreport.contract import CountedComponent, CounterInfo, Kind, Report, Tier
from tokenreport.render.markdown import (
    COMMENT_LIMIT,
    COMMENT_MARKER,
    render_comment,
    render_summary,
)


def report(components, *, model="claude-opus-5", exact=True, overhead=0, commit="a" * 40):
    return Report(
        model=model,
        counter=CounterInfo(name="anthropic" if exact else "tiktoken", exact=exact),
        commit=commit,
        tool_set_overhead=overhead,
        components=[
            CountedComponent(
                id=cid,
                tokens=tokens,
                tier=Tier.ON_DEMAND if cid.startswith("doc") else Tier.RESIDENT,
                kind=Kind.OTHER,
                group=cid.split(".")[0],
                source=f"{cid}.md",
                cache_prefix=not cid.startswith("doc"),
            )
            for cid, tokens in components.items()
        ],
    )


def rendered(head, base=None, budgets=Budgets(), **kwargs):
    comparison = compare(head, base)
    return render_comment(comparison, evaluate(comparison, budgets), **kwargs)


class TestCommentStructure:
    def test_starts_with_the_sticky_marker(self):
        # The marker is how the next run finds this comment instead of posting a new one.
        assert rendered(report({"a": 10})).startswith(COMMENT_MARKER)

    def test_headline_leads_with_resident_tokens(self):
        body = rendered(report({"agent.a": 2500, "doc.b": 9000}))
        headline = [line for line in body.splitlines() if "Resident context" in line][0]
        assert "2,500" in headline
        assert "9,000" not in headline

    def test_shows_growth_with_absolute_and_percentage(self):
        body = rendered(report({"a": 130}), report({"a": 100}))
        assert "+30" in body
        assert "+30.0%" in body

    def test_shows_shrinkage_distinctly_from_growth(self):
        body = rendered(report({"a": 70}), report({"a": 100}))
        assert "-30" in body
        assert "🔻" in body

    def test_unchanged_total_says_so(self):
        assert "unchanged" in rendered(report({"a": 100}), report({"a": 100}))

    def test_separates_the_two_tiers_in_the_totals_table(self):
        body = rendered(report({"a": 100, "doc.b": 900}))
        assert "paid every request" in body
        assert "paid when retrieved" in body

    def test_never_prints_a_combined_total_of_both_tiers(self):
        # 100 + 900 = 1000 would be a meaningless number: no request pays both.
        body = rendered(report({"a": 100, "doc.b": 900}))
        assert "1,000" not in body

    def test_cache_prefix_row_appears_when_relevant(self):
        assert "cache prefix" in rendered(report({"a": 100}))

    def test_tool_overhead_row_appears_only_when_nonzero(self):
        assert "tool-set overhead" not in rendered(report({"a": 100}))
        assert "tool-set overhead" in rendered(report({"a": 100}, overhead=12))

    def test_added_and_removed_components_are_labelled(self):
        body = rendered(report({"kept": 10, "fresh": 5}), report({"kept": 10, "old": 7}))
        assert "(new)" in body
        assert "(removed)" in body

    def test_unchanged_components_are_collapsed(self):
        body = rendered(report({"a": 10, "b": 20}), report({"a": 10, "b": 20}))
        assert "<details><summary>Unchanged (2)</summary>" in body

    def test_reports_when_nothing_changed_size(self):
        assert "No component changed size." in rendered(
            report({"a": 10}), report({"a": 10})
        )

    def test_footer_records_model_counter_and_commit(self):
        body = rendered(report({"a": 10}, commit="abc1234def"))
        assert "claude-opus-5" in body
        assert "anthropic" in body
        assert "abc1234" in body

    def test_summary_link_is_included_when_given(self):
        body = rendered(report({"a": 10}), summary_url="https://example.test/run/1")
        assert "https://example.test/run/1" in body


class TestCommentWarnings:
    def test_first_run_says_it_establishes_a_baseline(self):
        assert "establishes one" in rendered(report({"a": 10}))

    def test_an_inexact_counter_is_named_and_its_limits_stated(self):
        # It no longer says "not recorded" — inexact counters are recorded now — but the
        # reader still needs to know the absolute figures are not Claude's.
        body = rendered(report({"a": 10}, exact=False))
        assert "not\nClaude's tokenizer" in body or "not Claude's tokenizer" in body
        assert "delta" in body
        assert "not recorded in history" not in body

    def test_tokenizer_change_explains_the_missing_delta(self):
        body = rendered(report({"a": 130}, model="claude-fable-5"), report({"a": 100}))
        assert "No delta shown" in body
        assert "not comparable across tokenizers" in body

    def test_tokenizer_change_suppresses_the_percentage(self):
        body = rendered(report({"a": 130}, model="claude-fable-5"), report({"a": 100}))
        assert "+30.0%" not in body


class TestBudgetsInComment:
    def test_violations_are_listed_with_their_budget_name(self):
        body = rendered(report({"a": 5000}), budgets=Budgets(resident_total=1000))
        assert "budget violation" in body
        assert "resident_total" in body

    def test_plural_agreement_for_a_single_violation(self):
        body = rendered(report({"a": 5000}), budgets=Budgets(resident_total=1000))
        assert "1 budget violation" in body
        assert "1 budget violations" not in body

    def test_passing_budgets_are_acknowledged(self):
        body = rendered(report({"a": 100}), budgets=Budgets(resident_total=1000))
        assert "Within budget" in body

    def test_no_budget_section_when_none_configured(self):
        body = rendered(report({"a": 100}))
        assert "Within budget" not in body
        assert "budget violation" not in body

    def test_skipped_checks_are_disclosed_but_collapsed(self):
        body = rendered(report({"a": 100}), budgets=Budgets(component={"gone": 10}))
        assert "Skipped checks" in body
        assert "no such component" in body


class TestCommentTruncation:
    def big_report(self, count, *, tokens=100):
        return report({f"component.number.{i:04d}": tokens + i for i in range(count)})

    def test_a_large_report_still_fits_the_api_limit(self):
        # GitHub rejects comment bodies over 65,536 characters outright.
        head = self.big_report(4000)
        base = self.big_report(4000, tokens=50)
        body = rendered(head, base)
        assert len(body) <= COMMENT_LIMIT

    def test_truncation_says_how_many_rows_were_hidden(self):
        body = rendered(self.big_report(4000), self.big_report(4000, tokens=50))
        assert "more changed component" in body
        assert "job summary" in body

    def test_largest_changes_survive_truncation(self):
        # If rows must be dropped, the biggest movers are the ones worth keeping.
        head = report({f"c{i:04d}": 10 for i in range(3000)} | {"whale": 90000})
        base = report({f"c{i:04d}": 5 for i in range(3000)} | {"whale": 10})
        body = rendered(head, base)
        assert "`whale`" in body

    def test_headline_and_budgets_survive_truncation(self):
        head = self.big_report(4000)
        body = rendered(head, self.big_report(4000, tokens=50),
                        budgets=Budgets(resident_total=10))
        assert "Resident context" in body
        assert "resident_total" in body

    def test_unchanged_block_is_dropped_before_changed_rows(self):
        head = report({f"same{i:04d}": 10 for i in range(3000)} | {"moved": 500})
        base = report({f"same{i:04d}": 10 for i in range(3000)} | {"moved": 100})
        body = rendered(head, base)
        assert "`moved`" in body
        assert "<details><summary>Unchanged" not in body
        assert len(body) <= COMMENT_LIMIT

    @pytest.mark.parametrize("count", [1, 2, 50])
    def test_small_reports_are_not_truncated(self, count):
        body = rendered(self.big_report(count), self.big_report(count, tokens=50))
        assert "more changed component" not in body


class TestSummary:
    def test_groups_components_and_reports_per_group_totals(self):
        head = report({"agent.a": 100, "agent.b": 50, "doc.c": 900})
        summary = render_summary(compare(head, None), evaluate(compare(head, None), Budgets()))
        assert "### agent — 150 resident" in summary
        assert "900 on-demand" in summary

    def test_includes_every_component_without_truncation(self):
        head = report({f"c{i:04d}": 10 for i in range(500)})
        summary = render_summary(compare(head, None), evaluate(compare(head, None), Budgets()))
        assert summary.count("`c0") >= 500
        assert len(summary) > COMMENT_LIMIT  # deliberately unbounded

    def test_carries_the_same_headline_as_the_comment(self):
        head, base = report({"a": 130}), report({"a": 100})
        comparison = compare(head, base)
        summary = render_summary(comparison, evaluate(comparison, Budgets()))
        assert "Resident context" in summary
        assert "+30.0%" in summary

    def test_has_no_sticky_marker(self):
        # The summary is not a comment; a marker there would be meaningless.
        head = report({"a": 10})
        summary = render_summary(compare(head, None), evaluate(compare(head, None), Budgets()))
        assert COMMENT_MARKER not in summary
