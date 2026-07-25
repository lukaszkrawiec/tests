import pytest

from tokenreport.compare import ChangeKind, Severity, compare, evaluate
from tokenreport.config import Budgets
from tokenreport.contract import CountedComponent, CounterInfo, Kind, Report, Tier


def report(components, *, model="claude-opus-5", counter="anthropic", exact=True,
           overhead=0):
    return Report(
        model=model,
        counter=CounterInfo(name=counter, exact=exact),
        tool_set_overhead=overhead,
        components=[
            CountedComponent(
                id=cid,
                tokens=tokens,
                tier=Tier.ON_DEMAND if str(cid).startswith("doc") else Tier.RESIDENT,
                kind=Kind.OTHER,
                source=f"{cid}.md",
            )
            for cid, tokens in components.items()
        ],
    )


class TestCompareWithoutBaseline:
    def test_everything_is_an_addition(self):
        result = compare(report({"a": 10, "b": 20}), None)
        assert {c.kind for c in result.components} == {ChangeKind.ADDED}
        assert result.has_baseline is False

    def test_resident_delta_is_the_full_total(self):
        result = compare(report({"a": 10, "b": 20}), None)
        assert result.resident_delta == 30
        assert result.resident_pct is None

    def test_components_are_ordered_largest_first(self):
        result = compare(report({"small": 5, "big": 500}), None)
        assert [c.id for c in result.components] == ["big", "small"]


class TestCompareWithBaseline:
    def test_classifies_added_removed_changed_and_unchanged(self):
        head = report({"same": 10, "grew": 30, "new": 5})
        base = report({"same": 10, "grew": 20, "gone": 7})
        result = compare(head, base)
        kinds = {c.id: c.kind for c in result.components}
        assert kinds["same"] is ChangeKind.UNCHANGED
        assert kinds["grew"] is ChangeKind.CHANGED
        assert kinds["new"] is ChangeKind.ADDED
        assert kinds["gone"] is ChangeKind.REMOVED

    def test_delta_and_percentage(self):
        result = compare(report({"a": 110}), report({"a": 100}))
        change = result.components[0]
        assert change.delta == 10
        assert change.pct == pytest.approx(10.0)

    def test_percentage_is_none_when_baseline_is_zero(self):
        result = compare(report({"a": 10}), report({}))
        assert result.components[0].pct is None

    def test_removed_component_has_negative_delta(self):
        result = compare(report({}), report({"a": 40}))
        assert result.components[0].delta == -40

    def test_resident_totals_track_tool_set_overhead(self):
        head = compare(report({"a": 100}, overhead=20), report({"a": 100}, overhead=5))
        assert head.resident_delta == 15

    def test_on_demand_is_tracked_separately_from_resident(self):
        head = report({"a": 100, "doc1": 900})
        base = report({"a": 100, "doc1": 400})
        result = compare(head, base)
        assert result.resident_delta == 0
        assert result.on_demand_before == 400
        assert result.head.on_demand == 900

    def test_changed_is_sorted_by_absolute_delta(self):
        head = report({"tiny": 11, "huge": 500, "shrank": 1})
        base = report({"tiny": 10, "huge": 100, "shrank": 200})
        assert [c.id for c in compare(head, base).changed()] == ["huge", "shrank", "tiny"]

    def test_unchanged_is_excluded_from_changed(self):
        result = compare(report({"a": 10}), report({"a": 10}))
        assert result.changed() == []
        assert [c.id for c in result.unchanged()] == ["a"]


class TestComparability:
    def test_a_different_model_makes_the_comparison_invalid(self):
        # Claude 4.7+ tokenizes ~30% higher; reporting that as growth would be wrong.
        head = report({"a": 130}, model="claude-fable-5")
        base = report({"a": 100}, model="claude-opus-5")
        result = compare(head, base)
        assert result.comparable is False
        assert "not comparable across tokenizers" in result.incomparable_reason
        assert result.has_baseline is False
        assert result.resident_pct is None

    def test_a_different_counter_makes_the_comparison_invalid(self):
        # Switching counter re-bases every number, exactly as a model change does.
        result = compare(
            report({"a": 10}), report({"a": 10}, counter="tiktoken", exact=False)
        )
        assert result.comparable is False
        assert "tiktoken" in result.incomparable_reason
        assert "re-bases" in result.incomparable_reason

    def test_the_same_approximate_counter_on_both_sides_still_compares(self):
        # tiktoken is the default and is not exact, but a delta between two tiktoken
        # measurements is perfectly meaningful — comparability is about identity, not
        # absolute accuracy. Gating on exactness would disable deltas entirely.
        result = compare(
            report({"a": 130}, counter="tiktoken", exact=False),
            report({"a": 100}, counter="tiktoken", exact=False),
        )
        assert result.comparable is True
        assert result.resident_pct == pytest.approx(30.0)

    def test_component_deltas_are_still_computed_for_display(self):
        # The numbers are shown with a warning rather than withheld entirely.
        result = compare(report({"a": 130}, model="m2"), report({"a": 100}))
        assert result.components[0].delta == 30


class TestBudgets:
    def test_no_budgets_configured_passes(self):
        result = evaluate(compare(report({"a": 10}), None), Budgets())
        assert result.ok
        assert result.severity is Severity.OK
        assert result.checked == 0

    def test_resident_ceiling_fails_when_exceeded(self):
        result = evaluate(
            compare(report({"a": 5000}), None), Budgets(resident_total=4000)
        )
        assert not result.ok
        assert result.violations[0].budget == "resident_total"
        assert "5,000" in result.violations[0].message

    def test_resident_ceiling_passes_at_exactly_the_limit(self):
        result = evaluate(
            compare(report({"a": 4000}), None), Budgets(resident_total=4000)
        )
        assert result.ok

    def test_on_demand_ceiling_is_independent_of_resident(self):
        result = evaluate(
            compare(report({"doc1": 5000}), None),
            Budgets(resident_total=100, on_demand_total=1000),
        )
        assert [v.budget for v in result.violations] == ["on_demand_total"]

    def test_growth_budget_fails_on_excessive_growth(self):
        result = evaluate(
            compare(report({"a": 130}), report({"a": 100})),
            Budgets(resident_growth_pct=10),
        )
        assert not result.ok
        assert result.violations[0].budget == "resident_growth_pct"
        assert "+30.0%" in result.violations[0].message

    def test_growth_budget_passes_within_the_limit(self):
        result = evaluate(
            compare(report({"a": 105}), report({"a": 100})),
            Budgets(resident_growth_pct=10),
        )
        assert result.ok

    def test_shrinking_never_violates_a_growth_budget(self):
        result = evaluate(
            compare(report({"a": 50}), report({"a": 100})),
            Budgets(resident_growth_pct=10),
        )
        assert result.ok

    def test_growth_budget_is_skipped_without_a_baseline(self):
        # A first run has nothing to grow from; failing here would block every new branch.
        result = evaluate(compare(report({"a": 100}), None), Budgets(resident_growth_pct=10))
        assert result.ok
        assert any("no comparable baseline" in s for s in result.skipped)

    def test_growth_budget_is_skipped_across_a_tokenizer_change(self):
        result = evaluate(
            compare(report({"a": 130}, model="claude-fable-5"), report({"a": 100})),
            Budgets(resident_growth_pct=10),
        )
        assert result.ok
        assert result.skipped

    def test_component_budget_fails_and_carries_its_source(self):
        result = evaluate(
            compare(report({"prompt": 900}), None), Budgets(component={"prompt": 500})
        )
        assert not result.ok
        violation = result.violations[0]
        assert violation.component_id == "prompt"
        assert violation.source == "prompt.md"  # enables an inline diff annotation

    def test_component_budget_for_a_missing_component_is_skipped_not_failed(self):
        # Stale config should prompt a cleanup, not block a merge.
        result = evaluate(
            compare(report({"a": 10}), None), Budgets(component={"deleted": 100})
        )
        assert result.ok
        assert any("no such component" in s for s in result.skipped)

    def test_all_violations_are_reported_not_just_the_first(self):
        result = evaluate(
            compare(report({"a": 5000, "b": 900}), report({"a": 100})),
            Budgets(
                resident_total=1000,
                resident_growth_pct=5,
                component={"a": 100, "b": 100},
            ),
        )
        assert {v.budget for v in result.violations} == {
            "resident_total",
            "resident_growth_pct",
            "component",
        }
        assert len(result.violations) == 4
