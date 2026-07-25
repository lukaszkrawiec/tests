import pytest

from tokenreport.contract import (
    Component,
    ContractError,
    CountedComponent,
    CounterInfo,
    Kind,
    Report,
    Tier,
    validate,
)


def comp(**overrides):
    base = {"id": "a", "tier": "resident", "text": "hello world"}
    base.update(overrides)
    return base


class TestComponentParsing:
    def test_minimal_component(self):
        parsed = Component.parse(comp(), index=0)
        assert parsed.id == "a"
        assert parsed.tier is Tier.RESIDENT
        assert parsed.kind is Kind.OTHER
        assert parsed.text == "hello world"
        assert parsed.tools == ()

    def test_tools_component(self):
        parsed = Component.parse(
            {"id": "t", "tier": "resident", "tools": [{"name": "search"}]}, index=0
        )
        assert parsed.is_tools
        assert parsed.tools[0]["name"] == "search"

    def test_accepts_already_parsed_component(self):
        original = Component.parse(comp(), index=0)
        assert Component.parse(original, index=0) is original

    @pytest.mark.parametrize("bad_id", ["", "   ", None, 5])
    def test_id_must_be_non_empty_string(self, bad_id):
        with pytest.raises(ContractError, match="'id' must be a non-empty string"):
            Component.parse(comp(id=bad_id), index=0)

    def test_missing_tier_names_the_component_and_explains(self):
        raw = comp()
        del raw["tier"]
        with pytest.raises(ContractError) as exc:
            Component.parse(raw, index=0)
        message = str(exc.value)
        assert "component 'a'" in message
        assert "resident" in message and "on_demand" in message

    def test_unknown_tier_lists_allowed_values(self):
        with pytest.raises(ContractError, match="unknown tier 'sometimes'"):
            Component.parse(comp(tier="sometimes"), index=0)

    def test_text_and_tools_are_mutually_exclusive(self):
        with pytest.raises(ContractError, match="not both"):
            Component.parse(comp(tools=[{"name": "x"}]), index=0)

    def test_requires_one_of_text_or_tools(self):
        raw = comp()
        del raw["text"]
        with pytest.raises(ContractError, match="must set either 'text' or 'tools'"):
            Component.parse(raw, index=0)

    def test_rejects_typo_keys_rather_than_ignoring_them(self):
        # A silently ignored 'teir' would mis-tier a component and corrupt history.
        with pytest.raises(ContractError, match=r"unexpected key\(s\) \['teir'\]"):
            Component.parse(comp(teir="resident"), index=0)

    def test_empty_tools_list_is_an_error(self):
        raw = {"id": "t", "tier": "resident", "tools": []}
        with pytest.raises(ContractError, match="'tools' is empty"):
            Component.parse(raw, index=0)

    def test_tools_must_be_a_list_not_a_mapping(self):
        raw = {"id": "t", "tier": "resident", "tools": {"name": "x"}}
        with pytest.raises(ContractError, match="must be a list of tool schemas"):
            Component.parse(raw, index=0)

    def test_tool_requires_a_name(self):
        raw = {"id": "t", "tier": "resident", "tools": [{"description": "no name"}]}
        with pytest.raises(ContractError, match="needs a 'name'"):
            Component.parse(raw, index=0)

    def test_index_is_reported_when_id_is_unusable(self):
        with pytest.raises(ContractError, match="component at index 7"):
            Component.parse({"tier": "resident", "text": "x"}, index=7)

    def test_non_mapping_is_rejected(self):
        with pytest.raises(ContractError, match="expected a mapping"):
            Component.parse("just a string", index=0)


class TestValidate:
    def test_returns_components_in_order(self):
        parsed = validate([comp(id="a"), comp(id="b")])
        assert [c.id for c in parsed] == ["a", "b"]

    def test_duplicate_ids_are_rejected_with_both_positions(self):
        with pytest.raises(ContractError) as exc:
            validate([comp(id="dup"), comp(id="other"), comp(id="dup")])
        assert "duplicate component id 'dup'" in str(exc.value)
        assert "index 0" in str(exc.value)

    def test_empty_collection_is_an_error(self):
        with pytest.raises(ContractError, match="yielded no components"):
            validate([])

    def test_accepts_a_generator(self):
        assert len(validate(comp(id=f"c{i}") for i in range(3))) == 3


def counted(cid, tokens, tier=Tier.RESIDENT, **kwargs):
    return CountedComponent(
        id=cid, tokens=tokens, tier=tier, kind=Kind.OTHER, **kwargs
    )


class TestReportTotals:
    def build(self, **kwargs):
        return Report(
            model="claude-opus-5",
            counter=CounterInfo(name="length", exact=True),
            components=[
                counted("r1", 100),
                counted("r2", 50, cache_prefix=True),
                counted("d1", 900, tier=Tier.ON_DEMAND),
            ],
            **kwargs,
        )

    def test_resident_and_on_demand_are_kept_separate(self):
        report = self.build()
        assert report.resident == 150
        assert report.on_demand == 900

    def test_tool_set_overhead_counts_as_resident(self):
        # It is the cost of having tools enabled at all, paid on every request.
        report = self.build(tool_set_overhead=25)
        assert report.resident == 175

    def test_cache_prefix_tracks_only_resident_prefix_components(self):
        assert self.build().cache_prefix_tokens == 50

    def test_json_round_trip_preserves_totals_and_components(self):
        original = self.build(tool_set_overhead=25, commit="abc", ref="refs/heads/main")
        restored = Report.from_json(original.to_json())
        assert restored.resident == original.resident
        assert restored.on_demand == original.on_demand
        assert restored.tool_set_overhead == 25
        assert restored.commit == "abc"
        assert restored.by_id().keys() == original.by_id().keys()

    def test_totals_are_not_summed_into_one_number(self):
        # No request pays both tiers, so a combined total would be meaningless.
        assert "total" not in self.build().to_json()["totals"]

    def test_future_schema_version_is_refused(self):
        payload = self.build().to_json()
        payload["schema_version"] = 99
        with pytest.raises(ContractError, match="schema_version 99 is not supported"):
            Report.from_json(payload)
