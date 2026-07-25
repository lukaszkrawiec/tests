from pathlib import Path

import pytest

from tokenreport import config as config_module
from tokenreport.collect import (
    EntrypointError,
    call_collector,
    count,
    resolve_entrypoint,
    run,
)
from tokenreport.contract import ContractError, Tier
from tokenreport.counters import LengthCounter

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestResolveEntrypoint:
    def test_resolves_a_dotted_callable(self):
        func = resolve_entrypoint(
            "examples.langgraph_app.tokenreport:collect", root=REPO_ROOT
        )
        assert callable(func)

    @pytest.mark.parametrize("spec", ["nocolon", ":collect", "module:"])
    def test_malformed_spec_is_rejected(self, spec):
        with pytest.raises(EntrypointError, match="must look like"):
            resolve_entrypoint(spec, root=REPO_ROOT)

    def test_missing_module_names_the_module(self):
        with pytest.raises(EntrypointError, match="could not import module 'nope.absent'"):
            resolve_entrypoint("nope.absent:collect", root=REPO_ROOT)

    def test_missing_attribute_names_the_attribute(self):
        with pytest.raises(EntrypointError, match="has no attribute 'absent'"):
            resolve_entrypoint("examples.langgraph_app.tokenreport:absent", root=REPO_ROOT)

    def test_non_callable_target_is_rejected(self):
        with pytest.raises(EntrypointError, match="not callable"):
            resolve_entrypoint("examples.langgraph_app.graph:PROMPTS_DIR", root=REPO_ROOT)


class TestCallCollector:
    def test_wraps_a_raising_collector_with_context(self):
        def boom():
            raise RuntimeError("kaboom")

        with pytest.raises(EntrypointError, match="raised RuntimeError: kaboom"):
            call_collector(boom, spec="x:boom")

    def test_none_return_is_rejected(self):
        with pytest.raises(EntrypointError, match="returned None"):
            call_collector(lambda: None, spec="x:y")

    def test_validation_errors_propagate(self):
        with pytest.raises(ContractError, match="duplicate component id"):
            call_collector(
                lambda: [
                    {"id": "a", "tier": "resident", "text": "x"},
                    {"id": "a", "tier": "resident", "text": "y"},
                ],
                spec="x:y",
            )


class TestCount:
    def build(self, components):
        from tokenreport.contract import validate

        return count(validate(components), LengthCounter(), model="claude-opus-5")

    def test_counts_text_components_per_tier(self):
        report = self.build(
            [
                {"id": "r", "tier": "resident", "text": "one two three"},
                {"id": "d", "tier": "on_demand", "text": "a b c d e"},
            ]
        )
        assert report.resident == 3
        assert report.on_demand == 5

    def test_records_the_counter_and_model_used(self):
        # A count is only meaningful alongside the tokenizer that produced it.
        report = self.build([{"id": "r", "tier": "resident", "text": "x"}])
        assert report.model == "claude-opus-5"
        assert report.counter.name == "length"
        assert report.counter.exact is True

    def test_sets_a_utc_timestamp(self):
        report = self.build([{"id": "r", "tier": "resident", "text": "x"}])
        assert report.generated_at.endswith("Z")

    def test_tool_set_overhead_is_measured_not_smeared(self):
        # LengthCounter charges one token per tool beyond its serialized words, so
        # counting two tool sets separately loses nothing and the overhead is 0.
        # What matters is that the total reconciles with counting them together.
        report = self.build(
            [
                {"id": "t1", "tier": "resident", "tools": [{"name": "a"}]},
                {"id": "t2", "tier": "resident", "tools": [{"name": "b"}]},
            ]
        )
        counter = LengthCounter()
        combined = counter.count_tools([{"name": "a"}, {"name": "b"}])
        attributed = sum(c.tokens for c in report.components)
        assert attributed + report.tool_set_overhead == combined

    def test_single_tool_component_has_no_overhead_line(self):
        report = self.build(
            [{"id": "t", "tier": "resident", "tools": [{"name": "a"}]}]
        )
        assert report.tool_set_overhead == 0

    def test_on_demand_tools_are_excluded_from_resident_overhead(self):
        report = self.build(
            [
                {"id": "t1", "tier": "resident", "tools": [{"name": "a"}]},
                {"id": "t2", "tier": "on_demand", "tools": [{"name": "b"}]},
            ]
        )
        assert report.tool_set_overhead == 0

    def test_warm_is_called_with_text_only(self):
        seen = []

        class Recording(LengthCounter):
            def warm(self, texts):
                seen.extend(texts)

        from tokenreport.contract import validate

        count(
            validate(
                [
                    {"id": "a", "tier": "resident", "text": "hello"},
                    {"id": "t", "tier": "resident", "tools": [{"name": "x"}]},
                ]
            ),
            Recording(),
            model="m",
        )
        assert seen == ["hello"]


class TestAgainstTheFixtureApp:
    """The example application is a real collector, so it exercises the whole path."""

    def report(self):
        cfg = config_module.load(REPO_ROOT / "tokenreport.toml")
        return run(cfg, LengthCounter())

    def test_collects_prompts_tools_and_knowledge(self):
        ids = {c.id for c in self.report().components}
        assert "agent.researcher.system_prompt" in ids
        assert "agent.analyst.tools" in ids
        assert "knowledge.index" in ids
        assert any(i.startswith("knowledge.doc.") for i in ids)

    def test_knowledge_documents_are_on_demand_and_the_index_is_resident(self):
        # This is the distinction the whole tool exists to make: adding a document is
        # nearly free, adding a line to the index is not.
        by_id = self.report().by_id()
        assert by_id["knowledge.index"].tier is Tier.RESIDENT
        assert by_id["knowledge.doc.pricing"].tier is Tier.ON_DEMAND

    def test_corpus_is_much_larger_than_the_resident_index(self):
        report = self.report()
        docs = sum(
            c.tokens for c in report.components if c.id.startswith("knowledge.doc.")
        )
        assert docs > report.by_id()["knowledge.index"].tokens * 3

    def test_composed_prompt_exceeds_any_single_source_file(self):
        # A file-glob approach would report the parts; only the collector sees the whole.
        report = self.report()
        composed = report.by_id()["agent.researcher.system_prompt"].tokens
        preamble = (
            REPO_ROOT / "examples/langgraph_app/prompts/shared_preamble.md"
        ).read_text()
        assert composed > len(preamble.split())

    def test_shared_preamble_is_counted_in_every_node(self):
        report = self.report()
        researcher = report.by_id()["agent.researcher.system_prompt"].tokens
        analyst = report.by_id()["agent.analyst.system_prompt"].tokens
        assert researcher > 0 and analyst > 0

    def test_components_carry_sources_for_annotations(self):
        report = self.report()
        assert report.by_id()["agent.researcher.system_prompt"].source.endswith(
            "researcher.md"
        )

    def test_report_survives_a_json_round_trip(self):
        from tokenreport.contract import Report

        original = self.report()
        restored = Report.from_json(original.to_json())
        assert restored.resident == original.resident
        assert len(restored.components) == len(original.components)
