"""The collection contract.

A repository under measurement exposes a callable that yields the pieces of context
its agents assemble. We accept plain dicts so the contract adds no import dependency
to the application being measured.

The central distinction is ``Tier``: text that is resident in every request costs
orders of magnitude more over time than text fetched only when a tool call asks for
it. Conflating the two produces a number that flags harmless growth and hides real
growth, so the tier is required on every component.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

SCHEMA_VERSION = 1


class Tier(enum.Enum):
    """Whether a component is paid for on every request or only on retrieval."""

    RESIDENT = "resident"
    ON_DEMAND = "on_demand"

    @classmethod
    def parse(cls, value: object) -> "Tier":
        if isinstance(value, cls):
            return value
        for member in cls:
            if member.value == value:
                return member
        allowed = ", ".join(repr(m.value) for m in cls)
        raise ContractError(f"unknown tier {value!r} (expected one of {allowed})")


class Kind(enum.Enum):
    """What sort of context a component is. Advisory: used for grouping and display."""

    SYSTEM_PROMPT = "system_prompt"
    TOOL_SCHEMA = "tool_schema"
    KNOWLEDGE_DOC = "knowledge_doc"
    KNOWLEDGE_INDEX = "knowledge_index"
    FEW_SHOT = "few_shot"
    OTHER = "other"

    @classmethod
    def parse(cls, value: object) -> "Kind":
        if isinstance(value, cls):
            return value
        if value is None:
            return cls.OTHER
        for member in cls:
            if member.value == value:
                return member
        raise ContractError(f"unknown kind {value!r}")


class ContractError(ValueError):
    """Raised when a collected component does not satisfy the contract.

    Messages name the offending component so a CI failure is actionable without
    reading our source.
    """


@dataclass(frozen=True)
class Component:
    """One measurable piece of context.

    Exactly one of ``text`` or ``tools`` carries the payload. Tool schemas are kept
    separate from text because a tool definition costs more than its serialized JSON,
    and only the counter knows the difference.
    """

    id: str
    tier: Tier
    kind: Kind = Kind.OTHER
    group: str | None = None
    text: str | None = None
    tools: tuple[Mapping[str, Any], ...] = ()
    source: str | None = None
    cache_prefix: bool = False

    @property
    def is_tools(self) -> bool:
        return bool(self.tools)

    @classmethod
    def parse(cls, raw: object, *, index: int) -> "Component":
        where = f"component at index {index}"
        if isinstance(raw, Component):
            return raw
        if not isinstance(raw, Mapping):
            raise ContractError(
                f"{where}: expected a mapping or Component, got {type(raw).__name__}"
            )

        cid = raw.get("id")
        if not isinstance(cid, str) or not cid.strip():
            raise ContractError(f"{where}: 'id' must be a non-empty string, got {cid!r}")
        where = f"component {cid!r}"

        unknown = set(raw) - _ALLOWED_KEYS
        if unknown:
            raise ContractError(
                f"{where}: unexpected key(s) {sorted(unknown)}; "
                f"allowed keys are {sorted(_ALLOWED_KEYS)}"
            )

        if "tier" not in raw:
            raise ContractError(
                f"{where}: 'tier' is required — is this text sent on every request "
                f"('resident') or only when retrieved ('on_demand')?"
            )
        try:
            tier = Tier.parse(raw["tier"])
        except ContractError as exc:
            raise ContractError(f"{where}: {exc}") from None
        try:
            kind = Kind.parse(raw.get("kind"))
        except ContractError as exc:
            raise ContractError(f"{where}: {exc}") from None

        text, tools = raw.get("text"), raw.get("tools")
        has_text = text is not None
        has_tools = tools is not None
        if has_text and has_tools:
            raise ContractError(
                f"{where}: set either 'text' or 'tools', not both — "
                f"tool schemas are counted differently from prose"
            )
        if not has_text and not has_tools:
            raise ContractError(f"{where}: must set either 'text' or 'tools'")

        if has_text and not isinstance(text, str):
            raise ContractError(f"{where}: 'text' must be a string, got {type(text).__name__}")

        parsed_tools: tuple[Mapping[str, Any], ...] = ()
        if has_tools:
            if isinstance(tools, Mapping) or not isinstance(tools, Iterable):
                raise ContractError(f"{where}: 'tools' must be a list of tool schemas")
            parsed_tools = tuple(tools)
            if not parsed_tools:
                raise ContractError(f"{where}: 'tools' is empty")
            for tool in parsed_tools:
                if not isinstance(tool, Mapping):
                    raise ContractError(
                        f"{where}: each tool must be a mapping, got {type(tool).__name__}"
                    )
                if "name" not in tool:
                    raise ContractError(f"{where}: every tool schema needs a 'name'")

        for optional in ("group", "source"):
            value = raw.get(optional)
            if value is not None and not isinstance(value, str):
                raise ContractError(f"{where}: '{optional}' must be a string if given")

        return cls(
            id=cid,
            tier=tier,
            kind=kind,
            group=raw.get("group"),
            text=text if has_text else None,
            tools=parsed_tools,
            source=raw.get("source"),
            cache_prefix=bool(raw.get("cache_prefix", False)),
        )


_ALLOWED_KEYS = {
    "id",
    "tier",
    "kind",
    "group",
    "text",
    "tools",
    "source",
    "cache_prefix",
}


@dataclass(frozen=True)
class CountedComponent:
    """A component with its measured token cost."""

    id: str
    tokens: int
    tier: Tier
    kind: Kind
    group: str | None = None
    source: str | None = None
    cache_prefix: bool = False

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "tokens": self.tokens,
            "tier": self.tier.value,
            "kind": self.kind.value,
        }
        if self.group is not None:
            out["group"] = self.group
        if self.source is not None:
            out["source"] = self.source
        if self.cache_prefix:
            out["cache_prefix"] = True
        return out

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> "CountedComponent":
        return cls(
            id=raw["id"],
            tokens=int(raw["tokens"]),
            tier=Tier.parse(raw["tier"]),
            kind=Kind.parse(raw.get("kind")),
            group=raw.get("group"),
            source=raw.get("source"),
            cache_prefix=bool(raw.get("cache_prefix", False)),
        )


@dataclass(frozen=True)
class CounterInfo:
    """Which counter produced a report, and whether its numbers are exact.

    ``exact`` gates entry into history: approximate runs are reported but never
    recorded, so a trend series never mixes measured and estimated points.
    """

    name: str
    exact: bool
    detail: str | None = None

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name, "exact": self.exact}
        if self.detail:
            out["detail"] = self.detail
        return out

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> "CounterInfo":
        return cls(
            name=raw["name"], exact=bool(raw["exact"]), detail=raw.get("detail")
        )


@dataclass
class Report:
    """The single artifact every downstream stage reads."""

    model: str
    counter: CounterInfo
    components: list[CountedComponent] = field(default_factory=list)
    commit: str | None = None
    ref: str | None = None
    generated_at: str | None = None
    tool_set_overhead: int = 0
    schema_version: int = SCHEMA_VERSION

    def tier_total(self, tier: Tier) -> int:
        return sum(c.tokens for c in self.components if c.tier is tier)

    @property
    def resident(self) -> int:
        """Tokens paid on every request, including tool-set overhead.

        Tool-set overhead is resident by definition: it is the cost of having tools
        enabled at all, which every request carries.
        """
        return self.tier_total(Tier.RESIDENT) + self.tool_set_overhead

    @property
    def on_demand(self) -> int:
        return self.tier_total(Tier.ON_DEMAND)

    @property
    def cache_prefix_tokens(self) -> int:
        """Resident tokens inside the cacheable prefix.

        Growth here is materially cheaper than growth after the last cache
        breakpoint, so the two are tracked separately.
        """
        return sum(
            c.tokens
            for c in self.components
            if c.tier is Tier.RESIDENT and c.cache_prefix
        )

    def by_id(self) -> dict[str, CountedComponent]:
        return {c.id: c for c in self.components}

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "commit": self.commit,
            "ref": self.ref,
            "model": self.model,
            "counter": self.counter.to_json(),
            "totals": {
                "resident": self.resident,
                "on_demand": self.on_demand,
                "cache_prefix": self.cache_prefix_tokens,
                "tool_set_overhead": self.tool_set_overhead,
            },
            "components": [c.to_json() for c in self.components],
        }

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> "Report":
        version = int(raw.get("schema_version", 0))
        if version != SCHEMA_VERSION:
            raise ContractError(
                f"report schema_version {version} is not supported "
                f"(this build reads version {SCHEMA_VERSION})"
            )
        return cls(
            model=raw["model"],
            counter=CounterInfo.from_json(raw["counter"]),
            components=[CountedComponent.from_json(c) for c in raw["components"]],
            commit=raw.get("commit"),
            ref=raw.get("ref"),
            generated_at=raw.get("generated_at"),
            tool_set_overhead=int(raw.get("totals", {}).get("tool_set_overhead", 0)),
            schema_version=version,
        )


def validate(components: Iterable[object]) -> list[Component]:
    """Parse and validate a collector's output.

    Duplicate ids are rejected: history is keyed on id, so a duplicate would make one
    component silently shadow another and corrupt the series.
    """
    parsed: list[Component] = []
    seen: dict[str, int] = {}
    for index, raw in enumerate(components):
        component = Component.parse(raw, index=index)
        if component.id in seen:
            raise ContractError(
                f"duplicate component id {component.id!r} "
                f"(also at index {seen[component.id]}); ids are the tracking identity "
                f"and must be unique"
            )
        seen[component.id] = index
        parsed.append(component)
    if not parsed:
        raise ContractError("collector yielded no components")
    return parsed
