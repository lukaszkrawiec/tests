"""Per-commit history: the series behind the trend dashboard.

History lives as JSON on an orphan branch, so it is auditable in git and needs no
external service. Two properties matter more than they look:

* Every entry stamps the model and counter that produced it. Any counter may be
  recorded — what a series cannot survive is a tokenizer change going unnoticed, so
  comparisons refuse to cross one and the dashboard breaks its line there.
* Entries are keyed by commit. Re-running a workflow on the same commit replaces its
  entry rather than adding a duplicate point.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass, field
from typing import Any, Mapping

from .contract import CountedComponent, CounterInfo, Kind, Report, Tier
from .counters import counter_is_exact

SCHEMA_VERSION = 1


class HistoryError(ValueError):
    """Raised when a history file cannot be read or an entry cannot be recorded."""


@dataclass(frozen=True)
class HistoryEntry:
    commit: str
    timestamp: str
    model: str
    counter: str
    resident: int
    on_demand: int
    cache_prefix: int = 0
    tool_set_overhead: int = 0
    components: Mapping[str, int] = field(default_factory=dict)
    #  Tier per component, recorded only where it differs from the resident default so
    #  the file stays compact. Without this, a component that a later commit removes
    #  entirely cannot be tiered when rebuilding the baseline, and an on-demand document
    #  would be counted as resident — reporting a large resident drop that never happened.
    tiers: Mapping[str, str] = field(default_factory=dict)
    ref: str | None = None

    @property
    def moment(self) -> _dt.datetime:
        text = self.timestamp.replace("Z", "+00:00")
        parsed = _dt.datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=_dt.timezone.utc)
        return parsed

    @classmethod
    def from_report(cls, report: Report) -> "HistoryEntry":
        # Any counter may be recorded. What a series cannot survive is a counter
        # *change* going unnoticed, so every entry stamps the tokenizer that produced
        # it and comparisons refuse to cross a change. Gating on exactness instead
        # would mean a repository using tiktoken — the default — never records a single
        # point and its dashboard stays permanently empty.
        if not report.commit:
            raise HistoryError("cannot record a report with no commit sha")
        return cls(
            commit=report.commit,
            timestamp=report.generated_at
            or _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z"),
            model=report.model,
            counter=report.counter.name,
            resident=report.resident,
            on_demand=report.on_demand,
            cache_prefix=report.cache_prefix_tokens,
            tool_set_overhead=report.tool_set_overhead,
            components={c.id: c.tokens for c in report.components},
            tiers={
                c.id: c.tier.value
                for c in report.components
                if c.tier is not Tier.RESIDENT
            },
            ref=report.ref,
        )

    def to_report(self, *, metadata_from: "Report | None" = None) -> "Report":
        """Rebuild a Report from this entry, for use as a comparison baseline.

        History stores per-component counts, not their metadata, so tier and grouping are
        recovered from a current report when one is available. Recovering the tier
        matters: defaulting an on-demand knowledge document to resident would attribute
        its size to the total paid on every request and report growth in the wrong tier.
        """
        known = {
            c.id: (c.tier, c.kind, c.group, c.source, c.cache_prefix)
            for c in (metadata_from.components if metadata_from else [])
        }
        components = []
        for cid, tokens in self.components.items():
            if cid in known:
                tier, kind, group, source, prefix = known[cid]
            else:
                # Not in the current report — typically because this commit removes it.
                # The recorded tier is the only thing that can classify it correctly.
                tier = Tier.parse(self.tiers.get(cid, Tier.RESIDENT.value))
                kind, group, source, prefix = Kind.OTHER, None, None, False
            components.append(
                CountedComponent(
                    id=cid,
                    tokens=tokens,
                    tier=tier,
                    kind=kind,
                    group=group,
                    source=source,
                    cache_prefix=prefix,
                )
            )
        return Report(
            model=self.model,
            counter=CounterInfo(name=self.counter, exact=counter_is_exact(self.counter)),
            commit=self.commit,
            ref=self.ref,
            generated_at=self.timestamp,
            tool_set_overhead=self.tool_set_overhead,
            components=components,
        )

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "commit": self.commit,
            "timestamp": self.timestamp,
            "model": self.model,
            "counter": self.counter,
            "totals": {
                "resident": self.resident,
                "on_demand": self.on_demand,
                "cache_prefix": self.cache_prefix,
                "tool_set_overhead": self.tool_set_overhead,
            },
            "components": dict(self.components),
        }
        if self.tiers:
            out["tiers"] = dict(self.tiers)
        if self.ref:
            out["ref"] = self.ref
        return out

    @classmethod
    def from_json(cls, raw: Mapping[str, Any]) -> "HistoryEntry":
        totals = raw.get("totals", {})
        return cls(
            commit=raw["commit"],
            timestamp=raw["timestamp"],
            model=raw["model"],
            counter=raw.get("counter", "unknown"),
            resident=int(totals.get("resident", 0)),
            on_demand=int(totals.get("on_demand", 0)),
            cache_prefix=int(totals.get("cache_prefix", 0)),
            tool_set_overhead=int(totals.get("tool_set_overhead", 0)),
            components=dict(raw.get("components", {})),
            tiers=dict(raw.get("tiers", {})),
            ref=raw.get("ref"),
        )


@dataclass
class History:
    entries: list[HistoryEntry] = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def loads(cls, text: str | None) -> "History":
        if not text or not text.strip():
            return cls()
        try:
            raw = json.loads(text)
        except ValueError as exc:
            raise HistoryError(f"history file is not valid JSON: {exc}") from exc
        version = int(raw.get("schema_version", 0))
        if version != SCHEMA_VERSION:
            raise HistoryError(
                f"history schema_version {version} is not supported "
                f"(this build reads version {SCHEMA_VERSION})"
            )
        return cls(
            entries=[HistoryEntry.from_json(e) for e in raw.get("entries", [])],
            schema_version=version,
        )

    def dumps(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "entries": [e.to_json() for e in self.entries],
        }
        # Trailing newline and stable key order keep the branch diff readable.
        return json.dumps(payload, indent=2, sort_keys=False) + "\n"

    def sorted_entries(self) -> list[HistoryEntry]:
        return sorted(self.entries, key=lambda e: (e.moment, e.commit))

    def latest(self) -> HistoryEntry | None:
        entries = self.sorted_entries()
        return entries[-1] if entries else None

    def find(self, commit: str) -> HistoryEntry | None:
        for entry in self.entries:
            if entry.commit == commit:
                return entry
        return None

    def append(self, entry: HistoryEntry) -> "History":
        """Add or replace an entry, returning a new History."""
        kept = [e for e in self.entries if e.commit != entry.commit]
        return History(entries=[*kept, entry], schema_version=self.schema_version)

    def tokenizer_changes(self) -> list[str]:
        """Commits at which the tokenizer changed — a different model or counter.

        The dashboard breaks its line at these points. Two distinct causes, same
        consequence: Claude 4.7 and later tokenize roughly 30% higher for identical
        text, and switching counter (tiktoken to the Anthropic API, say) re-bases every
        number. Connecting across either would draw a measurement change as growth.
        """
        changes: list[str] = []
        previous: tuple[str, str] | None = None
        for entry in self.sorted_entries():
            current = (entry.model, entry.counter)
            if previous is not None and current != previous:
                changes.append(entry.commit)
            previous = current
        return changes

    def prune(
        self,
        *,
        retention_days: int = 90,
        max_entries: int = 500,
        now: _dt.datetime | None = None,
    ) -> "History":
        """Keep recent history at full resolution and downsample the rest.

        Without this the file grows without bound: slow to fetch on every run, and a
        chart too dense to read.
        """
        if max_entries <= 0 or retention_days <= 0:
            # Config validates these, but prune is reachable directly and the old
            # behaviour for max_entries=0 was to return an empty history — silently
            # deleting the whole series. Failing loudly is the safer default.
            raise HistoryError(
                f"prune needs positive bounds; got retention_days={retention_days!r}, "
                f"max_entries={max_entries!r}"
            )
        entries = self.sorted_entries()
        if not entries:
            return History(entries=[], schema_version=self.schema_version)

        moment = now or _dt.datetime.now(_dt.timezone.utc)
        cutoff = moment - _dt.timedelta(days=retention_days)

        recent = [e for e in entries if e.moment >= cutoff]
        older = [e for e in entries if e.moment < cutoff]

        # One point per ISO week for anything past the retention window; the last of
        # each week is the one that survived into the following week.
        weekly: dict[tuple[int, int], HistoryEntry] = {}
        for entry in older:
            iso = entry.moment.isocalendar()
            weekly[(iso[0], iso[1])] = entry
        downsampled = sorted(weekly.values(), key=lambda e: (e.moment, e.commit))

        kept = [*downsampled, *recent]
        if len(kept) > max_entries:
            # Drop from the oldest end; recent detail is what a reviewer looks at.
            kept = kept[len(kept) - max_entries :]
        return History(entries=kept, schema_version=self.schema_version)


def build_history_files(
    *,
    existing: str | None,
    report: Report,
    retention_days: int,
    max_entries: int,
    now: _dt.datetime | None = None,
) -> tuple[History, str]:
    """Merge a report into existing history text and return the new history and JSON."""
    history = History.loads(existing)
    updated = history.append(HistoryEntry.from_report(report)).prune(
        retention_days=retention_days, max_entries=max_entries, now=now
    )
    return updated, updated.dumps()
