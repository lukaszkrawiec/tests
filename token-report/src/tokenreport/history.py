"""Per-commit history: the series behind the trend dashboard.

History lives as JSON on an orphan branch, so it is auditable in git and needs no
external service. Two properties matter more than they look:

* Only exact counts are recorded. An approximate run would put a step change in the
  series that reflects the counter, not the prompts.
* Entries are keyed by commit. Re-running a workflow on the same commit replaces its
  entry rather than adding a duplicate point.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .contract import Report, Tier

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
        if not report.counter.exact:
            raise HistoryError(
                f"refusing to record an approximate run (counter "
                f"{report.counter.name!r}): a trend series must not mix measured and "
                f"estimated points"
            )
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
            ref=report.ref,
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

    def model_changes(self) -> list[str]:
        """Commits at which the model, and therefore the tokenizer, changed.

        The dashboard breaks its line at these points: Claude 4.7 and later tokenize
        roughly 30% higher for the same text, so connecting across a change would draw
        a model upgrade as a sudden regression.
        """
        changes: list[str] = []
        previous: str | None = None
        for entry in self.sorted_entries():
            if previous is not None and entry.model != previous:
                changes.append(entry.commit)
            previous = entry.model
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


def component_series(history: History) -> dict[str, list[int | None]]:
    """Per-component token counts aligned to the sorted entry list.

    A component missing from an entry yields None rather than 0, so a chart shows a gap
    where a component did not exist instead of implying it dropped to zero.
    """
    entries = history.sorted_entries()
    ids: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        for cid in entry.components:
            if cid not in seen:
                seen.add(cid)
                ids.append(cid)
    return {
        cid: [entry.components.get(cid) for entry in entries] for cid in sorted(ids)
    }


def resident_component_ids(report: Report) -> list[str]:
    return [c.id for c in report.components if c.tier is Tier.RESIDENT]


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


def iter_entries(history: History) -> Iterable[HistoryEntry]:
    return history.sorted_entries()
