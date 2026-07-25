"""Diff two reports, and evaluate budgets against the result.

Additions and removals are kept distinct from changes to existing components: a new
system prompt and a grown system prompt call for different reviews, and folding both into
a single total hides which happened.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Iterable

from .config import Budgets
from .contract import CountedComponent, Report, Tier


class ChangeKind(enum.Enum):
    ADDED = "added"
    REMOVED = "removed"
    CHANGED = "changed"
    UNCHANGED = "unchanged"


@dataclass(frozen=True)
class ComponentDelta:
    id: str
    kind: ChangeKind
    tier: Tier
    before: int | None
    after: int | None
    source: str | None = None
    group: str | None = None

    @property
    def delta(self) -> int:
        return (self.after or 0) - (self.before or 0)

    @property
    def pct(self) -> float | None:
        """Growth as a percentage, or None when there is no baseline to divide by."""
        if not self.before:
            return None
        return 100.0 * self.delta / self.before


@dataclass
class Comparison:
    """The difference between a head report and its baseline.

    ``comparable`` is False when the baseline was measured under a different tokenizer.
    Claude 4.7 and later tokenize roughly 30% higher than earlier models for identical
    text, so printing that as growth would report a model upgrade as a regression.
    """

    head: Report
    base: Report | None
    components: list[ComponentDelta] = field(default_factory=list)
    comparable: bool = True
    incomparable_reason: str | None = None

    @property
    def has_baseline(self) -> bool:
        return self.base is not None and self.comparable

    @property
    def resident_before(self) -> int | None:
        return self.base.resident if self.has_baseline else None

    @property
    def resident_delta(self) -> int:
        return self.head.resident - (self.resident_before or 0)

    @property
    def resident_pct(self) -> float | None:
        before = self.resident_before
        if not before:
            return None
        return 100.0 * (self.head.resident - before) / before

    @property
    def on_demand_before(self) -> int | None:
        return self.base.on_demand if self.has_baseline else None

    def changed(self) -> list[ComponentDelta]:
        """Components that moved, largest absolute change first."""
        moved = [c for c in self.components if c.kind is not ChangeKind.UNCHANGED]
        return sorted(moved, key=lambda c: (-abs(c.delta), c.id))

    def unchanged(self) -> list[ComponentDelta]:
        return sorted(
            (c for c in self.components if c.kind is ChangeKind.UNCHANGED),
            key=lambda c: (-(c.after or 0), c.id),
        )


def compare(head: Report, base: Report | None) -> Comparison:
    """Diff ``head`` against ``base``, tolerating a missing or incomparable baseline."""
    if base is None:
        return Comparison(
            head=head,
            base=None,
            components=[
                ComponentDelta(
                    id=c.id,
                    kind=ChangeKind.ADDED,
                    tier=c.tier,
                    before=None,
                    after=c.tokens,
                    source=c.source,
                    group=c.group,
                )
                for c in _sorted(head.components)
            ],
            comparable=True,
        )

    comparable, reason = _comparability(head, base)
    head_by_id, base_by_id = head.by_id(), base.by_id()
    deltas: list[ComponentDelta] = []

    for cid in sorted(head_by_id.keys() | base_by_id.keys()):
        after = head_by_id.get(cid)
        before = base_by_id.get(cid)
        if after is not None and before is None:
            kind = ChangeKind.ADDED
        elif after is None and before is not None:
            kind = ChangeKind.REMOVED
        elif after.tokens != before.tokens:
            kind = ChangeKind.CHANGED
        else:
            kind = ChangeKind.UNCHANGED
        reference = after or before
        deltas.append(
            ComponentDelta(
                id=cid,
                kind=kind,
                tier=reference.tier,
                before=before.tokens if before else None,
                after=after.tokens if after else None,
                source=reference.source,
                group=reference.group,
            )
        )

    return Comparison(
        head=head,
        base=base,
        components=deltas,
        comparable=comparable,
        incomparable_reason=reason,
    )


def _comparability(head: Report, base: Report) -> tuple[bool, str | None]:
    if head.model != base.model:
        return False, (
            f"baseline was measured on {base.model} and this run on {head.model}; "
            f"token counts are not comparable across tokenizers"
        )
    if head.counter.name != base.counter.name:
        return False, (
            f"baseline was measured with the {base.counter.name} counter and this run "
            f"with {head.counter.name}; switching counter re-bases every number"
        )
    # Exactness is deliberately not a gate here. An approximate counter still produces a
    # meaningful delta as long as both sides used the same one — which is precisely why
    # comparisons key on counter identity rather than on absolute accuracy.
    return True, None


def _sorted(components: Iterable[CountedComponent]) -> list[CountedComponent]:
    return sorted(components, key=lambda c: (-c.tokens, c.id))


class Severity(enum.Enum):
    OK = "ok"
    FAIL = "fail"


@dataclass(frozen=True)
class Violation:
    budget: str
    message: str
    limit: float
    actual: float
    component_id: str | None = None
    source: str | None = None


@dataclass
class BudgetResult:
    violations: list[Violation] = field(default_factory=list)
    checked: int = 0
    skipped: list[str] = field(default_factory=list)

    @property
    def severity(self) -> Severity:
        return Severity.FAIL if self.violations else Severity.OK

    @property
    def ok(self) -> bool:
        return not self.violations


def evaluate(comparison: Comparison, budgets: Budgets) -> BudgetResult:
    """Check a comparison against configured budgets.

    Growth budgets are skipped rather than failed when there is no comparable baseline:
    a first run, or a run whose baseline used a different tokenizer, has nothing to grow
    from, and failing there would block every new branch.
    """
    result = BudgetResult()
    head = comparison.head

    if budgets.resident_total is not None:
        result.checked += 1
        if head.resident > budgets.resident_total:
            result.violations.append(
                Violation(
                    budget="resident_total",
                    message=(
                        f"resident context is {head.resident:,} tokens, over the "
                        f"budget of {budgets.resident_total:,}"
                    ),
                    limit=budgets.resident_total,
                    actual=head.resident,
                )
            )

    if budgets.on_demand_total is not None:
        result.checked += 1
        if head.on_demand > budgets.on_demand_total:
            result.violations.append(
                Violation(
                    budget="on_demand_total",
                    message=(
                        f"on-demand context is {head.on_demand:,} tokens, over the "
                        f"budget of {budgets.on_demand_total:,}"
                    ),
                    limit=budgets.on_demand_total,
                    actual=head.on_demand,
                )
            )

    if budgets.resident_growth_pct is not None:
        pct = comparison.resident_pct
        if pct is None:
            result.skipped.append(
                "resident_growth_pct: no comparable baseline to measure growth against"
            )
        else:
            result.checked += 1
            if pct > budgets.resident_growth_pct:
                result.violations.append(
                    Violation(
                        budget="resident_growth_pct",
                        message=(
                            f"resident context grew {pct:+.1f}%, over the budget of "
                            f"{budgets.resident_growth_pct:+.1f}% "
                            f"({comparison.resident_before:,} → {head.resident:,})"
                        ),
                        limit=budgets.resident_growth_pct,
                        actual=pct,
                    )
                )

    by_id = head.by_id()
    for cid, limit in sorted(budgets.component.items()):
        component = by_id.get(cid)
        if component is None:
            # A budget for a component that no longer exists is stale config, not a
            # failure. Surfacing it as skipped prompts a cleanup without blocking.
            result.skipped.append(
                f"component budget for {cid!r}: no such component in this report"
            )
            continue
        result.checked += 1
        if component.tokens > limit:
            result.violations.append(
                Violation(
                    budget="component",
                    message=(
                        f"{cid} is {component.tokens:,} tokens, over its budget of "
                        f"{limit:,}"
                    ),
                    limit=limit,
                    actual=component.tokens,
                    component_id=cid,
                    source=component.source,
                )
            )

    return result
