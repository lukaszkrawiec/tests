"""Markdown rendering for the pull request comment and the job summary.

The comment and the summary show the same information at different lengths: a comment
body is capped at 65,536 characters by the GitHub API, so it carries the headline and the
changes, while the summary carries everything.
"""

from __future__ import annotations

from ..compare import BudgetResult, ChangeKind, Comparison, ComponentDelta
from ..contract import Tier

# Identifies our own comment so subsequent runs update it in place instead of posting a
# new one on every push.
COMMENT_MARKER = "<!-- tokenreport:comment -->"
COMMENT_LIMIT = 65_536
# Leaves room for the truncation notice and a safety margin under the hard API limit.
_TRUNCATION_BUDGET = 1_500

_KIND_LABEL = {
    ChangeKind.ADDED: "new",
    ChangeKind.REMOVED: "removed",
    ChangeKind.CHANGED: "",
    ChangeKind.UNCHANGED: "",
}


def thousands(value: int | None) -> str:
    return "—" if value is None else f"{value:,}"


def signed(value: int) -> str:
    return f"{value:+,}" if value else "0"


def delta_cell(delta: ComponentDelta) -> str:
    if delta.kind is ChangeKind.UNCHANGED:
        return "—"
    if delta.kind is ChangeKind.ADDED:
        return f"**{signed(delta.delta)}** (new)"
    if delta.kind is ChangeKind.REMOVED:
        return f"**{signed(delta.delta)}** (removed)"
    pct = delta.pct
    suffix = f" ({pct:+.1f}%)" if pct is not None else ""
    return f"**{signed(delta.delta)}**{suffix}"


def _totals_table(comparison: Comparison) -> list[str]:
    head = comparison.head
    base = comparison.base if comparison.has_baseline else None
    rows = [
        "| Tier | Base | Head | Δ |",
        "| --- | ---: | ---: | ---: |",
    ]

    def row(label: str, before: int | None, after: int) -> str:
        if before is None:
            return f"| {label} | — | {thousands(after)} | — |"
        change = after - before
        pct = f" ({100.0 * change / before:+.1f}%)" if before else ""
        cell = f"**{signed(change)}**{pct}" if change else "—"
        return f"| {label} | {thousands(before)} | {thousands(after)} | {cell} |"

    rows.append(
        row("**Resident** — paid every request", base.resident if base else None, head.resident)
    )
    if head.cache_prefix_tokens or (base and base.cache_prefix_tokens):
        rows.append(
            row(
                "↳ inside cache prefix",
                base.cache_prefix_tokens if base else None,
                head.cache_prefix_tokens,
            )
        )
    if head.tool_set_overhead or (base and base.tool_set_overhead):
        rows.append(
            row(
                "↳ tool-set overhead",
                base.tool_set_overhead if base else None,
                head.tool_set_overhead,
            )
        )
    rows.append(
        row(
            "**On-demand** — paid when retrieved",
            base.on_demand if base else None,
            head.on_demand,
        )
    )
    return rows


def _headline(comparison: Comparison) -> str:
    head = comparison.head
    if not comparison.has_baseline:
        return f"**Resident context: {thousands(head.resident)} tokens**"
    delta = comparison.resident_delta
    if delta == 0:
        return (
            f"**Resident context: {thousands(head.resident)} tokens** — unchanged"
        )
    pct = comparison.resident_pct
    arrow = "🔺" if delta > 0 else "🔻"
    suffix = f" ({pct:+.1f}%)" if pct is not None else ""
    return (
        f"{arrow} **Resident context: {thousands(head.resident)} tokens** "
        f"— {signed(delta)}{suffix} vs base"
    )


def _component_table(
    deltas: list[ComponentDelta],
    *,
    show_tier: bool = True,
    show_delta: bool = True,
) -> list[str]:
    """Render a component table.

    ``show_delta`` is False when the baseline used a different tokenizer: both counts
    are still worth showing, but their difference is not growth and must not be
    presented as though it were.
    """
    columns = ["Component"] + (["Tier"] if show_tier else []) + ["Base", "Head"]
    if show_delta:
        columns.append("Δ")
    alignments = ["---"] + (["---"] if show_tier else []) + ["---:", "---:"]
    if show_delta:
        alignments.append("---:")
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(alignments) + " |"]
    for delta in deltas:
        tier = "resident" if delta.tier is Tier.RESIDENT else "on-demand"
        cells = [f"`{delta.id}`"]
        if show_tier:
            cells.append(tier)
        cells += [thousands(delta.before), thousands(delta.after)]
        if show_delta:
            cells.append(delta_cell(delta))
        rows.append("| " + " | ".join(cells) + " |")
    return rows


def _budget_section(budgets: BudgetResult) -> list[str]:
    lines: list[str] = []
    if budgets.violations:
        count = len(budgets.violations)
        lines.append(
            f"### ❌ {count} budget violation{'s' if count != 1 else ''}"
        )
        lines.append("")
        lines.extend(f"- **{v.budget}** — {v.message}" for v in budgets.violations)
    elif budgets.checked:
        lines.append(
            f"### ✅ Within budget ({budgets.checked} check"
            f"{'s' if budgets.checked != 1 else ''})"
        )
    if budgets.skipped:
        lines.append("")
        lines.append("<details><summary>Skipped checks</summary>")
        lines.append("")
        lines.extend(f"- {note}" for note in budgets.skipped)
        lines.append("")
        lines.append("</details>")
    return lines


def _footer(comparison: Comparison) -> list[str]:
    head = comparison.head
    notes = [f"model `{head.model}`", f"counter `{head.counter.name}`"]
    if head.commit:
        notes.append(f"commit `{head.commit[:7]}`")
    lines = ["", "---", f"<sub>{' · '.join(notes)}</sub>"]
    if not head.counter.exact:
        lines.insert(
            1,
            f"> ℹ️ Counted with `{head.counter.name}`, which is reproducible but is not "
            f"Claude's tokenizer — treat the **delta** as the signal and absolute "
            f"figures as indicative. Set the counter to `anthropic` for exact counts.",
        )
    if comparison.base is not None and not comparison.comparable:
        lines.insert(
            1,
            f"> ⚠️ No delta shown: {comparison.incomparable_reason}.",
        )
    elif comparison.base is None:
        lines.insert(1, "> ℹ️ No baseline found, so this run establishes one.")
    return lines


def render_comment(
    comparison: Comparison,
    budgets: BudgetResult,
    *,
    limit: int = COMMENT_LIMIT,
    summary_url: str | None = None,
) -> str:
    """Render the sticky pull request comment, truncating the table if necessary."""
    changed = comparison.changed()
    # A baseline on a different tokenizer gives numbers but not a meaningful delta.
    show_delta = comparison.base is None or comparison.comparable
    head_parts = [
        COMMENT_MARKER,
        "## 🧮 Token Report",
        "",
        _headline(comparison),
        "",
        *_totals_table(comparison),
        "",
    ]
    budget_parts = _budget_section(budgets)
    footer_parts = _footer(comparison)
    if summary_url:
        footer_parts.insert(
            len(footer_parts) - 1, f"<sub>[Full breakdown]({summary_url})</sub>  "
        )

    fixed = "\n".join(head_parts + budget_parts + footer_parts)
    room = limit - len(fixed) - _TRUNCATION_BUDGET

    body_parts: list[str] = []
    if changed:
        body_parts.append(f"### Changed components ({len(changed)})")
        body_parts.append("")
        shown, hidden = _fit(changed, room, show_delta=show_delta)
        if shown:
            body_parts.extend(_component_table(shown, show_delta=show_delta))
        if hidden:
            body_parts.append("")
            body_parts.append(
                f"<sub>…and {hidden} more changed component"
                f"{'s' if hidden != 1 else ''} — see the job summary for the full "
                f"breakdown.</sub>"
            )
        body_parts.append("")
    elif comparison.has_baseline:
        body_parts.append("No component changed size.")
        body_parts.append("")

    unchanged = comparison.unchanged()
    if unchanged:
        block = [
            f"<details><summary>Unchanged ({len(unchanged)})</summary>",
            "",
            *_component_table(unchanged, show_delta=show_delta),
            "",
            "</details>",
            "",
        ]
        rendered = "\n".join(block)
        if len(rendered) < limit - len(fixed) - len("\n".join(body_parts)) - _TRUNCATION_BUDGET:
            body_parts.extend(block)

    return "\n".join(head_parts + body_parts + budget_parts + footer_parts)


def _fit(
    deltas: list[ComponentDelta], room: int, *, show_delta: bool = True
) -> tuple[list[ComponentDelta], int]:
    """Take as many rows as fit in ``room`` characters, largest changes first."""
    if room <= 0:
        return [], len(deltas)
    used = 0
    taken: list[ComponentDelta] = []
    for delta in deltas:
        row_length = len("\n".join(_component_table([delta], show_delta=show_delta)[2:])) + 1
        if used + row_length > room:
            break
        used += row_length
        taken.append(delta)
    return taken, len(deltas) - len(taken)


def render_summary(comparison: Comparison, budgets: BudgetResult) -> str:
    """Render the untruncated job summary, grouped by component group."""
    head = comparison.head
    show_delta = comparison.base is None or comparison.comparable
    lines = [
        "# 🧮 Token Report",
        "",
        _headline(comparison),
        "",
        *_totals_table(comparison),
        "",
        *_budget_section(budgets),
        "",
    ]

    changed = comparison.changed()
    if changed:
        lines += [
            f"## Changed components ({len(changed)})",
            "",
            *_component_table(changed, show_delta=show_delta),
            "",
        ]

    lines.append("## All components")
    lines.append("")
    groups: dict[str, list[ComponentDelta]] = {}
    for delta in comparison.components:
        groups.setdefault(delta.group or "ungrouped", []).append(delta)
    for group in sorted(groups):
        members = sorted(groups[group], key=lambda d: (-(d.after or 0), d.id))
        resident = sum(m.after or 0 for m in members if m.tier is Tier.RESIDENT)
        on_demand = sum(m.after or 0 for m in members if m.tier is Tier.ON_DEMAND)
        lines.append(
            f"### {group} — {thousands(resident)} resident, "
            f"{thousands(on_demand)} on-demand"
        )
        lines.append("")
        lines.extend(_component_table(members, show_delta=show_delta))
        lines.append("")

    lines.extend(_footer(comparison))
    return "\n".join(lines)
