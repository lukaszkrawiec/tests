"""Self-contained HTML trend dashboard.

Charts are generated as inline SVG with no scripts and no external requests, so the page
works as a workflow artifact opened from disk, as a GitHub Pages deploy, and inside a
strict content-security policy. Values are reachable from the table view as well as from
the marks, so nothing is gated behind a hover.

The x axis is commit *sequence*, not wall-clock time: commits are not evenly spaced in
time, and a time axis bunches a busy week into a few pixels while stretching a quiet
month across the plot.
"""

from __future__ import annotations

import datetime as _dt
import html
from dataclasses import dataclass

from ..contract import Report, Tier
from ..history import History, HistoryEntry

# The documented categorical order, in slot order. Both modes validated: worst adjacent
# CVD ΔE 9.1 light / 8.4 dark, worst adjacent normal-vision ΔE 19.6 / 19.3. Three light
# steps sit below 3:1 on the light surface, so the relief rule applies — the table view
# below the charts is that relief.
#
# These are the only place the palette is written down; the CSS custom properties are
# generated from them, so a slot cannot say one colour in the stylesheet and another in
# the legend.
_SERIES_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
                 "#4a3aa7", "#e34948"]
_SERIES_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300",
                "#9085e9", "#e66767"]
MAX_SERIES = len(_SERIES_LIGHT)


def _series_vars(palette: list[str]) -> str:
    return " ".join(f"--s{i}:{hex_};" for i, hex_ in enumerate(palette))

_PLOT = {"width": 960, "height": 300, "left": 64, "right": 16, "top": 16, "bottom": 44}


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


@dataclass(frozen=True)
class _Scale:
    lo: float
    hi: float
    pixels: int

    def y(self, value: float) -> float:
        if self.hi == self.lo:
            return float(self.pixels)
        return self.pixels * (1 - (value - self.lo) / (self.hi - self.lo))


def _nice_ceiling(value: int) -> int:
    """Round a maximum up to a readable axis bound."""
    if value <= 0:
        return 10
    magnitude = 10 ** (len(str(value)) - 1)
    for step in (1, 2, 2.5, 5, 10):
        candidate = magnitude * step
        if candidate >= value:
            return int(candidate)
    return int(magnitude * 10)


def _ticks(hi: int, count: int = 4) -> list[int]:
    return [round(hi * i / count) for i in range(count + 1)]


def _group_totals(entry: HistoryEntry, groups: list[str], grouping: dict[str, str],
                  resident_ids: set[str]) -> dict[str, int]:
    totals = {group: 0 for group in groups}
    for cid, tokens in entry.components.items():
        if cid not in resident_ids:
            continue
        group = grouping.get(cid, "other")
        if group in totals:
            totals[group] += tokens
    return totals


def _segments(count: int, breaks: set[int]) -> list[list[int]]:
    """Split indices into runs, cutting before each break.

    A tokenizer change makes the numbers either side incomparable, so the line must not
    connect across it — otherwise a model upgrade draws as a sudden 30% regression.
    """
    if count == 0:
        return []
    runs: list[list[int]] = [[]]
    for index in range(count):
        if index in breaks and runs[-1]:
            runs.append([])
        runs[-1].append(index)
    return [run for run in runs if run]


def _x_positions(count: int, width: int) -> list[float]:
    if count == 1:
        return [width / 2]
    return [width * i / (count - 1) for i in range(count)]


def _stacked_area_svg(
    history: History,
    *,
    groups: list[str],
    grouping: dict[str, str],
    resident_ids: set[str],
    breaks: set[int],
) -> str:
    entries = history.sorted_entries()
    inner_w = _PLOT["width"] - _PLOT["left"] - _PLOT["right"]
    inner_h = _PLOT["height"] - _PLOT["top"] - _PLOT["bottom"]

    per_entry = [_group_totals(e, groups, grouping, resident_ids) for e in entries]
    peak = max((sum(totals.values()) for totals in per_entry), default=0)
    hi = _nice_ceiling(peak)
    scale = _Scale(0, hi, inner_h)
    xs = _x_positions(len(entries), inner_w)

    parts: list[str] = []
    parts.append(
        f'<svg class="chart" viewBox="0 0 {_PLOT["width"]} {_PLOT["height"]}" '
        f'role="img" aria-label="Resident tokens by group over the last '
        f'{len(entries)} recorded commits" preserveAspectRatio="xMidYMid meet">'
    )
    parts.append(f'<g transform="translate({_PLOT["left"]},{_PLOT["top"]})">')

    # Gridlines and y axis: solid hairlines, one shade off the surface.
    for tick in _ticks(hi):
        y = scale.y(tick)
        parts.append(
            f'<line class="grid" x1="0" y1="{y:.1f}" x2="{inner_w}" y2="{y:.1f}"/>'
        )
        parts.append(
            f'<text class="tick ytick" x="-10" y="{y:.1f}">{tick:,}</text>'
        )

    # Stacked bands, drawn bottom-up so each sits on the previous.
    cumulative = [0.0] * len(entries)
    for slot, group in enumerate(groups):
        values = [totals.get(group, 0) for totals in per_entry]
        tops = [cumulative[i] + values[i] for i in range(len(entries))]
        for run in _segments(len(entries), breaks):
            if len(run) == 1:
                # A single point cannot form an area; a short mark keeps it visible.
                i = run[0]
                y_top, y_base = scale.y(tops[i]), scale.y(cumulative[i])
                parts.append(
                    f'<rect class="band s{slot}" x="{xs[i] - 3:.1f}" y="{y_top:.1f}" '
                    f'width="6" height="{max(1.0, y_base - y_top):.1f}"/>'
                )
                continue
            upper = " ".join(f"{xs[i]:.1f},{scale.y(tops[i]):.1f}" for i in run)
            lower = " ".join(
                f"{xs[i]:.1f},{scale.y(cumulative[i]):.1f}" for i in reversed(run)
            )
            parts.append(f'<polygon class="band s{slot}" points="{upper} {lower}"/>')
            # A 2px surface-coloured top edge separates adjacent segments without
            # drawing a border around the marks.
            parts.append(f'<polyline class="bandgap" points="{upper}"/>')
        for i, entry in enumerate(entries):
            parts.append(
                f'<circle class="pt s{slot}" cx="{xs[i]:.1f}" '
                f'cy="{scale.y(tops[i]):.1f}" r="7">'
                f"<title>{_esc(group)} · {values[i]:,} tokens · "
                f"{_esc(entry.commit[:7])}</title></circle>"
            )
        cumulative = tops

    parts.extend(
        _axis_and_markers(entries, xs, inner_w, inner_h, breaks, scale, hi)
    )
    parts.append("</g></svg>")
    return "".join(parts)


def _line_svg(
    history: History,
    *,
    values: list[int],
    label: str,
    breaks: set[int],
    slot: int = 1,
) -> str:
    entries = history.sorted_entries()
    inner_w = _PLOT["width"] - _PLOT["left"] - _PLOT["right"]
    inner_h = _PLOT["height"] - _PLOT["top"] - _PLOT["bottom"]
    hi = _nice_ceiling(max(values, default=0))
    scale = _Scale(0, hi, inner_h)
    xs = _x_positions(len(entries), inner_w)

    parts = [
        f'<svg class="chart" viewBox="0 0 {_PLOT["width"]} {_PLOT["height"]}" '
        f'role="img" aria-label="{_esc(label)} over the last {len(entries)} recorded '
        f'commits" preserveAspectRatio="xMidYMid meet">',
        f'<g transform="translate({_PLOT["left"]},{_PLOT["top"]})">',
    ]
    for tick in _ticks(hi):
        y = scale.y(tick)
        parts.append(
            f'<line class="grid" x1="0" y1="{y:.1f}" x2="{inner_w}" y2="{y:.1f}"/>'
        )
        parts.append(f'<text class="tick ytick" x="-10" y="{y:.1f}">{tick:,}</text>')

    for run in _segments(len(entries), breaks):
        points = " ".join(f"{xs[i]:.1f},{scale.y(values[i]):.1f}" for i in run)
        if len(run) == 1:
            i = run[0]
            parts.append(
                f'<circle class="pt s{slot} solo" cx="{xs[i]:.1f}" '
                f'cy="{scale.y(values[i]):.1f}" r="4"/>'
            )
        else:
            parts.append(f'<polyline class="line s{slot}" points="{points}"/>')

    for i, entry in enumerate(entries):
        parts.append(
            f'<circle class="pt s{slot}" cx="{xs[i]:.1f}" '
            f'cy="{scale.y(values[i]):.1f}" r="7">'
            f"<title>{values[i]:,} tokens · {_esc(entry.commit[:7])}</title></circle>"
        )

    parts.extend(_axis_and_markers(entries, xs, inner_w, inner_h, breaks, scale, hi))
    parts.append("</g></svg>")
    return "".join(parts)


def _axis_and_markers(entries, xs, inner_w, inner_h, breaks, scale, hi) -> list[str]:
    parts = [
        f'<line class="axis" x1="0" y1="{inner_h}" x2="{inner_w}" y2="{inner_h}"/>'
    ]
    # Label a handful of commits rather than all of them; the rest are in the table.
    count = len(entries)
    stride = max(1, count // 6)
    for i, entry in enumerate(entries):
        if i % stride and i != count - 1:
            continue
        # Built by hand rather than with %-d, which is not portable off glibc.
        date = f"{entry.moment:%b} {entry.moment.day}"
        # The outermost labels are anchored inward; centred, half of each would
        # fall outside the viewBox and get clipped.
        anchor = "start" if i == 0 else ("end" if i == count - 1 else "middle")
        parts.append(
            f'<text class="tick xtick" text-anchor="{anchor}" x="{xs[i]:.1f}" '
            f'y="{inner_h + 18}">{_esc(date)}</text>'
        )
        parts.append(
            f'<text class="tick xtick sha" text-anchor="{anchor}" x="{xs[i]:.1f}" '
            f'y="{inner_h + 32}">{_esc(entry.commit[:7])}</text>'
        )
    for index in sorted(breaks):
        if index >= len(xs):
            continue
        x = xs[index]
        parts.append(f'<line class="marker" x1="{x:.1f}" y1="0" x2="{x:.1f}" y2="{inner_h}"/>')
        # Near the right edge the label would overflow the viewBox, so it flips to the
        # left of its own marker line rather than being clipped.
        flip = x > inner_w - 100
        parts.append(
            f'<text class="markerlabel" text-anchor="{"end" if flip else "start"}" '
            f'x="{(x - 5) if flip else (x + 5):.1f}" y="12">tokenizer change</text>'
        )
    return parts


def _stat_tile(label: str, value: int, *, delta: int | None, note: str) -> str:
    if delta is None:
        change = '<span class="delta none">no prior point</span>'
    elif delta == 0:
        change = '<span class="delta none">unchanged</span>'
    else:
        direction = "up" if delta > 0 else "down"
        icon = "▲" if delta > 0 else "▼"
        change = (
            f'<span class="delta {direction}">{icon} {delta:+,}</span>'
        )
    return (
        '<div class="tile">'
        f'<div class="tile-label">{_esc(label)}</div>'
        f'<div class="tile-value">{value:,}</div>'
        f'<div class="tile-foot">{change}'
        f'<span class="tile-note">· {_esc(note)}</span></div>'
        "</div>"
    )


def _component_table(report: Report) -> str:
    rows = []
    for component in sorted(report.components, key=lambda c: (-c.tokens, c.id)):
        tier = "resident" if component.tier is Tier.RESIDENT else "on-demand"
        rows.append(
            "<tr>"
            f"<td><code>{_esc(component.id)}</code></td>"
            f'<td><span class="pill {tier.replace("-", "")}">{tier}</span></td>'
            f"<td>{_esc(component.group or '—')}</td>"
            f'<td class="num">{component.tokens:,}</td>'
            "</tr>"
        )
    return (
        '<table class="data"><thead><tr><th>Component</th><th>Tier</th>'
        '<th>Group</th><th class="num">Tokens</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


_STYLE = """
:root {
  --surface-1: #fcfcfb; --plane: #f9f9f7;
  --text-primary: #0b0b0b; --text-secondary: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10);
  --good: #0ca30c; --critical: #d03b3b; --up: #d03b3b; --down: #006300;
  /*SERIES-LIGHT*/
  color-scheme: light;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    --surface-1: #1a1a19; --plane: #0d0d0d;
    --text-primary: #ffffff; --text-secondary: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
    --good: #0ca30c; --critical: #d03b3b; --up: #e66767; --down: #0ca30c;
    /*SERIES-DARK*/
    color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --surface-1: #1a1a19; --plane: #0d0d0d;
  --text-primary: #ffffff; --text-secondary: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
  --good: #0ca30c; --critical: #d03b3b; --up: #e66767; --down: #0ca30c;
  /*SERIES-DARK*/
  color-scheme: dark;
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 2rem 1.25rem 4rem;
  background: var(--plane); color: var(--text-primary);
  font: 15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif;
}
.wrap { max-width: 1100px; margin: 0 auto; }
h1 { font-size: 1.5rem; margin: 0 0 .25rem; letter-spacing: -0.01em; }
h2 { font-size: 1.05rem; margin: 0 0 .2rem; }
.sub { color: var(--text-secondary); margin: 0 0 1.75rem; font-size: .9rem; }
.card {
  background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 10px; padding: 1.25rem; margin-bottom: 1.25rem;
}
.card-note { color: var(--muted); font-size: .82rem; margin: 0 0 1rem; }
.tiles { display: grid; gap: .875rem; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); margin-bottom: 1.25rem; }
.tile { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: .95rem 1.05rem; }
.tile-label { color: var(--text-secondary); font-size: .8rem; text-transform: uppercase; letter-spacing: .04em; }
.tile-value { font-size: 2rem; font-weight: 600; letter-spacing: -0.02em; margin: .15rem 0 .1rem; }
.tile-foot { display: flex; gap: .5rem; align-items: baseline; flex-wrap: wrap; font-size: .82rem; }
.delta { font-weight: 600; font-variant-numeric: tabular-nums; }
.delta.up { color: var(--up); } .delta.down { color: var(--down); } .delta.none { color: var(--muted); font-weight: 400; }
.tile-note { color: var(--muted); }
.chart-scroll { overflow-x: auto; }
.chart { width: 100%; min-width: 620px; height: auto; display: block; }
.grid { stroke: var(--grid); stroke-width: 1; }
.axis { stroke: var(--axis); stroke-width: 1; }
.marker { stroke: var(--muted); stroke-width: 1; }
.markerlabel { fill: var(--muted); font-size: 10px; }
.tick { fill: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums; }
.ytick { text-anchor: end; dominant-baseline: middle; }
.xtick { text-anchor: middle; }
.sha { font-size: 10px; opacity: .8; }
.band { stroke: none; }
.bandgap { fill: none; stroke: var(--surface-1); stroke-width: 2; }
.line { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
.pt { fill: transparent; stroke: none; }
.pt.solo { fill: currentColor; }
/*SERIES-SLOTS*/
polygon.band, rect.band { fill: currentColor; }
polyline.line { stroke: currentColor; }
.legend { display: flex; flex-wrap: wrap; gap: .25rem 1.1rem; margin: .9rem 0 0; padding: 0; list-style: none; font-size: .85rem; }
.legend li { display: flex; align-items: center; gap: .4rem; color: var(--text-secondary); }
.swatch { width: 11px; height: 11px; border-radius: 3px; background: currentColor; flex: none; }
table.data { width: 100%; border-collapse: collapse; font-size: .875rem; }
table.data th, table.data td { text-align: left; padding: .45rem .6rem; border-bottom: 1px solid var(--border); }
table.data th { color: var(--text-secondary); font-weight: 600; font-size: .8rem; text-transform: uppercase; letter-spacing: .03em; }
table.data .num { text-align: right; font-variant-numeric: tabular-nums; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .92em; }
.pill { display: inline-block; padding: .08rem .45rem; border-radius: 999px; font-size: .75rem; border: 1px solid var(--border); color: var(--text-secondary); }
.pill.resident { border-color: color-mix(in srgb, var(--s0) 45%, transparent); }
.status { display: inline-flex; align-items: center; gap: .4rem; font-weight: 600; font-size: .9rem; }
.status.ok { color: var(--good); } .status.fail { color: var(--critical); }
.violations { margin: .6rem 0 0; padding-left: 1.1rem; color: var(--text-secondary); font-size: .875rem; }
.empty { color: var(--text-secondary); padding: 1.5rem 0; text-align: center; }
footer { color: var(--muted); font-size: .8rem; margin-top: 1.5rem; }
"""


def render_dashboard(
    history: History,
    report: Report,
    *,
    title: str = "Token Report",
    repo: str | None = None,
    generated_at: _dt.datetime | None = None,
) -> str:
    """Render the trend dashboard as one self-contained HTML document."""
    entries = history.sorted_entries()
    latest = entries[-1] if entries else None
    # The dashboard is normally built just after this report was appended, so the last
    # entry is the report itself and the comparison point is the one before it. When the
    # report has not been recorded (an approximate run, or a preview), the last entry is
    # already the right baseline.
    prior = [e for e in entries if e.commit != report.commit]
    previous = prior[-1] if prior else None

    grouping = {c.id: (c.group or "other") for c in report.components}
    resident_ids = {c.id for c in report.components if c.tier is Tier.RESIDENT}

    # Groups ordered by current size so the largest band sits at the bottom of the
    # stack, and capped: a ninth colour would not be distinguishable.
    sizes: dict[str, int] = {}
    for component in report.components:
        if component.tier is Tier.RESIDENT:
            sizes[component.group or "other"] = (
                sizes.get(component.group or "other", 0) + component.tokens
            )
    groups = sorted(sizes, key=lambda g: (-sizes[g], g))
    if len(groups) > MAX_SERIES:
        groups = groups[: MAX_SERIES - 1] + ["other"]
        grouping = {
            cid: (group if group in groups else "other")
            for cid, group in grouping.items()
        }

    changed_at = set(history.tokenizer_changes())
    breaks = {i for i, entry in enumerate(entries) if entry.commit in changed_at}

    stamp = (generated_at or _dt.datetime.now(_dt.timezone.utc)).strftime(
        "%d %b %Y %H:%M UTC"
    )
    subject = f"{_esc(repo)} · " if repo else ""

    tiles = [
        _stat_tile(
            "Resident",
            report.resident,
            delta=(report.resident - previous.resident) if previous else None,
            note="every request",
        ),
        _stat_tile(
            "On-demand",
            report.on_demand,
            delta=(report.on_demand - previous.on_demand) if previous else None,
            note="when retrieved",
        ),
        _stat_tile(
            "Cache prefix",
            report.cache_prefix_tokens,
            delta=(report.cache_prefix_tokens - previous.cache_prefix) if previous else None,
            note="cacheable resident",
        ),
        _stat_tile(
            "Components",
            len(report.components),
            delta=(len(report.components) - len(previous.components)) if previous else None,
            note="tracked",
        ),
    ]

    body: list[str] = [
        '<div class="wrap">',
        f"<h1>{_esc(title)}</h1>",
        f'<p class="sub">{subject}{len(entries)} recorded commit'
        f"{'s' if len(entries) != 1 else ''} · generated {_esc(stamp)}</p>",
        f'<div class="tiles">{"".join(tiles)}</div>',
    ]

    if not entries:
        body.append(
            '<div class="card"><div class="empty">No history recorded yet. The first '
            "push to the default branch establishes the series.</div></div>"
        )
    elif len(entries) == 1:
        body.append(
            '<div class="card"><h2>Resident tokens over time</h2>'
            '<p class="card-note">One data point so far — a trend appears from the '
            "second recorded commit.</p>"
            f'<div class="chart-scroll">{_stacked_area_svg(history, groups=groups, grouping=grouping, resident_ids=resident_ids, breaks=breaks)}</div>'
            f"{_legend(groups)}</div>"
        )
    else:
        body.append(
            '<div class="card"><h2>Resident tokens over time</h2>'
            '<p class="card-note">Context sent on every request, stacked by group. '
            "This is the number that compounds.</p>"
            f'<div class="chart-scroll">{_stacked_area_svg(history, groups=groups, grouping=grouping, resident_ids=resident_ids, breaks=breaks)}</div>'
            f"{_legend(groups)}</div>"
        )
        body.append(
            '<div class="card"><h2>On-demand tokens over time</h2>'
            '<p class="card-note">Retrievable context, charted separately because no '
            "single request pays for both tiers.</p>"
            f'<div class="chart-scroll">'
            f'{_line_svg(history, values=[e.on_demand for e in entries], label="On-demand tokens", breaks=breaks)}'
            "</div></div>"
        )

    body.append(
        '<div class="card"><h2>Current components</h2>'
        '<p class="card-note">Every measured component at the latest recorded commit.'
        "</p>"
        f'<div class="chart-scroll">{_component_table(report)}</div></div>'
    )

    counter_note = (
        f"model <code>{_esc(report.model)}</code> · counter "
        f"<code>{_esc(report.counter.name)}</code>"
    )
    if latest:
        counter_note += f" · latest <code>{_esc(latest.commit[:7])}</code>"
    body.append(f"<footer>{counter_note}</footer>")
    body.append("</div>")

    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{_esc(title)}</title>\n"
        f"<style>{_style()}</style>\n</head>\n<body>\n"
        + "\n".join(body)
        + "\n</body>\n</html>\n"
    )


def _style() -> str:
    """Fill the stylesheet's palette placeholders from the slot lists."""
    slots = "\n".join(f".s{i} {{ color: var(--s{i}); }}" for i in range(MAX_SERIES))
    return (
        _STYLE.replace("/*SERIES-LIGHT*/", _series_vars(_SERIES_LIGHT))
        .replace("/*SERIES-DARK*/", _series_vars(_SERIES_DARK))
        .replace("/*SERIES-SLOTS*/", slots)
    )


def _legend(groups: list[str]) -> str:
    items = "".join(
        f'<li><span class="swatch s{slot}"></span>{_esc(group)}</li>'
        for slot, group in enumerate(groups)
    )
    return f'<ul class="legend">{items}</ul>'
