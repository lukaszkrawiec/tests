"""Layout regressions the validator cannot catch — clipping and overflow.

Every one of these was a real defect found by rendering the page and looking at it.
"""

from __future__ import annotations

import datetime as dt
import re

from tokenreport.contract import CountedComponent, CounterInfo, Kind, Report, Tier
from tokenreport.history import History, HistoryEntry
from tokenreport.render.dashboard import render_dashboard

NOW = dt.datetime(2026, 7, 1, tzinfo=dt.timezone.utc)
VIEWBOX_WIDTH = 960
PLOT_LEFT = 64
PLOT_RIGHT = 16
INNER_WIDTH = VIEWBOX_WIDTH - PLOT_LEFT - PLOT_RIGHT


def report(components):
    return Report(
        model="claude-opus-5",
        counter=CounterInfo(name="anthropic", exact=True),
        commit="head000",
        components=[
            CountedComponent(
                id=cid,
                tokens=tokens,
                tier=Tier.ON_DEMAND if cid.startswith("doc") else Tier.RESIDENT,
                kind=Kind.OTHER,
                group=cid.split(".")[0],
            )
            for cid, tokens in components.items()
        ],
    )


def entry(commit, days_ago, *, model="claude-opus-5", resident=100):
    return HistoryEntry(
        commit=commit,
        timestamp=(NOW - dt.timedelta(days=days_ago)).isoformat().replace("+00:00", "Z"),
        model=model,
        counter="anthropic",
        resident=resident,
        on_demand=40,
        components={"agent.a": resident, "doc.b": 40},
    )


def long_history(count=30, *, change_at=None):
    entries = []
    for i in range(count):
        model = (
            "claude-fable-5" if change_at is not None and i >= change_at else "claude-opus-5"
        )
        entries.append(entry(f"c{i:05d}", count - i, model=model, resident=100 + i * 10))
    return History(entries=entries)


def render(history):
    return render_dashboard(
        history, report({"agent.a": 400, "doc.b": 40}), generated_at=NOW
    )


def text_elements(html: str) -> list[tuple[float, str, str]]:
    """(x, anchor, content) for every SVG text node."""
    found = []
    for match in re.finditer(r"<text([^>]*)>([^<]*)</text>", html):
        attrs, content = match.group(1), match.group(2)
        x = float(re.search(r'x="(-?[\d.]+)"', attrs).group(1))
        anchor_match = re.search(r'text-anchor="(\w+)"', attrs)
        anchor = anchor_match.group(1) if anchor_match else "inherit"
        found.append((x, anchor, content))
    return found


class TestNoHorizontalClipping:
    def test_the_last_x_label_is_anchored_inward(self):
        # Centred on the final point, half the label would fall outside the viewBox.
        html = render(long_history())
        rightmost = [
            (x, anchor, text)
            for x, anchor, text in text_elements(html)
            if abs(x - INNER_WIDTH) < 1 and text
        ]
        assert rightmost, "expected labels at the right edge of the plot"
        assert all(anchor == "end" for _, anchor, _ in rightmost)

    def test_the_first_x_label_is_anchored_outward(self):
        html = render(long_history())
        leftmost = [
            (x, anchor, text)
            for x, anchor, text in text_elements(html)
            if x == 0.0 and text
        ]
        assert leftmost
        assert all(anchor == "start" for _, anchor, _ in leftmost)

    def test_a_marker_label_near_the_right_edge_flips_left(self):
        html = render(long_history(30, change_at=29))
        labels = [
            (x, anchor) for x, anchor, text in text_elements(html)
            if text == "tokenizer change"
        ]
        assert labels
        assert all(anchor == "end" for _, anchor in labels)

    def test_a_marker_label_in_the_middle_reads_left_to_right(self):
        html = render(long_history(30, change_at=15))
        labels = [
            (x, anchor) for x, anchor, text in text_elements(html)
            if text == "tokenizer change"
        ]
        assert labels
        assert all(anchor == "start" for _, anchor in labels)

    def test_no_text_is_positioned_outside_the_plot_area(self):
        html = render(long_history(30, change_at=20))
        for x, _, text in text_elements(html):
            if not text:
                continue
            # Y-axis labels sit in the left gutter; nothing may exceed the plot width.
            assert -PLOT_LEFT <= x <= INNER_WIDTH + PLOT_RIGHT, (x, text)


class TestLegend:
    def test_swatches_carry_the_series_colour_not_the_text_ink(self):
        # The label text keeps text ink; the swatch beside it carries identity.
        html = render(long_history())
        swatches = re.findall(r'<span class="swatch (s\d)"></span>', html)
        assert swatches, "legend swatches must be slot-coloured"
        assert len(set(swatches)) == len(swatches), "each group needs a distinct slot"

    def test_legend_labels_are_not_coloured_by_series(self):
        html = render(long_history())
        assert '<li><span class="swatch' in html
