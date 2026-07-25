"""Prompt composition for the example graph.

This stands in for the part of a real application that assembles a system prompt from
several sources. It is the reason the contract is a Python callable rather than a set of
file globs: no glob can see that a preamble, a role prompt, and a rendered knowledge
index become one resident block.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import tools as _tools
from .knowledge import build_index, load_documents

PROMPTS_DIR = Path(__file__).parent / "prompts"


def _read(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8").strip()


@dataclass(frozen=True)
class Node:
    name: str
    system_prompt: str
    tools: list[dict[str, Any]]
    prompt_sources: list[str]


def build_nodes() -> list[Node]:
    documents = load_documents()
    index = build_index(documents)
    preamble = _read("shared_preamble")

    def compose(role: str) -> str:
        # Stable content first so a cache prefix stays valid for as long as possible.
        return "\n\n".join([preamble, _read(role), index])

    return [
        Node(
            name="researcher",
            system_prompt=compose("researcher"),
            tools=_tools.RESEARCHER_TOOLS,
            prompt_sources=[
                "token-report/examples/langgraph_app/prompts/shared_preamble.md",
                "token-report/examples/langgraph_app/prompts/researcher.md",
            ],
        ),
        Node(
            name="analyst",
            system_prompt=compose("analyst"),
            tools=_tools.ANALYST_TOOLS,
            prompt_sources=[
                "token-report/examples/langgraph_app/prompts/shared_preamble.md",
                "token-report/examples/langgraph_app/prompts/analyst.md",
            ],
        ),
    ]
