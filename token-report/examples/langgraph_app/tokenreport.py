"""The collector this application exposes to Token Report.

This is the file a user of the tool writes. It is deliberately small: walk your own
graph, and for each piece of context say what it is, how it is identified, and whether
it is resident in every request or fetched on demand.
"""

from __future__ import annotations

from typing import Any, Iterator

from .graph import build_nodes
from .knowledge import build_index, load_documents


def collect() -> Iterator[dict[str, Any]]:
    documents = load_documents()

    # The index is resident: it ships in every system prompt so the model knows what
    # exists. Its growth is linear in the number of documents and is paid on every
    # request, which is the growth worth alerting on.
    yield {
        "id": "knowledge.index",
        "kind": "knowledge_index",
        "tier": "resident",
        "group": "knowledge",
        "text": build_index(documents),
        "source": "token-report/examples/langgraph_app/knowledge.py",
        "cache_prefix": True,
    }

    # The documents themselves are fetched by tool call, so a request pays for one only
    # when it asks for it. Reported for visibility, never summed with resident tokens.
    for doc in documents:
        yield {
            "id": f"knowledge.doc.{doc.slug}",
            "kind": "knowledge_doc",
            "tier": "on_demand",
            "group": "knowledge",
            "text": doc.body,
            "source": f"token-report/examples/langgraph_app/knowledge/{doc.slug}.md",
        }

    for node in build_nodes():
        yield {
            "id": f"agent.{node.name}.system_prompt",
            "kind": "system_prompt",
            "tier": "resident",
            "group": node.name,
            "text": node.system_prompt,
            "source": node.prompt_sources[-1],
            "cache_prefix": True,
        }
        yield {
            "id": f"agent.{node.name}.tools",
            "kind": "tool_schema",
            "tier": "resident",
            "group": node.name,
            "tools": node.tools,
            "source": "token-report/examples/langgraph_app/tools.py",
            "cache_prefix": True,
        }
