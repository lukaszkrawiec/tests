"""Tool schemas bound to graph nodes.

Tools are bound per node rather than globally so that a node does not pay for
capabilities it cannot use. The report reflects that grouping.
"""

from __future__ import annotations

from typing import Any

SEARCH_TOOL: dict[str, Any] = {
    "name": "search",
    "description": (
        "Search the web for pages relevant to a query. Returns ranked results with "
        "title, url, and an extract. Use one deliberate query per hypothesis rather "
        "than many broad queries."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query."},
            "recency": {
                "type": "string",
                "enum": ["any", "year", "month", "week"],
                "description": "Restrict results by age.",
            },
        },
        "required": ["query"],
    },
}

READ_KNOWLEDGE_TOOL: dict[str, Any] = {
    "name": "read_knowledge",
    "description": (
        "Fetch the full text of a knowledge-base document by slug. Slugs are listed in "
        "the knowledge base index in the system prompt."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "slug": {"type": "string", "description": "Document slug from the index."}
        },
        "required": ["slug"],
    },
}

HANDOFF_TOOL: dict[str, Any] = {
    "name": "handoff_brief",
    "description": (
        "Hand a structured research brief to the analyst node. Call this once, when "
        "evidence gathering is complete."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "The question as understood."},
            "claims": {
                "type": "array",
                "description": "Evidence grouped by claim.",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim": {"type": "string"},
                        "sources": {"type": "array", "items": {"type": "string"}},
                        "verified": {"type": "boolean"},
                    },
                    "required": ["claim", "sources", "verified"],
                },
            },
            "gaps": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Questions the evidence does not answer.",
            },
        },
        "required": ["question", "claims", "gaps"],
    },
}

CITE_TOOL: dict[str, Any] = {
    "name": "cite",
    "description": "Attach a source to a claim in the final answer.",
    "input_schema": {
        "type": "object",
        "properties": {
            "claim": {"type": "string"},
            "url": {"type": "string"},
            "measured_at": {
                "type": "string",
                "description": "When the cited figure was measured, if applicable.",
            },
        },
        "required": ["claim", "url"],
    },
}

RESEARCHER_TOOLS = [SEARCH_TOOL, READ_KNOWLEDGE_TOOL, HANDOFF_TOOL]
ANALYST_TOOLS = [READ_KNOWLEDGE_TOOL, CITE_TOOL]
