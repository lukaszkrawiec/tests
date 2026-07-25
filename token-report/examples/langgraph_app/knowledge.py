"""Knowledge base with progressive disclosure.

Documents live on disk and are fetched by tool call. Only a compact index is resident in
the system prompt: one line per document, enough for the model to decide whether to
fetch it. The index is the part whose growth costs something on every request.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"
_FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)


@dataclass(frozen=True)
class Document:
    slug: str
    title: str
    summary: str
    body: str
    path: Path

    @property
    def relative_path(self) -> str:
        return str(self.path.relative_to(KNOWLEDGE_DIR.parent.parent.parent))


def _parse(path: Path) -> Document:
    raw = path.read_text(encoding="utf-8")
    match = _FRONT_MATTER.match(raw)
    if match is None:
        raise ValueError(f"{path} is missing front matter with title and summary")
    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    missing = {"title", "summary"} - meta.keys()
    if missing:
        raise ValueError(f"{path} front matter is missing {sorted(missing)}")
    return Document(
        slug=path.stem,
        title=meta["title"],
        summary=meta["summary"],
        body=raw[match.end() :].strip(),
        path=path,
    )


def load_documents() -> list[Document]:
    return [_parse(p) for p in sorted(KNOWLEDGE_DIR.glob("*.md"))]


def build_index(documents: list[Document]) -> str:
    """Render the resident index.

    Deliberately terse. Each extra line here is paid on every request by every node,
    unlike the documents themselves.
    """
    lines = [
        "## Knowledge base",
        "",
        "Fetch a document with `read_knowledge(slug)` when it is relevant.",
        "",
    ]
    lines.extend(f"- `{doc.slug}` — {doc.title}: {doc.summary}" for doc in documents)
    return "\n".join(lines)
