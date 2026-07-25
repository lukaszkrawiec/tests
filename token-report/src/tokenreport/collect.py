"""Resolve a collector entrypoint and turn its components into a counted report."""

from __future__ import annotations

import datetime as _dt
import importlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, Iterable

from .config import Config
from .contract import (
    Component,
    ContractError,
    CountedComponent,
    Report,
    Tier,
    validate,
)
from .counters import Counter


class EntrypointError(ContractError):
    """Raised when the configured entrypoint cannot be imported or called."""


def resolve_entrypoint(spec: str, *, root: Path | None = None) -> Callable[[], Iterable[object]]:
    """Import ``module:callable``.

    The repository root and a ``src`` layout directory are prepended to ``sys.path``
    so an application that has not been installed still imports, which is the common
    case in CI before a build step.
    """
    module_name, _, attr = spec.partition(":")
    if not module_name or not attr:
        raise EntrypointError(f"entrypoint {spec!r} must look like 'package.module:callable'")

    if root is not None:
        for candidate in (root, root / "src"):
            resolved = str(candidate.resolve())
            if candidate.is_dir() and resolved not in sys.path:
                sys.path.insert(0, resolved)

    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise EntrypointError(
            f"could not import module {module_name!r} from entrypoint {spec!r}: {exc}"
        ) from exc

    try:
        func = getattr(module, attr)
    except AttributeError:
        raise EntrypointError(
            f"module {module_name!r} has no attribute {attr!r}"
        ) from None

    if not callable(func):
        raise EntrypointError(f"{spec!r} resolved to {type(func).__name__}, which is not callable")
    return func


def call_collector(func: Callable[[], Iterable[object]], *, spec: str) -> list[Component]:
    try:
        raw = func()
    except Exception as exc:  # noqa: BLE001 - surfaced with context, re-raised below
        raise EntrypointError(f"collector {spec!r} raised {type(exc).__name__}: {exc}") from exc
    if raw is None:
        raise EntrypointError(f"collector {spec!r} returned None; it must return components")
    return validate(raw)


def count(
    components: Iterable[Component],
    counter: Counter,
    *,
    model: str,
    commit: str | None = None,
    ref: str | None = None,
    now: _dt.datetime | None = None,
) -> Report:
    """Measure each component and assemble a report.

    Tool components are counted individually for attribution, then the whole tool set
    is counted once more. The difference is recorded as ``tool_set_overhead`` rather
    than smeared across the individual tools, because enabling tools carries a
    one-time cost that belongs to no single tool. Without this the totals would not
    reconcile with what the model actually charges.
    """
    components = list(components)
    counter.warm(c.text for c in components if c.text)

    counted: list[CountedComponent] = []
    tool_components: list[Component] = []

    for component in components:
        if component.is_tools:
            tokens = counter.count_tools(list(component.tools))
            tool_components.append(component)
        else:
            tokens = counter.count_text(component.text or "")
        counted.append(
            CountedComponent(
                id=component.id,
                tokens=tokens,
                tier=component.tier,
                kind=component.kind,
                group=component.group,
                source=component.source,
                cache_prefix=component.cache_prefix,
            )
        )

    overhead = 0
    resident_tools = [c for c in tool_components if c.tier is Tier.RESIDENT]
    if len(resident_tools) > 1:
        every_tool = [t for c in resident_tools for t in c.tools]
        combined = counter.count_tools(every_tool)
        attributed = sum(
            c.tokens for c in counted if c.id in {rc.id for rc in resident_tools}
        )
        overhead = max(0, combined - attributed)

    timestamp = (now or _dt.datetime.now(_dt.timezone.utc)).replace(microsecond=0)
    return Report(
        model=model,
        counter=counter.info,
        components=counted,
        commit=commit,
        ref=ref,
        generated_at=timestamp.isoformat().replace("+00:00", "Z"),
        tool_set_overhead=overhead,
    )


def git_commit(root: Path | None = None) -> str | None:
    """Best-effort current commit sha, preferring the PR head over the merge commit."""
    for env_var in ("GITHUB_SHA",):
        value = os.environ.get(env_var)
        if value:
            # On pull_request events GITHUB_SHA is the ephemeral merge commit, which
            # will never exist again. The head sha is what the report should name.
            head = os.environ.get("GITHUB_HEAD_SHA")
            return head or value
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    return result.stdout.strip() or None


def run(config: Config, counter: Counter) -> Report:
    """Collect and count according to ``config``."""
    func = resolve_entrypoint(config.entrypoint, root=config.root)
    components = call_collector(func, spec=config.entrypoint)
    return count(
        components,
        counter,
        model=config.model,
        commit=git_commit(config.root),
        ref=os.environ.get("GITHUB_REF"),
    )
