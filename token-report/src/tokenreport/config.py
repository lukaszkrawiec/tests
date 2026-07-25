"""Configuration loading.

Config lives in ``tokenreport.toml`` at the repository root, or under
``[tool.tokenreport]`` in ``pyproject.toml``. The model is part of the configuration
because the tokenizer is model-specific: Claude 4.7 and later tokenize roughly 30%
higher than earlier models for identical text, so a count is only meaningful alongside
the model it was taken under.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

DEFAULT_MODEL = "claude-opus-5"
CONFIG_FILENAME = "tokenreport.toml"


class ConfigError(ValueError):
    """Raised when configuration is missing or malformed."""


@dataclass(frozen=True)
class Budgets:
    """Thresholds that turn a report into a pass/fail check.

    Both an absolute ceiling and a growth rate are supported because either alone has
    a blind spot: growth-only lets a prompt creep to 50k in 9% steps, ceiling-only
    gives no signal until the day it hard-fails.
    """

    resident_total: int | None = None
    resident_growth_pct: float | None = None
    on_demand_total: int | None = None
    component: Mapping[str, int] = field(default_factory=dict)

    @property
    def empty(self) -> bool:
        return (
            self.resident_total is None
            and self.resident_growth_pct is None
            and self.on_demand_total is None
            and not self.component
        )

    @classmethod
    def from_raw(cls, raw: Mapping[str, Any]) -> "Budgets":
        component = raw.get("component", {}) or {}
        if not isinstance(component, Mapping):
            raise ConfigError("[budgets.component] must be a table of id = max_tokens")
        parsed: dict[str, int] = {}
        for key, value in component.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ConfigError(
                    f"budget for component {key!r} must be a positive integer, got {value!r}"
                )
            parsed[key] = value

        def positive_int(name: str) -> int | None:
            value = raw.get(name)
            if value is None:
                return None
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ConfigError(f"budgets.{name} must be a positive integer, got {value!r}")
            return value

        growth = raw.get("resident_growth_pct")
        if growth is not None:
            if isinstance(growth, bool) or not isinstance(growth, (int, float)) or growth <= 0:
                raise ConfigError(
                    f"budgets.resident_growth_pct must be a positive number, got {growth!r}"
                )
            growth = float(growth)

        return cls(
            resident_total=positive_int("resident_total"),
            resident_growth_pct=growth,
            on_demand_total=positive_int("on_demand_total"),
            component=parsed,
        )


@dataclass(frozen=True)
class Config:
    entrypoint: str
    model: str = DEFAULT_MODEL
    budgets: Budgets = field(default_factory=Budgets)
    history_branch: str = "token-report"
    history_path: str = "history.json"
    retention_days: int = 90
    retention_max_entries: int = 500
    root: Path = field(default_factory=Path.cwd)

    @classmethod
    def from_raw(cls, raw: Mapping[str, Any], *, root: Path) -> "Config":
        entrypoint = raw.get("entrypoint")
        if not isinstance(entrypoint, str) or ":" not in entrypoint:
            raise ConfigError(
                "entrypoint is required and must look like 'package.module:callable' "
                f"(got {entrypoint!r})"
            )
        model = raw.get("model", DEFAULT_MODEL)
        if not isinstance(model, str) or not model:
            raise ConfigError(f"model must be a non-empty string, got {model!r}")

        budgets_raw = raw.get("budgets", {}) or {}
        if not isinstance(budgets_raw, Mapping):
            raise ConfigError("[budgets] must be a table")

        history = raw.get("history", {}) or {}
        if not isinstance(history, Mapping):
            raise ConfigError("[history] must be a table")

        return cls(
            entrypoint=entrypoint,
            model=model,
            budgets=Budgets.from_raw(budgets_raw),
            history_branch=history.get("branch", "token-report"),
            history_path=history.get("path", "history.json"),
            retention_days=int(history.get("retention_days", 90)),
            retention_max_entries=int(history.get("max_entries", 500)),
            root=root,
        )


def find_config(start: Path | None = None) -> Path:
    """Locate config by walking up from ``start``, preferring tokenreport.toml."""
    current = (start or Path.cwd()).resolve()
    for directory in [current, *current.parents]:
        candidate = directory / CONFIG_FILENAME
        if candidate.is_file():
            return candidate
        pyproject = directory / "pyproject.toml"
        if pyproject.is_file():
            with pyproject.open("rb") as handle:
                data = tomllib.load(handle)
            if "tokenreport" in data.get("tool", {}):
                return pyproject
    raise ConfigError(
        f"no {CONFIG_FILENAME} found in {current} or any parent directory, and no "
        f"[tool.tokenreport] section in a pyproject.toml"
    )


def load(path: Path | None = None) -> Config:
    resolved = path if path is not None else find_config()
    if resolved.is_dir():
        resolved = find_config(resolved)
    with resolved.open("rb") as handle:
        data = tomllib.load(handle)
    if resolved.name == "pyproject.toml":
        raw = data.get("tool", {}).get("tokenreport")
        if raw is None:
            raise ConfigError(f"{resolved} has no [tool.tokenreport] section")
    else:
        raw = data
    return Config.from_raw(raw, root=resolved.parent)
