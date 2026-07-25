"""Token Report — CI accounting for the context an agent application assembles.

Measures the token cost of system prompts, tool schemas, and knowledge-base documents,
separating context that is resident in every request from context fetched only on
demand, and reports growth per commit inside GitHub.
"""

from .contract import (
    SCHEMA_VERSION,
    Component,
    ContractError,
    CountedComponent,
    CounterInfo,
    Kind,
    Report,
    Tier,
    validate,
)

__all__ = [
    "SCHEMA_VERSION",
    "Component",
    "ContractError",
    "CountedComponent",
    "CounterInfo",
    "Kind",
    "Report",
    "Tier",
    "validate",
]
