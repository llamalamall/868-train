"""Typed game-state schema with per-field extraction status."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

FieldStatus = Literal["ok", "missing", "invalid"]


@dataclass(frozen=True)
class FieldState:
    """One extracted state field with explicit status metadata."""

    value: Any | None
    status: FieldStatus
    error_code: str | None = None
    error: str | None = None
    address: int | None = None
    source_field: str | None = None


@dataclass(frozen=True)
class GameStateSnapshot:
    """Normalized state snapshot consumed by control/training loops."""

    timestamp_utc: str
    health: FieldState
    energy: FieldState
    currency: FieldState
    fail_state: FieldState
    extra_fields: dict[str, FieldState] = field(default_factory=dict)
