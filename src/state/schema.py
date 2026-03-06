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
class ProgInventoryItem:
    """One decoded collected prog entry."""

    prog_id: int
    count: int
    name: str | None = None
    known: bool = False
    flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class InventoryState:
    """Decoded inventory/prog section extracted from memory."""

    status: FieldStatus
    collected_progs: tuple[ProgInventoryItem, ...] = ()
    raw_prog_ids: tuple[int, ...] = ()
    unknown_prog_ids: tuple[int, ...] = ()
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
    inventory: InventoryState = field(default_factory=lambda: InventoryState(status="missing"))
    extra_fields: dict[str, FieldState] = field(default_factory=dict)
