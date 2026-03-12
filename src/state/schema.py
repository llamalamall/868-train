"""Typed game-state schema with per-field extraction status."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

FieldStatus = Literal["ok", "missing", "invalid"]
GridIntMap = tuple[tuple[int, ...], ...]
GridProgMap = tuple[tuple[tuple[int, ...], ...], ...]
GridEnemyMap = tuple[tuple[tuple[int, ...], ...], ...]


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
class GridPosition:
    """One logical grid coordinate in the 6x6 map."""

    x: int
    y: int


@dataclass(frozen=True)
class MapCellState:
    """Decoded per-cell map state."""

    position: GridPosition
    cell_type: int
    tile_variant: int
    wall_state: int
    prog_id: int | None = None
    credits: int = 0
    energy: int = 0
    points: int = 0
    threat: int = 0
    special_state: int = 0
    has_siphon: bool = False
    has_exit_overlay: bool = False
    is_wall: bool = False
    is_exit: bool = False


@dataclass(frozen=True)
class WallCellState:
    """Decoded wall metadata for one wall cell."""

    position: GridPosition
    wall_type: Literal["prog_wall", "point_wall", "unknown_wall"]
    wall_state: int
    prog_id: int | None = None
    points: int = 0
    threat: int = 0


@dataclass(frozen=True)
class ResourceCellState:
    """Resources currently available in a non-wall cell."""

    position: GridPosition
    credits: int = 0
    energy: int = 0
    points: int = 0


@dataclass(frozen=True)
class EnemyState:
    """One active hostile entity tracked by the runtime entity table."""

    slot: int
    type_id: int
    position: GridPosition
    hp: int
    state: int
    in_bounds: bool


@dataclass(frozen=True)
class MapLayersState:
    """Layered board representation used by controllers and UI."""

    obstacle_map: GridIntMap = ()
    player_position_map: GridIntMap = ()
    enemy_position_map: GridEnemyMap = ()
    goal_map: GridIntMap = ()
    energy_map: GridIntMap = ()
    credits_map: GridIntMap = ()
    progs_map: GridProgMap = ()
    points_map: GridIntMap = ()
    siphon_penalty_map: GridIntMap = ()


@dataclass(frozen=True)
class MapLayerRefreshState:
    """Flags describing which map-layer groups were recomputed this snapshot."""

    obstacles_updated: bool = True
    player_and_enemy_updated: bool = True
    goals_updated: bool = True
    siphon_outcomes_updated: bool = True


@dataclass(frozen=True)
class MapState:
    """Decoded map section extracted from memory."""

    status: FieldStatus
    width: int = 6
    height: int = 6
    cells: tuple[MapCellState, ...] = ()
    siphons: tuple[GridPosition, ...] = ()
    walls: tuple[WallCellState, ...] = ()
    resource_cells: tuple[ResourceCellState, ...] = ()
    player_position: GridPosition | None = None
    exit_position: GridPosition | None = None
    enemies: tuple[EnemyState, ...] = ()
    layers: MapLayersState = field(default_factory=MapLayersState)
    layer_refresh: MapLayerRefreshState = field(default_factory=MapLayerRefreshState)
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
    map: MapState = field(default_factory=lambda: MapState(status="missing"))
    can_siphon_now: bool | None = None
    prog_slots_available_mask: int | None = None
    extra_fields: dict[str, FieldState] = field(default_factory=dict)
